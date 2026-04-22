from contextlib import nullcontext
from data.utils import get_dataloader

import torch
import torch.distributed
import torch.nn.functional as F
import wandb
import time 
import itertools
import copy
import traceback

import random
import os
import numpy as np
from .utils import eval, get_batch, save_checkpoint


def train_base(model, opt, data, data_seed, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, 
        distributed_backend,extra_args, itr=0,rng_state_dict=None):
    
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=extra_args.dtype)  # extra_args.dtype)
    best_val_loss, text_table = float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 
    substep = itr * acc_steps
    data_train, train_sampler = data["train"]
    
    data_val, val_sampler = data["val"]

    num_substeps_per_epoch = len(data_train)
    train_epochs = substep//num_substeps_per_epoch
    
    if rng_state_dict is not None and  rng_state_dict.get("train_sampler_state", None) is not None:
        train_sampler.generator.set_state(rng_state_dict["train_sampler_state"])
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(train_epochs)
        sampler_state_before_iter = None
    else:
        sampler_state_before_iter = train_sampler.generator.get_state()  
    data_train_iter = iter(data_train)
    
    # for val data we don't care about epochs? just cycle through (no need to set_epoch to reshuffle)
    data_val_iter = itertools.cycle(data_val)

    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}

    # --- Gradient norm tracking for spike detection ---
    # LN-CoTFormer exhibits "benign spikes" (paper §3.3). Tracking grad norms
    # helps diagnose whether spikes originate from exploding gradients (attention)
    # or data anomalies, which informs MLA compression design.
    grad_norm_accumulator = []

   
    if extra_args.compile:
        print(f"Compiling model ...")
        import torch._dynamo as torchdynamo
        torchdynamo.config.guard_nn_modules = True
        model = torch.compile(model) # requires pytorch 2.0+

    model.train()

    t0 = time.time()
    
    if rng_state_dict is not None:
        if "cpu_rng_state" in rng_state_dict:
            torch.set_rng_state(rng_state_dict["cpu_rng_state"])
        if "numpy_rng_state" in rng_state_dict:
            np.random.set_state(rng_state_dict["numpy_rng_state"])
        if "py_rng_state" in rng_state_dict:
            random.setstate(rng_state_dict["py_rng_state"])
        # GPU RNG: per-rank restore from gathered states (new checkpoints),
        # or rank-0-only restore from legacy single-state checkpoints.
        if "gpu_rng_states" in rng_state_dict:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank < len(rng_state_dict["gpu_rng_states"]):
                torch.cuda.set_rng_state(rng_state_dict["gpu_rng_states"][rank])
        elif "gpu_rng_state" in rng_state_dict:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                torch.cuda.set_rng_state(rng_state_dict["gpu_rng_state"])
    for _ in range(substep % num_substeps_per_epoch):
        get_batch(data_train_iter, device=extra_args.device)

    while itr < iterations:
            
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y = get_batch(data_train_iter, device=extra_args.device)
            
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                    if getattr(distributed_backend.get_raw_model(model), "needs_iter", False):
                        outputs = model(x, targets=y, iter=itr)
                    else:
                        outputs = model(x, targets=y)

            loss = outputs['loss'] / acc_steps
            loss.backward()
            substep += 1
            if substep % len(data_train) == 0:
                train_epochs += 1
                print(f"Train epoch {train_epochs} done (full pass over training data)")
                if hasattr(train_sampler, "set_epoch"):
                    # set epoch for reshuffling between epochs
                    train_sampler.set_epoch(train_epochs)
                    sampler_state_before_iter = None
                else:
                    sampler_state_before_iter = train_sampler.generator.get_state()
                data_train_iter = iter(data_train)


        # Compute gradient norm (before clipping) for diagnostics
        if extra_args.grad_clip != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        grad_norm_accumulator.append(grad_norm)
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        # Lightweight progress line between full evals (every 10 steps)
        if itr % 10 == 0 and itr % eval_freq != 0 and distributed_backend.is_master_process():
            elapsed = time.time() - t0
            ms_per_step = elapsed * 1000 / (itr % eval_freq or 10)
            steps_left = iterations - itr
            eta_h = (steps_left * ms_per_step) / 3.6e6
            print(f"  step {itr}/{iterations}  loss={loss.item() * acc_steps:.3f}  grad={grad_norm:.3f}  {ms_per_step:.0f}ms/step  ETA ~{eta_h:.1f}h", flush=True)

        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_substeps_per_epoch

                model.eval()
                train_loss = loss.detach().cpu().item() * acc_steps
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                eval_steps = (
                    24 if itr < iterations else len(data_val)
                )
                # If we are at the last iteration, re-initialize the data iterator
                if itr == iterations:
                    data_val_iter = iter(data_val)

                val_acc, val_loss, val_perplexity, avg_depth = eval(
                    distributed_backend.get_raw_model(model),
                    data_val_iter,
                    extra_args.device,
                    max_num_batches=eval_steps,
                    ctx=type_ctx,
                )

                # Gradient norm stats over the eval interval
                recent_norms = grad_norm_accumulator[-eval_freq:]
                cur_grad_norm = recent_norms[-1] if recent_norms else 0.0
                max_grad_norm = max(recent_norms) if recent_norms else 0.0
                mean_grad_norm = sum(recent_norms) / len(recent_norms) if recent_norms else 0.0

                print_string = f"{epoch}/{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}, avg_depth={(avg_depth or extra_args.n_layer):.3f}"
                print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print_string += f" [grad_norm] {cur_grad_norm:.4f} [max_grad_norm] {max_grad_norm:.4f}"
                print(print_string)

                stats["train_loss"].append(train_loss)
                stats["val_loss"].append(val_loss)
                stats["val_pp"].append(val_perplexity)
                stats["val_acc"].append(val_acc)

                # Flag spikes: grad norm > 5x running mean (benign spikes in LN-CoTFormer)
                if len(grad_norm_accumulator) > eval_freq and max_grad_norm > 5 * mean_grad_norm:
                    print(f"  >> SPIKE WARNING: max_grad_norm={max_grad_norm:.4f} is {max_grad_norm/mean_grad_norm:.1f}x the mean ({mean_grad_norm:.4f})")

                if extra_args.wandb:
                    logs = {
                        "iter": itr,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "val/avg_depth": avg_depth or extra_args.n_layer,
                        "lr": current_lr,
                        "train/grad_norm": cur_grad_norm,
                        "train/max_grad_norm": max_grad_norm,
                        "train/mean_grad_norm": mean_grad_norm,
                    }

                    if itr == iterations:
                        logs["val/final-ppl"] = val_perplexity
                        logs["val/final-acc"] = val_acc
                        logs["val/final-loss"] = val_loss

                    wandb.log(logs)

                    if extra_args.eval_seq_prefix != 'none' and (itr % (eval_freq * 5) == 0 or itr == iterations):
                        if text_table is None:
                            text_table = wandb.Table(columns=["itr", "val-pp", "text"])
                        if False:
                            try:
                                out_str = distributed_backend.get_raw_model(model).generate_from_string(
                                    extra_args.eval_seq_prefix, max_new_tokens=40, temperature=0.9, top_k=None)
                                text_table.add_data(itr, val_perplexity, out_str)
                                # why a copy? see github.com/wandb/wandb/issues/2981
                                wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
                            except Exception as e:
                                print(e)
                                try:
                                    traceback.print_exc()
                                except:
                                    pass

                model.train()
                t0 = time.time()
        # Checkpoint save: all ranks evaluate the frequency check together
        # because the GPU RNG gather is a collective op.
        if extra_args.save_checkpoint_freq is not None and itr % extra_args.save_checkpoint_freq == 0:
            # Gather per-rank CUDA RNG states via Gloo (CPU tensors, no NCCL).
            # Previous versions used NCCL all_gather on CUDA ByteTensors, which
            # triggered an LL-protocol hang after ~52K gradient AllReduces on
            # NCCL 2.15-2.17 / CUDA 11.8 (see reprod-notes B9).
            if distributed_backend.get_world_size() > 1:
                local_rng = torch.cuda.get_rng_state()  # CPU ByteTensor, fixed size
                _gpu_rng_gathered = [torch.empty_like(local_rng) for _ in range(distributed_backend.get_world_size())]
                torch.distributed.all_gather(_gpu_rng_gathered, local_rng, group=distributed_backend.cpu_group)
            else:
                _gpu_rng_gathered = [torch.cuda.get_rng_state()]

            if distributed_backend.is_master_process():
                print(f"saving checkpoint to {ckpt_path}/ckpt_{itr}.pt")
                save_checkpoint(distributed_backend=distributed_backend,
                                model=model,
                                opt=opt,
                                scheduler=scheduler,
                                itr=itr,
                                cpu_rng_state=torch.get_rng_state(),
                                gpu_rng_states=_gpu_rng_gathered,
                                numpy_rng_state=np.random.get_state(),
                                py_rng_state=random.getstate(),
                                train_sampler_state=sampler_state_before_iter,
                                ckpt_path=os.path.join(ckpt_path, f"ckpt_{itr}.pt"))

            # Barrier: wait for rank 0 torch.save before next training step.
            # Without this, non-master ranks race to the next gradient ALLREDUCE
            # while rank 0 is still writing to disk (see reprod-notes B9).
            if distributed_backend.get_world_size() > 1:
                distributed_backend.sync()

    # Final checkpoint — gather GPU RNG from all ranks before master writes.
    # Gloo (CPU) gather — same rationale as periodic checkpoint above (see reprod-notes B9).
    if distributed_backend.get_world_size() > 1:
        local_rng = torch.cuda.get_rng_state()  # CPU ByteTensor, fixed size
        _gpu_rng_gathered = [torch.empty_like(local_rng) for _ in range(distributed_backend.get_world_size())]
        torch.distributed.all_gather(_gpu_rng_gathered, local_rng, group=distributed_backend.cpu_group)
    else:
        _gpu_rng_gathered = [torch.cuda.get_rng_state()]

    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(distributed_backend=distributed_backend,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        itr=itr,
                        cpu_rng_state=torch.get_rng_state(),
                        gpu_rng_states=_gpu_rng_gathered,
                        numpy_rng_state=np.random.get_state(),
                        py_rng_state=random.getstate(),
                        train_sampler_state=sampler_state_before_iter,
                        ckpt_path=f"{ckpt_path}/ckpt.pt")
        if extra_args.remove_intermediary_checkpoints_at_end:
            for file_ in os.listdir(ckpt_path):
                 if 'ckpt_' in file_:
                    os.remove(os.path.join(ckpt_path, file_))

    # Barrier: wait for rank 0 final save before process cleanup (see reprod-notes B9).
    if distributed_backend.get_world_size() > 1:
        distributed_backend.sync()

    return stats
