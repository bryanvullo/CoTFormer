import os
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(dataloader, device="cpu"):
    x, y = next(dataloader)
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def eval(model, data_val_iter, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    avg_depth_list = []

    for _ in range(max_num_batches): 
        x, y = get_batch(data_val_iter, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['cross_entropy_loss'] if 'cross_entropy_loss' in outputs else outputs['loss'] 
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())
        if 'average_depth' in outputs:
            avg_depth_list.append(outputs['average_depth'])

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss
    avg_depth = torch.stack(avg_depth_list).float().mean().item() if avg_depth_list else None

    return val_acc, val_loss, val_perplexity, avg_depth

def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    # Save to local /tmp first, then atomic-copy to target (Lustre).
    # torch.save() uses zip serialization that buffers the entire state dict
    # before flushing — a single ~2 GB write to Lustre can stall for 30+ min
    # under cluster I/O contention, causing the post-save DDP barrier to
    # timeout (see reprod-notes B11).
    import shutil, tempfile, time

    t0 = time.time()
    with tempfile.NamedTemporaryFile(dir='/tmp', suffix='.pt', delete=False) as tmp:
        tmp_path = tmp.name
    torch.save(checkpoint, tmp_path)

    ckpt_tmp = ckpt_path + '.tmp'
    shutil.copy2(tmp_path, ckpt_tmp)
    os.rename(ckpt_tmp, ckpt_path)
    os.remove(tmp_path)

    elapsed = time.time() - t0
    print(f"  checkpoint saved in {elapsed:.1f}s ({os.path.getsize(ckpt_path) / 1e6:.0f} MB)")
