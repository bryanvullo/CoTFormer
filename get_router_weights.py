"""Extract router weight distributions from a trained Adaptive LN-CoTFormer.

Hooks into MoDBlock.mod_router (nn.Linear) to capture raw router logits BEFORE
top-k selection, then applies sigmoid to get the actual router scores for ALL
tokens at each repeat.

Output:
    <checkpoint_dir>/router_weights.npy  -- numpy object array of n_repeat elements:
        [None, scores_repeat_1, scores_repeat_2, ..., scores_repeat_{n-1}]
    Each element is a 1-D float32 array of sigmoid scores (0=halt, 1=continue).
    Cascaded minimums are applied: a token halted at repeat 2 stays halted at 3+.

    Also saves router_logits.npz with raw logits per router (for threshold sweeps).

Usage:
    python get_router_weights.py --checkpoint /path/to/ckpt_dir
    python get_router_weights.py --checkpoint /path/to/ckpt_dir/ckpt_40000.pt
"""
import argparse
import numpy as np
from tqdm import tqdm
from data.utils import get_dataset, prepare_dataset
from contextlib import nullcontext

import torch
import models
import json
import os
import sys
import random
import distributed


def none_or_str(value):
    if value == 'None':
        return None
    return value


def iceildiv(x, y):
    return (x + y - 1) // y


def get_as_batch(data, seq_length, batch_size, device='cpu', sample_size=None):
    all_ix = list(range(0, len(data), seq_length))
    assert all_ix[-1] + seq_length + 1 > len(data)
    all_ix.pop()
    if sample_size is not None:
        all_ix = np.random.choice(all_ix, size=sample_size // seq_length, replace=False).tolist()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([i + seq_length + 1 <= len(data) for i in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.device = torch.device(args.device)
    torch.cuda.set_device(args.device)
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset '{args.dataset}'")
    if distributed_backend.is_master_process():
        prepare_dataset(args)
    distributed_backend.sync()
    data = get_dataset(args)
    print(f"Num validation tokens: {len(data['val'])}")

    model = models.make_model_from_args(args).to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(os.path.join(args.checkpoint, args.checkpoint_filename))
        model.load_state_dict(
            {k: v for k, v in checkpoint['model'].items()
             if "attn.bias" not in k and "wpe" not in k},
            strict=False)

    raw_model = model  # keep unwrapped ref for hook registration

    # --- Verify this is an adaptive model with MoDBlocks ---
    if not hasattr(raw_model, 'transformer') or not hasattr(raw_model.transformer, 'mod'):
        print("ERROR: Model does not have a router (transformer.mod).")
        print("       This script is for adaptive CoTFormer models only.")
        sys.exit(1)

    n_routers = len(raw_model.transformer.mod)
    n_repeat = raw_model.n_repeat
    print(f"Found {n_routers} MoDBlock routers (n_repeat={n_repeat})")

    # --- Register hooks on each MoDBlock.mod_router (nn.Linear) ---
    # The Linear output is the raw router logits: (B, T_active, 1)
    # With eval capacity_factor=1.0, T_active=T for all routers (no top-k pruning).
    router_logits_per_repeat = {i: [] for i in range(n_routers)}

    def make_hook(router_idx):
        def hook_fn(module, input, output):
            # output shape: (B, T_active, 1) -- raw logits before top-k + sigmoid
            router_logits_per_repeat[router_idx].append(
                output.detach().squeeze(-1).cpu()  # (B, T_active)
            )
        return hook_fn

    hooks = []
    for i, mod_block in enumerate(raw_model.transformer.mod):
        h = mod_block.mod_router.register_forward_hook(make_hook(i))
        hooks.append(h)

    # --- Forward pass on VALIDATION data ---
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=args.dtype)
    seq_length = getattr(args, 'eval_seq_length', None) or args.sequence_length

    model.eval()
    total_batches = iceildiv(
        iceildiv(len(data['val']), seq_length),
        args.batch_size
    )

    print(f"Running forward pass on val data ({total_batches} batches, bs={args.batch_size})...")
    with torch.no_grad():
        for x, y in tqdm(
            get_as_batch(data['val'], seq_length, args.batch_size,
                         device=args.device),
            total=total_batches
        ):
            with type_ctx:
                _ = model(x, targets=y, get_logits=False)

    # --- Remove hooks ---
    for h in hooks:
        h.remove()

    # --- Post-process: build per-repeat score arrays ---
    # Router i (0-indexed) decides whether tokens proceed from repeat i+1 to i+2.
    # Apply sigmoid to get scores, then cascade minimums:
    #   effective_score[repeat_k] = min(score[1], score[2], ..., score[k])
    # A token halted at repeat 2 (low score) stays halted at repeats 3, 4, 5.
    raw_logits = {}
    raw_scores = {}
    for i in range(n_routers):
        # Verify T_active is uniform across batches (requires capacity_factor=1.0)
        shapes = [t.shape[1] for t in router_logits_per_repeat[i]]
        assert len(set(shapes)) == 1, (
            f"Router {i} has variable T_active across batches ({set(shapes)}). "
            f"Ensure eval uses capacity_factor=1.0 (eval_length_factor=[1]*{n_routers})."
        )
        all_logits = torch.cat(router_logits_per_repeat[i], dim=0)  # (N_batches*B, T)
        raw_logits[i] = all_logits.reshape(-1).numpy()
        raw_scores[i] = torch.sigmoid(all_logits).reshape(-1).numpy()

    # Build the cascaded output array: [None, scores_1, scores_2, ..., scores_{n-1}]
    # Matches the repeat indexing: element 0 = repeat 1 (no router), element k = repeat k+1
    all_router_weights = [None]  # repeat 1 has no router
    for i in range(n_routers):
        if i == 0:
            cascaded = raw_scores[i].copy()
        else:
            cascaded = np.minimum(raw_scores[i], all_router_weights[-1])
        all_router_weights.append(cascaded)

    # --- Print summary stats ---
    for i in range(n_routers):
        scores = all_router_weights[i + 1]
        print(f"  Router {i+1} (repeat {i+1}->{i+2}): "
              f"n={len(scores)}, mean={scores.mean():.4f}, "
              f"median={np.median(scores):.4f}, "
              f"frac<0.1={np.mean(scores < 0.1):.3f}, "
              f"frac>0.9={np.mean(scores > 0.9):.3f}")

    # The last router's cascaded scores are what Figure 5 plots
    last_scores = all_router_weights[-1]
    print(f"\n  Figure 5 data (last repeat entry, n={len(last_scores)}):")
    print(f"    frac_score=0 (halted): {np.mean(last_scores < 0.01):.3f}")
    print(f"    frac_score>0.5 (active): {np.mean(last_scores > 0.5):.3f}")

    # --- Save ---
    output_dir = getattr(args, 'output_dir', None) or args.checkpoint
    os.makedirs(output_dir, exist_ok=True)

    # Legacy format: numpy object array (adm-exp4/exp5 job.sh expect this)
    weights_path = os.path.join(output_dir, "router_weights.npy")
    np.save(weights_path, np.array(all_router_weights, dtype=object))
    print(f"\nRouter weights saved to: {weights_path}")

    # Also save raw logits for threshold sweep experiments (adm-exp4 phase 2)
    logits_path = os.path.join(output_dir, "router_logits.npz")
    np.savez(logits_path, **{f'router_{i+1}': raw_logits[i] for i in range(n_routers)})
    print(f"Raw logits saved to: {logits_path}")

    distributed_backend.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--checkpoint', type=none_or_str, required=True)
    parser.add_argument('--config_format', type=str, required=False)
    parser.add_argument('--output_dir', type=none_or_str, default=None)

    args, rem_args = parser.parse_known_args()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            args.checkpoint, args.checkpoint_filename = os.path.split(args.checkpoint)
        else:
            args.checkpoint_filename = "ckpt.pt"

        with open(os.path.join(args.checkpoint, "summary.json")) as f:
            summary = json.load(f)

        for k, v in summary['args'].items():
            if k == "config_format" and args.config_format is not None:
                continue
            if k not in ["device", "dtype", "distributed_backend"]:
                setattr(args, k, v)

    import config as cfg_module
    args = cfg_module.parse_args_with_format(
        format=args.config_format,
        base_parser=argparse.ArgumentParser(allow_abbrev=False),
        args=rem_args, namespace=args)

    main(args)
