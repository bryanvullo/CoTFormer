"""Extract router weight distributions using the AUTHORS' ORIGINAL method.

This script is a faithful reproduction of the original get_router_weights.py
from commit 2723e30 (the paper's authors), adapted only for CLI compatibility
with our evaluation pipeline (--split, --distributed_backend, --output_dir).

WHY THIS EXISTS:
    The original code contains a cascading bug that produces the SAME distribution
    shown in Figure 5 of the paper. When the MoDBlock selects tokens via top-k,
    it REORDERS them by score. The original cascading logic takes element-wise
    minimums at the same POSITION INDEX across repeats, but after reordering,
    position k at repeat i corresponds to a DIFFERENT token than position k at
    repeat i-1. The result is a minimum of randomly-paired scores across tokens,
    producing a broader distribution than correct per-token cascading.

    Our rewritten get_router_weights.py (hook-based, commit a9df8fd) fixes this
    bug by hooking into mod_router directly and cascading per-logit. The correct
    version produces distributions concentrated near zero (density ~29 at x=0),
    while the buggy version produces the broader distribution seen in the paper
    (density ~0.6, tail extending to x=0.5).

    We retain this buggy extraction for scientific reproducibility: it recovers
    the paper's Figure 5 distribution, confirming that the paper's figure is an
    artifact of this cascading bug. See reprod-notes.md C1 for full analysis.

DIFFERENCES FROM ORIGINAL (commit 2723e30):
    1. Added --split (train/val) argument (original hardcoded data['train'])
    2. Added --distributed_backend None support (original had no distributed)
    3. Added --output_dir to control save location
    4. Uses args.seed instead of hardcoded seed=0
    5. Minor formatting/comments -- logic is identical

Output:
    <output_dir>/router_weights.npy  -- numpy object array of n_repeat elements.
    The cascading is POSITION-INDEXED (the bug): element-wise min across repeats
    after token reordering, so scores at the same array index correspond to
    DIFFERENT tokens. This produces the same distribution as the paper's Figure 5.

Usage:
    python broken_get_router_weights.py --checkpoint /path/to/ckpt_dir
    python broken_get_router_weights.py --checkpoint /path/to/ckpt_dir --split val
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
        ix = all_ix[idx:idx + batch_size]
        assert all([i + seq_length + 1 <= len(data) for i in ix])
        x = torch.stack([torch.from_numpy((data[i:i + seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y


def custom_forward(model, idx, targets):
    """Replicate the model's forward pass, capturing router weights per repeat.

    This is the authors' original extraction method. It manually runs the
    model's repeat loop with capacity_factor=1.0 (all tokens selected at each
    router), collecting the MoDBlock's returned router_weights tensors.

    The returned router_weights[i] contains sigmoid scores for the tokens as
    they are ordered AFTER top-k reordering at repeat i. Because top-k
    reorders tokens by score, position k at repeat i is a different token
    than position k at repeat i-1. This is the source of the cascading bug.
    """
    device = idx.device
    b, t = idx.size()

    # Forward: embeddings
    index_shift = 0
    cache_context = None
    if getattr(model.transformer.wpe, "needs_iter", False):
        idx, pos_emb_closure = model.transformer.wpe(idx, iter=None)
    else:
        idx, pos_emb_closure = model.transformer.wpe(idx)
    x = model.transformer.wte(idx)
    x = model.transformer.drop(x)
    x = pos_emb_closure.adapt_model_input(x, start_index=index_shift)

    # Forward: prefix blocks
    for block in model.transformer.h_begin:
        x = block(x, pos_emb_closure, cache_context, start_index=index_shift)

    B, T, D = x.shape
    active_indices = (index_shift + torch.arange(T, device=x.device)).unsqueeze(0).repeat(B, 1).view(B, T)

    router_weights = None
    total_expected_length = T * model.n_repeat
    for block in model.transformer.h_mid:
        block.attn.init_cache(total_expected_length)

    all_router_weights = []
    for rep_idx in range(1, model.n_repeat + 1):
        x_in = x
        if model.depth_emb is not None:
            x = model.depth_emb(x, indices=torch.full_like(active_indices, model.n_repeat - rep_idx))
        for block in model.transformer.h_mid:
            x = block(x, pos_emb_closure, cache_context, start_index=None, indices=active_indices)
        x = model.transformer.ln_mid(x)
        if router_weights is not None:
            x = x_in * (1 - router_weights) + x * router_weights
        all_router_weights.append(router_weights)

        if rep_idx < model.n_repeat:
            # BUG SITE: MoDBlock returns weights for tokens in TOP-K order.
            # With capacity_factor=1.0, all T tokens are selected but their
            # order within the returned tensors depends on topk(sorted=False).
            # The subsequent take_along_dim reorders x, so the NEXT repeat's
            # router operates on differently-ordered tokens.
            is_final, selected_indices, router_weights = model.transformer.mod[rep_idx - 1](x, capacity_factor=1.0)
        else:
            is_final = x.new_ones((B, x.shape[1])) == 1.
            selected_indices = x.new_ones((B, 0, 1)).long()
            router_weights = None

        # Reorder tokens by router score for next repeat
        x = x.take_along_dim(selected_indices, dim=1)
        active_indices = active_indices.take_along_dim(selected_indices.squeeze(2), dim=1)

    for block in model.transformer.h_mid:
        block.attn.drop_cache()

    return all_router_weights


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
    split = args.split
    print(f"Running on '{split}' split ({len(data[split])} tokens)")

    model = models.make_model_from_args(args).to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(os.path.join(args.checkpoint, args.checkpoint_filename))
        model.load_state_dict(
            {k: v for k, v in checkpoint['model'].items()
             if "attn.bias" not in k and "wpe" not in k},
            strict=False)

    model.eval()

    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=args.dtype)
    seq_length = getattr(args, 'eval_seq_length', None) or args.sequence_length

    # Subsample train to val size (matches original code's convention)
    sample_size = len(data['val']) if split == 'train' else None
    eval_tokens = len(data['val'])
    total_batches = iceildiv(
        iceildiv(eval_tokens, seq_length),
        args.batch_size
    )

    # Accumulate router weights across batches
    all_router_weights = [[] for _ in range(model.n_repeat)]

    print(f"Running ORIGINAL (buggy) forward pass on {split} data "
          f"({total_batches} batches, bs={args.batch_size})...")
    for x, y in tqdm(
        get_as_batch(data[split], seq_length, args.batch_size,
                     device=args.device, sample_size=sample_size),
        total=total_batches
    ):
        with torch.no_grad():
            with type_ctx:
                router_weights = custom_forward(model, x, y)

        # CASCADING BUG: take element-wise minimum at the same POSITION INDEX
        # across repeats. After top-k reordering, position k at repeat i is a
        # DIFFERENT token than position k at repeat i-1. This compares scores
        # of randomly-paired tokens, producing a broader distribution than
        # correct per-token cascading.
        for i in range(2, len(router_weights)):
            for j in range(len(router_weights[i])):
                router_weights[i][j] = torch.minimum(
                    router_weights[i][j], router_weights[i - 1][j]
                )

        for rw_list, rw_tensor in zip(all_router_weights, router_weights):
            if rw_tensor is not None:
                rw_list += rw_tensor.detach().float().view(-1).tolist()

    all_router_weights = [
        np.array(x, dtype=np.float32) if len(x) > 0 else None
        for x in all_router_weights
    ]

    # Print summary stats
    for i in range(1, len(all_router_weights)):
        scores = all_router_weights[i]
        if scores is not None:
            print(f"  Router {i} (repeat {i}->{i+1}): "
                  f"n={len(scores)}, mean={scores.mean():.4f}, "
                  f"median={np.median(scores):.4f}, "
                  f"frac<0.1={np.mean(scores < 0.1):.3f}, "
                  f"frac>0.9={np.mean(scores > 0.9):.3f}")

    last_scores = all_router_weights[-1]
    if last_scores is not None:
        print(f"\n  Figure 5 data (last repeat, n={len(last_scores)}):")
        print(f"    frac_score=0 (halted): {np.mean(last_scores < 0.01):.3f}")
        print(f"    frac_score>0.5 (active): {np.mean(last_scores > 0.5):.3f}")

    # Save
    output_dir = getattr(args, 'output_dir', None) or args.checkpoint
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, "router_weights.npy")
    np.save(weights_path, np.array(all_router_weights, dtype=object))
    print(f"\nRouter weights (buggy cascaded) saved to: {weights_path}")

    distributed_backend.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--checkpoint', type=none_or_str, required=True)
    parser.add_argument('--config_format', type=str, required=False)
    parser.add_argument('--output_dir', type=none_or_str, default=None)
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Data split (default: train, matching original)')

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
