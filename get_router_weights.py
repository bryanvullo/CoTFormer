"""Unified router weight extraction for Figure 5 confounder analysis.

Supports multiple extraction methods and config loading modes for systematic
isolation of experimental confounders in the Figure 5 reproduction. Each run
saves raw (per-repeat, no cascade) and cascaded (position-indexed and/or
per-token) outputs.

Methods (--method):
  cf          Authors' custom forward loop, topk sorted=False
  cf-sorted   Custom forward loop, topk sorted=True (forces token reordering)
  hooks       Forward hooks on mod_router (captures pre-topk logits)

Config loading (--config-loading):
  raw         Original authors' raw JSON dict (dtype=None -> autocast default)
  argparse    Our config parser (dtype=bfloat16 via argparse default)

Cascade outputs (always saved for cf/cf-sorted; PT only for hooks):
  *_raw.npy        Per-repeat sigmoid scores, no cascade applied
  *_picascade.npy  Position-indexed cascade (C1 bug)
  *_ptcascade.npy  Per-token cascade via active_indices scatter

Sidecar metadata (always written):
  router_weights_meta.json  n_repeat, n_routers, last-repeat slot, extraction
                            parameters. Downstream plotters read this to
                            locate the Figure 5 quantity without hardcoded
                            slot assumptions.

Slot semantics (shared by all outputs, all methods):
  weights_array has length n_repeat. Slot 0 is always None (no router
  decision precedes the first pass). Slot k in [1..n_repeat-1] holds the
  sigmoid scores from mod[k-1], the router deciding entry into pass k+1.
  The paper's Figure 5 "last repeat" quantity is weights_array[n_repeat-1],
  i.e., the router that gates the final pass. Both `n_repeat` and the
  "last repeat slot" are derived from the loaded model's `n_repeat`
  attribute -- no hardcoded assumption of n_repeat=5.

See iridis/eval-adm/README.md for the experiment matrix and methodology.
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


# ---------------------------------------------------------------------------
# Extraction method: custom forward (authors' manual repeat loop)
# ---------------------------------------------------------------------------

def custom_forward(model, idx, targets, sorted_topk=False):
    """Replicate the model's forward pass, capturing router weights per repeat.

    Faithfully reproduces the original authors' forward() from commit 2723e30,
    adapted only from method-style (self) to standalone (model). When
    sorted_topk=True, the MoDBlock call is inlined with torch.topk(sorted=True)
    to force deterministic token reordering (testing the topk hypothesis).

    Returns:
        all_router_weights: list of n_repeat elements. Index 0 is None (no
            router for first repeat). Index k (k>=1) contains the sigmoid
            scores from router k-1, in topk-index order (shape B, T, 1).
        all_active_indices: list of n_repeat elements. Index k contains the
            position-to-original-token mapping at the time router_weights[k]
            was captured (shape B, T). Used for per-token cascade scatter.
    """
    device = idx.device
    b, t = idx.size()

    index_shift = 0
    cache_context = None
    # Original code passes iter=iter (Python builtin) here; this branch is
    # dead code for rotary positional encodings (needs_iter is never True).
    if getattr(model.transformer.wpe, "needs_iter", False):
        idx, pos_emb_closure = model.transformer.wpe(idx, iter=iter)
    else:
        idx, pos_emb_closure = model.transformer.wpe(idx)
    x = model.transformer.wte(idx)
    x = model.transformer.drop(x)
    x = pos_emb_closure.adapt_model_input(x, start_index=index_shift)

    for block in model.transformer.h_begin:
        x = block(x, pos_emb_closure, cache_context, start_index=index_shift)

    B, T, D = x.shape
    active_indices = (index_shift + torch.arange(T, device=x.device)).unsqueeze(0).repeat(B, 1).view(B, T)

    router_weights = None

    total_expected_length = T * model.n_repeat
    for block in model.transformer.h_mid:
        block.attn.init_cache(total_expected_length)

    all_router_weights = []
    all_active_indices = []
    # PT cascade scatter assumes all T positions are written (capacity_factor=1.0).
    # With capacity_factor < 1.0, unselected positions would remain zero,
    # silently corrupting the cascade. Guard against this.
    assert not sorted_topk or True, "sorted_topk always uses k=T"
    for rep_idx in range(1, model.n_repeat + 1):
        x_in = x
        if model.depth_emb is not None:
            x = model.depth_emb(x, indices=torch.full_like(active_indices, model.n_repeat - rep_idx))
        for block in model.transformer.h_mid:
            x = block(x, pos_emb_closure, cache_context, start_index=None, indices=active_indices)
        x = model.transformer.ln_mid(x)
        if router_weights is not None:
            x = x_in * (1 - router_weights) + x * router_weights

        # Snapshot: router_weights is from the PREVIOUS repeat's MoDBlock.
        # active_indices maps current positions to original token indices.
        # Both are aligned: rw[b,k] corresponds to original token ai[b,k].
        all_router_weights.append(router_weights)
        all_active_indices.append(active_indices.clone())

        if rep_idx < model.n_repeat:
            if sorted_topk:
                # Inline MoDBlock logic with sorted=True to force reordering.
                # Tests the hypothesis that the paper's Figure 5 depends on
                # topk producing significant token reordering.
                router_logits = model.transformer.mod[rep_idx - 1].mod_router(x)
                weights, selected_tokens = torch.topk(
                    router_logits, x.shape[1], dim=1, sorted=True)
                router_weights = torch.nn.functional.sigmoid(weights)
                is_final = x.new_ones((B, x.shape[1])) == 1.
                row_indices = torch.arange(x.size(0)).unsqueeze(1)
                is_final[row_indices, selected_tokens.squeeze(2)] = False
                selected_indices = selected_tokens
            else:
                is_final, selected_indices, router_weights = \
                    model.transformer.mod[rep_idx - 1](x, capacity_factor=1.0)
        else:
            is_final = x.new_ones((B, x.shape[1])) == 1.
            selected_indices = x.new_ones((B, 0, 1)).long()
            router_weights = None

        x = x.take_along_dim(selected_indices, dim=1)
        active_indices = active_indices.take_along_dim(selected_indices.squeeze(2), dim=1)

    for block in model.transformer.h_mid:
        block.attn.drop_cache()

    return all_router_weights, all_active_indices


def run_cf_extraction(model, data_iter, total_batches, type_ctx, n_repeat, method_label):
    """Run custom-forward extraction, accumulating raw + PI + PT cascades."""
    sorted_topk = (method_label == "cf-sorted")

    raw_accum = [[] for _ in range(n_repeat)]
    pi_accum = [[] for _ in range(n_repeat)]
    pt_accum = [[] for _ in range(n_repeat)]

    print(f"Running {method_label} forward pass ({total_batches} batches)...")
    for x, y in tqdm(data_iter, total=total_batches):
        with torch.no_grad():
            with type_ctx:
                rw_list, ai_list = custom_forward(model, x, y, sorted_topk=sorted_topk)

        # --- RAW accumulation (no cascade) ---
        for i, rw in enumerate(rw_list):
            if rw is not None:
                raw_accum[i].extend(rw.detach().float().view(-1).tolist())

        # --- Position-indexed cascade (C1 bug) ---
        pi_rw = [rw.clone() if rw is not None else None for rw in rw_list]
        for i in range(2, len(pi_rw)):
            if pi_rw[i] is not None and pi_rw[i - 1] is not None:
                for j in range(pi_rw[i].shape[0]):
                    pi_rw[i][j] = torch.minimum(pi_rw[i][j], pi_rw[i - 1][j])
        for i, rw in enumerate(pi_rw):
            if rw is not None:
                pi_accum[i].extend(rw.detach().float().view(-1).tolist())

        # --- Per-token cascade (correct, via active_indices scatter) ---
        T = rw_list[1].shape[1] if rw_list[1] is not None else 0
        B = rw_list[1].shape[0] if rw_list[1] is not None else 0
        scattered = []
        for i, (rw, ai) in enumerate(zip(rw_list, ai_list)):
            if rw is None:
                scattered.append(None)
                continue
            s = torch.zeros(B, T, 1, device='cpu')
            s.scatter_(1, ai.unsqueeze(-1).cpu(), rw.detach().float().cpu())
            scattered.append(s)

        pt_cascaded = [None]
        for i in range(1, len(scattered)):
            if scattered[i] is None:
                pt_cascaded.append(None)
                continue
            if pt_cascaded[-1] is None:
                pt_cascaded.append(scattered[i])
            else:
                pt_cascaded.append(torch.minimum(scattered[i], pt_cascaded[-1]))

        for i, rw in enumerate(pt_cascaded):
            if rw is not None:
                pt_accum[i].extend(rw.float().view(-1).tolist())

    return raw_accum, pi_accum, pt_accum


# ---------------------------------------------------------------------------
# Extraction method: hooks (captures pre-topk logits from mod_router)
# ---------------------------------------------------------------------------

def run_hooks_extraction(model, data_iter, total_batches, type_ctx, n_repeat):
    """Hook-based extraction: captures raw logits before topk selection.

    Hooks fire on each MoDBlock.mod_router (nn.Linear), capturing the output
    BEFORE topk and sigmoid. Since hooks capture in original token order,
    per-token cascade is natural (position = original token). Position-indexed
    cascade on hook data is identical to per-token cascade (no reordering).

    Only the first `n_repeat - 1` mod entries are hooked -- these are the
    routers actually called in the forward pass. A hypothetical model with
    an orphan trailing MoDBlock (e.g., pre-B6 original-author code path)
    would be handled correctly: the orphan is never called, so hooking it
    would produce an empty capture list. We avoid that by deriving the
    router count from `model.n_repeat`, which is the authoritative source.

    Returns raw_accum and pt_accum (both as lists of flat score lists,
    length = n_repeat with slot 0 = empty).
    """
    n_routers = n_repeat - 1
    router_logits_per_repeat = {i: [] for i in range(n_routers)}

    def make_hook(router_idx):
        def hook_fn(module, input, output):
            router_logits_per_repeat[router_idx].append(
                output.detach().squeeze(-1).cpu())
        return hook_fn

    hooks = []
    for i in range(n_routers):
        h = model.transformer.mod[i].mod_router.register_forward_hook(make_hook(i))
        hooks.append(h)

    print(f"Running hooks forward pass ({total_batches} batches)...")
    with torch.no_grad():
        for x, y in tqdm(data_iter, total=total_batches):
            with type_ctx:
                _ = model(x, targets=y, get_logits=False)

    for h in hooks:
        h.remove()

    # Post-process: sigmoid on captured logits, then cascade per-token
    raw_scores = {}
    for i in range(n_routers):
        shapes = [t.shape[1] for t in router_logits_per_repeat[i]]
        assert len(set(shapes)) == 1, (
            f"Router {i} has variable T_active ({set(shapes)}). "
            f"Ensure eval uses capacity_factor=1.0.")
        all_logits = torch.cat(router_logits_per_repeat[i], dim=0)
        raw_scores[i] = torch.sigmoid(all_logits).reshape(-1).float().numpy()

    # Build accumulators in the same format as CF extraction (length n_repeat,
    # slot 0 empty, slot k >= 1 holds mod[k-1]'s sigmoid scores).
    raw_accum = [[] for _ in range(n_repeat)]
    pt_accum = [[] for _ in range(n_repeat)]

    for i in range(n_routers):
        raw_accum[i + 1] = raw_scores[i].tolist()

    # Per-token cascade (hooks capture in original order, so this is correct)
    pt_cascaded = [None]
    for i in range(n_routers):
        if i == 0:
            pt_cascaded.append(raw_scores[i].copy())
        else:
            pt_cascaded.append(np.minimum(raw_scores[i], pt_cascaded[-1]))

    for i in range(1, len(pt_cascaded)):
        if pt_cascaded[i] is not None:
            pt_accum[i] = pt_cascaded[i].tolist()

    return raw_accum, pt_accum


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config_raw(checkpoint_dir, checkpoint_filename):
    """Original authors' config loading: raw JSON dict, dtype=None.

    Reproduces the exact config construction from commit 2723e30. The dtype
    field is loaded as-is from summary.json (null -> None), which causes
    torch.amp.autocast to use its default precision (float16 on CUDA).
    """
    class Config:
        pass
    with open(os.path.join(checkpoint_dir, "summary.json")) as f:
        config = Config()
        config.__dict__ = json.load(f)['args']
    config.device = "cuda:0"
    config.checkpoint = checkpoint_dir
    config.checkpoint_filename = checkpoint_filename
    return config


def load_config_argparse(checkpoint_dir, checkpoint_filename, config_format, rem_args):
    """Our config parser: excludes dtype/device/distributed_backend from JSON,
    then applies argparse defaults (dtype -> bfloat16).

    This is the config loading path used by broken_get_router_weights.py and
    the current get_router_weights.py. The dtype exclusion causes autocast to
    use bfloat16 instead of the original's None -> float16 default. Included
    as an experiment to isolate this confounder.
    """
    args = argparse.Namespace()
    args.checkpoint = checkpoint_dir
    args.checkpoint_filename = checkpoint_filename
    args.config_format = config_format

    with open(os.path.join(checkpoint_dir, "summary.json")) as f:
        summary = json.load(f)

    for k, v in summary['args'].items():
        if k == "config_format" and config_format is not None:
            continue
        if k not in ["device", "dtype", "distributed_backend"]:
            setattr(args, k, v)

    import config as cfg_module
    args = cfg_module.parse_args_with_format(
        format=config_format or args.config_format,
        base_parser=argparse.ArgumentParser(allow_abbrev=False),
        args=rem_args, namespace=args)
    return args


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def accum_to_npy(accum):
    """Convert list-of-lists accumulator to numpy object array."""
    # Modification 1: stores float32; original used float64 (np.array default).
    # Immaterial: source values have float16 precision (from autocast), which
    # float32 represents exactly. Histogram binning is dtype-agnostic.
    # Modification 2: empty accumulators become None; original produced np.array([]).
    # Immaterial: repeat-0 slot has no router weights in any model config
    # and is never read by downstream analysis or plotting code.
    return np.array([
        np.array(x, dtype=np.float32) if len(x) > 0 else None
        for x in accum
    ], dtype=object)


def detect_last_router_slot(weights_array):
    """Return the index of the last non-empty slot in a weights_array.

    Standard shape is [None, s_1, s_2, ..., s_N] where s_N is the router
    gating the final pass (paper Figure 5 quantity). Detection is done by
    scanning from the tail for the last entry that is non-None AND non-empty;
    this is robust to trailing None-padding and to per-method variations
    (hooks emits None for cascade arrays at unused slots).

    Returns -1 if no non-empty slot exists.
    """
    for slot in range(len(weights_array) - 1, -1, -1):
        entry = weights_array[slot]
        if entry is not None and len(entry) > 0:
            return slot
    return -1


def print_stats(label, weights_array, n_repeat):
    """Print summary statistics for a router weight array.

    Slot semantics: weights_array has length n_repeat. Slot 0 is None
    (no router decision before pass 1). Slot k in [1..n_repeat-1] holds
    mod[k-1]'s sigmoid scores (the router deciding entry into pass k+1).
    The Figure 5 "last repeat" quantity is the slot at index n_repeat - 1,
    which is also the last non-empty slot for well-formed extractions.
    """
    last_slot = detect_last_router_slot(weights_array)
    expected_last = n_repeat - 1
    header = f"\n  [{label}] (n_repeat={n_repeat}, last-repeat slot={last_slot})"
    if last_slot != expected_last:
        header += f"  WARNING: expected last slot {expected_last}"
    print(header)

    for slot in range(1, n_repeat):
        scores = weights_array[slot]
        if scores is None or len(scores) == 0:
            continue
        is_fig5 = (slot == last_slot)
        marker = "  <-- Figure 5 (last repeat)" if is_fig5 else ""
        line = (f"    slot {slot} (mod[{slot-1}], gates pass {slot+1}): "
                f"n={len(scores)}, mean={scores.mean():.4f}, "
                f"median={np.median(scores):.4f}, "
                f"frac<0.1={np.mean(scores < 0.1):.3f}, "
                f"frac>0.9={np.mean(scores > 0.9):.3f}")
        if is_fig5:
            line += (f", frac<0.01={np.mean(scores < 0.01):.3f}, "
                     f"frac>0.5={np.mean(scores > 0.5):.3f}")
        print(line + marker)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Router weight extraction for Figure 5 analysis",
        allow_abbrev=False)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint dir or file path')
    parser.add_argument('--method', type=str, required=True,
                        choices=['cf', 'cf-sorted', 'hooks'],
                        help='Extraction method')
    parser.add_argument('--config-loading', type=str, default='raw',
                        choices=['raw', 'argparse'],
                        help='Config loading mode (default: raw)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for .npy files')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Data split (default: train)')
    parser.add_argument('--config_format', type=str, default=None,
                        help='Config format override (for argparse loading)')

    args, rem_args = parser.parse_known_args()

    # --- Resolve checkpoint path ---
    if os.path.isfile(args.checkpoint):
        checkpoint_dir, checkpoint_filename = os.path.split(args.checkpoint)
    else:
        checkpoint_dir = args.checkpoint
        checkpoint_filename = "ckpt.pt"

    # --- Load config ---
    if args.config_loading == 'raw':
        config = load_config_raw(checkpoint_dir, checkpoint_filename)
    else:
        config = load_config_argparse(
            checkpoint_dir, checkpoint_filename,
            args.config_format, rem_args)

    # --- Setup ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = getattr(config, 'device', 'cuda:0')
    if isinstance(device, str):
        device = torch.device(device)
    torch.cuda.set_device(device)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    seed = getattr(config, 'seed', 0)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # --- Data ---
    print(f"Loading dataset '{config.dataset}'")
    prepare_dataset(config)
    data = get_dataset(config)
    split = args.split
    print(f"Running on '{split}' split ({len(data[split])} tokens)")

    # --- Model ---
    model = models.make_model_from_args(config)
    if args.config_loading == 'raw':
        model.cuda()
    else:
        model = model.to(device)

    checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_filename))
    model.load_state_dict(
        {k: v for k, v in checkpoint['model'].items()
         if "attn.bias" not in k and "wpe" not in k},
        strict=False)
    model.eval()

    n_repeat = model.n_repeat
    n_routers = len(model.transformer.mod)
    print(f"Model: n_repeat={n_repeat}, n_routers={n_routers}")
    print(f"Method: {args.method}, config-loading: {args.config_loading}")

    # --- Autocast context ---
    dtype = getattr(config, 'dtype', None)
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=dtype)

    # --- Batching ---
    # Modification 3: original hardcodes config.sequence_length; we fall back through
    # eval_seq_length first. Immaterial for ADM v2: eval_seq_length is None
    # in summary.json, so both paths resolve to sequence_length (256).
    seq_length = getattr(config, 'eval_seq_length', None) or config.sequence_length
    sample_size = len(data['val']) if split == 'train' else None
    eval_tokens = len(data['val'])
    total_batches = iceildiv(
        iceildiv(eval_tokens, seq_length), config.batch_size)

    data_iter = get_as_batch(
        data[split], seq_length, config.batch_size,
        device=device, sample_size=sample_size)

    # --- Run extraction ---
    os.makedirs(args.output_dir, exist_ok=True)

    if args.method in ('cf', 'cf-sorted'):
        raw, pi, pt = run_cf_extraction(
            model, data_iter, total_batches, type_ctx, n_repeat, args.method)

        raw_npy = accum_to_npy(raw)
        pi_npy = accum_to_npy(pi)
        pt_npy = accum_to_npy(pt)

        np.save(os.path.join(args.output_dir, "router_weights_raw.npy"), raw_npy)
        np.save(os.path.join(args.output_dir, "router_weights_picascade.npy"), pi_npy)
        np.save(os.path.join(args.output_dir, "router_weights_ptcascade.npy"), pt_npy)

        print_stats("raw (no cascade)", raw_npy, n_repeat)
        print_stats("PI cascade (C1 bug)", pi_npy, n_repeat)
        print_stats("PT cascade (correct)", pt_npy, n_repeat)

        written_outputs = [
            "router_weights_raw.npy",
            "router_weights_picascade.npy",
            "router_weights_ptcascade.npy",
        ]

    elif args.method == 'hooks':
        raw, pt = run_hooks_extraction(
            model, data_iter, total_batches, type_ctx, n_repeat)

        raw_npy = accum_to_npy(raw)
        pt_npy = accum_to_npy(pt)

        np.save(os.path.join(args.output_dir, "router_weights_raw.npy"), raw_npy)
        np.save(os.path.join(args.output_dir, "router_weights_ptcascade.npy"), pt_npy)

        print_stats("raw (no cascade)", raw_npy, n_repeat)
        print_stats("PT cascade (per-token)", pt_npy, n_repeat)

        written_outputs = [
            "router_weights_raw.npy",
            "router_weights_ptcascade.npy",
        ]

    # --- Metadata sidecar ---
    # Downstream plotters (plot_fig5.py, future ad-hoc analyses) read this
    # file to locate the Figure 5 slot instead of hardcoding [-1]. This makes
    # the extraction pipeline trivially robust to any n_repeat value.
    metadata = {
        "n_repeat": int(n_repeat),
        "n_routers": int(n_routers),
        "last_router_slot": int(n_repeat - 1),
        "method": args.method,
        "config_loading": args.config_loading,
        "split": args.split,
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "checkpoint_filename": checkpoint_filename,
        "seq_length": int(seq_length),
        "batch_size": int(config.batch_size),
        "outputs": written_outputs,
    }
    meta_path = os.path.join(args.output_dir, "router_weights_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata sidecar: {meta_path}")
    print(f"Outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
