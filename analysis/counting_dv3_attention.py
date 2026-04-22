"""DV-3 -- Attention-pattern analysis at the penultimate repeat (RQ9).

Scope
-----
Addresses DV-3 of RQ9 per DEC-027. Loads a counting-task checkpoint
(trained via the `iridis/counting-sweep/` package), runs a non-flash
forward pass over the OOD test set (variable-length sequences in
`[51, 200]`), and extracts per-head attention at the **penultimate
repeat** of the CoTFormer mid-block stack. For each ``(mid layer,
head)`` pair the script reports three attention-pattern descriptors:

- **Attention entropy** at the last-meaningful-input-token query
  position (position ``L`` per sample, where ``L`` is the sampled
  counting length in ``[51, 200]``). Low entropy =
  concentrated attention; high entropy = diffuse.
- **Top-k concentration** -- the fraction of attention mass assigned
  to the top ``k`` key positions for ``k in {1, 3, 5}``.
- **Target-position alignment** -- attention mass at the position
  whose input token is the start-integer (position 0), interpreted
  as the "tracking" signal. If the recurrent model tracks the
  running count internally, attention at the last-input query
  should be concentrated near the start-integer position; a
  non-tracking baseline would show diffuse attention.

Per DEC-027, DV-3 is the structural / mechanistic DV triangulating
DV-1 (exact-match accuracy) and DV-2 (cross-entropy); a CoTFormer
that beats the 4L baseline on DV-1 without a corresponding DV-3
signature would falsify the "recurrent inductive bias for
counting" interpretation and suggest a non-counting shortcut.

Falsifiability relevance
------------------------
Per `docs/extend-notes.md` §1.2 RQ9 DV-3 row, the attention-pattern
signature is one of three orthogonal measurements triangulating the
RQ9 H0. A positive RQ9 result (CoTFormer OOD accuracy above the
4L baseline by at least 5 pp) must be accompanied by a clean DV-3
signature at the penultimate repeat or the claim is downgraded per
the triangulation matrix at `docs/extend-notes.md` §1.6.

Collector dependency
--------------------
Uses ``analysis.common.collector.ActivationCollector`` at
``ActivationSite.ATTN_WEIGHTS`` with ``non_flash=True`` (per the
non-flash requirement documented in the collector). The
``CausalSelfAttention.forward`` monkey-patch installed by the
collector re-runs the Q/K math to recover the attention matrix
post-softmax; the captured shape is ``(N_tokens, n_head, T_k)``
with ``N_tokens == B * T_q``, from which the script selects the
per-sample query rows at position ``L``.

Padding caveat
--------------
The CoTFormer variants (V1-V4 in DIR-001) do NOT accept an
``attention_mask`` kwarg on ``GPTBase.forward``. Their collector-
captured attention distributions therefore include pad-token key
positions. The analysis compensates by truncating each per-sample
attention distribution to the first ``L + 1`` key positions (the
meaningful input) and renormalising before computing entropy,
top-k, and target alignment.

The Arm A baseline (``but_full_depth``) DOES accept an
``attention_mask`` kwarg; when the loaded checkpoint's model is
``but_full_depth``, the forward is called with ``attention_mask =
pad_mask`` and no post-capture truncation is required. See
`docs/extend-technical.md` §8 if a follow-up DIR adds attention_mask
support to the CoTFormer classes.

Smoke test
----------
Synthetic-attention-matrix tests live under ``_smoke_test``; invoked
via ``python -m analysis.counting_dv3_attention --smoke-test``. The
smoke path constructs a known ``(B, H, T_q, T_k)`` array, calls
``analyse_attention_at_query`` with hand-picked query positions and
length ``L``, and verifies that the reported entropy, top-k, and
target-mass match the analytic values.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from typing import Any

import numpy as np


OOD_LENGTH_RANGE = (51, 200)
DEFAULT_SMOKE_N_SAMPLES = 8


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for DV-3."""
    parser = argparse.ArgumentParser(
        description="DV-3 attention-pattern analysis at the penultimate repeat"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False,
        help="Checkpoint directory containing summary.json + checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="ckpt.pt",
        help="Checkpoint filename within --checkpoint (default ckpt.pt)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=False,
        help="Directory for counting_dv3_attention_results.json + PNG",
    )
    parser.add_argument("--seed", type=int, default=19937,
                        help="OOD test-set seed (project OOD convention)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of OOD samples to draw (default 500)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-batch sample count (default 4; kept small "
                             "so the attention-weight buffer fits in memory "
                             "at seq_length=256 x n_head=4/8)")
    parser.add_argument("--sequence-length", type=int, default=256,
                        help="Pad length (default 256; must be >= max OOD "
                             "length 201)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device string; 'cpu' forces CPU run")
    parser.add_argument("--config-mode", type=str, default="raw",
                        choices=["raw", "argparse"])
    parser.add_argument("--module-path", type=str,
                        default="model.transformer.h_mid")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run the synthetic-attention smoke test and exit")
    return parser


# ---------------------------------------------------------------------------
# Pure analysis primitives (operate on numpy arrays; smoke-testable)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AttentionStats:
    """Per-sample per-head attention-distribution descriptors."""

    entropy: float
    top1: float
    top3: float
    top5: float
    target_mass: float


def analyse_attention_at_query(
    att: np.ndarray,
    query_idx: int,
    length_L: int,
    target_position: int = 0,
) -> list[AttentionStats]:
    """Compute per-head DV-3 descriptors from a 4-D attention tensor.

    Parameters
    ----------
    att : ndarray of shape (B, H, T_q, T_k), float
        Per-sample per-head attention matrix (post-softmax).
    query_idx : int
        Query row index to analyse (e.g. the last meaningful input
        position, which is ``L`` for a counting sample with length
        ``L``; position 0 holds the start-integer, positions 1..L
        are the 'a' input tokens, so the "last input" query is
        ``L``).
    length_L : int
        Sampled counting length for this batch element. Key positions
        beyond ``L`` are pad tokens and must be excluded before the
        entropy / top-k / target-mass descriptors are computed; the
        attention row is renormalised over ``[0, L + 1)``.
    target_position : int
        Key position whose attention mass represents the "tracking"
        signal. Defaults to 0, the start-integer input position.

    Returns
    -------
    list[AttentionStats]
        One ``AttentionStats`` per (sample, head) pair, flattened
        so ``result[b * H + h]`` gives sample ``b`` head ``h``. Each
        entry reports the four descriptors on the truncated,
        renormalised attention distribution restricted to
        ``[0, L + 1)`` keys.
    """
    if att.ndim != 4:
        raise ValueError(
            f"analyse_attention_at_query: expected 4-D (B, H, T_q, T_k), "
            f"got shape {att.shape}"
        )
    B, H, T_q, T_k = att.shape
    if not 0 <= query_idx < T_q:
        raise IndexError(
            f"query_idx={query_idx} out of range for T_q={T_q}"
        )
    if length_L < 0 or length_L + 1 > T_k:
        raise ValueError(
            f"length_L={length_L} incompatible with T_k={T_k} "
            f"(need length_L + 1 <= T_k)"
        )
    if not 0 <= target_position < length_L + 1:
        raise ValueError(
            f"target_position={target_position} must lie in "
            f"[0, length_L + 1) = [0, {length_L + 1})"
        )

    # (B, H, T_k) -- the attention row at the chosen query position.
    rows = att[:, :, query_idx, :].astype(np.float64)

    # Truncate to meaningful key range [0, L + 1) and renormalise so
    # each (sample, head) row sums to 1. This compensates for the
    # pad-token key contamination documented in the module docstring.
    truncated = rows[:, :, : length_L + 1]
    mass = truncated.sum(axis=-1, keepdims=True)
    mass = np.where(mass > 0.0, mass, 1.0)  # guard against zero-mass rows
    dist = truncated / mass

    # Entropy (natural log; units = nats).
    safe = np.clip(dist, 1e-12, 1.0)
    entropy = -(dist * np.log(safe)).sum(axis=-1)  # (B, H)

    # Top-k concentration. Use argsort descending; clip k to the
    # truncated key count so tiny L values don't overflow.
    sorted_desc = np.sort(dist, axis=-1)[..., ::-1]  # (B, H, L + 1)
    cumsum = np.cumsum(sorted_desc, axis=-1)
    keys_available = length_L + 1

    def _top_k(k: int) -> np.ndarray:
        idx = min(k, keys_available) - 1
        if idx < 0:
            return np.zeros(cumsum.shape[:2], dtype=np.float64)
        return cumsum[..., idx]

    top1 = _top_k(1)  # (B, H)
    top3 = _top_k(3)
    top5 = _top_k(5)

    # Target-position alignment.
    target_mass = dist[:, :, target_position]

    results: list[AttentionStats] = []
    for b in range(B):
        for h in range(H):
            results.append(
                AttentionStats(
                    entropy=float(entropy[b, h]),
                    top1=float(top1[b, h]),
                    top3=float(top3[b, h]),
                    top5=float(top5[b, h]),
                    target_mass=float(target_mass[b, h]),
                )
            )
    return results


def aggregate_stats(samples: list[AttentionStats]) -> dict[str, float]:
    """Return mean + std of every descriptor over a sample list."""
    if not samples:
        return {
            "n": 0, "entropy_mean": float("nan"), "entropy_std": float("nan"),
            "top1_mean": float("nan"), "top3_mean": float("nan"),
            "top5_mean": float("nan"), "target_mass_mean": float("nan"),
            "target_mass_std": float("nan"),
        }
    arrs = {
        "entropy": np.array([s.entropy for s in samples]),
        "top1": np.array([s.top1 for s in samples]),
        "top3": np.array([s.top3 for s in samples]),
        "top5": np.array([s.top5 for s in samples]),
        "target_mass": np.array([s.target_mass for s in samples]),
    }
    return {
        "n": len(samples),
        "entropy_mean": float(arrs["entropy"].mean()),
        "entropy_std": float(arrs["entropy"].std(ddof=1) if len(samples) > 1 else 0.0),
        "top1_mean": float(arrs["top1"].mean()),
        "top3_mean": float(arrs["top3"].mean()),
        "top5_mean": float(arrs["top5"].mean()),
        "target_mass_mean": float(arrs["target_mass"].mean()),
        "target_mass_std": float(arrs["target_mass"].std(ddof=1) if len(samples) > 1 else 0.0),
    }


# ---------------------------------------------------------------------------
# Checkpoint-driven analysis (requires a loaded CoTFormer / standard model)
# ---------------------------------------------------------------------------


def _resolve_penultimate_repeat_label(n_repeat: int) -> int:
    """Map ``n_repeat`` to the penultimate repeat label used by the collector.

    The ``ActivationCollector`` labels captures ``r=1..n_repeat``.
    The penultimate repeat is ``n_repeat - 1`` for ``n_repeat >= 2``.
    For a standard 4L baseline (n_repeat = 1) the concept of
    "penultimate REPEAT" collapses; the caller interprets the only
    captured repeat ``r=1`` as the "last-layer" site and selects
    the penultimate LAYER in the mid-block list instead (handled
    downstream in ``analyse_checkpoint``).
    """
    if n_repeat >= 2:
        return n_repeat - 1
    return 1


def analyse_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Load a checkpoint, run DV-3, return a result dict.

    Not invoked in smoke-test mode; separated from ``main`` for
    testability. Delegates heavy lifting to
    ``analysis.common.{loader, collector, sites}`` and to
    ``data.counting.CountingDataset`` for the OOD iterator.
    """
    # Deferred imports: the smoke-test mode does not need torch model
    # loading or the shared analysis infrastructure, so torch stays
    # out of module top-level for CPU-only smoke-test environments.
    import torch  # noqa: F401 (consumed by torch.no_grad() below)
    from torch.utils.data import DataLoader
    from analysis.common.collector import ActivationCollector
    from analysis.common.loader import load_model_from_checkpoint
    from analysis.common.sites import ActivationSite
    from data.counting import CountingDataset, TE200_MAX_OUT

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model, config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        config_mode=args.config_mode,
        device=device,
        module_path=args.module_path,
    )
    model.eval()
    n_repeat = int(getattr(model, "n_repeat", 1) or 1)
    penultimate_r = _resolve_penultimate_repeat_label(n_repeat)

    # Detect the Arm A / Arm B attention-mask pathway. ``but_full_depth``
    # accepts ``attention_mask``; the four CoTFormer variants (V1-V4)
    # do NOT as of DIR-001. Downstream we skip the kwarg for the
    # latter and fall back to post-capture key truncation.
    model_name = getattr(config, "model", "unknown")
    accepts_attention_mask = model_name == "but_full_depth"

    # Build the OOD iterator directly (``analysis.common.data`` is
    # OWT2-shaped and not compatible with the counting 4-tuple).
    dataset = CountingDataset(
        split="ood",
        seed=int(args.seed),
        num_samples=int(args.n_samples),
        sequence_length=int(args.sequence_length),
        max_out=TE200_MAX_OUT,
        length_range=OOD_LENGTH_RANGE,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    # Install ATTN_WEIGHTS + run forward.
    collector = ActivationCollector(
        model=model,
        sites=[ActivationSite.ATTN_WEIGHTS],
        non_flash=True,
        module_path=args.module_path,
    )

    # Per-sample (L, sample_index) bookkeeping so we can stride into
    # the collector's (N_tokens, n_head, T_k) buffers: the buffer's
    # leading axis is ordered as ``batch_0_token_0..T_q, batch_0_..,
    # batch_1_token_0..T_q, ...`` and ``analyse_attention_at_query``
    # operates on a 4-D view reshaped back to (B, H, T_q, T_k).
    per_sample_L: list[int] = []
    per_layer_head_stats: dict[str, list[AttentionStats]] = {}
    total_samples = 0

    with torch.no_grad(), collector:
        for batch in loader:
            x, y, pad_mask, _loss_mask = batch
            x = x.to(device)
            pad_mask_d = pad_mask.to(device)

            if accepts_attention_mask:
                _ = model(x, attention_mask=pad_mask_d, get_logits=False)
            else:
                _ = model(x, get_logits=False)

            # pad_mask shape (B, T). Sample length L = pad_mask.sum() - 1
            # (the mask is 1.0 on positions 0..L inclusive = L + 1
            # positions; L is the count of 'a' input tokens).
            batch_L = (pad_mask.sum(dim=1).long() - 1).tolist()
            per_sample_L.extend(batch_L)
            total_samples += len(batch_L)

        # After the forward loop, drain the collector's per-layer
        # attention buffers and analyse at the penultimate repeat.
        # Buffer naming follows the collector: ``attn_weights_mid_l<L>_r<R>``.
        buffer_keys = collector.buffer_keys()
        penultimate_keys = [
            k for k in buffer_keys if k.endswith(f"_r{penultimate_r}")
        ]
        for key in penultimate_keys:
            buf = collector.buffer_for(key)  # (N_tokens_flat, H, T_k)
            # N_tokens_flat == total_samples * T_q; reshape to (B, T_q, H, T_k)
            # then permute to (B, H, T_q, T_k) for analyse_attention_at_query.
            B_total, H, T_k = buf.shape
            T_q = B_total // max(1, total_samples)
            if T_q * total_samples != B_total:
                raise RuntimeError(
                    f"DV-3 collector buffer for key={key!r} has "
                    f"N_tokens={B_total} that is not divisible by the "
                    f"sample count {total_samples}; cannot reshape to "
                    f"(B, T_q, H, T_k)."
                )
            att_4d = buf.reshape(total_samples, T_q, H, T_k).transpose(0, 2, 1, 3)

            layer_head_samples: list[AttentionStats] = []
            # Per-sample analysis at the sample's own L (variable per sample).
            for sample_idx, L in enumerate(per_sample_L):
                per_sample_att = att_4d[sample_idx : sample_idx + 1]  # (1, H, T_q, T_k)
                stats_list = analyse_attention_at_query(
                    att=per_sample_att,
                    query_idx=L,  # last meaningful input token
                    length_L=L,
                    target_position=0,  # start-integer position
                )
                # stats_list has H entries; attribute them to per-head buckets.
                for head_idx, stat in enumerate(stats_list):
                    bucket_key = f"{key}_h{head_idx}"
                    per_layer_head_stats.setdefault(bucket_key, []).append(stat)
                layer_head_samples.extend(stats_list)

    # Aggregate.
    per_layer_head_summary: dict[str, dict[str, float]] = {}
    for bucket_key, stat_list in sorted(per_layer_head_stats.items()):
        per_layer_head_summary[bucket_key] = aggregate_stats(stat_list)

    return {
        "checkpoint": args.checkpoint,
        "checkpoint_file": args.checkpoint_file,
        "model": model_name,
        "n_repeat": n_repeat,
        "penultimate_repeat_label": penultimate_r,
        "n_samples": total_samples,
        "accepts_attention_mask": accepts_attention_mask,
        "per_layer_head": per_layer_head_summary,
        "meta": {
            "ood_length_range": list(OOD_LENGTH_RANGE),
            "sequence_length": int(args.sequence_length),
            "seed": int(args.seed),
            "module_path": args.module_path,
        },
    }


# ---------------------------------------------------------------------------
# Smoke test (synthetic attention matrices)
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Verify ``analyse_attention_at_query`` on hand-built arrays.

    The smoke test constructs three canonical attention distributions
    and checks each descriptor matches its analytic value.
    """
    # Build (B=3, H=1, T_q=8, T_k=8) with three known query rows at
    # query_idx=3, target_position=0, length_L=5 (so keys 0..5 are
    # meaningful; keys 6, 7 are "pads" and get stripped before renorm).
    T = 8
    att = np.zeros((3, 1, T, T), dtype=np.float64)
    # Sample 0: uniform over keys 0..5 (6 positions) -- max entropy.
    att[0, 0, 3, :6] = 1.0 / 6
    # Sample 1: delta on key 0 -- zero entropy, target_mass=1.
    att[1, 0, 3, 0] = 1.0
    # Sample 2: 50-50 between key 0 (target) and key 5 -- intermediate.
    att[2, 0, 3, 0] = 0.5
    att[2, 0, 3, 5] = 0.5

    results: list[AttentionStats] = []
    for b in range(3):
        stats = analyse_attention_at_query(
            att=att[b : b + 1],
            query_idx=3,
            length_L=5,
            target_position=0,
        )
        results.extend(stats)

    # Expected values:
    # Sample 0: entropy = ln(6) ~= 1.7918; top1 = 1/6; top3 = 3/6; top5 = 5/6; target_mass = 1/6.
    # Sample 1: entropy = 0; top1 = 1; top3 = 1; top5 = 1; target_mass = 1.
    # Sample 2: entropy = ln(2) ~= 0.6931; top1 = 0.5; top3 = 1; top5 = 1; target_mass = 0.5.
    tol = 1e-6
    assert abs(results[0].entropy - math.log(6)) < tol, results[0].entropy
    assert abs(results[0].top1 - 1 / 6) < tol, results[0].top1
    assert abs(results[0].top3 - 3 / 6) < tol, results[0].top3
    assert abs(results[0].top5 - 5 / 6) < tol, results[0].top5
    assert abs(results[0].target_mass - 1 / 6) < tol, results[0].target_mass

    assert abs(results[1].entropy - 0.0) < tol, results[1].entropy
    assert abs(results[1].top1 - 1.0) < tol, results[1].top1
    assert abs(results[1].target_mass - 1.0) < tol, results[1].target_mass

    assert abs(results[2].entropy - math.log(2)) < tol, results[2].entropy
    assert abs(results[2].top1 - 0.5) < tol, results[2].top1
    assert abs(results[2].top3 - 1.0) < tol, results[2].top3
    assert abs(results[2].target_mass - 0.5) < tol, results[2].target_mass

    # Aggregation smoke: aggregate_stats over three known samples.
    agg = aggregate_stats(results)
    expected_entropy_mean = (math.log(6) + 0.0 + math.log(2)) / 3
    assert abs(agg["entropy_mean"] - expected_entropy_mean) < tol, agg

    # Pad-position exclusion: build a row where mass is split between
    # a meaningful key and a "pad" key (index > L); the analysis
    # should renormalise to remove the pad mass.
    att_pad = np.zeros((1, 1, T, T), dtype=np.float64)
    att_pad[0, 0, 3, 0] = 0.5
    att_pad[0, 0, 3, 7] = 0.5  # key 7 is pad for L=5
    stats_pad = analyse_attention_at_query(
        att=att_pad, query_idx=3, length_L=5, target_position=0,
    )[0]
    # After truncation to [0, 6) and renorm, row has mass 1.0 at key 0
    # (the pad mass was discarded + renormalised). Entropy = 0,
    # target_mass = 1, top1 = 1.
    assert abs(stats_pad.entropy - 0.0) < tol, stats_pad.entropy
    assert abs(stats_pad.target_mass - 1.0) < tol, stats_pad.target_mass
    assert abs(stats_pad.top1 - 1.0) < tol, stats_pad.top1

    print("DV-3 smoke test PASS")
    print(
        f"  Sample 0 (uniform over 6): entropy={results[0].entropy:.4f} "
        f"(expected {math.log(6):.4f})"
    )
    print(
        f"  Sample 1 (delta on target): entropy={results[1].entropy:.4f} "
        f"target_mass={results[1].target_mass:.4f}"
    )
    print(
        f"  Sample 2 (50-50 target + far): entropy={results[2].entropy:.4f} "
        f"target_mass={results[2].target_mass:.4f}"
    )
    print(f"  Pad-key exclusion: renormalised target_mass={stats_pad.target_mass:.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.smoke_test:
        _smoke_test()
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required when --smoke-test is not set")
    if not args.output_dir:
        parser.error("--output-dir is required when --smoke-test is not set")

    os.makedirs(args.output_dir, exist_ok=True)
    results = analyse_checkpoint(args)

    out_json = os.path.join(args.output_dir, "counting_dv3_attention_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"DV-3 results written to {out_json}")


if __name__ == "__main__":
    main()
