"""Protocol A -- Logit Lens (unweighted).

Scope
-----
Addresses RQ1 (prediction convergence across repeats). Runs a
CoTFormer variant over 2048 tokens of OWT2 validation data and
captures the residual stream at two sites per (mid-layer, repeat)
pair: the post-h_mid residual (pre-`ln_mid`) and the post-`ln_mid`
residual. Each site is projected through the frozen ``lm_head``
twice -- once with the model's trained ``ln_f`` (the "standard"
logit lens) and once directly on the raw residual (the "lens with
no ln_f"). Top-1 accuracy against the ground-truth next token,
KL divergence between successive repeats, and entropy are computed
over the same token set.

Falsifiability relevance
------------------------
H0: "top-1 accuracy flat across repeats" is rejected under a paired
t-test between repeats 1 and ``n_repeat`` at p < 0.01 AND a
monotonically non-decreasing accuracy curve (per
`docs/extend-notes.md` §1.2 RQ1 "Statistical test" and "Falsification
threshold"). The dual measurement (with and without `ln_f`) isolates
the normalisation confounder; see
`docs/extend-notes.md` §1.2 RQ1 "Confounders".

Ontological purpose
-------------------
Determines whether iterative repeats implement refinement toward a
fixed point (diminishing returns profile) or qualitative
restructuring; this is the central distinction of SQ1 and decides
whether the paper's "chain-of-thought" analogy is accurate for the
weight-tied recurrent transformer. Triangulated with Protocol A-ext
(Tuned Lens) per the `docs/extend-notes.md` §1.6 triangulation matrix.

Workspace outputs
-----------------
When ``--workspace`` is provided, the forward pass residuals and
targets are persisted for downstream consumers (Protocol A-ext
``analysis.tuned_lens`` and Protocol E ``analysis.residual_diagnostics``).
See ``analysis.common.collector`` for the workspace layout.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from analysis.common.collector import ActivationCollector
from analysis.common.data import iterate_owt2_val
from analysis.common.loader import load_model_from_checkpoint
from analysis.common.plotting import palette_for_repeats, savefig, setup_figure
from analysis.common.sites import ActivationSite
from analysis.common.stats import paired_t_test


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol A.

    Expected inputs per `docs/extend-notes.md` §1.3 Protocol A row:
    ``--checkpoint``, ``--checkpoint-file``, ``--workspace``,
    ``--output-dir``, ``--seed``, ``--max-tokens``.
    """
    parser = argparse.ArgumentParser(
        description="Protocol A -- unweighted Logit Lens on CoTFormer residuals"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Checkpoint directory containing summary.json + ckpt file",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="ckpt.pt",
        help="Checkpoint filename within --checkpoint (default ckpt.pt)",
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Workspace directory on /scratch. When provided, the "
             "collector persists per-site residuals for downstream "
             "protocol consumers (tuned_lens, residual_diagnostics).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for logit_lens_results.json + PNG",
    )
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Tokens per batch (default 2048)")
    parser.add_argument("--n-batches", type=int, default=4,
                        help="Inter-batch robustness batch count (default 4)")
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config-mode", type=str, default="raw",
                        choices=["raw", "argparse"])
    parser.add_argument("--module-path", type=str,
                        default="model.transformer.h_mid")
    parser.add_argument("--cv-threshold", type=float, default=0.05,
                        help="Inter-batch CV gate; failed metrics flagged")
    return parser


def _logits_from_residuals(
    residuals: torch.Tensor,
    lm_head: torch.nn.Linear,
    ln_f: torch.nn.Module,
    via_lnf: bool,
) -> torch.Tensor:
    """Project ``residuals`` of shape ``(N_tokens, D)`` to logits.

    ``via_lnf=True`` applies the model's trained ``ln_f`` before
    ``lm_head``; ``via_lnf=False`` applies ``lm_head`` directly
    (the raw-residual lens, which tests whether ``ln_f`` contributes
    the systematic monotonicity).
    """
    x = residuals
    if via_lnf:
        x = ln_f(x)
    return lm_head(x)


def _metrics_at_site(
    residuals: torch.Tensor,
    targets: torch.Tensor,
    lm_head: torch.nn.Linear,
    ln_f: torch.nn.Module,
    prev_log_softmax_lnf: torch.Tensor | None,
    prev_log_softmax_raw: torch.Tensor | None,
) -> dict[str, Any]:
    """Compute per-site metrics in a single pass.

    Returns a dict with ``top1_lnf``, ``top1_raw``, ``entropy_lnf``,
    ``entropy_raw``, ``kl_lnf``, ``kl_raw``, plus the log-softmax
    tensors at the current site so the caller can pass them as
    ``prev_log_softmax_*`` for the next repeat's KL computation.
    """
    with torch.no_grad():
        logits_lnf = _logits_from_residuals(residuals, lm_head, ln_f, via_lnf=True)
        logits_raw = _logits_from_residuals(residuals, lm_head, ln_f, via_lnf=False)

        top1_lnf = (logits_lnf.argmax(dim=-1) == targets).float().mean().item()
        top1_raw = (logits_raw.argmax(dim=-1) == targets).float().mean().item()

        log_softmax_lnf = F.log_softmax(logits_lnf, dim=-1)
        log_softmax_raw = F.log_softmax(logits_raw, dim=-1)

        softmax_lnf = log_softmax_lnf.exp()
        softmax_raw = log_softmax_raw.exp()

        entropy_lnf = float((-softmax_lnf * log_softmax_lnf).sum(dim=-1).mean().item())
        entropy_raw = float((-softmax_raw * log_softmax_raw).sum(dim=-1).mean().item())

        if prev_log_softmax_lnf is not None:
            # D_KL(P_r || P_{r-1}) = sum_i P_r(i) * (log P_r(i) - log P_{r-1}(i))
            kl_lnf = float(
                (softmax_lnf * (log_softmax_lnf - prev_log_softmax_lnf)).sum(dim=-1).mean().item()
            )
        else:
            kl_lnf = float("nan")

        if prev_log_softmax_raw is not None:
            kl_raw = float(
                (softmax_raw * (log_softmax_raw - prev_log_softmax_raw)).sum(dim=-1).mean().item()
            )
        else:
            kl_raw = float("nan")

    return {
        "top1_lnf": float(top1_lnf),
        "top1_raw": float(top1_raw),
        "entropy_lnf": entropy_lnf,
        "entropy_raw": entropy_raw,
        "kl_lnf": kl_lnf,
        "kl_raw": kl_raw,
        "_log_softmax_lnf": log_softmax_lnf,
        "_log_softmax_raw": log_softmax_raw,
    }


def _run_one_batch(
    model: torch.nn.Module,
    config,
    args: argparse.Namespace,
    start_offset: int,
    device: str,
) -> dict[str, Any]:
    """Run the collector on one batch, compute per-site metrics.

    Returns a dict holding per-(layer, repeat) metric arrays and the
    accumulated targets and residuals (for optional workspace write).
    """
    transformer = model.transformer
    ln_f = transformer.ln_f
    lm_head = model.lm_head

    mid_path = args.module_path
    if mid_path.startswith("model."):
        mid_path = mid_path[len("model.") :]
    cursor = model
    for name in mid_path.split("."):
        cursor = getattr(cursor, name)
    n_layer_mid = len(cursor)
    n_repeat = int(getattr(model, "n_repeat", 1))

    # RESIDUAL_PRE_LN_F is required so the workspace carries the pre-``ln_f``
    # residual that ``analysis.tuned_lens`` consumes as its target residual
    # (canonical Belrose 2023 convention). Without this site the downstream
    # Protocol A-ext run would fail at ``load_residuals``.
    sites = [
        ActivationSite.RESIDUAL_POST_MID,
        ActivationSite.RESIDUAL_POST_LN_MID,
        ActivationSite.RESIDUAL_PRE_LN_F,
    ]
    collector = ActivationCollector(
        model, sites, non_flash=False, module_path=args.module_path
    )

    # Capture targets inline: we use a pre-forward hook on the data
    # iterator rather than modifying the collector.
    captured_targets: list[np.ndarray] = []

    dtype = getattr(config, "dtype", None)
    if isinstance(dtype, str):
        try:
            dtype = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}[dtype]
        except KeyError:
            dtype = None
    if device.startswith("cuda") and dtype is not None:
        type_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        type_ctx = nullcontext()

    data_iter_source = iterate_owt2_val(
        data_dir=args.data_dir,
        config=config,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        total_tokens=args.max_tokens,
        device=device,
        start_offset=start_offset,
        split="val",
    )

    def _iter_with_target_capture():
        for x, y in data_iter_source:
            captured_targets.append(y.detach().cpu().reshape(-1).numpy())
            yield x, y

    with collector:
        total = collector.run(_iter_with_target_capture(), type_ctx=type_ctx,
                              max_tokens=args.max_tokens)

    targets_np = np.concatenate(captured_targets, axis=0) if captured_targets else np.zeros(0, dtype=np.int64)
    targets_torch = torch.from_numpy(targets_np.astype(np.int64)).to(device)

    # Compute metrics per (mid layer, repeat); for each layer iterate
    # repeats in order so the previous-repeat log_softmax stays in
    # scope for the KL computation.
    top1_lnf = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float32)
    top1_raw = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float32)
    entropy_lnf = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float32)
    entropy_raw = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float32)
    kl_lnf = np.full((n_layer_mid, n_repeat - 1), np.nan, dtype=np.float32)
    kl_raw = np.full((n_layer_mid, n_repeat - 1), np.nan, dtype=np.float32)

    ln_mid_top1_lnf = np.full((n_repeat,), np.nan, dtype=np.float32)
    ln_mid_top1_raw = np.full((n_repeat,), np.nan, dtype=np.float32)
    ln_mid_entropy_lnf = np.full((n_repeat,), np.nan, dtype=np.float32)
    ln_mid_kl_lnf = np.full((n_repeat - 1,), np.nan, dtype=np.float32)

    lm_head_eval = lm_head.to(device).eval()
    ln_f_eval = ln_f.to(device).eval()

    for layer_idx in range(n_layer_mid):
        prev_ls_lnf = None
        prev_ls_raw = None
        for repeat_idx in range(1, n_repeat + 1):
            key = f"residual_mid_l{layer_idx}_r{repeat_idx}"
            try:
                h_np = collector.buffer_for(key)
            except KeyError:
                continue
            h = torch.from_numpy(h_np).to(device=device, dtype=lm_head_eval.weight.dtype)
            if h.shape[0] != targets_torch.shape[0]:
                trim = min(h.shape[0], targets_torch.shape[0])
                h = h[:trim]
                targets_slice = targets_torch[:trim]
            else:
                targets_slice = targets_torch
            site_metrics = _metrics_at_site(
                h, targets_slice, lm_head_eval, ln_f_eval,
                prev_ls_lnf, prev_ls_raw,
            )
            top1_lnf[layer_idx, repeat_idx - 1] = site_metrics["top1_lnf"]
            top1_raw[layer_idx, repeat_idx - 1] = site_metrics["top1_raw"]
            entropy_lnf[layer_idx, repeat_idx - 1] = site_metrics["entropy_lnf"]
            entropy_raw[layer_idx, repeat_idx - 1] = site_metrics["entropy_raw"]
            if repeat_idx >= 2:
                kl_lnf[layer_idx, repeat_idx - 2] = site_metrics["kl_lnf"]
                kl_raw[layer_idx, repeat_idx - 2] = site_metrics["kl_raw"]
            prev_ls_lnf = site_metrics["_log_softmax_lnf"]
            prev_ls_raw = site_metrics["_log_softmax_raw"]

    # ln_mid sites (one per repeat). Project lm_head(ln_f(h_post_ln_mid)).
    prev_ls_lnf = None
    prev_ls_raw = None
    for repeat_idx in range(1, n_repeat + 1):
        key = f"residual_ln_mid_r{repeat_idx}"
        try:
            h_np = collector.buffer_for(key)
        except KeyError:
            continue
        h = torch.from_numpy(h_np).to(device=device, dtype=lm_head_eval.weight.dtype)
        if h.shape[0] != targets_torch.shape[0]:
            trim = min(h.shape[0], targets_torch.shape[0])
            h = h[:trim]
            targets_slice = targets_torch[:trim]
        else:
            targets_slice = targets_torch
        site_metrics = _metrics_at_site(
            h, targets_slice, lm_head_eval, ln_f_eval,
            prev_ls_lnf, prev_ls_raw,
        )
        ln_mid_top1_lnf[repeat_idx - 1] = site_metrics["top1_lnf"]
        ln_mid_top1_raw[repeat_idx - 1] = site_metrics["top1_raw"]
        ln_mid_entropy_lnf[repeat_idx - 1] = site_metrics["entropy_lnf"]
        if repeat_idx >= 2:
            ln_mid_kl_lnf[repeat_idx - 2] = site_metrics["kl_lnf"]
        prev_ls_lnf = site_metrics["_log_softmax_lnf"]
        prev_ls_raw = site_metrics["_log_softmax_raw"]

    return {
        "n_layer_mid": int(n_layer_mid),
        "n_repeat": int(n_repeat),
        "total_tokens": int(total),
        "start_offset": int(start_offset),
        "top1_lnf": top1_lnf,
        "top1_raw": top1_raw,
        "entropy_lnf": entropy_lnf,
        "entropy_raw": entropy_raw,
        "kl_lnf": kl_lnf,
        "kl_raw": kl_raw,
        "ln_mid_top1_lnf": ln_mid_top1_lnf,
        "ln_mid_top1_raw": ln_mid_top1_raw,
        "ln_mid_entropy_lnf": ln_mid_entropy_lnf,
        "ln_mid_kl_lnf": ln_mid_kl_lnf,
        "targets": targets_np,
        "collector": collector,
    }


def _aggregate_per_batch(
    per_batch: list[dict[str, Any]],
    cv_threshold: float,
) -> dict[str, Any]:
    """Compute mean and coefficient-of-variation across batches."""
    metric_keys = [
        "top1_lnf", "top1_raw", "entropy_lnf", "entropy_raw",
        "kl_lnf", "kl_raw",
        "ln_mid_top1_lnf", "ln_mid_top1_raw", "ln_mid_entropy_lnf",
        "ln_mid_kl_lnf",
    ]
    aggregate: dict[str, Any] = {}
    failed_cv: list[str] = []

    for key in metric_keys:
        stack = np.stack([batch[key] for batch in per_batch], axis=0)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(np.abs(mean) > 1e-12, std / np.abs(mean), np.nan)
        aggregate[f"mean_{key}"] = mean.tolist()
        aggregate[f"cv_{key}"] = cv.tolist()
        max_cv = float(np.nanmax(cv)) if np.isfinite(cv).any() else float("nan")
        aggregate[f"max_cv_{key}"] = max_cv
        if np.isfinite(max_cv) and max_cv > cv_threshold:
            failed_cv.append(key)

    aggregate["failed_cv_metrics"] = failed_cv
    aggregate["cv_threshold"] = cv_threshold
    return aggregate


def _h0_verdict(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Paired t-test on per-layer top-1 accuracy at r=1 vs r=n_repeat.

    The paired observations are the 21 mid-layer top-1 values at the
    first and last repeat (via ``ln_f`` projection, the canonical
    logit lens). H0 rejected when p < 0.01 AND mean(r_last) >
    mean(r_first).
    """
    mean_top1_lnf = np.asarray(aggregate["mean_top1_lnf"], dtype=np.float64)
    if mean_top1_lnf.ndim != 2:
        return {"method": "paired_t_test", "skipped_reason": "expected 2-D array"}
    n_layer_mid, n_repeat = mean_top1_lnf.shape
    if n_repeat < 2:
        return {"method": "paired_t_test", "skipped_reason": "n_repeat < 2"}
    first = mean_top1_lnf[:, 0]
    last = mean_top1_lnf[:, -1]
    mask = np.isfinite(first) & np.isfinite(last)
    if mask.sum() < 2:
        return {"method": "paired_t_test", "skipped_reason": "insufficient finite pairs"}

    t_stat, p_value = paired_t_test(last[mask], first[mask])
    monotonic = bool(np.all(np.diff(np.nanmean(mean_top1_lnf, axis=0)) >= -1e-4))
    rejects_flat = bool(p_value < 0.01 and float(np.nanmean(last) - np.nanmean(first)) > 0.0)

    return {
        "method": "paired_t_test",
        "sample_space_layers": int(mask.sum()),
        "mean_r1": float(np.nanmean(first)),
        "mean_rlast": float(np.nanmean(last)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "threshold_alpha": 0.01,
        "rejects_flat": rejects_flat,
        "monotonic_non_decreasing": monotonic,
    }


def _persist_workspace(
    workspace_dir: str,
    per_batch: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Persist the first batch's collector state + targets for downstream.

    Protocol A-ext (Tuned Lens) and Protocol E (Residual Diagnostics)
    consume this workspace directly (they do not re-run the forward).
    When the collector has already run and its buffers are populated,
    ``save`` flushes them as ``.npy`` files per §4.3 layout.
    """
    first = per_batch[0]
    collector: ActivationCollector = first["collector"]
    collector.save(
        workspace_dir,
        meta_extra={
            "produced_by": "analysis.logit_lens",
            "checkpoint": args.checkpoint,
            "checkpoint_file": args.checkpoint_file,
            "seed": args.seed,
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
            "start_offset": int(first["start_offset"]),
            "n_batches_in_study": len(per_batch),
        },
    )
    targets_path = os.path.join(workspace_dir, "targets.npy")
    np.save(targets_path, first["targets"])


def _plot_trajectories(
    aggregate: dict[str, Any],
    output_dir: str,
    n_layer_mid: int,
    n_repeat: int,
) -> None:
    """Save the per-layer top-1 trajectory + entropy trajectory PNGs."""
    mean_top1 = np.asarray(aggregate["mean_top1_lnf"], dtype=np.float64)
    mean_entropy = np.asarray(aggregate["mean_entropy_lnf"], dtype=np.float64)

    repeat_palette = palette_for_repeats(n_repeat)

    fig, axes = setup_figure(1, 2, size=(14.0, 5.0))
    ax_top1, ax_ent = axes

    for repeat_idx in range(n_repeat):
        ax_top1.plot(
            np.arange(n_layer_mid),
            mean_top1[:, repeat_idx],
            color=repeat_palette[repeat_idx % len(repeat_palette)],
            label=f"r={repeat_idx + 1}",
        )
    ax_top1.set_xlabel("Mid-block layer")
    ax_top1.set_ylabel("Top-1 accuracy (logit lens via ln_f)")
    ax_top1.set_title("Per-layer per-repeat logit-lens top-1 accuracy")
    ax_top1.legend()
    ax_top1.grid(True, alpha=0.3)

    for repeat_idx in range(n_repeat):
        ax_ent.plot(
            np.arange(n_layer_mid),
            mean_entropy[:, repeat_idx],
            color=repeat_palette[repeat_idx % len(repeat_palette)],
            label=f"r={repeat_idx + 1}",
        )
    ax_ent.set_xlabel("Mid-block layer")
    ax_ent.set_ylabel("Entropy (nats)")
    ax_ent.set_title("Per-layer per-repeat logit-lens entropy")
    ax_ent.legend()
    ax_ent.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    savefig(fig, os.path.join(output_dir, "logit_lens_trajectories.png"))


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        config_mode=args.config_mode,
        device=args.device,
        module_path=args.module_path,
    )

    per_batch: list[dict[str, Any]] = []
    for batch_idx in range(args.n_batches):
        start_offset = batch_idx * args.max_tokens
        per_batch.append(
            _run_one_batch(model, config, args, start_offset, device=args.device)
        )

    aggregate = _aggregate_per_batch(per_batch, cv_threshold=args.cv_threshold)
    h0 = _h0_verdict(aggregate)

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "checkpoint": args.checkpoint,
        "checkpoint_file": args.checkpoint_file,
        "seed": args.seed,
        "n_batches": args.n_batches,
        "max_tokens": args.max_tokens,
        "n_layer_mid": int(per_batch[0]["n_layer_mid"]),
        "n_repeat": int(per_batch[0]["n_repeat"]),
        "per_batch": [
            {
                "batch_idx": idx,
                "start_offset": int(batch["start_offset"]),
                "total_tokens": int(batch["total_tokens"]),
                "top1_lnf": batch["top1_lnf"].tolist(),
                "top1_raw": batch["top1_raw"].tolist(),
                "entropy_lnf": batch["entropy_lnf"].tolist(),
                "entropy_raw": batch["entropy_raw"].tolist(),
                "kl_lnf": batch["kl_lnf"].tolist(),
                "kl_raw": batch["kl_raw"].tolist(),
                "ln_mid_top1_lnf": batch["ln_mid_top1_lnf"].tolist(),
                "ln_mid_top1_raw": batch["ln_mid_top1_raw"].tolist(),
                "ln_mid_entropy_lnf": batch["ln_mid_entropy_lnf"].tolist(),
                "ln_mid_kl_lnf": batch["ln_mid_kl_lnf"].tolist(),
            }
            for idx, batch in enumerate(per_batch)
        ],
        "aggregate": aggregate,
        "h0_test": h0,
    }
    with open(os.path.join(args.output_dir, "logit_lens_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _plot_trajectories(
        aggregate,
        args.output_dir,
        per_batch[0]["n_layer_mid"],
        per_batch[0]["n_repeat"],
    )

    if args.workspace:
        _persist_workspace(args.workspace, per_batch, args)


if __name__ == "__main__":
    main()
