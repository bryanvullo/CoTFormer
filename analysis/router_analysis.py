"""Protocol D -- Router Policy Analysis (ADM variants only).

Scope
-----
Analyses the per-token routing decisions produced by the ADM (Adaptive
Depth Model) router stack at ``model.transformer.mod[r].mod_router``.
ADM variants (C4, C5 in the Analysis Matrix; the 60k-step v1 and v2
checkpoints) expose ``n_repeat - 1`` routers, each an ``nn.Linear``
producing a per-token continue / halt logit for entering the next
repeat. C1, C2, C3 do NOT instantiate ``transformer.mod`` and this
script short-circuits to a ``not_applicable`` verdict for those
checkpoints rather than raising.

Three per-router metrics are computed over the OWT2 validation forward:

- **Decisiveness** (``mean |sigmoid(logit) - 0.5|``): average absolute
  distance from the indifference point. A decisive router produces
  outputs near 0 or 1; a near-random router clusters around 0.5. The
  metric is bounded in ``[0, 0.5]``; values near 0.5 indicate a
  near-deterministic binary gate.
- **Bernoulli entropy** (``- p log2 p - (1 - p) log2 (1 - p)``
  averaged across tokens): information content of the continue / halt
  decision. Uniform random (``p = 0.5``) yields 1 bit; a deterministic
  gate yields 0 bits.
- **Continue rate**: fraction of tokens whose sigmoid-scored logit
  exceeds 0.5 (the implicit continue-threshold inherited from the
  training-time topk cascade).

The cross-repeat policy stability is measured by the **Jaccard overlap
matrix** between continue-sets: for each pair of routers ``(r_i, r_j)``,
``J_{i,j} = |C_i \\cap C_j| / |C_i \\cup C_j|`` where
``C_k = {t : sigmoid(logit_k(t)) > 0.5}``. A matrix near the identity
indicates each router learns a distinct policy; a matrix with high
off-diagonal entries indicates router redundancy across repeat
positions.

Falsifiability relevance
------------------------
H0 "ADM routers implement interchangeable near-random policies" is
rejected when:

- mean Decisiveness across routers exceeds 0.2 (i.e. average distance
  from 0.5 is bounded well away from 0); AND
- Bernoulli entropy mean is below 0.8 bits across routers (i.e. the
  policy is more decisive than a fair coin); AND
- the off-diagonal Jaccard overlap does not uniformly exceed 0.7 (i.e.
  routers do not make near-identical decisions for most tokens).

The cross-version comparison (C4 v1 vs C5 v2) is exposed via the
``--compare-against`` flag: given a completed prior-checkpoint JSON
results file, the script computes the per-repeat Jaccard overlap
between the current version's continue-set and the prior version's
continue-set at the same router index. Per DIR-001 T3 the H0
"router decisions identical between v1 and v2 at the same repeat
index" is rejected at ``min overlap across routers < 0.7``; this is
recorded as ``cross_version_verdict`` in the output JSON.

Ontological purpose
-------------------
The paper's "budget-adaptive" claim on ADM presumes routers learn
useful token-difficulty proxies. Protocol D provides a first-order
empirical check of that claim: if routers are near-random
(Bernoulli entropy near 1 bit; decisiveness near 0) the adaptive
budget reduces to a stochastic thinning of the mid-block and the
claim fails. Triangulated with the full RQ4 partial-correlation
(per-token CE loss vs halting depth) and RQ5 context-preservation
analyses; those two DVs live in later-wave Protocol D extensions
(DIR-002 reserve).

Cross-reference
---------------
``get_router_weights.py`` at repo root carries the original
author-derived router-extraction logic. The three `` # Modification``
comments at top of that file document the deltas from the paper's
reference code; Protocol D honours those deltas implicitly via the
collector's ``ROUTER_LOGITS`` site (which hooks the same
``mod[k].mod_router`` module but captures raw logits before the topk
cascade, matching the `hooks` extraction method in
``get_router_weights.py`` line 256 onwards). If the shapes written by
the collector disagree with ``get_router_weights.py``'s output at
capacity_factor=1.0, the discrepancy must be escalated rather than
silently papered over.

Entry point
-----------
``python -m analysis.router_analysis \\
    --checkpoint <ckpt_dir> \\
    --checkpoint-file <ckpt_file> \\
    --workspace <path> \\
    --output-dir <path> \\
    [--compare-against <prior router_analysis_results.json>]``

Short-circuit behaviour: if the loaded model has no
``transformer.mod`` attribute (i.e. non-ADM), the script writes
``{"verdict": "not_applicable", ...}`` and exits cleanly without
raising.

Outputs
-------
- ``router_analysis_results.json``: per-router decisiveness / entropy /
  continue-rate plus the Jaccard overlap matrix and the optional
  cross-version verdict.
- ``router_policy_heatmap.png``: (n_router, n_router) Jaccard overlap
  heatmap.
- ``router_decisiveness_bars.png``: per-router decisiveness, entropy,
  and continue-rate triple-bar chart.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from typing import Any

import numpy as np


# --------------------------------------------------------------
# Frozen configuration constants (DIR-001 commit)
# --------------------------------------------------------------

CONTINUE_THRESHOLD = 0.5         # sigmoid(logit) > this -> continue
DECISIVENESS_H0_FLOOR = 0.2      # mean decisiveness must exceed to reject H0
ENTROPY_H0_CEILING_BITS = 0.8    # mean Bernoulli entropy must fall below
JACCARD_REDUNDANCY_CEILING = 0.7 # off-diagonal Jaccard must not exceed this
JACCARD_STABILITY_FLOOR = 0.7    # DIR-001 cross-version stability threshold
EPS = 1e-12


# --------------------------------------------------------------
# Router-metric primitives (pure numpy)
# --------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid computation for logits of arbitrary magnitude."""
    # For x >= 0: 1 / (1 + exp(-x)); for x < 0: exp(x) / (1 + exp(x))
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def decisiveness(logits: np.ndarray) -> float:
    """Mean absolute distance from 0.5 of sigmoid(logits).

    Bounded in ``[0, 0.5]``; a deterministic gate (all tokens either
    strongly continue or strongly halt) scores near 0.5, a near-random
    coin-flip scores near 0.
    """
    if logits.size == 0:
        return float("nan")
    p = _sigmoid(logits.astype(np.float64))
    return float(np.mean(np.abs(p - 0.5)))


def bernoulli_entropy_bits(logits: np.ndarray) -> float:
    """Mean per-token Bernoulli entropy in bits.

    ``H(p) = - p log2 p - (1 - p) log2 (1 - p)`` with the usual
    ``0 log 0 = 0`` convention; uniform (p = 0.5) gives 1 bit, a
    deterministic gate gives 0.
    """
    if logits.size == 0:
        return float("nan")
    p = _sigmoid(logits.astype(np.float64))
    p = np.clip(p, EPS, 1.0 - EPS)
    h = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return float(np.mean(h))


def continue_rate(logits: np.ndarray, threshold: float = CONTINUE_THRESHOLD) -> float:
    """Fraction of tokens with ``sigmoid(logit) > threshold``."""
    if logits.size == 0:
        return float("nan")
    p = _sigmoid(logits.astype(np.float64))
    return float(np.mean(p > threshold))


def continue_mask(logits: np.ndarray, threshold: float = CONTINUE_THRESHOLD) -> np.ndarray:
    """Boolean mask ``sigmoid(logit) > threshold`` per token."""
    p = _sigmoid(logits.astype(np.float64))
    return p > threshold


def jaccard_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Jaccard overlap ``|A \\cap B| / |A \\cup B|`` on boolean masks.

    Returns 1.0 when both masks are empty (by the conventional
    ``0 / 0 = 1`` extension; no tokens to disagree on). The expected
    overlap under independent uniform-random routers with
    ``P(continue) = p`` on both sides is ``p^2 / (2 p - p^2)`` by
    simple set algebra; uniform 50-50 routers expect
    ``0.25 / 0.75 = 0.333`` and highly-positive routers
    (``P = 0.9`` both sides) expect ``0.81 / 0.99 = 0.818``.
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"jaccard_overlap: shape mismatch {mask_a.shape} vs {mask_b.shape}"
        )
    inter = int(np.sum(mask_a & mask_b))
    union = int(np.sum(mask_a | mask_b))
    if union == 0:
        return 1.0
    return inter / union


def jaccard_matrix(masks: list[np.ndarray]) -> np.ndarray:
    """Full symmetric Jaccard overlap matrix for a list of boolean masks."""
    n = len(masks)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            out[i, j] = jaccard_overlap(masks[i], masks[j])
            out[j, i] = out[i, j]
    return out


# --------------------------------------------------------------
# Workspace I/O
# --------------------------------------------------------------


def _discover_routers(workspace: str) -> dict[str, Any]:
    """Scan ``workspace`` for ``router_mod<K>.npy`` entries.

    The collector's ROUTER_LOGITS site produces one file per router at
    ``mod[k].mod_router`` following the key naming in
    ``analysis.common.collector.ActivationCollector._make_site_hook``.
    """
    if not os.path.isdir(workspace):
        raise FileNotFoundError(
            f"router_analysis: workspace {workspace!r} does not exist"
        )
    files = os.listdir(workspace)
    prefix = "router_mod"
    suffix = ".npy"
    indexed: dict[int, str] = {}
    for name in files:
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        stem = name[len(prefix):-len(suffix)]
        try:
            idx = int(stem)
        except ValueError:
            continue
        indexed[idx] = os.path.join(workspace, name)
    ordered = [indexed[k] for k in sorted(indexed)]
    return {
        "count": len(ordered),
        "indices": sorted(indexed),
        "files": ordered,
    }


def _load_router_arrays(discovered: dict[str, Any]) -> list[np.ndarray]:
    """Load each ``router_mod<K>.npy`` into a 1D logit array."""
    arrays: list[np.ndarray] = []
    for path in discovered["files"]:
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[-1] == 1:
            arr = arr.reshape(-1)
        elif arr.ndim == 1:
            pass
        else:
            raise ValueError(
                f"router_analysis: expected (N, 1) or (N,) shape at "
                f"{path!r}; got {arr.shape}"
            )
        arrays.append(arr.astype(np.float64))
    return arrays


# --------------------------------------------------------------
# Capture stage (calls collector) -- only run when workspace is empty
# --------------------------------------------------------------


def _run_capture_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Run the ROUTER_LOGITS capture. Returns the model's ``n_repeat``.

    Short-circuits to ``{"short_circuit": True, "reason": ...}`` when
    the loaded model has no ``transformer.mod`` list (non-ADM).
    """
    import torch

    from analysis.common.collector import ActivationCollector
    from analysis.common.data import iterate_owt2_val
    from analysis.common.loader import load_model_from_checkpoint
    from analysis.common.sites import ActivationSite

    model, config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        device=args.device,
        config_mode=args.config_mode,
        module_path=args.module_path,
    )
    model.eval()

    mod_list = getattr(getattr(model, "transformer", None), "mod", None)
    if mod_list is None or len(mod_list) == 0:
        return {
            "short_circuit": True,
            "reason": "non-ADM checkpoint, no mod[] router layers",
            "n_repeat": int(getattr(model, "n_repeat", 1)),
        }

    dtype = getattr(config, "dtype", None)
    if isinstance(dtype, str):
        dtype = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}.get(dtype)
    if args.device.startswith("cuda") and dtype is not None:
        type_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        type_ctx = nullcontext()

    data_iter = iterate_owt2_val(
        data_dir=args.data_dir,
        config=config,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        total_tokens=args.max_tokens,
        device=args.device,
        start_offset=0,
        split="val",
    )

    collector = ActivationCollector(
        model,
        [ActivationSite.ROUTER_LOGITS],
        non_flash=False,
        module_path=args.module_path,
    )
    os.makedirs(args.workspace, exist_ok=True)
    with collector:
        total = collector.run(data_iter, type_ctx=type_ctx, max_tokens=args.max_tokens)
    collector.save(
        args.workspace,
        meta_extra={
            "protocol": "D",
            "analysis": "router_analysis",
            "seq_length": int(args.seq_length),
            "max_tokens": int(args.max_tokens),
            "checkpoint_dir": str(args.checkpoint),
            "checkpoint_filename": str(args.checkpoint_file),
            "n_tokens_captured": int(total),
        },
    )
    return {
        "short_circuit": False,
        "n_repeat": int(getattr(model, "n_repeat", 1)),
    }


# --------------------------------------------------------------
# Analysis stage
# --------------------------------------------------------------


def _analyse_routers(workspace: str) -> dict[str, Any]:
    """Load router logits from workspace; compute metrics + overlap matrix."""
    discovered = _discover_routers(workspace)
    if discovered["count"] == 0:
        return {
            "verdict": "not_applicable",
            "reason": (
                "no router_mod*.npy files in workspace; either the "
                "capture short-circuited on a non-ADM checkpoint, or the "
                "workspace path is wrong"
            ),
        }
    arrays = _load_router_arrays(discovered)
    n_routers = len(arrays)
    per_router: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    for idx, logits in zip(discovered["indices"], arrays):
        dec = decisiveness(logits)
        ent = bernoulli_entropy_bits(logits)
        cr = continue_rate(logits)
        masks.append(continue_mask(logits))
        per_router.append({
            "router_index": int(idx),
            "n_tokens": int(logits.size),
            "decisiveness": dec,
            "bernoulli_entropy_bits": ent,
            "continue_rate": cr,
        })

    overlap = jaccard_matrix(masks)
    # Off-diagonal summary statistics: these are the scientifically
    # meaningful entries (diagonal is always 1.0 by construction).
    if n_routers >= 2:
        iu = np.triu_indices(n_routers, k=1)
        off_diag = overlap[iu]
    else:
        off_diag = np.asarray([])

    mean_dec = float(np.nanmean([r["decisiveness"] for r in per_router]))
    mean_ent = float(np.nanmean([r["bernoulli_entropy_bits"] for r in per_router]))
    mean_cont = float(np.nanmean([r["continue_rate"] for r in per_router]))

    # H0 "routers interchangeable near-random" rejection verdict.
    verdict_components = {
        "mean_decisiveness_above_floor": bool(mean_dec > DECISIVENESS_H0_FLOOR),
        "mean_entropy_below_ceiling": bool(mean_ent < ENTROPY_H0_CEILING_BITS),
        "off_diag_jaccard_below_ceiling": bool(
            off_diag.size == 0
            or float(np.max(off_diag)) < JACCARD_REDUNDANCY_CEILING
        ),
    }
    rejects_interchangeable_null = all(verdict_components.values())

    return {
        "verdict": "applicable",
        "n_routers": n_routers,
        "router_indices": discovered["indices"],
        "per_router": per_router,
        "overlap_matrix": overlap.tolist(),
        "off_diagonal_jaccard_mean": float(np.mean(off_diag)) if off_diag.size else float("nan"),
        "off_diagonal_jaccard_max": float(np.max(off_diag)) if off_diag.size else float("nan"),
        "off_diagonal_jaccard_min": float(np.min(off_diag)) if off_diag.size else float("nan"),
        "mean_decisiveness": mean_dec,
        "mean_bernoulli_entropy_bits": mean_ent,
        "mean_continue_rate": mean_cont,
        "rejects_interchangeable_null": rejects_interchangeable_null,
        "null_components": verdict_components,
        "constants": {
            "continue_threshold": CONTINUE_THRESHOLD,
            "decisiveness_H0_floor": DECISIVENESS_H0_FLOOR,
            "entropy_H0_ceiling_bits": ENTROPY_H0_CEILING_BITS,
            "jaccard_redundancy_ceiling": JACCARD_REDUNDANCY_CEILING,
        },
        "continue_masks_sha1": [
            _mask_fingerprint(mask) for mask in masks
        ],
    }


def _mask_fingerprint(mask: np.ndarray) -> str:
    """Short SHA-1 fingerprint of the continue-mask for reproducibility checks."""
    import hashlib

    return hashlib.sha1(mask.astype(np.uint8).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------
# Cross-version comparison (DIR-001 T3 RQ3-like verdict)
# --------------------------------------------------------------


def _load_prior_continue_masks(
    prior_results_path: str,
    current_workspace: str,
    current_masks: list[np.ndarray],
) -> list[np.ndarray] | None:
    """Attempt to rebuild the prior-checkpoint continue masks.

    The prior JSON alone does not carry the raw masks (they are
    too large for a JSON blob); the prior ``--workspace`` path is
    recorded in the prior JSON's ``meta.workspace`` field if the user
    ran the prior analysis with the ``--record-workspace-path`` flag.
    If unavailable, we fall back to the mask fingerprint as an
    equality probe (strict; only detects exact matches, not partial
    overlap).
    """
    if not os.path.isfile(prior_results_path):
        return None
    try:
        with open(prior_results_path) as fh:
            prior = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    prior_workspace = prior.get("meta", {}).get("workspace")
    if prior_workspace and os.path.isdir(prior_workspace):
        prior_discovered = _discover_routers(prior_workspace)
        if prior_discovered["count"] == len(current_masks):
            prior_arrays = _load_router_arrays(prior_discovered)
            return [continue_mask(a) for a in prior_arrays]
    return None


def _cross_version_verdict(
    current_masks: list[np.ndarray],
    prior_masks: list[np.ndarray] | None,
    prior_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Per-router Jaccard overlap between current and prior masks.

    Per DIR-001 T3: H0 "router decisions identical between versions at
    the same repeat index" rejected at ``min overlap < 0.7``. Falls
    back to the fingerprint-equality probe if raw masks are
    unavailable.
    """
    if prior_masks is not None:
        if len(prior_masks) != len(current_masks):
            return {
                "status": "router_count_mismatch",
                "current_n_routers": len(current_masks),
                "prior_n_routers": len(prior_masks),
            }
        per_router: list[dict[str, Any]] = []
        overlaps: list[float] = []
        for idx, (cur, prev) in enumerate(zip(current_masks, prior_masks)):
            if cur.shape != prev.shape:
                # N_tokens mismatch: fall back to mask-fingerprint probe.
                per_router.append({
                    "router_index": idx,
                    "status": "token_count_mismatch",
                    "current_n_tokens": int(cur.size),
                    "prior_n_tokens": int(prev.size),
                })
                continue
            overlap = jaccard_overlap(cur, prev)
            overlaps.append(overlap)
            per_router.append({
                "router_index": idx,
                "jaccard": float(overlap),
                "below_stability_floor": bool(overlap < JACCARD_STABILITY_FLOOR),
            })
        if overlaps:
            min_overlap = float(np.min(overlaps))
            mean_overlap = float(np.mean(overlaps))
        else:
            min_overlap = float("nan")
            mean_overlap = float("nan")
        return {
            "status": "direct_jaccard",
            "per_router": per_router,
            "min_overlap": min_overlap,
            "mean_overlap": mean_overlap,
            "stability_floor": JACCARD_STABILITY_FLOOR,
            "rejects_identical_null": bool(
                math.isfinite(min_overlap)
                and min_overlap < JACCARD_STABILITY_FLOOR
            ),
        }
    if prior_results is None:
        return {
            "status": "no_prior",
            "reason": "no prior results file passed or file missing",
        }
    # Fingerprint-equality probe: compare fingerprints per router.
    prior_fps = prior_results.get("continue_masks_sha1", [])
    current_fps = [_mask_fingerprint(m) for m in current_masks]
    if len(prior_fps) != len(current_fps):
        return {
            "status": "fingerprint_router_count_mismatch",
            "current_n_routers": len(current_fps),
            "prior_n_routers": len(prior_fps),
        }
    per_router = [
        {
            "router_index": i,
            "current_fingerprint": cur,
            "prior_fingerprint": prev,
            "identical_policies": cur == prev,
        }
        for i, (cur, prev) in enumerate(zip(current_fps, prior_fps))
    ]
    return {
        "status": "fingerprint_only",
        "reason": (
            "prior workspace raw masks unavailable; fingerprint-equality "
            "probe is a strict all-or-nothing check. For the Jaccard-threshold "
            "verdict re-run the prior protocol with --record-workspace-path."
        ),
        "per_router": per_router,
        "all_identical": all(row["identical_policies"] for row in per_router),
    }


# --------------------------------------------------------------
# Plotting
# --------------------------------------------------------------


def _render_policy_heatmap(results: dict[str, Any], output_dir: str) -> str | None:
    """Render the (n_router, n_router) Jaccard overlap heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if results.get("verdict") != "applicable":
        return None
    overlap = np.asarray(results["overlap_matrix"], dtype=np.float64)
    n = overlap.shape[0]
    fig, ax = plt.subplots(figsize=(max(3, n * 0.8), max(3, n * 0.8)))
    im = ax.imshow(overlap, vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Jaccard overlap")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{overlap[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if overlap[i, j] < 0.6 else "black")
    labels = [f"mod[{i}]" for i in results["router_indices"]]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_title("Router continue-set Jaccard overlap")
    path = os.path.join(output_dir, "router_policy_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_decisiveness_bars(results: dict[str, Any], output_dir: str) -> str | None:
    """Per-router decisiveness / entropy / continue-rate triple-bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if results.get("verdict") != "applicable":
        return None
    per_router = results["per_router"]
    n = len(per_router)
    indices = np.arange(n)
    width = 0.25
    dec_vals = [r["decisiveness"] for r in per_router]
    ent_vals = [r["bernoulli_entropy_bits"] for r in per_router]
    cont_vals = [r["continue_rate"] for r in per_router]

    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), 4))
    ax.bar(indices - width, dec_vals, width, label="Decisiveness", color="#1b9e77")
    ax.bar(indices, ent_vals, width, label="Bernoulli entropy (bits)", color="#d95f02")
    ax.bar(indices + width, cont_vals, width, label="Continue rate", color="#7570b3")
    ax.axhline(DECISIVENESS_H0_FLOOR, color="#1b9e77", linestyle="--", linewidth=0.8,
               label=f"Dec floor = {DECISIVENESS_H0_FLOOR}")
    ax.axhline(ENTROPY_H0_CEILING_BITS, color="#d95f02", linestyle="--", linewidth=0.8,
               label=f"Ent ceiling = {ENTROPY_H0_CEILING_BITS} bits")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"mod[{r['router_index']}]" for r in per_router])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Per-router decisiveness, Bernoulli entropy, continue rate")
    ax.legend(loc="best", fontsize=8)
    path = os.path.join(output_dir, "router_decisiveness_bars.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------
# CLI + main
# --------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """CLI for Protocol D.

    Required: ``--checkpoint``, ``--workspace``, ``--output-dir``.
    Optional: ``--checkpoint-file``, ``--compare-against``,
    ``--skip-capture``, ``--record-workspace-path``.
    """
    parser = argparse.ArgumentParser(
        description="Protocol D -- ADM router policy analysis"
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
        "--workspace", type=str, required=True,
        help="Workspace directory for router_mod*.npy files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for router_analysis_*.json / *.png",
    )
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config-mode", type=str, default="raw",
                        choices=["raw", "argparse"])
    parser.add_argument("--module-path", type=str,
                        default="model.transformer.h_mid")
    parser.add_argument(
        "--skip-capture", action="store_true",
        help="Skip the ROUTER_LOGITS capture stage; assume workspace is populated",
    )
    parser.add_argument(
        "--compare-against", type=str, default=None,
        help=(
            "Path to a prior router_analysis_results.json for the "
            "cross-version stability verdict (DIR-001 T3 RQ3-like)"
        ),
    )
    parser.add_argument(
        "--record-workspace-path", action="store_true",
        help=(
            "Record the current workspace path in the JSON meta block. "
            "The next cross-version comparison can then rebuild the raw "
            "continue masks; omit if the workspace path is ephemeral."
        ),
    )
    return parser


def _write_short_circuit(args: argparse.Namespace, reason: str, n_repeat: int) -> None:
    """Persist the not-applicable verdict and exit cleanly."""
    payload = {
        "verdict": "not_applicable",
        "reason": reason,
        "n_repeat": int(n_repeat),
        "checkpoint": str(args.checkpoint),
        "checkpoint_file": str(args.checkpoint_file),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "router_analysis_results.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.workspace, exist_ok=True)

    discovered_before = _discover_routers(args.workspace)
    has_cache = discovered_before["count"] > 0

    if args.skip_capture and not has_cache:
        # Skip-capture on an empty workspace: attempt to infer whether
        # this is a non-ADM short-circuit vs a missing-capture error by
        # reading the workspace meta.json written by a prior capture.
        meta_path = os.path.join(args.workspace, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
                if meta.get("short_circuit"):
                    _write_short_circuit(
                        args,
                        reason=meta.get(
                            "reason",
                            "workspace meta.json records a prior short-circuit",
                        ),
                        n_repeat=int(meta.get("n_repeat", 1)),
                    )
                    return
            except (json.JSONDecodeError, OSError):
                pass
        raise FileNotFoundError(
            "router_analysis: --skip-capture requested but no "
            f"router_mod*.npy files in {args.workspace!r}. Run the "
            "capture stage first or point --workspace at the correct "
            "collector output directory."
        )

    if not args.skip_capture and not has_cache:
        capture_outcome = _run_capture_stage(args)
        if capture_outcome.get("short_circuit"):
            _write_short_circuit(
                args,
                reason=capture_outcome["reason"],
                n_repeat=int(capture_outcome.get("n_repeat", 1)),
            )
            return

    results = _analyse_routers(args.workspace)

    if results.get("verdict") == "not_applicable":
        # The analysis stage observed no router_mod*.npy; write the
        # short-circuit JSON and exit. This path is hit if skip-capture
        # is set AND the workspace was prepared by a non-ADM run that
        # left meta.json behind without files.
        _write_short_circuit(
            args,
            reason=str(results.get("reason", "no router_mod*.npy in workspace")),
            n_repeat=int(results.get("n_repeat", 1)),
        )
        return

    # Cross-version comparison (optional).
    if args.compare_against:
        # Re-load current masks for the comparison (they are not
        # persisted by _analyse_routers beyond fingerprints).
        current_arrays = _load_router_arrays(_discover_routers(args.workspace))
        current_masks = [continue_mask(a) for a in current_arrays]
        prior_results: dict[str, Any] | None = None
        if os.path.isfile(args.compare_against):
            try:
                with open(args.compare_against) as fh:
                    prior_results = json.load(fh)
            except (json.JSONDecodeError, OSError):
                prior_results = None
        prior_masks = _load_prior_continue_masks(
            args.compare_against, args.workspace, current_masks
        )
        results["cross_version"] = _cross_version_verdict(
            current_masks, prior_masks, prior_results
        )
    else:
        results["cross_version"] = {
            "status": "not_run",
            "reason": "no --compare-against flag",
        }

    # Meta block + optional workspace-path record for downstream
    # cross-version comparisons.
    meta: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_file": str(args.checkpoint_file),
        "max_tokens": int(args.max_tokens),
        "seq_length": int(args.seq_length),
    }
    if args.record_workspace_path:
        meta["workspace"] = os.path.abspath(args.workspace)
    results["meta"] = meta

    path = os.path.join(args.output_dir, "router_analysis_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    _render_policy_heatmap(results, args.output_dir)
    _render_decisiveness_bars(results, args.output_dir)


if __name__ == "__main__":
    main()
