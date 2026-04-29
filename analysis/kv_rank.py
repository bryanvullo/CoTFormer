"""Protocol G -- KV Compressibility (architectural-extension MLA design, RQ6 cross-validation).

Scope
-----
Replaces the legacy ``analyze_kv_compression.py`` monolith. Two
deliberately-separated passes measure different ontological objects:

- **Type A (weight-level)**: SVD of the fused ``c_attn`` matrix's K
  and V slices separately. No data; measures the intrinsic rank of
  the linear projection.
- **Type B (activation-level)**: SVD over real OWT2 KV vectors
  captured by the Protocol A collector. Per-(layer, repeat); computes
  the KV-CoRE normalised effective rank (Chen et al. 2026 [8])
  alongside the classical 0.90 / 0.95 / 0.99 thresholds.

Falsifiability relevance
------------------------
Prediction 1 "weight decay drives QK/OV to low rank" is rejected
(reported as "inconsistent with literature") if Type A yields
``rank_95 > 40`` (more than about 63 per cent of nominal
``d_head = 64``, per the two-tier threshold in
`docs/extend-notes.md` §1.4 Prediction 1). The prediction is framed
observationally: we compare the frozen-checkpoint rank profile at
``wd = 0.1`` against Kobayashi et al. 2024 [11]'s rank reductions
at the same weight-decay value on a 125M model with identical
``d_head = 64``. The verdict is exactly one of three
mutually-exclusive labels emitted by ``_prediction_1_verdict``:
"evidence-of-compounding" (every mid-layer rank_95 below 40,
Kobayashi-bracket strictly improved upon),
"consistent-with-literature" (no mid-layer rank_95 exceeds 48,
Kobayashi-bracket respected),
"inconsistent-with-literature" (any mid-layer rank_95 exceeds 48,
literature threshold crossed).

RQ6 cross-validation (`docs/extend-notes.md` §1.2 RQ6 "Rogue
dimensions and three-way cross-validation"): Type B rank at later
repeats should track the Chun participation-ratio trend from
Protocol F within 10 per cent. Disagreement flags a measurement
artefact rather than a conflict; the three metrics (Chun PR,
Kobayashi pseudo-rank, Chen NER) measure different aspects of the
spectrum and are expected to agree qualitatively on trend sign.

Ontological purpose
-------------------
Determines whether the provisional uniform MLA rank
(``kv_lora_rank = 192``) is sufficient, or whether per-layer /
per-repeat rank allocation is warranted. If Type B ranks are
substantially lower than Type A at later repeats, the activation
distribution is MORE compressible than the weight matrix; this
finding would support aggressive per-repeat compression.

Novelty
-------
Novelty Claim 3 (post-hostile-review narrow reframing): first
KV-CoRE on a cross-repeat cache. Chen et al. 2026 [8] benchmark
KV-CoRE on standard multi-layer transformers; the present
per-(layer, repeat) analysis operates on the CoTFormer's
weight-tied cross-repeat KV cache, where the cache at repeat
``r`` depends on repeats ``1..r-1`` in a way that standard caches
do not exhibit.

Implementation
--------------
Type A consumes the loaded model's ``c_attn`` weight matrix (no
data required, CPU only). Type B consumes
``kv_mid_l<L>_r<R>.npy`` caches from the workspace populated by
Protocol A's collector; sample-size convergence is verified by
reporting rank at ``N in {1024, 2048, 4096}`` where the workspace
permits. The centring step before SVD is mandatory (removes the
trivial mean-component from a rank-one bias direction).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

from analysis.common.plotting import (
    palette_for_repeats,
    savefig,
    setup_figure,
    site_label,
)
from analysis.common.sites import discover_workspace_sites
from analysis.common.spectral import compute_effective_rank, kv_core_rank


# -----------------------------------------------------------------------------
# Type A: weight-level SVD
# -----------------------------------------------------------------------------

def _extract_qkv_weights(c_attn_weight: np.ndarray, n_embd: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the fused ``c_attn.weight`` into Q, K, V slices.

    Supports both ``(3*n_embd, n_embd)`` (out, in) and
    ``(n_embd, 3*n_embd)`` (in, out) layouts. nn.Linear in PyTorch
    stores weight as ``(out, in)`` by default; the explicit shape check
    absorbs legacy checkpoints saved with the transposed layout.
    """
    W = np.asarray(c_attn_weight, dtype=np.float64)
    if W.shape[0] == 3 * n_embd:
        Wq = W[:n_embd, :]
        Wk = W[n_embd : 2 * n_embd, :]
        Wv = W[2 * n_embd :, :]
    elif W.shape[1] == 3 * n_embd:
        Wq = W[:, :n_embd]
        Wk = W[:, n_embd : 2 * n_embd]
        Wv = W[:, 2 * n_embd :]
    else:
        raise ValueError(
            f"_extract_qkv_weights: unexpected c_attn shape {W.shape} for n_embd={n_embd}"
        )
    return Wq, Wk, Wv


def _analyse_weight_svd(model: Any) -> list[dict]:
    """SVD of W_K and W_V for every mid-block attention layer.

    Returns a list of per-layer entries. Each entry contains the K
    and V singular values, effective ranks at 90 / 95 / 99 per cent
    cumulative variance, and the cumulative-variance curve.
    """
    results: list[dict] = []
    n_embd = int(model.config.n_embd)

    # Collect every attention block across h_begin, h_mid, h_end. The
    # main analysis sub-stream focuses on h_mid (21 mid layers for C3),
    # but weight-level rank is cheap so we measure h_begin and h_end
    # too for synthesis context.
    layer_groups: list[tuple[str, int, Any]] = []
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise AttributeError("_analyse_weight_svd: model.transformer missing")

    h_begin = getattr(transformer, "h_begin", None) or []
    h_mid = getattr(transformer, "h_mid", None) or getattr(transformer, "h", []) or []
    h_end = getattr(transformer, "h_end", None) or []
    for idx in range(len(h_begin)):
        layer_groups.append(("begin", idx, h_begin[idx]))
    for idx in range(len(h_mid)):
        layer_groups.append(("mid", idx, h_mid[idx]))
    for idx in range(len(h_end)):
        layer_groups.append(("end", idx, h_end[idx]))

    for group, idx, block in layer_groups:
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "c_attn"):
            continue
        weight = attn.c_attn.weight.detach().float().cpu().numpy()
        try:
            Wq, Wk, Wv = _extract_qkv_weights(weight, n_embd)
        except ValueError:
            continue

        sv_k = np.linalg.svd(Wk, full_matrices=False, compute_uv=False)
        sv_v = np.linalg.svd(Wv, full_matrices=False, compute_uv=False)

        ranks_k, cumvar_k = compute_effective_rank(sv_k)
        ranks_v, cumvar_v = compute_effective_rank(sv_v)

        results.append(
            {
                "group": group,
                "layer": idx,
                "label": f"{group}[{idx}]",
                "W_k_shape": list(Wk.shape),
                "W_v_shape": list(Wv.shape),
                "sv_k": sv_k.tolist(),
                "sv_v": sv_v.tolist(),
                "rank_k": {str(t): int(v) for t, v in ranks_k.items()},
                "rank_v": {str(t): int(v) for t, v in ranks_v.items()},
                "cumvar_k": cumvar_k.tolist(),
                "cumvar_v": cumvar_v.tolist(),
            }
        )
    return results


# -----------------------------------------------------------------------------
# Type B: activation-level SVD per (layer, repeat)
# -----------------------------------------------------------------------------

def _discover_kv_sites(workspace_dir: str) -> list[tuple[str, int, int]]:
    """Scan for ``kv_mid_l<L>_r<R>.npy`` files.

    Returns sorted ``(path, layer, repeat)`` tuples. The collector
    writes the fused Q||K||V matrix to each file; the caller splits
    per-slice. Thin wrapper around
    ``analysis.common.sites.discover_workspace_sites`` with the
    ``"kv_mid"`` capture prefix.
    """
    return [
        (path, layer, repeat)
        for layer, repeat, path in discover_workspace_sites(
            workspace_dir, "kv_mid"
        )
    ]


def _analyse_activation_one(
    kv_fused: np.ndarray,
    n_embd: int,
    sample_sizes: list[int],
) -> dict:
    """Per-site SVD of K and V activation slices with sample-size scan.

    Parameters
    ----------
    kv_fused : np.ndarray
        Shape ``(N_tokens, 3 * n_embd)`` fused Q||K||V.
    n_embd : int
        Hidden dimension; K slice is ``[:, n_embd : 2 * n_embd]``.
    sample_sizes : list[int]
        Row counts at which to report effective rank (convergence
        diagnostic). Values larger than ``N_tokens`` are clipped.
    """
    kv = np.asarray(kv_fused, dtype=np.float64)
    N, D3 = kv.shape
    if D3 != 3 * n_embd:
        raise ValueError(
            f"_analyse_activation_one: width {D3} != 3 * n_embd ({3*n_embd})"
        )
    K = kv[:, n_embd : 2 * n_embd]
    V = kv[:, 2 * n_embd : 3 * n_embd]

    convergence_k: dict[int, dict] = {}
    convergence_v: dict[int, dict] = {}

    for requested_N in sample_sizes:
        n_use = min(int(requested_N), N)
        if n_use < 4:
            continue
        K_slice = K[:n_use]
        V_slice = V[:n_use]
        K_c = K_slice - K_slice.mean(axis=0, keepdims=True)
        V_c = V_slice - V_slice.mean(axis=0, keepdims=True)

        sv_k = np.linalg.svd(K_c, full_matrices=False, compute_uv=False)
        sv_v = np.linalg.svd(V_c, full_matrices=False, compute_uv=False)
        ranks_k, _ = compute_effective_rank(sv_k)
        ranks_v, _ = compute_effective_rank(sv_v)

        convergence_k[n_use] = {str(t): int(v) for t, v in ranks_k.items()}
        convergence_v[n_use] = {str(t): int(v) for t, v in ranks_v.items()}

    # Primary measurement uses the full sample. Also compute KV-CoRE NER.
    K_full = K - K.mean(axis=0, keepdims=True)
    V_full = V - V.mean(axis=0, keepdims=True)
    sv_k_full = np.linalg.svd(K_full, full_matrices=False, compute_uv=False)
    sv_v_full = np.linalg.svd(V_full, full_matrices=False, compute_uv=False)
    ranks_k_full, cumvar_k = compute_effective_rank(sv_k_full)
    ranks_v_full, cumvar_v = compute_effective_rank(sv_v_full)
    ner_k_rank, ner_k = kv_core_rank(K_full, target_fidelity=0.99)
    ner_v_rank, ner_v = kv_core_rank(V_full, target_fidelity=0.99)

    return {
        "n_tokens": int(N),
        "rank_k": {str(t): int(v) for t, v in ranks_k_full.items()},
        "rank_v": {str(t): int(v) for t, v in ranks_v_full.items()},
        "cumvar_k_head": cumvar_k[:64].tolist(),
        "cumvar_v_head": cumvar_v[:64].tolist(),
        "sv_k_head": sv_k_full[:64].tolist(),
        "sv_v_head": sv_v_full[:64].tolist(),
        "convergence_k_by_N": convergence_k,
        "convergence_v_by_N": convergence_v,
        "ner_k_rank_99": int(ner_k_rank),
        "ner_v_rank_99": int(ner_v_rank),
        "ner_k_value": float(ner_k),
        "ner_v_value": float(ner_v),
    }


def _analyse_activation_svd(
    workspace_dir: str, n_embd: int, sample_sizes: list[int]
) -> dict:
    """Iterate over every ``kv_mid_l<L>_r<R>.npy`` and measure rank.

    Returns a nested dict keyed by ``site_label("mid", layer, repeat)``.
    """
    sites = _discover_kv_sites(workspace_dir)
    per_site: dict[str, dict] = {}
    for path, layer, repeat in sites:
        kv = np.load(path)
        entry = _analyse_activation_one(kv, n_embd, sample_sizes)
        entry["layer"] = int(layer)
        entry["repeat"] = int(repeat)
        key = site_label("mid", layer, repeat)
        per_site[key] = entry
    return per_site


# -----------------------------------------------------------------------------
# Prediction 1 verdict
# -----------------------------------------------------------------------------

def _prediction_1_verdict(
    weight_results: list[dict], d_head_nominal: int
) -> dict:
    """Two-tier threshold verdict per `docs/extend-notes.md` §1.4.

    Emits exactly one of three mutually-exclusive labels (matching the
    module-level docstring's three-case framing):

    - "evidence-of-compounding": ``n_above_48 == 0`` AND
      ``n_below_40 == n_total``. Strongest reading -- every mid-layer
      KV rank lies BELOW Kobayashi 2024's 125M observation, indicating
      cross-repeat compounding has driven the rank profile down.
    - "consistent-with-literature": ``n_above_48 == 0`` AND at least
      one mid-layer rank_95 in [40, 48]. The Kobayashi-bracket is
      respected (no rank exceeds 48) but compounding is not yet evident.
    - "inconsistent-with-literature": ``n_above_48 > 0``. At least one
      mid-layer rank_95 crosses the strict literature threshold.

    The collapse to three cases is intentional: ``n_above_48 > 0``
    implies ``n_above_40 > 0``, so the previously-defensive
    ``n_above_40 == 0 and n_above_48 > 0`` branch is unreachable.
    """
    mid_entries = [e for e in weight_results if e["group"] == "mid"]
    if not mid_entries:
        return {"verdict": "no_mid_layers", "n_mid": 0}

    rank95_k = [int(e["rank_k"]["0.95"]) for e in mid_entries]
    rank95_v = [int(e["rank_v"]["0.95"]) for e in mid_entries]
    combined = rank95_k + rank95_v

    n_above_40 = int(sum(1 for r in combined if r > 40))
    n_above_48 = int(sum(1 for r in combined if r > 48))
    n_below_40 = int(sum(1 for r in combined if r < 40))
    n_total = int(len(combined))

    # Three-tier verdict: once any rank exceeds 48 the sample has crossed
    # the literature threshold and is classified inconsistent. The
    # n_above_40 == 0 branch was previously defensively duplicated here
    # but is unreachable (n_above_48 > 0 implies n_above_40 > 0), so the
    # classification collapses to three mutually-exclusive cases.
    if n_above_48 == 0 and n_below_40 == n_total:
        verdict = "evidence-of-compounding"
    elif n_above_48 == 0:
        verdict = "consistent-with-literature"
    else:
        verdict = "inconsistent-with-literature"

    return {
        "verdict": verdict,
        "d_head_nominal": int(d_head_nominal),
        "n_mid": int(len(mid_entries)),
        "n_measurements": int(n_total),
        "rank95_k_mid": rank95_k,
        "rank95_v_mid": rank95_v,
        "n_rank95_above_40": n_above_40,
        "n_rank95_above_48": n_above_48,
        "n_rank95_below_40": n_below_40,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _plot_kv_rank(
    weight_results: list[dict],
    activation_results: dict,
    target_rank: int,
    output_path: str,
) -> None:
    """Multi-panel: SV spectra, per-layer rank bars, per-(layer, repeat) heatmap."""
    fig, axes = setup_figure(2, 2, size=(14.0, 10.0))

    # (0, 0): Weight-level K SV spectra (log scale, normalised)
    ax = axes[0, 0]
    for entry in weight_results:
        sv = np.asarray(entry["sv_k"], dtype=np.float64)
        if sv.size == 0 or sv[0] <= 0:
            continue
        ax.semilogy(sv / sv[0], alpha=0.6, linewidth=0.8, label=entry["label"])
    if target_rank > 0:
        ax.axvline(x=target_rank, color="red", linestyle="--", alpha=0.7, label=f"target={target_rank}")
    ax.set_xlabel("SV index")
    ax.set_ylabel("SV (normalised by SV[0], log)")
    ax.set_title("W_K SV spectrum per layer (Type A)")

    # (0, 1): Weight-level rank at 90/95/99 per layer (bar group)
    ax2 = axes[0, 1]
    labels = [e["label"] for e in weight_results]
    x = np.arange(len(weight_results))
    width = 0.25
    k90 = [int(e["rank_k"]["0.9"]) for e in weight_results]
    k95 = [int(e["rank_k"]["0.95"]) for e in weight_results]
    k99 = [int(e["rank_k"]["0.99"]) for e in weight_results]
    ax2.bar(x - width, k90, width, label="K 90%", alpha=0.8)
    ax2.bar(x, k95, width, label="K 95%", alpha=0.8)
    ax2.bar(x + width, k99, width, label="K 99%", alpha=0.8)
    if target_rank > 0:
        ax2.axhline(y=target_rank, color="red", linestyle="--", alpha=0.7, label=f"MLA={target_rank}")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Effective rank")
    ax2.set_title("W_K rank at 90/95/99 per layer (Type A)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.legend(fontsize=7)

    # (1, 0): Activation rank (K, rank_95) per (layer, repeat) heatmap
    # Build a layer x repeat matrix. Unpopulated cells = NaN.
    if activation_results:
        layers = sorted({e["layer"] for e in activation_results.values()})
        repeats = sorted({e["repeat"] for e in activation_results.values()})
        mat = np.full((len(layers), len(repeats)), np.nan, dtype=np.float64)
        for key, entry in activation_results.items():
            li = layers.index(int(entry["layer"]))
            ri = repeats.index(int(entry["repeat"]))
            mat[li, ri] = float(entry["rank_k"]["0.95"])
        ax3 = axes[1, 0]
        im = ax3.imshow(mat, aspect="auto", cmap="viridis")
        ax3.set_xticks(range(len(repeats)))
        ax3.set_xticklabels([f"r={r}" for r in repeats])
        ax3.set_yticks(range(len(layers)))
        ax3.set_yticklabels([f"l{l}" for l in layers], fontsize=7)
        ax3.set_xlabel("Repeat")
        ax3.set_ylabel("Layer")
        ax3.set_title("K activation rank_95 per (layer, repeat) (Type B)")
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    else:
        axes[1, 0].set_title("Type B activation rank (no workspace)")
        axes[1, 0].axis("off")

    # (1, 1): KV-CoRE NER per (layer, repeat) heatmap
    if activation_results:
        layers = sorted({e["layer"] for e in activation_results.values()})
        repeats = sorted({e["repeat"] for e in activation_results.values()})
        mat = np.full((len(layers), len(repeats)), np.nan, dtype=np.float64)
        for key, entry in activation_results.items():
            li = layers.index(int(entry["layer"]))
            ri = repeats.index(int(entry["repeat"]))
            mat[li, ri] = float(entry["ner_k_value"])
        ax4 = axes[1, 1]
        im = ax4.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax4.set_xticks(range(len(repeats)))
        ax4.set_xticklabels([f"r={r}" for r in repeats])
        ax4.set_yticks(range(len(layers)))
        ax4.set_yticklabels([f"l{l}" for l in layers], fontsize=7)
        ax4.set_xlabel("Repeat")
        ax4.set_ylabel("Layer")
        ax4.set_title("KV-CoRE NER (K) per (layer, repeat) (Chen 2026)")
        fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    else:
        axes[1, 1].set_title("KV-CoRE NER (no workspace)")
        axes[1, 1].axis("off")

    fig.suptitle("Protocol G -- KV rank (Type A + Type B + KV-CoRE NER)", fontsize=13)
    savefig(fig, output_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol G."""
    parser = argparse.ArgumentParser(
        description="Protocol G -- KV compressibility (Type A + Type B + KV-CoRE)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory (must contain summary.json and ckpt file)",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="ckpt.pt",
        help="Checkpoint filename within --checkpoint (default: ckpt.pt)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="",
        help=(
            "Analysis workspace directory containing kv_mid_l*_r*.npy. When "
            "empty or --skip-activations is passed, Type B is skipped."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for kv_rank_results.json and kv_rank_plots.png",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="RNG seed; currently unused but reserved for future bootstrap",
    )
    parser.add_argument(
        "--target-rank",
        type=int,
        default=192,
        help="Reference MLA kv_lora_rank for the comparison line (default 192)",
    )
    parser.add_argument(
        "--skip-activations",
        action="store_true",
        help="Skip Type B activation-level SVD (useful for weight-only runs)",
    )
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated sample sizes for Type B convergence scan",
    )
    parser.add_argument(
        "--d-head-nominal",
        type=int,
        default=64,
        help="Nominal d_head for Prediction 1 comparison (default 64)",
    )
    # ---- KV capture (only when workspace does not already contain kv files) ----
    parser.add_argument(
        "--capture-kv",
        action="store_true",
        help=(
            "Run the collector to populate --workspace with kv_mid_l*_r*.npy "
            "if those files are missing. Requires --data-dir and a GPU."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset directory for --capture-kv (default: use config.data_dir)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for --capture-kv forward pass; ignored when Type B reads existing cache",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=256,
        help="Sequence length per batch during --capture-kv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size during --capture-kv",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Per-batch token budget during --capture-kv (default 2048)",
    )
    return parser


def _capture_kv_activations(
    checkpoint_dir: str,
    checkpoint_file: str,
    workspace_dir: str,
    data_dir: str | None,
    device: str,
    seq_length: int,
    batch_size: int,
    max_tokens: int,
) -> None:
    """Collector pass that writes kv_mid_l<L>_r<R>.npy into ``workspace_dir``.

    Invoked only when ``--capture-kv`` is set and the workspace is missing
    the ``kv_mid_*.npy`` files. Uses the shared
    ``analysis.common.collector.ActivationCollector`` at the
    ``ActivationSite.KV_C_ATTN`` site.

    Collector-only principle (see docs/extend-technical.md §8.2): the
    collector is the single site that OWNS activation capture; protocol
    scripts extract from the workspace cache it produces. This helper
    is a documented exception justified by the CV-gate accounting
    constraint: Protocol A's 4-batch inter-batch CV threshold is
    per-residual-site, and adding KV_C_ATTN to the Protocol A capture
    set would change the per-batch buffer shape and invalidate the
    pre-registered CV computation. The exception is local to this
    script, runs only when the workspace lacks the cache, and uses the
    shared ``ActivationCollector`` class (never a bespoke capture path).
    """
    import torch

    from analysis.common.collector import ActivationCollector
    from analysis.common.data import iterate_owt2_val
    from analysis.common.loader import load_model_from_checkpoint
    from analysis.common.sites import ActivationSite

    model, config = load_model_from_checkpoint(
        checkpoint_dir,
        checkpoint_file=checkpoint_file,
        config_mode="raw",
        device=device,
    )
    if device != "cpu":
        type_ctx = torch.amp.autocast(
            device_type="cuda" if device.startswith("cuda") else device,
            dtype=torch.bfloat16,
        )
    else:
        from contextlib import nullcontext

        type_ctx = nullcontext()

    with ActivationCollector(
        model=model,
        sites=[ActivationSite.KV_C_ATTN],
        non_flash=False,
        module_path="model.transformer.h_mid",
    ) as collector:
        data_iter = iterate_owt2_val(
            data_dir=data_dir,
            config=config,
            seq_length=seq_length,
            batch_size=batch_size,
            total_tokens=max_tokens,
            device=device,
        )
        total = collector.run(data_iter, type_ctx=type_ctx, max_tokens=max_tokens)
        collector.save(workspace_dir, meta_extra={"captured_by": "kv_rank"})
        print(f"kv_rank: captured {total} tokens into {workspace_dir}")


def _workspace_has_kv(workspace_dir: str) -> bool:
    """Return True if any ``kv_mid_l<L>_r<R>.npy`` exists in workspace_dir."""
    from analysis.common.sites import discover_workspace_sites
    try:
        return bool(discover_workspace_sites(workspace_dir, "kv_mid"))
    except FileNotFoundError:
        return False


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # If --capture-kv is set and the workspace lacks the kv cache, run the
    # collector pass first. This pre-empty-workspace hand-off pattern keeps
    # the GPU stage single-step from the job.sh caller's point of view.
    if (
        args.capture_kv
        and args.workspace
        and not args.skip_activations
        and not _workspace_has_kv(args.workspace)
    ):
        os.makedirs(args.workspace, exist_ok=True)
        _capture_kv_activations(
            checkpoint_dir=args.checkpoint,
            checkpoint_file=args.checkpoint_file,
            workspace_dir=args.workspace,
            data_dir=args.data_dir,
            device=args.device,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )

    # Deferred torch/model imports so the pure-numpy path can be tested on
    # environments without torch installed.
    import torch  # noqa: F401  -- needed by common.loader

    from analysis.common.loader import load_model_from_checkpoint

    print(f"kv_rank: loading checkpoint {args.checkpoint}")
    model, config = load_model_from_checkpoint(
        args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        config_mode="raw",
        device="cpu",
    )

    print("kv_rank: Type A -- weight-level SVD")
    weight_results = _analyse_weight_svd(model)
    if not weight_results:
        print("kv_rank: no weight SVD results (empty block lists)", file=sys.stderr)

    # Free the model before heavy activation work to reduce peak memory.
    del model

    activation_results: dict[str, dict] = {}
    if args.workspace and not args.skip_activations:
        n_embd = int(config.n_embd)
        sample_sizes = [int(x) for x in args.sample_sizes.split(",") if x.strip()]
        print(
            f"kv_rank: Type B -- activation SVD over {args.workspace} "
            f"at N in {sample_sizes}"
        )
        activation_results = _analyse_activation_svd(
            args.workspace, n_embd, sample_sizes
        )
    else:
        print("kv_rank: skipping Type B (no --workspace or --skip-activations set)")

    verdict = _prediction_1_verdict(weight_results, args.d_head_nominal)

    payload: dict[str, Any] = {
        "n_embd": int(config.n_embd),
        "n_layer": int(getattr(config, "n_layer", 0)),
        "target_rank_kv_lora": int(args.target_rank),
        "d_head_nominal": int(args.d_head_nominal),
        "weight_analysis": weight_results,
        "activation_analysis": activation_results,
        "prediction_1_verdict": verdict,
    }

    json_path = os.path.join(args.output_dir, "kv_rank_results.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"kv_rank: wrote {json_path}")

    png_path = os.path.join(args.output_dir, "kv_rank_plots.png")
    _plot_kv_rank(weight_results, activation_results, args.target_rank, png_path)
    print(f"kv_rank: wrote {png_path}")

    print(
        f"kv_rank: Prediction 1 verdict = {verdict.get('verdict')} "
        f"(rank_95 > 40 count: {verdict.get('n_rank95_above_40')} / "
        f"{verdict.get('n_measurements')})"
    )


if __name__ == "__main__":
    main()
