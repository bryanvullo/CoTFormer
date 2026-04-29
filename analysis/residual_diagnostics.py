"""Protocol E -- Residual Diagnostics (SQ1, RQ6 pre-check).

Scope
-----
Addresses SQ1 residual-stream diagnostics and serves as the RQ6
pre-check. CPU-only post-processor over Protocol A's residual cache.
For each (layer, repeat) site it computes per-token quantities
(mean L2 norm, per-token L2 variance, L_inf norm), the top-2-PC
variance fraction, and the mean cosine similarity between block
input and block output (residual entering layer L at repeat R
versus residual entering layer L+1 at repeat R; the boundary cells
at the last mid-layer are emitted as NaN because the residual
flow there crosses an optional ``ln_mid`` block and/or the
``h_end`` stack, conflating single-block cosine drift with
multi-block drift). Produces curve plots across repeats plus an
optional cosine-sim heatmap.

Falsifiability relevance
------------------------
H0 "residual L2 norm constant within 5 per cent across repeats within
each layer" is rejected if the within-layer coefficient of variation
exceeds 0.1 (per `docs/extend-notes.md` §1.3 "Protocol E operational
spec"). Surfaces exponential norm growth -- which would invalidate
the participation-ratio-vs-repeat trend regression used in RQ6 -- and
rogue-direction emergence -- which RQ6 handles via the
top-2-PC-removed PR cross-check.

Ontological purpose
-------------------
Cheapest diagnostic of whether the model is "doing anything" at
later repeats. Near-zero norm change between repeats 3 and 5 would
reveal degenerate iterative compute, supporting Prediction 1
(Kobayashi et al. 2024 low-rank limit).

Inputs
------
Reads residuals from the workspace directory produced by a prior
``analysis.logit_lens`` run (which runs the collector and persists
``residual_*_l{L}_r{R}.npy`` and ``residual_ln_mid_r{R}.npy`` files
per §4.3 of ``docs/extend-technical.md``). The ``residual_*`` group
prefix is selected by ``--module-path``: ``residual_mid_*`` for the
CoTFormer ``h_mid`` sub-stream (default), ``residual_flat_*`` for
the standard ``transformer.h`` (Protocol-D Tier 1 substrate).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from analysis.common.plotting import palette_for_repeats, savefig, setup_figure
from analysis.common.sites import residual_key_prefix


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol E.

    Expected inputs: ``--workspace``, ``--output-dir``, ``--seed``,
    ``--normalise-by-h-begin`` (remove the trivial depth baseline).
    """
    parser = argparse.ArgumentParser(
        description="Protocol E -- residual diagnostics (L2, variance, L_inf)"
    )
    parser.add_argument(
        "--workspace", type=str, required=True,
        help="Workspace directory containing residuals produced by "
             "a prior analysis.logit_lens run",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for residual_diagnostics_results.json + 3 PNGs",
    )
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--max-layers", type=int, default=64,
                        help="Upper bound on scanned mid-layer index")
    parser.add_argument("--max-repeat", type=int, default=16,
                        help="Upper bound on scanned repeat index")
    parser.add_argument(
        "--normalise-by-h-begin", action="store_true",
        help="Subtract the h_begin last-block residual mean L2 from "
             "each site's mean L2 (removes the trivial depth baseline; "
             "§1.3 Protocol E operational spec)",
    )
    parser.add_argument(
        "--rogue-threshold", type=float, default=0.5,
        help="Top-2 PC variance fraction threshold above which a site "
             "is flagged as rogue-dominated (default 0.5)",
    )
    parser.add_argument(
        "--cv-threshold", type=float, default=0.1,
        help="Within-layer CV gate for the mean-L2-constant H0 (§1.3)",
    )
    parser.add_argument(
        "--module-path",
        type=str,
        default="model.transformer.h_mid",
        help="Dotted module path the upstream collector hooked into; "
             "selects the residual buffer-key prefix "
             "(``model.transformer.h_mid`` -> residual_mid_*; "
             "``model.transformer.h`` -> residual_flat_*).",
    )
    return parser


def _load_residual(workspace: str, key: str) -> np.ndarray | None:
    """Load ``workspace/<key>.npy`` or return None when missing."""
    path = os.path.join(workspace, f"{key}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def _compute_site_metrics(h: np.ndarray) -> dict[str, float]:
    """Return ``mean_l2``, ``var_l2``, ``linf`` over the (N_tokens, D) tensor."""
    l2_per_token = np.linalg.norm(h, axis=-1)
    linf_per_token = np.max(np.abs(h), axis=-1)
    mean_l2 = float(np.mean(l2_per_token))
    var_l2 = float(np.var(l2_per_token, ddof=1)) if l2_per_token.size > 1 else 0.0
    linf = float(np.max(linf_per_token))
    return {
        "mean_l2": mean_l2,
        "var_l2": var_l2,
        "linf": linf,
    }


def _rogue_flag(h: np.ndarray, threshold: float) -> tuple[bool, float]:
    """Return ``(is_rogue, top2_fraction)`` for the centred covariance.

    A rogue-dominated site is one where the top-2 principal components
    explain more than ``threshold`` fraction of the total variance;
    this signals emergent low-rank outlier directions that invalidate
    the standard PR estimator and triggers RQ6's top-2-PC-removed
    cross-check.
    """
    centred = h - h.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centred, full_matrices=False, compute_uv=False)
    power = sv ** 2
    total = float(power.sum())
    if total <= 0.0:
        return False, 0.0
    top2 = float(power[:2].sum())
    fraction = top2 / total
    return bool(fraction > threshold), fraction


def _mean_cosine_sim(h_in: np.ndarray, h_out: np.ndarray) -> float:
    """Mean per-token cosine similarity between two ``(N_tokens, D)`` tensors.

    Uses the numpy idiom ``sum(h_in * h_out, axis=-1) / (||h_in|| * ||h_out||)``
    per token, then averages over tokens. Rows where either operand has
    zero norm are dropped (cosine undefined); if every row is zero-norm
    the function returns ``NaN`` to signal an uninformative cell.
    """
    if h_in.shape != h_out.shape:
        return float("nan")
    norm_in = np.linalg.norm(h_in, axis=-1)
    norm_out = np.linalg.norm(h_out, axis=-1)
    denom = norm_in * norm_out
    valid = denom > 0.0
    if not np.any(valid):
        return float("nan")
    dot = np.sum(h_in * h_out, axis=-1)
    cos = dot[valid] / denom[valid]
    return float(np.mean(cos))


def _array_to_json_safe_list(a: np.ndarray) -> list:
    """Convert array to nested list with NaN -> None for RFC 8259 conformance."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        v = float(arr)
        return None if not np.isfinite(v) else v
    obj = arr.astype(object)
    obj[~np.isfinite(arr)] = None
    return obj.tolist()


def _compute_block_io_cosine_sim(
    workspace: str,
    n_layer_mid: int,
    n_repeat: int,
    residual_prefix: str,
) -> dict[tuple[int, int], float]:
    """Mean per-token cosine similarity between block input and block output.

    Cell ``(L, R)`` records cosine similarity between
    ``<residual_prefix>_l<L>_r<R>`` (output of mid-block ``L`` =
    input to mid-block ``L+1``) and ``<residual_prefix>_l<L+1>_r<R>``
    (output of mid-block ``L+1``). Therefore the cell at row ``L``
    represents the IO of mid-block ``L+1``, NOT mid-block ``L``;
    heatmap row labels should be interpreted accordingly.
    ``residual_prefix`` is ``"residual_mid"`` for ``h_mid`` and
    ``"residual_flat"`` for standard ``transformer.h``.

    Boundary handling at ``L = n_layer_mid - 1``
    --------------------------------------------
    Both intermediate-repeat (``R < n_repeat``) and last-repeat
    (``R == n_repeat``) cells at the last mid-layer are emitted as
    ``NaN``. For ``R < n_repeat`` the next residual flowing through
    the network is the start of repeat ``R+1`` *after* an optional
    ``ln_mid`` block (present in CoTFormer variants C2--C5, absent
    in C1). For ``R == n_repeat``, the canonical Belrose-2023 target
    residual ``residual_pre_ln_f`` is the input to ``ln_f`` -- but
    for V3/V4 (with ``ln_mid``) the residual flow is
    ``h_mid_last -> ln_mid -> h_end -> ln_f``, so cosine similarity
    between ``residual_mid_l<last>_r<last>`` and
    ``residual_pre_ln_f`` measures multi-block drift (``ln_mid +
    h_end + ...``), NOT single-block IO. Both boundary cells are
    emitted as ``NaN`` rather than mixing apples and oranges into
    the summary statistics.

    Returns ``{(layer, repeat): mean_cosine_sim}`` with one entry per
    valid cell; missing residuals or shape mismatches yield ``NaN``.
    """
    cosine_by_cell: dict[tuple[int, int], float] = {}

    for layer_idx in range(n_layer_mid):
        for repeat_idx in range(1, n_repeat + 1):
            h_in = _load_residual(
                workspace, f"{residual_prefix}_l{layer_idx}_r{repeat_idx}"
            )
            if h_in is None:
                cosine_by_cell[(layer_idx, repeat_idx)] = float("nan")
                continue

            if layer_idx < n_layer_mid - 1:
                h_out = _load_residual(
                    workspace,
                    f"{residual_prefix}_l{layer_idx + 1}_r{repeat_idx}",
                )
            else:
                # At (n_layer_mid - 1, n_repeat) the model has run h_mid_last + ln_mid
                # (when present) + ALL of h_end + then ln_f reads its input. Cosine sim
                # between residual_mid_l<last>_r<last> and residual_pre_ln_f therefore
                # measures multi-block drift (ln_mid + h_end + ...), not single-block
                # IO. Emit NaN with documented rationale rather than mix apples and
                # oranges into the summary statistics. The intermediate-repeat
                # boundary (R < n_repeat) is similarly undefined because of the
                # optional ln_mid block between this residual and the next-repeat
                # input.
                cosine_by_cell[(layer_idx, repeat_idx)] = float("nan")
                continue

            if h_out is None:
                cosine_by_cell[(layer_idx, repeat_idx)] = float("nan")
                continue
            cosine_by_cell[(layer_idx, repeat_idx)] = _mean_cosine_sim(h_in, h_out)

    return cosine_by_cell


def _plot_cosine_heatmap(
    grid: np.ndarray,
    n_layer_mid: int,
    n_repeat: int,
    output_path: str,
) -> None:
    """Render ``grid`` (shape ``(n_layer_mid, n_repeat)``) as a heatmap.

    Skipped silently if matplotlib is unavailable; the JSON summary is
    the authoritative artefact and the heatmap is informational.
    """
    try:
        fig, axes = setup_figure(1, 1, size=(8.0, 6.0))
    except Exception:  # noqa: BLE001 - matplotlib import is optional here
        return
    ax = axes
    # Diverging colourmap centred at 0: cosine sim has natural symmetry
    # around 0 (orthogonal). Sequential viridis biased the visual reading
    # (cos=0 looked "low" rather than "neutral").
    import matplotlib.colors as _mcolors
    norm = _mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm,
    )
    ax.set_xticks(np.arange(n_repeat))
    ax.set_xticklabels([str(r) for r in range(1, n_repeat + 1)])
    ax.set_yticks(np.arange(n_layer_mid))
    ax.set_yticklabels([f"L{l}" for l in range(n_layer_mid)])
    ax.set_xlabel("Repeat index")
    ax.set_ylabel("Mid-layer index")
    ax.set_title("Block input/output mean cosine similarity")
    fig.colorbar(im, ax=ax, label="cos(h_in, h_out)")
    savefig(fig, output_path)


def _discover_site_grid(
    workspace: str,
    max_layers: int,
    max_repeat: int,
    residual_prefix: str,
) -> tuple[int, int]:
    """Infer ``(n_layer_mid, n_repeat)`` from the workspace file layout.

    ``residual_prefix`` selects between ``"residual_mid"`` (CoTFormer
    ``h_mid``) and ``"residual_flat"`` (standard ``transformer.h``);
    see ``analysis.common.sites.residual_key_prefix``.
    """
    n_layer_found = 0
    n_repeat_found = 0
    for layer_idx in range(max_layers):
        for repeat_idx in range(1, max_repeat + 1):
            key = f"{residual_prefix}_l{layer_idx}_r{repeat_idx}"
            if _load_residual(workspace, key) is None:
                continue
            n_layer_found = max(n_layer_found, layer_idx + 1)
            n_repeat_found = max(n_repeat_found, repeat_idx)
    if n_layer_found == 0 or n_repeat_found == 0:
        raise FileNotFoundError(
            f"_discover_site_grid: workspace {workspace} has no "
            f"{residual_prefix}_l{{L}}_r{{R}}.npy files"
        )
    return n_layer_found, n_repeat_found


def _plot_curve(
    grid: np.ndarray,
    n_layer_mid: int,
    n_repeat: int,
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    """Plot 21 mid-layer curves across repeats on one panel."""
    fig, axes = setup_figure(1, 1, size=(12.0, 6.0))
    ax = axes  # single axis

    palette = palette_for_repeats(n_repeat)
    for layer_idx in range(n_layer_mid):
        ax.plot(
            np.arange(1, n_repeat + 1),
            grid[layer_idx],
            alpha=0.55,
            label=f"L{layer_idx}" if layer_idx < 3 or layer_idx == n_layer_mid - 1 else None,
        )

    mean_curve = np.nanmean(grid, axis=0)
    ax.plot(np.arange(1, n_repeat + 1), mean_curve, color="black",
            linewidth=2.5, label="mean-across-layers")

    ax.set_xlabel("Repeat index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(1, n_repeat + 1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    savefig(fig, output_path)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    residual_prefix = residual_key_prefix(args.module_path)

    n_layer_mid, n_repeat = _discover_site_grid(
        args.workspace, args.max_layers, args.max_repeat, residual_prefix
    )

    mean_l2_grid = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float64)
    var_l2_grid = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float64)
    linf_grid = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float64)
    top2_grid = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float64)
    rogue_flags: list[dict[str, Any]] = []

    for layer_idx in range(n_layer_mid):
        for repeat_idx in range(1, n_repeat + 1):
            h = _load_residual(
                args.workspace, f"{residual_prefix}_l{layer_idx}_r{repeat_idx}"
            )
            if h is None:
                continue
            metrics = _compute_site_metrics(h)
            is_rogue, top2 = _rogue_flag(h, args.rogue_threshold)
            mean_l2_grid[layer_idx, repeat_idx - 1] = metrics["mean_l2"]
            var_l2_grid[layer_idx, repeat_idx - 1] = metrics["var_l2"]
            linf_grid[layer_idx, repeat_idx - 1] = metrics["linf"]
            top2_grid[layer_idx, repeat_idx - 1] = top2
            if is_rogue:
                rogue_flags.append(
                    {"layer": layer_idx, "repeat": repeat_idx, "top2_fraction": top2}
                )

    cosine_io_grid = np.full(
        (n_layer_mid, n_repeat), np.nan, dtype=np.float64
    )
    cosine_by_cell = _compute_block_io_cosine_sim(
        args.workspace, n_layer_mid, n_repeat, residual_prefix
    )
    for (layer_idx, repeat_idx), value in cosine_by_cell.items():
        cosine_io_grid[layer_idx, repeat_idx - 1] = value

    baseline_mean_l2: float | None = None
    h_begin = _load_residual(args.workspace, "residual_h_begin")
    if h_begin is not None:
        baseline_mean_l2 = float(np.mean(np.linalg.norm(h_begin, axis=-1)))

    if args.normalise_by_h_begin and baseline_mean_l2 is not None:
        mean_l2_plot_grid = mean_l2_grid - baseline_mean_l2
    else:
        mean_l2_plot_grid = mean_l2_grid

    # Within-layer CV gate (§1.3 Protocol E)
    with np.errstate(divide="ignore", invalid="ignore"):
        layer_mean = np.nanmean(mean_l2_grid, axis=1)
        layer_std = np.nanstd(mean_l2_grid, axis=1)
        layer_cv = np.where(
            np.abs(layer_mean) > 1e-12, layer_std / np.abs(layer_mean), np.nan
        )
    n_cv_fail = int(np.sum(np.isfinite(layer_cv) & (layer_cv > args.cv_threshold)))

    _plot_curve(
        mean_l2_plot_grid, n_layer_mid, n_repeat,
        title="Mean residual L2 per (layer, repeat)"
              + (" (normalised by h_begin)" if args.normalise_by_h_begin else ""),
        ylabel="Mean ||x||_2"
               + (" - ||h_begin||_2" if args.normalise_by_h_begin else ""),
        output_path=os.path.join(args.output_dir,
                                 "residual_diagnostics_mean_l2.png"),
    )
    _plot_curve(
        var_l2_grid, n_layer_mid, n_repeat,
        title="Per-token L2 variance per (layer, repeat)",
        ylabel="Var_t ||x||_2",
        output_path=os.path.join(args.output_dir,
                                 "residual_diagnostics_var_l2.png"),
    )
    _plot_curve(
        linf_grid, n_layer_mid, n_repeat,
        title="L_inf norm per (layer, repeat)",
        ylabel="max_t |x|_inf",
        output_path=os.path.join(args.output_dir,
                                 "residual_diagnostics_linf.png"),
    )
    _plot_cosine_heatmap(
        cosine_io_grid, n_layer_mid, n_repeat,
        output_path=os.path.join(args.output_dir,
                                 "residual_cosine_sim_io.png"),
    )

    finite_cos = cosine_io_grid[np.isfinite(cosine_io_grid)]
    if finite_cos.size > 0:
        cosine_summary = {
            "mean": float(np.mean(finite_cos)),
            "median": float(np.median(finite_cos)),
            "min": float(np.min(finite_cos)),
            "max": float(np.max(finite_cos)),
            "n_cells_with_value": int(finite_cos.size),
            "n_cells_below_0_5": int(np.sum(finite_cos < 0.5)),
        }
    else:
        cosine_summary = {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "n_cells_with_value": 0,
            "n_cells_below_0_5": 0,
        }
    cosine_per_cell = [
        {
            "layer": int(layer_idx),
            "repeat": int(repeat_idx),
            "mean_cosine_sim_block_input_output": (
                None
                if not np.isfinite(cosine_io_grid[layer_idx, repeat_idx - 1])
                else float(cosine_io_grid[layer_idx, repeat_idx - 1])
            ),
        }
        for layer_idx in range(n_layer_mid)
        for repeat_idx in range(1, n_repeat + 1)
    ]

    results = {
        "workspace": args.workspace,
        "seed": args.seed,
        "n_layer_mid": int(n_layer_mid),
        "n_repeat": int(n_repeat),
        "normalise_by_h_begin": bool(args.normalise_by_h_begin),
        "h_begin_mean_l2": baseline_mean_l2,
        "mean_l2_grid": _array_to_json_safe_list(mean_l2_grid),
        "var_l2_grid": _array_to_json_safe_list(var_l2_grid),
        "linf_grid": _array_to_json_safe_list(linf_grid),
        "top2_pc_fraction_grid": _array_to_json_safe_list(top2_grid),
        "rogue_flags": rogue_flags,
        "layer_cv": _array_to_json_safe_list(layer_cv),
        "cv_threshold": args.cv_threshold,
        "n_layers_failing_cv_gate": n_cv_fail,
        "h0_reject_flat_norm": bool(n_cv_fail > 0),
        "cosine_sim_io_grid": _array_to_json_safe_list(cosine_io_grid),
        "cosine_sim_io_per_cell": cosine_per_cell,
        "summary": {
            "cosine_sim_block_input_output": cosine_summary,
        },
    }
    with open(
        os.path.join(args.output_dir, "residual_diagnostics_results.json"), "w"
    ) as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
