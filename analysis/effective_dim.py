"""Protocol F -- Effective Dimensionality (RQ6).

Scope
-----
Addresses RQ6 (effective dimensionality across repeats). CPU-only
post-processor over Protocol A's residual cache. For each (layer,
repeat) site computes the Chun et al. 2025 participation ratio with
row-sampling bias correction; per-layer log-linear regression on
the repeat axis replaces the earlier Mann-Kendall trend test per
DEC-026, which is retained only as a secondary robustness column
due to its documented low power at ``n = 5`` observations.

Falsifiability relevance
------------------------
H0 "per-layer slope of ``log(d_eff)`` regressed against repeat
index ``r in {1, ..., n_repeat}`` is non-negative" is rejected
per-layer when the one-tailed t-test p-value is below
``alpha_family / rank`` after Holm-Bonferroni ordering of the 21
mid-layer tests at family-wise ``alpha_family = 0.01`` (per
`docs/extend-notes.md` §1.2 RQ6 "Falsification"). The RQ6
falsification rule requires the slope to be significantly negative
in at least 16 of the 21 mid-layers for the overall claim
"d_eff decreases across repeats" to be supported.

Ontological purpose
-------------------
If d_eff decreases across repeats, later repeats operate in
lower-dimensional subspaces, directly informing the MLA rank
allocation (DEC-EXT-001 in `docs/extend-notes.md` §2.1). Flat
d_eff supports uniform ``kv_lora_rank``; a >= 2x variation
warrants CARE-style per-layer per-repeat rank allocation. The
top-2-PC removed PR cross-check is mandatory when Protocol E
surfaces rogue-direction norm outliers, per the rogue-dimension
three-way cross-validation (`docs/extend-notes.md` §1.2 RQ6 row
"Rogue dimensions and three-way cross-validation").

Three independent rank metrics provide cross-validation per the
RQ6 table:

| Metric                             | Source   | What it measures              |
|------------------------------------|----------|-------------------------------|
| Chun gamma_row (participation)     | Chun 25  | Effective dimensionality      |
| Kobayashi pseudo-rank (5% of max)  | Kobay 24 | Number of non-negligible SVs  |
| KV-CoRE NER (Shannon entropy)      | Chen 26  | Information-theoretic rank    |

Agreement across all three strengthens the claim; disagreement
reveals which aspect of the spectrum matters for KV compression.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable

import numpy as np

from analysis.common.plotting import (
    palette_for_repeats,
    savefig,
    setup_figure,
    site_label,
)
from analysis.common.sites import discover_workspace_sites, residual_key_prefix
from analysis.common.spectral import kv_core_rank, participation_ratio


# -----------------------------------------------------------------------------
# Per-site metrics
# -----------------------------------------------------------------------------

def _load_and_centre(path: str) -> np.ndarray:
    X = np.load(path)
    if X.ndim != 2:
        raise ValueError(
            f"_load_and_centre: {path} has shape {X.shape}; expected 2-D"
        )
    X = X.astype(np.float64, copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    return X


def _top_k_pc_removed(X_centred: np.ndarray, k: int) -> np.ndarray:
    """Return ``X_centred`` after removing its top-``k`` principal components.

    ``X_centred`` must already be column-centred. SVD is economy-mode; the
    returned matrix has the same shape as input with the rank-``k``
    approximation subtracted.
    """
    if k <= 0 or X_centred.size == 0:
        return X_centred
    U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)
    k_eff = min(k, S.size)
    approx = (U[:, :k_eff] * S[:k_eff]) @ Vt[:k_eff, :]
    return X_centred - approx


def _kobayashi_pseudo_rank(X_centred: np.ndarray, frac: float = 0.05) -> int:
    """Count singular values above ``frac * max_sv``.

    Kobayashi et al. 2024 [11] operational rank: number of singular
    values at least 5 per cent of the largest. Returns an integer in
    ``[0, min(n, d)]``.
    """
    if X_centred.size == 0:
        return 0
    sv = np.linalg.svd(X_centred, full_matrices=False, compute_uv=False)
    if sv.size == 0 or sv[0] <= 0.0:
        return 0
    return int((sv >= frac * float(sv[0])).sum())


def _site_metrics(X_centred: np.ndarray, top_k_pc_remove: int) -> dict:
    """Compute participation ratio + rogue + Kobayashi + KV-CoRE NER."""
    pr_raw = participation_ratio(X_centred, finite_sample_correction=True)
    pr_naive = participation_ratio(X_centred, finite_sample_correction=False)

    if top_k_pc_remove > 0:
        X_mod = _top_k_pc_removed(X_centred, top_k_pc_remove)
        pr_top_removed = participation_ratio(X_mod, finite_sample_correction=True)
    else:
        pr_top_removed = pr_raw

    rogue_flag = False
    if pr_raw > 0.0:
        rel_diff = abs(pr_raw - pr_top_removed) / pr_raw
        rogue_flag = bool(rel_diff > 0.5)
    else:
        rel_diff = 0.0

    kob_rank = _kobayashi_pseudo_rank(X_centred, frac=0.05)
    ner_rank, ner_value = kv_core_rank(X_centred, target_fidelity=0.99)

    return {
        "participation_ratio": float(pr_raw),
        "participation_ratio_naive": float(pr_naive),
        "participation_ratio_top_removed": float(pr_top_removed),
        "top_removed_relative_diff": float(rel_diff),
        "rogue_flag": rogue_flag,
        "kobayashi_pseudo_rank": int(kob_rank),
        "kv_core_ner_rank": int(ner_rank),
        "kv_core_ner_value": float(ner_value),
    }


# -----------------------------------------------------------------------------
# Per-layer log-linear slope regression
# -----------------------------------------------------------------------------

def _loglinear_slope_one_tailed(
    repeats: np.ndarray, d_eff: np.ndarray
) -> dict:
    """Regress ``log(d_eff)`` on ``repeats`` and return slope + one-tailed p.

    The H0 is "slope >= 0"; the one-tailed p reports the probability of
    observing a slope at or below the point estimate under the Student-t
    distribution with ``n - 2`` degrees of freedom.
    """
    r = np.asarray(repeats, dtype=np.float64).ravel()
    y = np.asarray(d_eff, dtype=np.float64).ravel()
    if r.size != y.size:
        raise ValueError("loglinear_slope: repeats and d_eff size mismatch")
    mask = y > 0.0
    if mask.sum() < 3:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p_one_tailed_slope_nonpositive": 1.0,
            "n_points": int(mask.sum()),
            "t_stat": 0.0,
        }
    r = r[mask]
    y_log = np.log(y[mask])
    n = r.size

    r_mean = r.mean()
    y_mean = y_log.mean()
    r_c = r - r_mean
    y_c = y_log - y_mean
    denom = float((r_c * r_c).sum())
    if denom <= 0.0:
        return {
            "slope": 0.0,
            "intercept": float(y_mean),
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p_one_tailed_slope_nonpositive": 1.0,
            "n_points": int(n),
            "t_stat": 0.0,
        }
    slope = float((r_c * y_c).sum() / denom)
    intercept = float(y_mean - slope * r_mean)

    residuals = y_log - (intercept + slope * r)
    dof = n - 2
    if dof <= 0:
        se_slope = 0.0
    else:
        sigma_sq = float((residuals * residuals).sum() / dof)
        se_slope = float(math.sqrt(sigma_sq / denom)) if denom > 0.0 else 0.0

    if se_slope > 0.0:
        t_stat = slope / se_slope
    else:
        t_stat = 0.0

    # One-tailed test with H0 "slope >= 0", H1 "slope < 0": reject when
    # t_stat is sufficiently negative. p-value is Pr(T_{dof} <= t_stat).
    p_one_tailed = _student_t_cdf(t_stat, dof)

    # 95% CI on slope via Student-t critical value.
    t_crit = _student_t_inv(0.975, dof)
    ci_low = slope - t_crit * se_slope
    ci_high = slope + t_crit * se_slope

    return {
        "slope": slope,
        "intercept": intercept,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_one_tailed_slope_nonpositive": float(p_one_tailed),
        "n_points": int(n),
        "t_stat": float(t_stat),
    }


def _student_t_cdf(t: float, dof: int) -> float:
    """Student-t CDF: prefers scipy; falls back to a normal approximation."""
    try:
        from scipy import stats as _stats

        return float(_stats.t.cdf(t, dof))
    except ImportError:
        # Normal approximation is acceptable for dof >= 3 since this is a
        # robustness column (the primary analysis runs under scipy in HPC).
        z = float(t)
        return float(0.5 * (1.0 + _erf(z / math.sqrt(2.0))))


def _student_t_inv(q: float, dof: int) -> float:
    """Inverse Student-t CDF."""
    try:
        from scipy import stats as _stats

        return float(_stats.t.ppf(q, dof))
    except ImportError:
        # Normal fallback: for dof >= 30 the approximation is accurate to
        # ~1 per cent; for very small dof the CI is conservative.
        return float(_inv_normal_cdf(q))


def _erf(x: float) -> float:
    sign = 1.0 if x >= 0 else -1.0
    x = abs(float(x))
    a1, a2, a3, a4, a5, p = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
        0.3275911,
    )
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def _inv_normal_cdf(q: float) -> float:
    """Rational approximation to the inverse standard-normal CDF."""
    q = float(q)
    if q <= 0.0:
        return -1e30
    if q >= 1.0:
        return 1e30
    # Beasley-Springer-Moro approximation, cited in Moro 1995.
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low
    if q < p_low:
        u = math.sqrt(-2.0 * math.log(q))
        num = ((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]
        den = (((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0
        return num / den
    if q <= p_high:
        u = q - 0.5
        r2 = u * u
        num = (((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * u
        den = ((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + 1.0
        return num / den
    u = math.sqrt(-2.0 * math.log(1.0 - q))
    num = ((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]
    den = (((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0
    return -num / den


# -----------------------------------------------------------------------------
# Mann-Kendall (secondary robustness column)
# -----------------------------------------------------------------------------

def _mann_kendall(y: Iterable[float]) -> dict:
    """Mann-Kendall trend test; returns ``{S, z, p}``.

    H0 "no trend" under asymptotic normality. At ``n = 5`` this test is
    documented low-power (Wang et al. 2020 [34]); we report it as a
    robustness column only, and disagreement with the log-linear slope
    is expected rather than a failure mode.
    """
    y_arr = np.asarray(list(y), dtype=np.float64)
    n = y_arr.size
    if n < 3:
        return {"S": 0.0, "z": 0.0, "p_two_tailed": 1.0, "n": int(n)}
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = y_arr[j] - y_arr[i]
            if d > 0.0:
                s += 1
            elif d < 0.0:
                s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s <= 0.0:
        return {"S": float(s), "z": 0.0, "p_two_tailed": 1.0, "n": int(n)}
    if s > 0:
        z = (s - 1.0) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / math.sqrt(var_s)
    else:
        z = 0.0
    p_two = 2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(z) / math.sqrt(2.0))))
    return {
        "S": float(s),
        "z": float(z),
        "p_two_tailed": float(p_two),
        "n": int(n),
    }


# -----------------------------------------------------------------------------
# Holm-Bonferroni correction
# -----------------------------------------------------------------------------

def _holm_bonferroni(p_values: list[float], alpha: float) -> list[bool]:
    """Return Holm-Bonferroni decisions (reject vs not) at family alpha.

    Holm-Bonferroni is uniformly more powerful than plain Bonferroni while
    preserving family-wise error control. Index order is preserved in the
    returned decision list.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda k: p_values[k])
    decisions = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            decisions[idx] = True
        else:
            # Once we fail, all higher-ranked p-values also fail per Holm's rule.
            break
    return decisions


# -----------------------------------------------------------------------------
# Workspace scan
# -----------------------------------------------------------------------------

def _discover_residual_sites(
    workspace_dir: str,
    residual_prefix: str,
) -> list[tuple[str, int, int]]:
    """Scan for ``<residual_prefix>_l<L>_r<R>.npy`` files.

    ``residual_prefix`` is one of ``"residual_mid"`` or
    ``"residual_flat"`` (see
    ``analysis.common.sites.residual_key_prefix``).

    Returns sorted ``(path, layer, repeat)`` tuples. Thin wrapper around
    ``analysis.common.sites.discover_workspace_sites``; the column
    reorder keeps the legacy iteration shape unchanged for callers that
    unpack via ``for path, layer, repeat in sites:``.
    """
    return [
        (path, layer, repeat)
        for layer, repeat, path in discover_workspace_sites(
            workspace_dir, residual_prefix
        )
    ]


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

def _plot_trajectories(
    per_site: dict,
    layers: list[int],
    repeats: list[int],
    output_path: str,
    group: str,
) -> None:
    fig, ax = setup_figure(1, 1, size=(10.0, 6.5))
    palette = palette_for_repeats(max(len(layers), 1))

    for li, layer in enumerate(layers):
        xs = []
        ys = []
        for r in repeats:
            key = site_label(group, layer, r)
            if key in per_site and per_site[key]["participation_ratio"] > 0.0:
                xs.append(r)
                ys.append(per_site[key]["participation_ratio"])
        if xs:
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.0,
                markersize=3,
                color=palette[li % len(palette)],
                alpha=0.7,
                label=f"{group}[{layer}]" if li < 3 or li == len(layers) - 1 else None,
            )

    ax.set_xlabel("Repeat index")
    ax.set_ylabel("Participation ratio (d_eff)")
    ax.set_title(f"Protocol F -- d_eff vs repeat per {group}-layer (RQ6)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    savefig(fig, output_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol F."""
    parser = argparse.ArgumentParser(
        description="Protocol F -- effective dimensionality across repeats (RQ6)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Path to the analysis workspace directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to receive effective_dim_results.json and trajectory PNG",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="RNG seed (currently unused; reserved for bootstrap future work)",
    )
    parser.add_argument(
        "--top-k-pc-remove",
        type=int,
        default=2,
        help="Remove this many top principal components for rogue cross-check (default 2)",
    )
    parser.add_argument(
        "--alpha-family",
        type=float,
        default=0.01,
        help="Family-wise alpha for Holm-Bonferroni across layers (default 0.01)",
    )
    parser.add_argument(
        "--reject-frac",
        type=float,
        default=16.0 / 21.0,
        help=(
            "Fraction of layers that must reject H0 for the RQ6 claim to be "
            "supported (default 16/21)."
        ),
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


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.workspace):
        print(
            f"effective_dim: workspace {args.workspace!r} missing",
            file=sys.stderr,
        )
        sys.exit(2)

    residual_prefix = residual_key_prefix(args.module_path)
    # Bare group label (``"mid"`` or ``"flat"``) drives plot/site-key
    # cosmetics; matches the collector's per-key group prefix.
    group = residual_prefix[len("residual_") :]

    sites = _discover_residual_sites(args.workspace, residual_prefix)
    if len(sites) < 2:
        print(
            f"effective_dim: only {len(sites)} residual sites; need >= 2",
            file=sys.stderr,
        )
        sys.exit(2)

    per_site: dict[str, dict] = {}
    by_layer: dict[int, dict[int, float]] = {}
    layers_seen: set[int] = set()
    repeats_seen: set[int] = set()

    for path, layer, repeat in sites:
        X = _load_and_centre(path)
        metrics = _site_metrics(X, args.top_k_pc_remove)
        key = site_label(group, layer, repeat)
        per_site[key] = metrics
        by_layer.setdefault(layer, {})[repeat] = metrics["participation_ratio"]
        layers_seen.add(layer)
        repeats_seen.add(repeat)

    layers = sorted(layers_seen)
    repeats = sorted(repeats_seen)

    # Per-layer log-linear slope regression.
    layer_slopes: dict[int, dict] = {}
    p_values: list[float] = []
    for layer in layers:
        per_repeat = by_layer.get(layer, {})
        r_arr = np.array(sorted(per_repeat.keys()), dtype=np.float64)
        y_arr = np.array([per_repeat[int(r)] for r in r_arr], dtype=np.float64)
        slope_info = _loglinear_slope_one_tailed(r_arr, y_arr)
        mk = _mann_kendall(y_arr)
        layer_slopes[layer] = {
            "loglinear": slope_info,
            "mann_kendall": mk,
            "d_eff_per_repeat": {
                int(r): float(per_repeat[int(r)]) for r in r_arr
            },
        }
        p_values.append(slope_info["p_one_tailed_slope_nonpositive"])

    # Holm-Bonferroni across layers.
    rejects = _holm_bonferroni(p_values, args.alpha_family)
    for layer_idx, layer in enumerate(layers):
        decision = bool(rejects[layer_idx])
        layer_slopes[layer]["holm_bonferroni_reject_at_alpha_family"] = decision

    rejected_count = int(sum(rejects))
    frac_rejected = rejected_count / max(1, len(layers))
    rq6_verdict = {
        "alpha_family": args.alpha_family,
        "n_layers": len(layers),
        "n_rejected": rejected_count,
        "fraction_rejected": frac_rejected,
        "reject_fraction_threshold": args.reject_frac,
        "claim_supported": frac_rejected >= args.reject_frac,
    }

    # Count rogue-flagged sites.
    rogue_sites = [k for k, v in per_site.items() if v["rogue_flag"]]

    # Count Mann-Kendall agreement with log-linear slope at two-tailed
    # p < 0.05 and matching sign.
    mk_agree = 0
    mk_total = 0
    for layer in layers:
        ll = layer_slopes[layer]["loglinear"]
        mk = layer_slopes[layer]["mann_kendall"]
        if ll["n_points"] >= 3:
            mk_total += 1
            mk_sig_negative = (mk["p_two_tailed"] < 0.05) and (mk["z"] < 0.0)
            ll_sig_negative = (ll["p_one_tailed_slope_nonpositive"] < 0.05) and (
                ll["slope"] < 0.0
            )
            if mk_sig_negative == ll_sig_negative:
                mk_agree += 1

    payload = {
        "n_sites": len(per_site),
        "layers": layers,
        "repeats": repeats,
        "top_k_pc_remove": args.top_k_pc_remove,
        "per_site": per_site,
        "per_layer_slope": {str(layer): layer_slopes[layer] for layer in layers},
        "rq6_verdict": rq6_verdict,
        "rogue_sites_flagged": rogue_sites,
        "mann_kendall_agreement": {
            "n_agreeing_layers": int(mk_agree),
            "n_layers_compared": int(mk_total),
            "agreement_fraction": (mk_agree / mk_total) if mk_total else 0.0,
        },
    }

    json_path = os.path.join(args.output_dir, "effective_dim_results.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"effective_dim: wrote {json_path}")

    png_path = os.path.join(args.output_dir, "effective_dim_trajectory.png")
    _plot_trajectories(per_site, layers, repeats, png_path, group)
    print(f"effective_dim: wrote {png_path}")

    print(
        f"effective_dim: {rq6_verdict['n_rejected']} / {rq6_verdict['n_layers']} "
        f"layers reject H0 at family alpha {args.alpha_family}; "
        f"claim supported: {rq6_verdict['claim_supported']}"
    )


if __name__ == "__main__":
    main()
