"""Protocol B -- Centred Kernel Alignment (RQ2).

Scope
-----
Addresses RQ2 (representation structural similarity across repeats).
CPU-only post-processor over the residual-stream cache produced by
Protocol A's collector: computes debiased HSIC U-statistic CKA with
linear and RBF kernels over all (layer, repeat) site pairs,
producing a 105x105 matrix for the canonical C3 LN-CoTFormer at
60k (21 mid-layers x 5 repeats), plus intra-layer, cross-layer, and
grand views.

Falsifiability relevance
------------------------
H0 "observed CKA above the 99th-percentile of the per-pair
permutation null at every cross-repeat pair within each layer"
is rejected when any adjacent-repeat pair within any mid-layer
yields CKA below the 90th-percentile of its own permutation null
(per `docs/extend-notes.md` §1.2 RQ2 "Falsifies H0"). Debiased
U-statistic + permutation-null zone thresholds are per DEC-026;
without the debias, finite-sample CKA carries a positive bias that
would reject a null the bias itself created.

Ontological purpose
-------------------
Resolves whether cross-repeat representations are redundant (high
CKA -- MLA-style compression is trivial) or uniquely informative
(low CKA -- the model does distinct work at each repeat, and the
layer-resolved map shows where).

Implementation
--------------
- **Primary statistic**: debiased linear CKA via the Song et al.
  2012 / Murphy et al. 2024 unbiased HSIC U-statistic. For centred
  Gram matrices with zero diagonals,

      HSIC_u(X, Y) = 1 / (n * (n - 3)) *
          [ tr(K_tilde L_tilde)
            + (1^T K_tilde 1)(1^T L_tilde 1) / ((n - 1)(n - 2))
            - 2 / (n - 2) * 1^T K_tilde L_tilde 1 ]

  with `K_tilde = X X^T` after its diagonal is zeroed. CKA_u is
  `HSIC_u(X, Y) / sqrt(HSIC_u(X, X) * HSIC_u(Y, Y))`. This estimator
  is consistent as n goes to infinity and removes the finite-sample
  positive bias of the plain biased estimator.
- **Robustness column**: RBF-kernel CKA with median-heuristic
  bandwidth `sigma = median(pairwise_distances) / sqrt(2)`. Reported
  per-pair alongside the linear value; the linear estimator is
  primary because the debiasing is published analytically only for
  the linear form.
- **Permutation null**: for each pair (i, j) we shuffle j's token
  index ``n_permutations`` times and recompute the debiased linear
  CKA on the shuffled pair; the resulting distribution defines the
  per-pair 90th- and 99th-percentile zone cutoffs:
  - Redundant:  observed CKA >= 99th-percentile of the null.
  - Partially distinct:  90th <= observed CKA < 99th-percentile.
  - Structurally distinct:  observed CKA < 90th-percentile.
- **Permutation symmetry**: because ``K_tilde_Y`` is the Gram matrix
  of Y's rows, permuting Y's rows is equivalent to permuting the
  rows AND columns of ``K_tilde_Y`` by the same index array (the
  permutation is a similarity transform on the Gram matrix). We
  therefore pre-compute ``K_tilde_Y`` once per site and index-
  permute the Gram matrix in-place, avoiding any recomputation of
  ``Y Y^T`` inside the 1000-sample null loop.
- **Cosine-similarity subsidiary** (per RQ2 DV): for every pair of
  adjacent-repeat sites within the same mid-layer
  ((layer, r), (layer, r+1)) we additionally record the per-token
  cosine similarity `<x_t^(r), x_t^(r+1)> / (||x^(r)|| ||x^(r+1)||)`
  mean and standard deviation. This is an operationally-independent
  complement to the CKA statistic (different invariance class:
  cosine is per-token and not orthogonal-invariant at the
  population level).

Compute budget
--------------
For C3 (60k LN-CoTFormer), there are 21 mid layers x 5 repeats =
105 sites, giving `C(105, 2) = 5460` unordered pairs. Each pair
costs one (debiased CKA, RBF CKA, cosine) measurement plus
``n_permutations`` extra debiased-CKA evaluations on the shuffled
Gram. With N=2048 tokens per site, each debiased CKA is O(n^2) in
the Gram-matrix element-wise product path; 1000 permutations per
pair x 5460 pairs lands within a ~15 h single-core CPU envelope
and scales linearly under ``multiprocessing.Pool``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from typing import Iterable

import numpy as np

from analysis.common.plotting import savefig, setup_figure
from analysis.common.sites import discover_workspace_sites, residual_key_prefix


# -----------------------------------------------------------------------------
# HSIC U-statistic (Song 2012 / Murphy 2024)
# -----------------------------------------------------------------------------

def _hsic_u_from_zero_diag_gram(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC U-statistic from zero-diagonal Gram matrices.

    Requires ``K`` and ``L`` to have zero on their diagonals (the
    ``K_tilde`` / ``L_tilde`` of Song 2012 eq. 14). Both matrices
    must be ``n x n`` with the same ``n``.

    Returns the scalar HSIC_u in the native scale of ``K`` and ``L``
    (no normalisation to the square-root denominator).
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError(
            f"_hsic_u_from_zero_diag_gram: n={n} too small; HSIC_u needs n >= 4"
        )

    term_trace = float(np.sum(K * L))

    sum_K = float(K.sum())
    sum_L = float(L.sum())
    term_sums = sum_K * sum_L / ((n - 1.0) * (n - 2.0))

    # 1^T (K L) 1 = sum_j (K row-sum)[j] * (L column-sum)[j]. Symmetric K and L
    # satisfy row-sum == column-sum so either axis works.
    row_sum_K = K.sum(axis=1)
    row_sum_L = L.sum(axis=1)
    term_cross = 2.0 * float(row_sum_K @ row_sum_L) / (n - 2.0)

    hsic = (term_trace + term_sums - term_cross) / (n * (n - 3.0))
    return hsic


def _zero_diag(G: np.ndarray) -> np.ndarray:
    """Return a copy of ``G`` with zero on its diagonal."""
    out = G.copy()
    np.fill_diagonal(out, 0.0)
    return out


def _linear_cka_debiased_from_grams(K_tilde: np.ndarray, L_tilde: np.ndarray) -> float:
    """Debiased linear CKA from already-zero-diagonal Grams.

    Allows the full signed range CKA in [-1, 1]: HSIC_u admits negative
    values under finite-sample estimator noise (and for genuinely
    anti-correlated kernel pairs); clipping ``h_XY <= 0`` to zero
    destroys the symmetry of the downstream null distribution and
    biases right-tailed quantile thresholds. ``denom`` (the product
    of two HSIC-self-similarity estimators) is population-non-negative;
    if it slips below 0 due to estimator noise the ratio is undefined
    and we signal NaN so callers can exclude the cell from aggregates.

    Consistent with the inline regime in ``_compute_pair``: the RBF
    debiased-CKA path (``_rbf_cka_debiased``) consumes this helper
    and therefore inherits the unclipped semantics.
    """
    h_XY = _hsic_u_from_zero_diag_gram(K_tilde, L_tilde)
    h_XX = _hsic_u_from_zero_diag_gram(K_tilde, K_tilde)
    h_YY = _hsic_u_from_zero_diag_gram(L_tilde, L_tilde)
    denom = h_XX * h_YY
    if denom <= 0.0:
        return float("nan")
    return float(h_XY / np.sqrt(denom))


def _linear_cka_debiased(X: np.ndarray, Y: np.ndarray) -> float:
    """Convenience wrapper: build Grams, zero diag, apply debiased CKA."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "linear_cka_debiased: X and Y must have the same number of rows"
        )
    K = X @ X.T
    L = Y @ Y.T
    return _linear_cka_debiased_from_grams(_zero_diag(K), _zero_diag(L))


# -----------------------------------------------------------------------------
# RBF CKA with median heuristic
# -----------------------------------------------------------------------------

def _median_rbf_sigma(X: np.ndarray) -> float:
    """Median pairwise-distance bandwidth: sigma = median(dist) / sqrt(2)."""
    n = X.shape[0]
    # Squared pairwise distances via the identity ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    sq_norm = np.sum(X * X, axis=1)
    dist_sq = sq_norm[:, None] + sq_norm[None, :] - 2.0 * (X @ X.T)
    dist_sq = np.clip(dist_sq, 0.0, None)
    # Use upper triangle only (exclude diagonal) for the median.
    iu = np.triu_indices(n, k=1)
    pairwise_sq = dist_sq[iu]
    median_sq = float(np.median(pairwise_sq))
    if median_sq <= 0.0:
        return 1.0
    return float(np.sqrt(median_sq) / np.sqrt(2.0))


def _rbf_gram(X: np.ndarray, sigma: float) -> np.ndarray:
    """RBF Gram matrix K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))."""
    sq_norm = np.sum(X * X, axis=1)
    dist_sq = sq_norm[:, None] + sq_norm[None, :] - 2.0 * (X @ X.T)
    dist_sq = np.clip(dist_sq, 0.0, None)
    return np.exp(-dist_sq / (2.0 * sigma * sigma))


def _rbf_cka_debiased(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased RBF CKA using Murphy 2024's unbiased HSIC U-statistic."""
    sigma_X = _median_rbf_sigma(X)
    sigma_Y = _median_rbf_sigma(Y)
    K = _rbf_gram(X, sigma_X)
    L = _rbf_gram(Y, sigma_Y)
    return _linear_cka_debiased_from_grams(_zero_diag(K), _zero_diag(L))


# -----------------------------------------------------------------------------
# Cosine-similarity subsidiary metric (RQ2 DV-2)
# -----------------------------------------------------------------------------

def _per_token_cosine_stats(X: np.ndarray, Y: np.ndarray) -> dict[str, float]:
    """Return mean and std of per-token cosine similarity.

    Both ``X`` and ``Y`` must have shape ``(n_tokens, d)`` with equal
    ``n_tokens``. For each row index t, ``sim_t = <X[t], Y[t]> /
    (||X[t]|| ||Y[t]||)`` is computed; rows with zero norm on either
    side contribute ``0.0``.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape != Y.shape:
        raise ValueError("per_token_cosine_stats: X and Y shapes must match")
    # Local centring keeps the cosine-side cosine_mean / cosine_std
    # comparable across pairs irrespective of the residual-stream mean
    # drift across the layer stack. The CKA U-statistic upstream operates
    # on the RAW row matrix because it absorbs centring algebraically;
    # cosine has no equivalent absorption, so we centre here.
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    nx = np.linalg.norm(X, axis=1)
    ny = np.linalg.norm(Y, axis=1)
    denom = nx * ny
    mask = denom > 0.0
    sim = np.zeros(X.shape[0], dtype=np.float64)
    sim[mask] = np.sum(X[mask] * Y[mask], axis=1) / denom[mask]
    return {"mean": float(sim.mean()), "std": float(sim.std(ddof=1) if sim.size > 1 else 0.0)}


# -----------------------------------------------------------------------------
# Workspace loading
# -----------------------------------------------------------------------------

def _discover_residual_sites(
    workspace_dir: str,
    residual_prefix: str,
) -> list[tuple[str, str, int, int]]:
    """Scan the workspace for ``<residual_prefix>_l<L>_r<R>.npy`` files.

    ``residual_prefix`` is one of ``"residual_mid"`` (CoTFormer
    ``h_mid`` sub-stream) or ``"residual_flat"`` (standard
    ``transformer.h``, e.g. Protocol-D Tier 1 GPT-2-large), as
    returned by ``analysis.common.sites.residual_key_prefix``.

    Returns a list of ``(site_key, path, layer, repeat)`` tuples
    sorted by ``(layer, repeat)``. The canonical ``site_key`` mirrors
    ``analysis.common.plotting.site_label`` with the same group
    string the prefix encodes (``"mid[L] r=R"`` or
    ``"flat[L] r=R"``). Wraps the architecture-agnostic
    ``analysis.common.sites.discover_workspace_sites`` and decorates
    each row with the CKA-specific display key.
    """
    # ``residual_prefix`` is always ``"residual_<group>"`` (mid/flat);
    # strip the literal ``"residual_"`` head to recover the bare group.
    group = residual_prefix[len("residual_") :]
    return [
        (f"{group}[{layer}] r={repeat}", path, layer, repeat)
        for layer, repeat, path in discover_workspace_sites(
            workspace_dir, residual_prefix
        )
    ]


def _load_residual(path: str) -> np.ndarray:
    """Load an ``.npy`` residual capture as float64, RAW (no centring).

    Per Song et al. 2012 / Murphy 2024 PML2 §3.2, the unbiased HSIC
    U-statistic estimator (``_hsic_u_from_zero_diag_gram``) is derived
    for the RAW (uncentred) kernel: the three-term correction
    ``(term_trace + term_sums - term_cross) / (n (n-3))`` absorbs the
    centring implicitly. Pre-centring the row matrix and then applying
    the U-statistic over-corrects -- the bias-correction terms presume
    an uncentred Gram, and applying them after column-centring yields
    a hybrid that is no longer unbiased for HSIC. The literature-
    standard recipe is therefore to feed the RAW row matrix and let
    the U-statistic do the centring work; that is what this loader
    provides.
    """
    X = np.load(path)
    if X.ndim != 2:
        raise ValueError(
            f"_load_residual: {path} has shape {X.shape}; expected 2-D"
        )
    X = X.astype(np.float64, copy=False)
    return X


# -----------------------------------------------------------------------------
# Pair worker
# -----------------------------------------------------------------------------

# Pool-global state populated by ``_pool_init``.
_POOL_K_TILDES: list[np.ndarray] | None = None
_POOL_H_XX: list[float] | None = None
_POOL_X_MATRICES: list[np.ndarray] | None = None
_POOL_N_PERMS: int = 0
_POOL_SEED: int = 0
_POOL_ALSO_RBF: bool = False
_POOL_ALSO_COS: bool = False
_POOL_SITE_META: list[tuple[int, int]] | None = None


def _pool_init(
    k_tildes: list[np.ndarray],
    h_xx: list[float],
    x_matrices: list[np.ndarray] | None,
    n_perms: int,
    seed: int,
    also_rbf: bool,
    also_cos: bool,
    site_meta: list[tuple[int, int]],
) -> None:
    """Initialise per-worker globals used by ``_pair_worker``."""
    global _POOL_K_TILDES, _POOL_H_XX, _POOL_X_MATRICES
    global _POOL_N_PERMS, _POOL_SEED, _POOL_ALSO_RBF, _POOL_ALSO_COS
    global _POOL_SITE_META
    _POOL_K_TILDES = k_tildes
    _POOL_H_XX = h_xx
    _POOL_X_MATRICES = x_matrices
    _POOL_N_PERMS = n_perms
    _POOL_SEED = seed
    _POOL_ALSO_RBF = also_rbf
    _POOL_ALSO_COS = also_cos
    _POOL_SITE_META = site_meta


def _compute_pair(
    i: int,
    j: int,
    k_tildes: list[np.ndarray],
    h_xx: list[float],
    x_matrices: list[np.ndarray] | None,
    n_perms: int,
    seed: int,
    also_rbf: bool,
    also_cos: bool,
    site_meta: list[tuple[int, int]],
) -> dict:
    """Compute observed CKA, permutation null, and (optional) side-metrics."""
    K_i = k_tildes[i]
    K_j = k_tildes[j]
    h_ii = h_xx[i]
    h_jj = h_xx[j]

    h_ij = _hsic_u_from_zero_diag_gram(K_i, K_j)
    denom = h_ii * h_jj
    # Allow negative debiased CKA in [-1, 1]: HSIC_u is defined to admit
    # negative values for anti-correlated kernels, and the U-statistic
    # estimator can dip below zero for any kernel pair under finite n.
    # Clipping h_ij < 0 to 0 (the previous regime) destroyed the symmetry
    # of the null distribution and biased right-tailed q90 / q99
    # thresholds DOWNWARD, producing an artificially permissive null.
    # ``denom`` is the product of two estimators of HSIC(K_i, K_i) and
    # HSIC(K_j, K_j) which are population-non-negative; if it slips
    # below 0 due to estimator noise the ratio is undefined and we
    # signal NaN to downstream so the cell can be excluded from
    # aggregates (q90 / q99 use ``nanquantile``; null_mean / null_std
    # use ``nanmean`` / ``nanstd``).
    if denom > 0.0:
        cka_linear_obs = float(h_ij / np.sqrt(denom))
    else:
        cka_linear_obs = float("nan")

    # Permutation null: permute the rows AND columns of K_j by the same
    # index array. Equivalent to shuffling j's token order and recomputing
    # the Gram. h_jj is permutation-invariant so we reuse it.
    n = K_j.shape[0]
    # SeedSequence mixes the (seed, i, j) tuple via SplitMix64 + spawn_key
    # logic, giving high-entropy decorrelation across (i, j) cells. The
    # previous linear hash ``seed + 1_000_003 * i + 7 * j`` was prone to
    # near-collisions when two cells differed in (i, j) by a small linear
    # combination, producing partially-correlated null permutations.
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(i), int(j)]))
    null_vals = np.empty(n_perms, dtype=np.float64)
    for p in range(n_perms):
        perm = rng.permutation(n)
        K_j_perm = K_j[perm[:, None], perm[None, :]]
        h_ij_p = _hsic_u_from_zero_diag_gram(K_i, K_j_perm)
        # Same regime as the observed-value branch: keep the full
        # signed range so the null retains its symmetry around zero;
        # mark NaN only when the denominator is non-positive (rare
        # estimator-noise edge case) and use nanquantile downstream.
        if denom > 0.0:
            null_vals[p] = float(h_ij_p / np.sqrt(denom))
        else:
            null_vals[p] = float("nan")

    q90 = float(np.nanquantile(null_vals, 0.90))
    q99 = float(np.nanquantile(null_vals, 0.99))
    if not np.isfinite(cka_linear_obs):
        zone = "undefined_denom"
    elif cka_linear_obs >= max(q99, 0.0):
        # Semantically "redundant" requires high POSITIVE similarity.
        # Without the max(., 0.0) floor, the >= q99 branch could fire
        # when both observed and q99 are negative (anti-correlated
        # kernels under finite-n estimator noise) -- mislabeling
        # structural anti-correlation as redundancy.
        zone = "redundant"
    elif cka_linear_obs >= max(q90, 0.0):
        zone = "partially_distinct"
    else:
        zone = "structurally_distinct"

    out = {
        "i": int(i),
        "j": int(j),
        "layer_i": int(site_meta[i][0]),
        "repeat_i": int(site_meta[i][1]),
        "layer_j": int(site_meta[j][0]),
        "repeat_j": int(site_meta[j][1]),
        "cka_linear_debiased": cka_linear_obs,
        "null_q90": q90,
        "null_q99": q99,
        "null_mean": float(np.nanmean(null_vals)),
        "null_std": float(np.nanstd(null_vals, ddof=1)) if n_perms > 1 else 0.0,
        "zone": zone,
    }

    if also_rbf and x_matrices is not None:
        out["cka_rbf_debiased"] = _rbf_cka_debiased(x_matrices[i], x_matrices[j])

    # Cosine side-metric only for adjacent-repeat pairs within the same layer.
    if (
        also_cos
        and x_matrices is not None
        and site_meta[i][0] == site_meta[j][0]
        and abs(site_meta[i][1] - site_meta[j][1]) == 1
    ):
        stats = _per_token_cosine_stats(x_matrices[i], x_matrices[j])
        out["cosine_mean"] = stats["mean"]
        out["cosine_std"] = stats["std"]

    return out


def _pair_worker(pair: tuple[int, int]) -> dict:
    """Pool worker wrapper: consume the pool globals."""
    i, j = pair
    assert _POOL_K_TILDES is not None and _POOL_H_XX is not None
    assert _POOL_SITE_META is not None
    return _compute_pair(
        i,
        j,
        _POOL_K_TILDES,
        _POOL_H_XX,
        _POOL_X_MATRICES,
        _POOL_N_PERMS,
        _POOL_SEED,
        _POOL_ALSO_RBF,
        _POOL_ALSO_COS,
        _POOL_SITE_META,
    )


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

_ZONE_COLORS = {
    "redundant": "#2ca02c",              # green
    "partially_distinct": "#ff7f0e",     # orange
    "structurally_distinct": "#d62728",  # red
}


def _plot_cka_heatmap(
    cka_matrix: np.ndarray,
    zone_matrix: np.ndarray,
    site_keys: list[str],
    output_path: str,
) -> None:
    """Render the 105x105 CKA matrix with zone-colour overlay."""
    fig, axes = setup_figure(1, 2, size=(22.0, 10.0))

    ax = axes[0]
    im = ax.imshow(cka_matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title("Debiased linear CKA")
    ax.set_xlabel("Site j")
    ax.set_ylabel("Site i")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA_u")

    ax2 = axes[1]
    # Map the zone strings to RGB floats for a categorical heatmap.
    rgb = np.zeros(zone_matrix.shape + (3,), dtype=np.float64)
    hex_to_rgb = {
        "#2ca02c": (0.17, 0.63, 0.17),
        "#ff7f0e": (1.0, 0.50, 0.05),
        "#d62728": (0.84, 0.15, 0.16),
        "#ffffff": (1.0, 1.0, 1.0),
    }
    for zi in range(zone_matrix.shape[0]):
        for zj in range(zone_matrix.shape[1]):
            zone = zone_matrix[zi, zj]
            col = _ZONE_COLORS.get(zone, "#ffffff")
            rgb[zi, zj] = hex_to_rgb.get(col, (1.0, 1.0, 1.0))
    ax2.imshow(rgb, aspect="equal")
    ax2.set_title("Zone (green=redundant, orange=partially_distinct, red=structurally_distinct)")
    ax2.set_xlabel("Site j")
    ax2.set_ylabel("Site i")

    # Thin tick labels (every 10th) to keep 105-axis readable.
    step = max(1, len(site_keys) // 10)
    tick_idx = list(range(0, len(site_keys), step))
    tick_lbl = [site_keys[k] for k in tick_idx]
    for a in (ax, ax2):
        a.set_xticks(tick_idx)
        a.set_xticklabels(tick_lbl, rotation=60, ha="right", fontsize=7)
        a.set_yticks(tick_idx)
        a.set_yticklabels(tick_lbl, fontsize=7)

    fig.suptitle("Protocol B -- 105x105 CKA pair matrix (RQ2)", fontsize=13)
    savefig(fig, output_path)


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol B."""
    parser = argparse.ArgumentParser(
        description="Protocol B -- debiased CKA matrix over residual sites (RQ2)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Path to the analysis workspace directory (contains "
             "residual_*_l*_r*.npy; the prefix follows --module-path)",
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
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to receive cka_results.json and cka_heatmap.png",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Seed for the permutation-null RNG (default: 1729)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Permutation-null sample count per pair (default: 1000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="multiprocessing Pool size; 1 = serial (default: 1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Cap the row count per site to this many (0 = use all; default: 0)",
    )
    parser.add_argument(
        "--skip-rbf",
        action="store_true",
        help="Skip the RBF robustness column (default: compute RBF too)",
    )
    parser.add_argument(
        "--skip-cosine",
        action="store_true",
        help="Skip the per-token cosine subsidiary metric (default: compute)",
    )
    return parser


def _prepare_grams(
    sites: list[tuple[str, str, int, int]],
    max_tokens: int,
    keep_x_matrices: bool,
) -> tuple[list[np.ndarray], list[float], list[np.ndarray] | None, int]:
    """Load all residual sites and precompute zero-diagonal Grams.

    Returns
    -------
    k_tildes : list[np.ndarray]
        Zero-diagonal Gram matrices per site.
    h_xx : list[float]
        ``HSIC_u(K_tilde, K_tilde)`` per site (used as CKA denominator).
    x_matrices : list[np.ndarray] | None
        The raw row matrices themselves (kept only when
        ``keep_x_matrices`` is true; used by RBF CKA and cosine
        side-metrics, which apply their own local centring as needed).
    n : int
        Common row count.
    """
    matrices: list[np.ndarray] = []
    for _, path, _, _ in sites:
        X = _load_residual(path)
        if max_tokens > 0 and X.shape[0] > max_tokens:
            X = X[:max_tokens]
        matrices.append(X)

    # Align to the minimum row count if they are unequal.
    n_tokens_per = [X.shape[0] for X in matrices]
    n_common = min(n_tokens_per)
    matrices = [X[:n_common] for X in matrices]

    if n_common < 4:
        raise ValueError(
            f"_prepare_grams: common row count {n_common} < 4; HSIC_u undefined"
        )

    k_tildes: list[np.ndarray] = []
    h_xx: list[float] = []
    for X in matrices:
        K = X @ X.T
        np.fill_diagonal(K, 0.0)
        k_tildes.append(K)
        h_xx.append(_hsic_u_from_zero_diag_gram(K, K))

    kept = matrices if keep_x_matrices else None
    return k_tildes, h_xx, kept, n_common


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.workspace):
        print(
            f"cka: workspace {args.workspace!r} missing or not a directory",
            file=sys.stderr,
        )
        sys.exit(2)

    residual_prefix = residual_key_prefix(args.module_path)
    sites = _discover_residual_sites(args.workspace, residual_prefix)
    if len(sites) < 2:
        print(
            f"cka: discovered {len(sites)} sites in {args.workspace}; need >= 2",
            file=sys.stderr,
        )
        sys.exit(2)

    site_keys = [t[0] for t in sites]
    site_meta = [(t[2], t[3]) for t in sites]
    n_sites = len(sites)

    print(f"cka: discovered {n_sites} residual sites")
    t0 = time.time()

    need_matrices = not (args.skip_rbf and args.skip_cosine)
    k_tildes, h_xx, x_matrices, n_common = _prepare_grams(
        sites, args.max_tokens, keep_x_matrices=need_matrices
    )
    t1 = time.time()
    print(
        f"cka: prepared {n_sites} Gram matrices "
        f"(n_tokens_common={n_common}); {(t1 - t0):.1f}s"
    )

    pairs = [(i, j) for i in range(n_sites) for j in range(i + 1, n_sites)]
    n_pairs = len(pairs)
    print(
        f"cka: running {n_pairs} pair measurements x "
        f"{args.n_permutations} permutations; workers={args.workers}"
    )

    results: list[dict]
    if args.workers <= 1:
        results = []
        for k, (i, j) in enumerate(pairs):
            results.append(
                _compute_pair(
                    i,
                    j,
                    k_tildes,
                    h_xx,
                    x_matrices,
                    args.n_permutations,
                    args.seed,
                    not args.skip_rbf,
                    not args.skip_cosine,
                    site_meta,
                )
            )
            if (k + 1) % max(1, n_pairs // 20) == 0:
                frac = (k + 1) / n_pairs
                print(f"cka: serial progress {100 * frac:.1f}%", flush=True)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_pool_init,
            initargs=(
                k_tildes,
                h_xx,
                x_matrices,
                args.n_permutations,
                args.seed,
                not args.skip_rbf,
                not args.skip_cosine,
                site_meta,
            ),
        ) as pool:
            results = []
            for k, res in enumerate(pool.imap_unordered(_pair_worker, pairs, chunksize=8)):
                results.append(res)
                if (k + 1) % max(1, n_pairs // 20) == 0:
                    frac = (k + 1) / n_pairs
                    print(f"cka: parallel progress {100 * frac:.1f}%", flush=True)

    t2 = time.time()
    print(f"cka: pair phase took {(t2 - t1):.1f}s")

    cka_matrix = np.eye(n_sites, dtype=np.float64)
    zone_matrix = np.array([["redundant"] * n_sites for _ in range(n_sites)], dtype=object)
    q90_matrix = np.zeros((n_sites, n_sites), dtype=np.float64)
    q99_matrix = np.zeros((n_sites, n_sites), dtype=np.float64)
    for r in results:
        i, j = r["i"], r["j"]
        v = r["cka_linear_debiased"]
        cka_matrix[i, j] = v
        cka_matrix[j, i] = v
        zone_matrix[i, j] = r["zone"]
        zone_matrix[j, i] = r["zone"]
        q90_matrix[i, j] = r["null_q90"]
        q90_matrix[j, i] = r["null_q90"]
        q99_matrix[i, j] = r["null_q99"]
        q99_matrix[j, i] = r["null_q99"]

    zone_counts = {
        "redundant": int(sum(1 for r in results if r["zone"] == "redundant")),
        "partially_distinct": int(
            sum(1 for r in results if r["zone"] == "partially_distinct")
        ),
        "structurally_distinct": int(
            sum(1 for r in results if r["zone"] == "structurally_distinct")
        ),
    }

    # H0 test: any adjacent-repeat pair within any layer below 90th-percentile
    # null rejects H0 "redundant through shared weights".
    rejects_h0 = False
    falsifying_pairs: list[dict] = []
    for r in results:
        if r["layer_i"] == r["layer_j"] and abs(r["repeat_i"] - r["repeat_j"]) == 1:
            if r["cka_linear_debiased"] < r["null_q90"]:
                rejects_h0 = True
                falsifying_pairs.append(
                    {
                        "layer": r["layer_i"],
                        "repeat_i": r["repeat_i"],
                        "repeat_j": r["repeat_j"],
                        "cka": r["cka_linear_debiased"],
                        "null_q90": r["null_q90"],
                    }
                )

    payload = {
        "n_sites": n_sites,
        "site_keys": site_keys,
        "n_permutations": args.n_permutations,
        "n_tokens_common": n_common,
        "seed": args.seed,
        "timing_sec": {
            "prepare": round(t1 - t0, 2),
            "pair_phase": round(t2 - t1, 2),
        },
        "cka_matrix": cka_matrix.tolist(),
        "zone_matrix": zone_matrix.tolist(),
        "null_q90_matrix": q90_matrix.tolist(),
        "null_q99_matrix": q99_matrix.tolist(),
        "pairs": results,
        "zone_counts": zone_counts,
        "h0_verdict": {
            "rejects_h0_redundant_same_weights": rejects_h0,
            "falsifying_adjacent_repeat_pairs": falsifying_pairs,
        },
    }

    json_path = os.path.join(args.output_dir, "cka_results.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"cka: wrote {json_path}")

    png_path = os.path.join(args.output_dir, "cka_heatmap.png")
    _plot_cka_heatmap(cka_matrix, zone_matrix, site_keys, png_path)
    print(f"cka: wrote {png_path}")


if __name__ == "__main__":
    main()
