"""Statistical helpers -- partial correlation, paired t-test, z-score.

Scope
-----
Three canonical hypothesis-test helpers consumed by Protocols A, D,
F and by the synthesis script. Each function returns a
``(statistic, p_value)`` tuple with consistent ordering so the
protocol code can pattern-match uniformly.

Falsifiability relevance
------------------------
- ``partial_correlation`` is RQ4's position-confounder control:
  partial |r| >= 0.05 after position control is the RQ4 falsification
  threshold (`docs/extend-notes.md` §1.2 RQ4).
- ``paired_t_test`` is RQ1's per-repeat significance test: H0
  "mean(top-1 at repeat 1) == mean(top-1 at n_repeat)" is rejected
  at p < 0.01 (`docs/extend-notes.md` §1.2 RQ1).
- ``z_score_per_sequence`` is RQ5's sequence-difficulty confounder
  control; without it, context-survival vs target-loss correlations
  are dominated by sequence-level variance.

Ontological purpose
-------------------
Centralises hypothesis-test utilities so that a future statistical
refinement (e.g., bootstrap vs permutation nulls) propagates through
every protocol consistently.
"""

from __future__ import annotations

import numpy as np


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, float]:
    """Pearson partial correlation r(x, y | z) with two-sided p-value.

    Implementation: linearly regress out ``z`` from both ``x`` and
    ``y`` (ordinary-least-squares with intercept), then compute the
    Pearson correlation of the two residual vectors. The p-value uses
    the Student-t distribution with ``n - 3`` degrees of freedom (two
    regression coefficients plus the intercept consume three degrees
    from the n-sample vector).

    Parameters
    ----------
    x, y, z : np.ndarray
        One-dimensional arrays of equal length ``n >= 4``.

    Returns
    -------
    r_partial : float
        Pearson partial correlation in ``[-1, 1]``.
    p_value : float
        Two-sided t-test p-value on the partial correlation under
        the H0 ``r(x, y | z) == 0``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    n = x.shape[0]
    if y.shape[0] != n or z.shape[0] != n:
        raise ValueError(
            "partial_correlation: x, y, z must be equal-length 1-D arrays"
        )
    if n < 4:
        raise ValueError("partial_correlation: need n >= 4 samples")

    design = np.column_stack([np.ones(n), z])
    # solve least-squares for x ~ z and y ~ z
    beta_x, *_ = np.linalg.lstsq(design, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, y, rcond=None)
    res_x = x - design @ beta_x
    res_y = y - design @ beta_y

    num = float(np.dot(res_x, res_y))
    denom = float(np.linalg.norm(res_x) * np.linalg.norm(res_y))
    if denom == 0.0:
        return 0.0, 1.0
    r = num / denom
    r = float(np.clip(r, -1.0, 1.0))

    df = n - 3
    if abs(r) >= 1.0 or df <= 0:
        p = 0.0 if abs(r) >= 1.0 else 1.0
        return r, float(p)
    t_stat = r * np.sqrt(df / max(1.0 - r * r, 1e-300))

    try:
        from scipy import stats as _scipy_stats

        p_value = float(2.0 * _scipy_stats.t.sf(abs(t_stat), df))
    except ImportError:
        # Fallback: normal-approximation p-value when scipy is unavailable.
        p_value = float(
            2.0 * (1.0 - 0.5 * (1.0 + _erf_like(abs(t_stat) / np.sqrt(2.0))))
        )
    return r, p_value


def paired_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t-test on equal-length vectors ``a``, ``b``.

    Parameters
    ----------
    a, b : np.ndarray
        One-dimensional paired sample vectors of equal length
        ``n >= 2``.

    Returns
    -------
    t_statistic : float
        Paired-t statistic ``mean(d) / (std(d) / sqrt(n))`` where
        ``d = a - b``.
    p_value : float
        Two-sided p-value under the Student-t distribution with
        ``n - 1`` degrees of freedom.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "paired_t_test: a and b must have the same length "
            f"(got {a.shape[0]} vs {b.shape[0]})"
        )
    n = a.shape[0]
    if n < 2:
        raise ValueError("paired_t_test: need n >= 2 paired samples")

    try:
        from scipy import stats as _scipy_stats

        result = _scipy_stats.ttest_rel(a, b)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        d = a - b
        d_mean = float(np.mean(d))
        d_std = float(np.std(d, ddof=1))
        if d_std == 0.0:
            if d_mean == 0.0:
                return 0.0, 1.0
            return float("inf") if d_mean > 0 else float("-inf"), 0.0
        t_stat = d_mean / (d_std / np.sqrt(n))
        df = n - 1
        p_value = float(
            2.0 * (1.0 - 0.5 * (1.0 + _erf_like(abs(t_stat) / np.sqrt(2.0))))
        )
        return float(t_stat), p_value


def z_score_per_sequence(
    loss: np.ndarray,
    sequence_ids: np.ndarray,
) -> np.ndarray:
    """Standardise ``loss`` within each unique ``sequence_ids`` group.

    Returns an array of the same shape as ``loss`` where each element
    has been mean-centred and divided by the within-sequence standard
    deviation. Sequences with standard deviation zero (constant loss)
    map to zero.

    Parameters
    ----------
    loss : np.ndarray
        One-dimensional loss vector of length ``n``.
    sequence_ids : np.ndarray
        One-dimensional integer vector of length ``n`` identifying
        the sequence each loss entry belongs to.

    Returns
    -------
    z : np.ndarray
        Z-scored loss, shape ``(n,)``, dtype float64.
    """
    loss = np.asarray(loss, dtype=np.float64).ravel()
    sequence_ids = np.asarray(sequence_ids).ravel()
    if loss.shape[0] != sequence_ids.shape[0]:
        raise ValueError(
            "z_score_per_sequence: loss and sequence_ids must be "
            f"equal-length (got {loss.shape[0]} vs {sequence_ids.shape[0]})"
        )

    out = np.zeros_like(loss)
    for seq_id in np.unique(sequence_ids):
        mask = sequence_ids == seq_id
        values = loss[mask]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        if std == 0.0:
            out[mask] = 0.0
        else:
            out[mask] = (values - mean) / std
    return out


def _erf_like(x: float) -> float:
    """Small-dependency erf for the no-scipy fallback path."""
    # Abramowitz and Stegun 7.1.26 numerical approximation.
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
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return float(sign * y)
