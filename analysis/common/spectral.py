"""Spectral helpers -- effective rank, participation ratio, KV-CoRE.

Scope
-----
Three pure-Python spectral utilities consumed by Protocols F, G and
the synthesis script:

- ``compute_effective_rank``  -- cumulative-variance threshold search.
- ``participation_ratio``     -- finite-sample-corrected PR (Chun et al. 2025).
- ``kv_core_rank``            -- KV-CoRE normalised effective rank (Chen et al. 2026).

Each function accepts NumPy input only; the collector performs the
GPU-to-CPU transfer once in ``ActivationCollector.save``.

Falsifiability relevance
------------------------
- Protocol F / RQ6: H0 "d_eff constant within 5 per cent across
  repeats" is rejected per-layer when ``participation_ratio``
  trajectories fail the per-layer log-linear regression test
  (`docs/extend-notes.md` §1.2 RQ6).
- Protocol G: Prediction 1 "weight decay drives QK/OV to low rank"
  is rejected when ``compute_effective_rank`` at the 0.95 threshold
  exceeds 40 components on the weight-level SVD.

Ontological purpose
-------------------
Collects all spectral algebra in one module so that a future
methodological refinement (e.g., a switch from PCA-based PR to
eigenvalue-based PR) updates every protocol consistently.

Body deferred to Phase 2.
"""

from __future__ import annotations

import numpy as np


def compute_effective_rank(
    sv: np.ndarray,
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> tuple[dict[float, int], np.ndarray]:
    """Return per-threshold effective rank and the cumulative-variance curve.

    Parameters
    ----------
    sv : np.ndarray
        Singular values, one-dimensional, sorted descending. NaN / Inf
        entries must be removed upstream.
    thresholds : tuple of float
        Fractions of cumulative variance at which to report the
        effective rank. Values must lie in (0, 1].

    Returns
    -------
    ranks : dict[float, int]
        Mapping threshold -> effective rank (smallest number of
        components whose squared singular values sum to the threshold
        fraction of the total).
    cumulative : np.ndarray
        Cumulative-variance curve (same length as ``sv``) for plotting.
    """
    raise NotImplementedError("body in Phase 2")


def participation_ratio(
    sv: np.ndarray,
    finite_sample_correction: bool = True,
) -> float:
    """Chun et al. 2025 finite-sample-corrected participation ratio.

    When ``finite_sample_correction=True``, applies the Chun et al.
    leave-one-out correction to remove the small-N positive bias
    present in the uncorrected PR estimator.
    """
    raise NotImplementedError("body in Phase 2")


def kv_core_rank(
    activations: np.ndarray,
    target_fidelity: float = 0.99,
) -> tuple[int, float]:
    """Chen et al. 2026 KV-CoRE normalised effective rank.

    Parameters
    ----------
    activations : np.ndarray
        Two-dimensional array of shape ``(n_tokens, feature_dim)``
        containing the KV vectors to analyse. Callers should centre
        per-(layer, repeat) before calling; centring removes the
        trivial mean-component that dominates a spurious rank-one.
    target_fidelity : float
        Fraction of activation energy the CoRE rank must preserve.

    Returns
    -------
    rank : int
        Number of components required to reach ``target_fidelity``.
    normalised_rank : float
        ``rank / feature_dim`` in [0, 1]; comparable across layers of
        different widths.
    """
    raise NotImplementedError("body in Phase 2")
