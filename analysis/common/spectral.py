"""Spectral helpers -- effective rank, participation ratio, KV-CoRE.

Scope
-----
Three pure-NumPy spectral utilities consumed by Protocols F, G and
the synthesis script:

- ``compute_effective_rank``  -- cumulative-variance threshold search
  (skel-1 precedent; matches ``analyze_kv_compression.py:77-85``).
- ``participation_ratio``     -- Chun et al. 2025 row-sampling
  bias-corrected participation ratio, equivalent to the paper's
  ``gamma_row`` estimator in the "rows sampled, columns fully
  observed" regime that applies here (2048 tokens sampled from OWT2;
  ``n_embd=768`` features fully observed).
- ``kv_core_rank``            -- Chen et al. 2026 KV-CoRE Normalised
  Effective Rank ``NER(K) = erank(K) / r`` where
  ``erank(K) = exp(-sum p_i * ln p_i)`` with ``p_i = sigma_i / sum sigma_j``
  and ``r`` the numerical rank of the centred activation matrix.

Each function accepts NumPy input only; the collector performs the
GPU-to-CPU transfer once in ``ActivationCollector.save``.

Falsifiability relevance
------------------------
- Protocol F / RQ6: H0 "d_eff constant within 5 per cent across
  repeats" is rejected per-layer when ``participation_ratio``
  trajectories fail the per-layer log-linear regression test
  (`docs/extend-notes.md` §1.2 RQ6).
- Protocol G: Prediction 1 "weight decay correlates with low rank"
  is rejected when ``compute_effective_rank`` at the 0.95 threshold
  exceeds 40 components on the weight-level SVD (two-tier threshold
  per `docs/extend-notes.md` §1.4).
- Protocol G / RQ6 three-way cross-validation: ``kv_core_rank``
  returns the information-theoretic NER complement to the raw-rank
  measure (`docs/extend-notes.md` §1.2 RQ6 three-way table).

Ontological purpose
-------------------
Collects all spectral algebra in one module so that a future
methodological refinement (e.g. switching from row-only to full
Chun ``gamma_both`` bias correction) updates every protocol
consistently.

Source references
-----------------
- ``compute_effective_rank`` lifts the cumulative-variance search
  from the legacy ``analyze_kv_compression.py:77-85``. The skel-1
  committed signature preserves the ``(ranks_dict, cumvar_array)``
  return shape.
- ``participation_ratio`` references Chun et al. 2025 ``docs/
  Dimensionality-Estimation-Chun2025.pdf`` §2.2 (classical PR) and
  §4 (row-sampling correction). The implementation uses the
  row-Gram analytic form derived in the module-level docstring
  below.
- ``kv_core_rank`` references Chen et al. 2026 ``docs/Kv-CoRe-
  Benchmarking-Chen2026.pdf`` §2.4. The erank formula is
  attributed by Chen et al. to Roy and Vetterli 2007.
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
        Singular values, one-dimensional, sorted descending. NaN or Inf
        entries must be removed upstream.
    thresholds : tuple of float
        Fractions of cumulative variance at which to report the
        effective rank. Values must lie in ``(0, 1]``.

    Returns
    -------
    ranks : dict[float, int]
        Mapping threshold -> effective rank (smallest number of
        components whose squared singular values sum to at least the
        threshold fraction of the total). Keys are the input
        thresholds (as floats); rank is ``int`` in ``[1, len(sv)]``.
    cumulative : np.ndarray
        Cumulative-variance curve ``cumsum(sv**2) / sum(sv**2)``, same
        length as ``sv``, for plotting and debugging. This matches
        the return convention of the monolith predecessor
        ``analyze_kv_compression.py:77-85`` that this module supersedes.
    """
    sv = np.asarray(sv, dtype=np.float64).ravel()
    if sv.size == 0:
        return {float(t): 0 for t in thresholds}, np.zeros(0, dtype=np.float64)
    if np.any(sv < 0):
        raise ValueError(
            "compute_effective_rank: singular values must be non-negative"
        )
    for t in thresholds:
        if not (0.0 < t <= 1.0):
            raise ValueError(
                f"compute_effective_rank: threshold {t!r} outside (0, 1]"
            )

    sv2 = sv * sv
    total = float(sv2.sum())
    if total == 0.0:
        cumulative = np.zeros_like(sv2)
        return {float(t): 0 for t in thresholds}, cumulative
    cumulative = np.cumsum(sv2) / total

    ranks: dict[float, int] = {}
    for t in thresholds:
        # searchsorted on the cumulative curve returns the first index
        # whose cumulative value reaches `t`; +1 converts index to rank.
        idx = int(np.searchsorted(cumulative, float(t), side="left"))
        rank = min(idx + 1, int(sv.size))
        ranks[float(t)] = rank
    return ranks, cumulative


def participation_ratio(
    x: np.ndarray,
    finite_sample_correction: bool = True,
) -> float:
    """Chun et al. 2025 row-sampling bias-corrected participation ratio.

    For a centred data matrix ``X`` of shape ``(N, D)`` with
    per-sample row vectors, the classical (naive) participation ratio
    is ``PR_naive = (sum sv_i^2)^2 / sum sv_i^4`` where ``sv_i`` are
    singular values of the centred matrix. Chun et al. 2025
    ``docs/Dimensionality-Estimation-Chun2025.pdf`` §3 show that this
    estimator carries a strong finite-sample bias (driven by
    overlapping indices in the expansion of the trace terms); the
    corrected estimator ``gamma_both`` restricts the summation to
    mutually-distinct row and column indices.

    When the columns (features) are fully observed but rows
    (samples) are drawn iid -- exactly our setting with
    2048 token-rows drawn from OWT2 and ``n_embd=768`` hidden-dim
    columns exhausted -- Chun et al. 2025 §4 note that correcting
    only the row-index bias is sufficient and yields the
    ``gamma_row`` estimator:

    .. math::

        \\gamma_{\\mathrm{row}} \\;=\\; \\frac{S_1^2 \\,-\\, S_2}{T \\,-\\, S_2}

    where, with ``R = X X^T`` the row-Gram matrix,

    - ``S_1 = trace(R) = ||X||_F^2 = sum sv_i^2``
    - ``S_2 = sum_i R_{ii}^2`` (sum of squared row-sq-norms)
    - ``T   = ||R||_F^2 = sum sv_i^4``

    The ``- S_2`` terms drop the ``i == j`` diagonal contributions
    from both the numerator ``(sum sv^2)^2 = sum_{ij} R_{ii} R_{jj}``
    and denominator ``sum sv^4 = sum_{ij} R_{ij}^2``, which is the
    row-sampling bias correction in compact form. Agrees with
    ``PR_naive`` as ``N -> infinity``.

    Parameters
    ----------
    x : np.ndarray
        Two-dimensional array of shape ``(N, D)``. The matrix is
        centred column-wise inside the function (subtracting the
        per-column mean) to match Chun's "task dimensionality"
        convention (§2.2). Callers do not need to pre-centre.
    finite_sample_correction : bool
        When ``True``, return the Chun ``gamma_row`` corrected
        estimator. When ``False``, return the naive ``PR_naive``
        for debugging.

    Returns
    -------
    pr : float
        Estimated dimensionality in ``[1, min(N, D)]``. Near ``D``
        indicates a nearly-isotropic representation; near ``1``
        indicates a spectrum dominated by a single direction.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(
            f"participation_ratio: x must be 2-D (got {x.ndim}-D)"
        )
    n, d = int(x.shape[0]), int(x.shape[1])
    if n == 0 or d == 0:
        return 0.0

    x_centred = x - x.mean(axis=0, keepdims=True)

    if n <= 2 or not finite_sample_correction:
        # Naive PR via squared singular values. Equivalent to
        # (trace C)^2 / trace(C^2) up to (N-1) factors that cancel.
        sv = np.linalg.svd(x_centred, full_matrices=False, compute_uv=False)
        s2 = float((sv * sv).sum())
        s4 = float((sv * sv * sv * sv).sum())
        if s4 <= 0.0:
            return 0.0
        return s2 * s2 / s4

    # Row-gram (N, N). For N=2048 this is 2048x2048=32 MB at float64;
    # comfortably in RAM on the target CPU nodes.
    R = x_centred @ x_centred.T
    diag_R = np.diag(R)
    s1 = float(diag_R.sum())
    s2_diag = float((diag_R * diag_R).sum())
    t_full = float((R * R).sum())

    numerator = s1 * s1 - s2_diag
    denominator = t_full - s2_diag
    if denominator <= 0.0:
        # Degenerate case (e.g. constant rows); fall back to naive.
        sv = np.linalg.svd(x_centred, full_matrices=False, compute_uv=False)
        s_sum2 = float((sv * sv).sum())
        s_sum4 = float((sv * sv * sv * sv).sum())
        if s_sum4 <= 0.0:
            return 0.0
        return s_sum2 * s_sum2 / s_sum4
    return float(numerator / denominator)


def kv_core_rank(
    activations: np.ndarray,
    target_fidelity: float = 0.99,
) -> tuple[int, float]:
    """Chen et al. 2026 KV-CoRE Normalised Effective Rank.

    Computes two quantities per activation matrix:

    - ``rank``: the classical cumulative-variance threshold rank
      (smallest ``k`` whose squared singular values sum to at least
      ``target_fidelity`` of the total). This is the "how many
      components to keep" number a compression pipeline consumes.
    - ``ner``: the Chen et al. 2026 Normalised Effective Rank, an
      information-theoretic compressibility scalar in ``[1/r, 1]``
      complementary to ``rank``. Low NER indicates a spectrum
      dominated by a few large singular values (high compressibility);
      NER near ``1`` indicates a uniform spectrum (low
      compressibility).

    NER formula (Chen et al. 2026 ``docs/Kv-CoRe-Benchmarking-Chen2026.pdf``
    §2.4):

    .. math::

        \\mathrm{NER}(K) \\;=\\; \\frac{\\mathrm{erank}(K)}{r},
        \\qquad
        \\mathrm{erank}(K) \\;=\\; \\exp\\!\\left(-\\sum_{i=1}^{r} p_i \\ln p_i\\right),
        \\qquad
        p_i \\;=\\; \\frac{\\sigma_i}{\\sum_{j=1}^{r} \\sigma_j}

    where ``r`` is the numerical rank of the matrix (count of
    singular values above a tolerance threshold; ``r <= min(N, D)``),
    ``sigma_i`` are the raw (not squared) singular values of the
    **centred** activation matrix, and ``ln`` is the natural
    logarithm. The centring removes the trivial mean-component that
    would otherwise dominate a spurious rank-one.

    Parameters
    ----------
    activations : np.ndarray
        Two-dimensional array of shape ``(N, D)`` containing one
        activation vector per row. Centred column-wise inside the
        function; callers do not need to pre-centre.
    target_fidelity : float
        Fraction of activation energy the ``rank`` component must
        preserve, in ``(0, 1]``.

    Returns
    -------
    rank : int
        Number of components required to reach ``target_fidelity``
        in cumulative squared-singular-value variance. Consistent
        with ``compute_effective_rank``'s semantics.
    ner : float
        Chen 2026 NER in ``[1/r, 1]`` (or ``0.0`` for a degenerate
        zero-rank matrix). Reported as a third cross-validation
        metric per `docs/extend-notes.md` §1.2 RQ6 three-way table.
    """
    X = np.asarray(activations, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"kv_core_rank: activations must be 2-D (got {X.ndim}-D)"
        )
    if not (0.0 < target_fidelity <= 1.0):
        raise ValueError(
            f"kv_core_rank: target_fidelity {target_fidelity!r} outside (0, 1]"
        )
    n, d = int(X.shape[0]), int(X.shape[1])
    if n == 0 or d == 0:
        return 0, 0.0

    X_centred = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(X_centred, full_matrices=False, compute_uv=False)
    if sv.size == 0 or sv[0] == 0.0:
        return 0, 0.0

    # Numerical-rank tolerance consistent with ``np.linalg.matrix_rank``.
    tol = float(max(n, d)) * float(np.finfo(np.float64).eps) * float(sv[0])
    sv_nz = sv[sv > tol]
    r = int(sv_nz.size)
    if r == 0:
        return 0, 0.0

    # Cumulative-variance rank (classical):
    sv2 = sv_nz * sv_nz
    cumvar = np.cumsum(sv2) / float(sv2.sum())
    idx = int(np.searchsorted(cumvar, float(target_fidelity), side="left"))
    rank = min(idx + 1, r)

    # Chen 2026 NER (paper-canonical: raw singular values, natural log):
    p = sv_nz / float(sv_nz.sum())
    p_nonzero = p[p > 0.0]
    entropy_nat = -float((p_nonzero * np.log(p_nonzero)).sum())
    erank = float(np.exp(entropy_nat))
    ner = erank / float(r)
    return rank, float(ner)
