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
  partial |r| >= 0.1 after position control is the RQ4 falsification
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

Body deferred to Phase 2.
"""

from __future__ import annotations

import numpy as np


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, float]:
    """Pearson partial correlation r(x, y | z) with two-sided p-value."""
    raise NotImplementedError("body in Phase 2")


def paired_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t-test on equal-length vectors ``a``, ``b``."""
    raise NotImplementedError("body in Phase 2")


def z_score_per_sequence(
    loss: np.ndarray,
    sequence_ids: np.ndarray,
) -> np.ndarray:
    """Per-sequence z-score the loss vector using sequence-level means / stds."""
    raise NotImplementedError("body in Phase 2")
