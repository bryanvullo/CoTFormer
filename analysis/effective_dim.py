"""Protocol F -- Effective Dimensionality (RQ6).

Scope
-----
Addresses RQ6 (effective dimensionality across repeats). CPU-only
post-processor over Protocol A's residual cache. For each (layer,
repeat) site computes the participation ratio with the Chun et al.
2025 finite-sample correction; per-layer log-linear regression on
the repeat axis replaces the earlier Mann-Kendall trend test per
DEC-M2-026.

Falsifiability relevance
------------------------
H0 "d_eff constant within 5 per cent across repeats" is rejected
per-layer when the log-linear regression slope is significantly
different from zero at alpha=0.01 AND the d_eff change exceeds 5 per
cent of the baseline (per `docs/extend-notes.md` §1.2 RQ6).

Ontological purpose
-------------------
If d_eff decreases across repeats, later repeats operate in
lower-dimensional subspaces -- directly determining MLA rank
allocation (DEC-EXT-001). Stable d_eff supports uniform
``kv_lora_rank``; a >= 2x d_eff variation warrants CARE-style
per-layer/per-repeat rank allocation. The top-2-PC-removed PR
cross-check is mandatory when Protocol E surfaces rogue-direction
norm outliers.

Body deferred to Phase 2.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol F.

    Expected inputs: ``--workspace``, ``--output-dir``, ``--seed``,
    ``--top-k-pc-remove`` (0 by default; 2 for the rogue-direction
    cross-check when Protocol E flags norm outliers).
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
