"""Protocol E -- Residual Diagnostics (SQ1, RQ6 pre-check).

Scope
-----
Addresses SQ1 residual-stream diagnostics and serves as the RQ6
pre-check. CPU-only post-processor over Protocol A's residual cache.
For each (layer, repeat) site it computes three per-token
quantities (mean L2 norm, per-token L2 variance, L_inf norm) and
produces three plots with 21 mid-layer curves indexed by repeat.

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

Body deferred to Phase 2.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol E.

    Expected inputs: ``--workspace``, ``--output-dir``, ``--seed``,
    ``--normalise-by-h-begin`` (remove the trivial depth baseline).
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
