"""Protocol D -- Router Analysis (RQ4 + RQ5).

Scope
-----
Addresses RQ4 (router validity) and RQ5 (context preservation).
Runs the ADM checkpoint (C5 at 60k) over OWT2 validation, capturing
per-token router scores via the existing ``get_router_weights.py``
hook pattern on ``model.transformer.mod[r].mod_router`` AND
per-token CE loss via an ``F.cross_entropy(..., reduction='none')``
shim. Two analysis passes produce two JSON outputs
(``router_loss.json`` for RQ4, ``context_survival.json`` for RQ5)
from the shared forward.

Falsifiability relevance
------------------------
- RQ4 H0 "no correlation between halting depth and per-token CE
  loss" is rejected when the partial correlation |r| >= 0.1 after
  controlling for position, with bootstrap 95 per cent CI (per
  DEC-M2-026).
- RQ5 H0 "target loss independent of context-survival fraction" is
  rejected at Pearson |r| >= 0.1 across sequences satisfying the
  strict ``halt_depth(j) >= 5`` conditioning (per DEC-M2-026 RQ5
  resolution).

Ontological purpose
-------------------
- RQ4 decides whether the ADM router learns a meaningful difficulty
  proxy or a near-random gate -- a first-order question about
  whether the paper's "budget-adaptive" claim is empirically
  supported.
- RQ5 decides whether surviving tokens suffer "contextual
  isolation", a previously undocumented limitation of adaptive-depth
  transformers. RQ5 DV-2 is subject to Protocol D-calibration
  (Spearman + classifier gate); DV-3 cosine similarity is a
  reliability triangulation per DEC-M2-026.

Body deferred to Phase 2.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol D.

    Expected inputs: ``--checkpoint``, ``--checkpoint-file``,
    ``--workspace``, ``--output-dir``, ``--seed``,
    ``--min-halt-depth`` (RQ5 conditioning threshold),
    ``--n-bootstrap`` (for the 95 per cent CI).
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
