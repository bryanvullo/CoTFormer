"""Protocol A-ext -- Tuned Lens (Belrose et al. 2023).

Scope
-----
Addresses RQ1 triangulation. Trains per-layer affine translators on
the model's residual stream and reports Tuned-Lens-projected top-1
accuracy, KL divergence, and entropy alongside the unweighted
logit-lens curves of Protocol A. The translator-training recipe is
the DEC-M2-017 canonical setup: SGD with Nesterov momentum, lr=1.0,
with a Muon-optimiser fallback if convergence stalls.

Falsifiability relevance
------------------------
The Tuned Lens is the reliability triangulation for RQ1: if the
unweighted logit-lens accuracy curve and the Tuned-Lens curve
disagree in monotonicity or in the paired-t-test verdict, the
convergence claim fails the `docs/extend-notes.md` §1.6 triangulation
bar (a single-measurement result is downgraded to preliminary).

Ontological purpose
-------------------
Removes the "lm_head only trained against the final residual"
systematic baseline disadvantage present in Protocol A: every
per-repeat projection is trained against its own residual, so the
early-repeat decoding is as fair as the late-repeat decoding. Any
systematic monotonicity that survives both the unweighted and the
tuned lens is a fact about the model's representations, not a fact
about lm_head bias.

Body deferred to Phase 2. See `docs/extend-notes.md` §1.3 Protocol A-ext
row for the compute budget (~5 h CPU for translator training).
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol A-ext (Tuned Lens).

    Expected inputs: ``--checkpoint``, ``--checkpoint-file``,
    ``--workspace``, ``--output-dir``, ``--seed``,
    ``--translator-epochs``, ``--optimiser`` (``sgd-nesterov`` or
    ``muon``; DEC-M2-017 fallback).
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
