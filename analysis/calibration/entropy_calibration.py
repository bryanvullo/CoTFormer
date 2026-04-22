"""Protocol D-calibration entry point.

Scope
-----
Runs the three-tier substrate ladder (GPT-2-large -> ADM C5 -> 24L
standard, contingency only) across the two-condition synthetic
sequence set (Condition A sparse induction-head, Condition B
broad-integration multi-target) and computes the four-gate
validation ladder (baseline >= 0.5, sink-correction < 1 bit,
Tier 1 / Tier 2 triangulation, Spearman <= -0.5 AND classifier
>= 0.8).

Outputs
-------
- ``metrics.csv``       -- per-sequence entropy, H_no_sink, target
                            accuracy, attention-sink mass.
- ``spearman.json``     -- Spearman rho, 95 per cent bootstrap CI,
                            per-tier breakdown.
- ``classifier.json``   -- linear classifier accuracy on
                            (entropy, target_accuracy); the decision
                            boundary is reused for DV-2 at test time
                            when the PASS verdict holds.
- ``figure.png``        -- entropy-vs-accuracy scatter with decision
                            boundary overlay.

Falsifiability relevance
------------------------
If the Spearman gate fails (rho > -0.3), the monotone interpretation
is refuted on the calibrated substrate; DV-2 is replaced by the
TOP-K-MASS fallback (fraction of attention mass at repeat 5 landing
on context tokens surviving to the same repeat or deeper). See
`docs/extend-notes.md` §1.3 "D-cal four-gate validation ladder"
verdict table for the AMBIGUOUS and FAIL paths.

Ontological purpose
-------------------
Empirically grounds an otherwise unprincipled interpretation of
attention entropy: the published literature (Meister et al. [43],
Zhai et al. [44], Xiao et al. [26], Gu et al. [46]) does not support
a bidirectional monotone mapping a priori; the calibration is the
methodological prerequisite for any subsequent claim about DV-2.

Body deferred to Phase 2.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for the calibration entry point.

    Expected inputs: ``--ckpt`` (Tier 1 path to GPT-2-large snapshot;
    Tier 2 path to ADM C5; Tier 3 contingency), ``--tier`` (1 / 2 / 3),
    ``--n-per-condition`` (default 1000 per condition for the primary
    ladder; 4000 per condition for the AMBIGUOUS-verdict retry),
    ``--seed``, ``--out`` (output directory), ``--module-path``
    (forwarded to the common loader; Tier 1 uses
    ``model.transformer.h``, Tiers 2 and 3 use
    ``model.transformer.h_mid``).
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
