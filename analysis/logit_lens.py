"""Protocol A -- Logit Lens (unweighted).

Scope
-----
Addresses RQ1 (prediction convergence across repeats). Runs a
CoTFormer variant over 2048 tokens of OWT2 validation data and
captures the residual stream at two sites per repeat: the raw
post-`h_mid` residual and the post-`ln_mid` residual. Each site is
projected through the frozen ``lm_head`` to produce per-repeat token
distributions; top-1 accuracy against the ground-truth next token,
KL divergence between successive repeats, and entropy are computed
over the same token set.

Falsifiability relevance
------------------------
H0: "top-1 accuracy flat across repeats" is rejected under a paired
t-test between repeats 1 and ``n_repeat`` at p < 0.01 AND a
monotonically non-decreasing accuracy curve (per
`docs/extend-notes.md` §1.2 RQ1 "Statistical test" and "Falsification
threshold"). The dual measurement (with and without `ln_mid`)
isolates the normalisation confounder; see
`docs/extend-notes.md` §1.2 RQ1 "Confounders".

Ontological purpose
-------------------
Determines whether iterative repeats implement refinement toward a
fixed point (diminishing returns profile) or qualitative
restructuring; this is the central distinction of SQ1 and decides
whether the paper's "chain-of-thought" analogy is accurate for the
weight-tied recurrent transformer. Triangulated with Protocol A-ext
(Tuned Lens) per the `docs/extend-notes.md` §1.6 triangulation matrix.

Body deferred to Phase 2.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol A.

    Expected inputs per `docs/extend-notes.md` §1.3 Protocol A row:
    ``--checkpoint``, ``--checkpoint-file``, ``--workspace``,
    ``--output-dir``, ``--seed``, ``--max-tokens``.
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
