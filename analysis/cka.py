"""Protocol B -- Centred Kernel Alignment (RQ2).

Scope
-----
Addresses RQ2 (representation structural similarity across repeats).
CPU-only post-processor over the residual-stream cache produced by
Protocol A's collector: computes debiased HSIC U-statistic CKA with
linear and RBF kernels over all (layer, repeat) site pairs,
producing a 105x105 matrix for the canonical C3 LN-CoTFormer at
60k (21 mid-layers x 5 repeats), plus intra-layer, cross-layer, and
grand views.

Falsifiability relevance
------------------------
H0 "CKA > 0.9 across all cross-repeat pairs within each layer" is
rejected when any adjacent-repeat pair within any mid-layer yields
CKA < 0.7 (per `docs/extend-notes.md` §1.2 RQ2 "Falsification
threshold"). Debiased U-statistic + permutation-null zone thresholds
are per DEC-M2-026; without the debias, finite-sample CKA carries a
positive bias that would reject a null the bias itself created.

Ontological purpose
-------------------
Resolves whether cross-repeat representations are redundant (high
CKA -- MLA-style compression is trivial) or uniquely informative
(low CKA -- the model does distinct work at each repeat, and the
layer-resolved map shows where).

Body deferred to Phase 2.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol B.

    Expected inputs: ``--workspace``, ``--output-dir``, ``--seed``,
    ``--n-permutations`` (for the permutation-null zone).
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
