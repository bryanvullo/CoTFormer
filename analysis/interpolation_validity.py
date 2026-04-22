"""Protocol H -- Interpolation Validity (RQ7).

Scope
-----
Addresses RQ7 (clone-set interpolation entropy). Generates a
clone-set of sequences with controlled pair-distance in a shared
feature-space embedding, measures per-sample halting depth on the
ADM (C5) and entropy of the halting distribution within each
clone-set. Permutation test (N = 10000 per DEC-026) against a
seed-perturbed null.

Falsifiability relevance
------------------------
H0 "clone-set halting entropy indistinguishable from seed-perturbed
null" is rejected at the permutation p < 0.01 threshold
(`docs/extend-notes.md` §1.2 RQ7). The test tells whether ADM
halting depth is a coherent function of input semantics (non-null)
or an idiosyncratic artefact of weight-init randomness (null).

Ontological purpose
-------------------
A positive result means the adaptive-depth router implements a
learned interpolation on sequence semantics; a negative result means
routing depth is effectively noise at the clone-set granularity,
which bears directly on the RQ4 router-validity claim and on any
future work that treats halting depth as an interpretable signal.

Body deferred to Phase 2.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol H.

    Expected inputs: ``--checkpoint``, ``--checkpoint-file``,
    ``--workspace``, ``--output-dir``, ``--seed``, ``--clone-size``
    (sequences per clone-set), ``--n-permutations`` (10000 per
    DEC-026).
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
