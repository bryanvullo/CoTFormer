"""Protocol C -- Attention-Head Taxonomy (RQ3).

Scope
-----
Addresses RQ3 (attention-head specialisation). Runs a non-flash
forward pass (Flash Attention does not return per-head weights),
captures per-head attention weights at every mid-layer across C1,
C2 and C4 (40k checkpoints only per `docs/extend-notes.md` §1.5).
Aggregates the final-token attention distribution at the final
repeat over all (token, repeat) pairs and clusters the 252
head-profile vectors (12 heads x 21 layers) into 4-6 archetypes via
k-means with silhouette-based selection. Reports top-k attention
mass and Gini coefficient per head alongside the cluster assignments;
secondary output is per-head per-repeat attention entropy (Spearman
rho between repeat and entropy) to test Prediction 2 (Zucchet et al.
2025).

Falsifiability relevance
------------------------
H0 "cluster assignments uniformly distributed across layers" is
rejected when the layer-vs-cluster contingency table gives a
chi-squared p < 0.01 (`docs/extend-notes.md` §1.2 RQ3). Silhouette
permutation null per DEC-M2-026; cluster stability is reported under
bootstrap resampling. Prediction 2 H0 "attention entropy flat across
repeats" is rejected at Spearman rho < -0.8 per repeat-vs-entropy
ordering.

Ontological purpose
-------------------
If early (layers 3-7) and late (layers 17-21) mid-layers favour
different archetypes, the weight-tied transformer develops emergent
functional specialisation despite identical parameters -- a strong
finding about how cross-repeat KV caches carry structurally
different information.

Body deferred to Phase 2. Port from `origin/tak-ablation-code` --
see `docs/extend-notes.md` §1.7 "SHA pinning header convention" for
the required port-header format.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol C.

    Expected inputs: ``--checkpoint``, ``--checkpoint-file``,
    ``--workspace``, ``--output-dir``, ``--seed``, ``--n-clusters``
    (range for silhouette search), ``--non-flash`` (enforce
    non-flash attention; default ``True`` for this protocol).
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
