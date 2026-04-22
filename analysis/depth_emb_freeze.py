"""Protocol I -- Depth-Embedding 8-Condition Freeze Ablation (RQ8).

Scope
-----
Addresses RQ8 (causal necessity of the depth embedding). Runs the
LN-CoTFormer (C3 at 60k) under 8 freeze conditions produced by the
Cartesian product of {freeze / unfreeze} x {zero / preserve / random}
applied to the ``depth_embedding`` entry and two matched controls
(per DEC-026's zero-vector condition addition). Per-condition
perplexity and loss are reported with bootstrap 95 per cent CI.

Falsifiability relevance
------------------------
H0 "depth embedding is non-causal for the recurrent convergence
signature" is rejected if perplexity under the zero-condition is
significantly worse than the preserve-condition at paired bootstrap
p < 0.01 AND the effect exceeds the seed-variance baseline (per
`docs/extend-notes.md` §1.2 RQ8 "Statistical test"). The 8-condition
ANOVA is per-condition rather than the earlier 4-condition layout
per DEC-026.

Ontological purpose
-------------------
Isolates depth-embedding causality from init-order confounds
(pre-B6 vs post-B6 checkpoints have different init-order RNG). A
non-causal depth embedding is evidence that the paper's C3-vs-C2
gain is attributable to ``ln_mid`` alone, which bears on RQ10's
Table-2-unconfound agenda.

Body deferred to Phase 2. GOTCHA: the model's forward loop at
`models/cotformer_full_depth_lnmid_depthemb.py:354` uses reverse
indexing ``n_repeat - rep_idx`` to select the depth embedding; the
freeze condition must patch the correctly-indexed entry, not the
forward-indexed one. See hostile-review evidence citation.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol I.

    Expected inputs: ``--checkpoint``, ``--checkpoint-file``,
    ``--workspace``, ``--output-dir``, ``--seed``, ``--freeze-mode``
    (zero / preserve / random), ``--freeze-target`` (``depth_emb``
    entry index; 1..n_repeat using the reverse-index convention of
    the forward loop), ``--n-bootstrap``.
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
