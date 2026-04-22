"""Protocol G -- KV Compressibility (M3 MLA design, RQ6 cross-validation).

Scope
-----
Replaces the legacy ``analyze_kv_compression.py`` monolith. Two
deliberately-separated passes measure different ontological objects:

- **Type A (weight-level)**: SVD of the fused ``c_attn`` matrix's K
  and V slices separately. No data; measures the intrinsic rank of
  the linear projection.
- **Type B (activation-level)**: SVD over real OWT2 KV vectors
  captured by the collector. Measures the rank of the empirical KV
  distribution at a specific (layer, repeat) site. Computes the
  KV-CoRE normalised effective rank (Chen et al. 2026) alongside the
  classical 0.90 / 0.95 / 0.99 thresholds.

Falsifiability relevance
------------------------
Prediction 1 "weight decay drives QK/OV to low rank" is rejected if
Type A yields 0.95-threshold rank > 40 (> 63 per cent of nominal
d_head=64). RQ6 cross-validation: Type B rank at later repeats
should track the Protocol F participation-ratio trend within 10 per
cent; disagreement flags a measurement artefact.

Ontological purpose
-------------------
Determines whether the provisional uniform MLA rank
(``kv_lora_rank=192``) is sufficient or whether per-layer /
per-repeat allocation is warranted. If Type B ranks are
substantially lower than Type A at later repeats, the activation
distribution is MORE compressible than the weight matrix -- a
finding supporting aggressive per-repeat compression.

Body deferred to Phase 2. Centring activations before SVD is
mandatory (removes the trivial mean-component); sample-size
convergence is verified by reporting rank at N in {1024, 2048, 4096}.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol G.

    Expected inputs: ``--checkpoint``, ``--checkpoint-file``,
    ``--workspace``, ``--output-dir``, ``--seed``, ``--target-rank``
    (MLA ``kv_lora_rank`` comparison line; default 192),
    ``--n-batches``, ``--skip-activations``.
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
