"""Cross-checkpoint synthesis (plotting-only).

Scope
-----
Not a scientific protocol; presentation-only. Reads per-checkpoint
JSON artefacts produced by Protocols A through I and emits
cross-checkpoint comparative figures: RQ1 convergence curves
overlaid for C1 / C2 / C3 / C4 / C5, CKA grand matrices side-by-side,
RQ6 d_eff trajectories, Protocol G effective-rank summary, and an
overall ``synthesis_report.md`` with embedded figure references.

Ontological purpose
-------------------
Communication artefact; no variables, no hypotheses, no confounder
analysis. Exempt from the ontological audit of a scientific protocol
because it computes nothing -- every datum plotted is computed
upstream by the protocol scripts.

Body deferred to Phase 2.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for the synthesis driver.

    Expected inputs: ``--run-dir`` (the ``run_N/`` output root
    containing per-checkpoint subdirectories), ``--output-dir``
    (defaults to ``--run-dir``), ``--template`` (optional Jinja2
    template for the synthesis report).
    """
    raise NotImplementedError("body in Phase 2")


def main() -> None:
    raise NotImplementedError("body in Phase 2")


if __name__ == "__main__":
    main()
