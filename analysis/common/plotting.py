"""Plotting helpers -- figure setup, site labels, palette, save.

Scope
-----
Enforces visual consistency across the 11 protocol scripts and the
synthesis script. Every protocol calls ``setup_figure`` for its
canvas and ``savefig`` for its write; protocol-specific plot bodies
stay inside each protocol script.

Falsifiability relevance
------------------------
Plots are communication artefacts rather than falsification
instruments, but consistency is a reliability property:
inconsistent palettes across protocols invite misreads of
cross-protocol comparisons. The helpers below pin the conventions.

Ontological purpose
-------------------
Deduplicates figure-setup boilerplate across 12+ scripts; a future
style change (journal revision, viridis-to-cividis switch) updates
one module.

Body deferred to Phase 2.
"""

from __future__ import annotations

from typing import Any


def setup_figure(rows: int, cols: int, size: tuple[float, float] = (12.0, 8.0)) -> Any:
    """Return ``(fig, axes)`` from matplotlib with the project's defaults."""
    raise NotImplementedError("body in Phase 2")


def site_label(group: str, layer: int, repeat: int | None) -> str:
    """Format a human-readable site label, e.g. ``"mid[0] @ r3"``."""
    raise NotImplementedError("body in Phase 2")


def palette_for_repeats(n_repeat: int) -> list[str]:
    """Return an ``n_repeat``-long list of hex colours for repeat indexing."""
    raise NotImplementedError("body in Phase 2")


def savefig(fig: Any, path: str) -> None:
    """Save ``fig`` to ``path`` with the project-standard dpi and bbox."""
    raise NotImplementedError("body in Phase 2")
