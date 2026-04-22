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

Defaults
--------
- dpi = 150
- bbox_inches = "tight"
- Agg backend (no X server required on Iridis compute nodes)
- Repeat palette for ``n_repeat=5``: green, blue, orange, red,
  purple; matches the paper's Figure 5 40k/60k colour convention
  where applicable.
"""

from __future__ import annotations

from typing import Any


_REPEAT_5_PALETTE = ("#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd")


def _import_matplotlib():
    """Lazy-import matplotlib with the Agg backend pinned.

    Agg is the non-GUI backend used on Iridis compute nodes (no X
    server). Every protocol script must import matplotlib via this
    helper so the backend is set before the first ``pyplot`` call.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def setup_figure(
    rows: int,
    cols: int,
    size: tuple[float, float] = (12.0, 8.0),
) -> tuple[Any, Any]:
    """Return ``(fig, axes)`` from matplotlib with the project's defaults.

    Parameters
    ----------
    rows, cols : int
        Subplot grid size passed to ``plt.subplots``. Callers with a
        single axis should pass ``rows=1, cols=1`` and index with
        ``axes`` directly (no array wrapping).
    size : tuple of float
        Figure size in inches (default ``(12, 8)``). Override for
        multi-column prints or poster-format exports.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray[matplotlib.axes.Axes] or Axes
        ``np.ndarray`` when ``rows * cols > 1``; a single ``Axes`` when
        both are 1. ``plt.subplots`` governs the wrapping behaviour.
    """
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(rows, cols, figsize=size)
    return fig, axes


def site_label(group: str, layer: int, repeat: int | None) -> str:
    """Format a human-readable site label.

    ``site_label("mid", 3, 2) -> "mid[3] r=2"``. When ``repeat`` is
    ``None``, the repeat suffix is omitted (used for pre-`h_mid` sites
    such as ``h_begin_last`` and for post-`h_end` sites).
    """
    core = f"{group}[{layer}]"
    if repeat is None:
        return core
    return f"{core} r={repeat}"


def palette_for_repeats(n_repeat: int) -> list[str]:
    """Return ``n_repeat`` hex colours for repeat indexing.

    For ``n_repeat == 5`` returns the paper-Figure-5 palette; for
    other ``n_repeat`` values returns a viridis sample of length
    ``n_repeat`` so the colour mapping degrades gracefully (same
    ordering, different hues) if the checkpoint's ``n_repeat``
    differs from the paper's ``5``.
    """
    if n_repeat == 5:
        return list(_REPEAT_5_PALETTE)

    matplotlib, _ = _import_matplotlib()
    cmap = matplotlib.colormaps["viridis"]
    if n_repeat <= 1:
        return ["#1f77b4"]
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(round(255 * r)),
            int(round(255 * g)),
            int(round(255 * b)),
        )
        for r, g, b, _a in (
            cmap(i / max(1, n_repeat - 1)) for i in range(n_repeat)
        )
    ]


def savefig(fig: Any, path: str) -> None:
    """Save ``fig`` to ``path`` with dpi=150 and bbox_inches='tight'.

    Closes the figure after write to release RAM; protocol scripts
    that generate many plots rely on this to avoid the matplotlib
    ``> 20 figures open`` warning.
    """
    _, plt = _import_matplotlib()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
