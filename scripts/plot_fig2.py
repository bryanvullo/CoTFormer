#!/usr/bin/env python3
"""Plot Figure 2: Perplexity vs MACs for n_layer=12 and n_layer=24.

Reproduces Figure 2 of the CoTFormer paper (Section 4.2). Two subplots:
  (a) n_layer=12, x-axis log scale
  (b) n_layer=24, x-axis linear scale

Each subplot shows two curves:
  - Block Universal Transformer (blue #1f77b4, circle markers, solid)
  - CoTFormer (green #2ca02c, circle markers, solid)

Per-point text labels "NLxNR" placed slightly above each marker.

Produces:
  fig2a.png, fig2a.pdf
  fig2b.png, fig2b.pdf

Usage:
  python scripts/plot_fig2.py \\
      --results <results_table1_fig2.json> \\
      --macs <macs.json> \\
      --output-dir <run_N/figs/>
"""

import argparse
import json
import os


# ---------------------------------------------------------------------------
# Style constants (from schema_table1_fig23.md)
# ---------------------------------------------------------------------------

COLOR_BUT = "#1f77b4"
COLOR_COT = "#2ca02c"
MARKER = "o"
LINEWIDTH = 2.4
MARKERSIZE = 6
FIG_SIZE = (4.5, 3.4)

# Legibility params (font/tick/label/legend sizes) tuned so text remains
# readable when the figure is shrunk to ~1.6in (~0.24 textwidth) in the
# paper. figsize and data pipeline are unchanged. Sizes are capped below the
# point where matplotlib's bbox_inches="tight" (which intersects, but never
# expands, the tight bbox against the raw figsize canvas) would clip long
# label text; the MACs xlabel is wrapped onto two lines for the same reason.
LABEL_FONTSIZE = 14  # 16pt xlabel overran the canvas right edge (axes sit
                     # right-of-centre under the wide ylabel); 14pt measures
                     # 3.02in vs the 4.5in canvas, safe on both sides.
TICK_FONTSIZE = 14
XTICK_FONTSIZE_LOG = 11  # smaller x-tick font for fig2a's dense log-scale axis
LEGEND_FONTSIZE = 12
ANNOTATION_FONTSIZE = 11
CAPTION_FONTSIZE = 14


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_macs_lookup(macs_data: dict) -> dict:
    """Return dict: (family, n_layer, n_repeat, seq_len) -> macs (float, ×10⁹)."""
    lookup = {}
    for pt in macs_data["points"]:
        key = (pt["family"], pt["n_layer"], pt["n_repeat"], pt["seq_len"])
        lookup[key] = pt["macs"] / 1e9
    return lookup


def get_but_ppl(results: dict, n_layer: int, n_repeat: int):
    """Return PPL for BUT from paper_reference_table1 or paper_reference_fig2."""
    ref = results["paper_reference_table1"]
    key = f"block_universal_{n_layer}L"
    by_r = ref[key]["by_n_repeat"]
    r_str = str(n_repeat)
    if r_str in by_r:
        return by_r[r_str]["ppl"]
    # Extra points only in paper_reference_fig2 (12L, r=6 and r=15)
    extra = results.get("paper_reference_fig2", {}).get("but_12L_extra", {})
    if r_str in extra:
        return extra[r_str]["ppl_visual"]
    return None


def get_cot_ppl(results: dict, n_layer: int, n_repeat: int):
    """Return PPL for CoTFormer from ablations (ours) if available, else paper ref."""
    ablations = results["ablations"]
    for a in ablations:
        if a["family"] == "CoTFormer" and a["n_layer"] == n_layer and a["n_repeat"] == n_repeat:
            return a["eval"]["val_perplexity"]
    # Fall back to paper reference
    ref = results["paper_reference_table1"]
    key = f"cotformer_{n_layer}L_paper"
    by_r = ref[key]["by_n_repeat"]
    r_str = str(n_repeat)
    if r_str in by_r:
        return by_r[r_str]["ppl"]
    return None


# ---------------------------------------------------------------------------
# Subplot renderer
# ---------------------------------------------------------------------------

def _plot_subplot(
    ax,
    results: dict,
    macs_lookup: dict,
    n_layer: int,
    but_repeats: list,
    cot_repeats: list,
    xlog: bool,
) -> None:
    """Draw one subplot on ax."""
    import matplotlib.pyplot as plt  # noqa: F401 (matplotlib already imported by caller)

    # --- BUT curve ---
    but_x, but_y, but_labels = [], [], []
    for r in but_repeats:
        macs = macs_lookup.get(("BUT", n_layer, r, 256))
        ppl = get_but_ppl(results, n_layer, r)
        if macs is not None and ppl is not None:
            but_x.append(macs)
            but_y.append(ppl)
            but_labels.append(f"{n_layer}x{r}")

    # --- CoTFormer curve ---
    cot_x, cot_y, cot_labels = [], [], []
    for r in cot_repeats:
        macs = macs_lookup.get(("CoTFormer", n_layer, r, 256))
        ppl = get_cot_ppl(results, n_layer, r)
        if macs is not None and ppl is not None:
            cot_x.append(macs)
            cot_y.append(ppl)
            cot_labels.append(f"{n_layer}x{r}")

    # Plot lines
    ax.plot(
        but_x, but_y,
        color=COLOR_BUT,
        marker=MARKER,
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="Block Universal",
        zorder=3,
    )
    ax.plot(
        cot_x, cot_y,
        color=COLOR_COT,
        marker=MARKER,
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="CoTFormer",
        zorder=3,
    )

    # Per-point text labels (slightly above marker)
    y_range = max(but_y + cot_y) - min(but_y + cot_y) if (but_y + cot_y) else 1.0
    offset = y_range * 0.025
    # Per-point labels rendered in BLACK (not line color) for legibility — the
    # line color is reserved for the marker; text reads against the page bg.
    for x, y, lbl in zip(but_x, but_y, but_labels):
        ax.annotate(lbl, (x, y), xytext=(0, 6), textcoords="offset points",
                    fontsize=ANNOTATION_FONTSIZE, color="black", ha="center", va="bottom")
    # CoTFormer labels placed BELOW their markers (not above, like BUT): at
    # legibility-target font size the BUT/CoTFormer 12x2 (and 24x2) markers
    # sit close enough in y that two above-marker labels collide. CoTFormer
    # PPL is lower than BUT's at every plotted (n_layer, n_repeat) config, so
    # anchoring its label below the marker moves it away from BUT's label
    # without touching the data, axis ranges, or curve rendering.
    for i, (x, y, lbl) in enumerate(zip(cot_x, cot_y, cot_labels)):
        if i == len(cot_x) - 1:
            # Final (lowest, rightmost) point: label above-left, as in the
            # original figure, so it stays inside both axis edges.
            ax.annotate(lbl, (x, y), xytext=(-4, 8), textcoords="offset points",
                        fontsize=ANNOTATION_FONTSIZE, color="black", ha="right", va="bottom")
        else:
            ax.annotate(lbl, (x, y), xytext=(0, -6), textcoords="offset points",
                        fontsize=ANNOTATION_FONTSIZE, color="black", ha="center", va="top")

    # Extra margins so the below-marker label of the lowest/rightmost point
    # (12x15 / 24x5) stays inside the axes at legibility-target font sizes.
    ax.margins(x=0.07, y=0.14)

    if xlog:
        ax.set_xscale("log")

    # Wrapped onto two lines: at legibility-target font sizes the single-line
    # form is wider than the figure canvas, which matplotlib's
    # bbox_inches="tight" silently clips (it never expands past figsize).
    ax.set_xlabel(
        "Multiply-Accumulate Operations\n" r"($\times 10^9$)", fontsize=LABEL_FONTSIZE
    )
    ax.set_ylabel("Perplexity", fontsize=LABEL_FONTSIZE)
    # Legend placed ABOVE the axes (outside the data area) rather than in any
    # in-plot corner: at legibility-target font size the legend box is wide
    # enough that every in-plot corner ("upper right", "lower left", "best")
    # ends up overlapping either a curve or a point-label for at least one of
    # the two n_layer configs. Anchoring above the axes guarantees zero
    # overlap regardless of curve shape; bbox_inches="tight" simply grows the
    # saved image to include it.
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(False)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    if xlog:
        # Log-scale minor ticks (2x/3x/4x/6x within a decade) collide at
        # legibility-target font sizes; rotate to match fig3's convention for
        # dense tick axes instead of shrinking below a readable size.
        # which="both" is required -- most of these labels are minor ticks.
        ax.tick_params(axis="x", which="both", labelsize=XTICK_FONTSIZE_LOG, rotation=45)
    else:
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Figure 2: Perplexity vs MACs (a) 12L log, (b) 24L linear"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results_table1_fig2.json",
    )
    parser.add_argument(
        "--macs",
        type=str,
        required=True,
        help="Path to macs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where fig2a.png/.pdf and fig2b.png/.pdf are written",
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = TICK_FONTSIZE

    with open(args.results, "r") as fh:
        results = json.load(fh)
    with open(args.macs, "r") as fh:
        macs_data = json.load(fh)

    macs_lookup = build_macs_lookup(macs_data)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Fig 2(a): n_layer=12, x log scale ---
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    _plot_subplot(
        ax=ax,
        results=results,
        macs_lookup=macs_lookup,
        n_layer=12,
        but_repeats=[2, 3, 5, 6, 15],
        cot_repeats=[2, 3, 5, 15],
        xlog=True,
    )
    fig.text(
        0.5, -0.04,
        r"(a) $n_{\mathrm{layer}} = 12$ (x-axis is in log scale)",
        ha="center",
        fontsize=CAPTION_FONTSIZE,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"fig2a.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Written: {out}")
    plt.close()

    # --- Fig 2(b): n_layer=24, x linear scale ---
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    _plot_subplot(
        ax=ax,
        results=results,
        macs_lookup=macs_lookup,
        n_layer=24,
        but_repeats=[2, 3, 5],
        cot_repeats=[2, 3, 5],
        xlog=False,
    )
    fig.text(
        0.5, -0.04,
        r"(b) $n_{\mathrm{layer}} = 24$",
        ha="center",
        fontsize=CAPTION_FONTSIZE,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"fig2b.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Written: {out}")
    plt.close()


if __name__ == "__main__":
    main()
