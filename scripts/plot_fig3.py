#!/usr/bin/env python3
"""Plot Figure 3: MACs vs Sequence Length (log-log).

Reproduces Figure 3 of the CoTFormer paper (Section 4.2). Log-log plot
comparing compute cost as a function of sequence length:
  - Block Universal Transformer (12x5): blue #1f77b4
  - CoTFormer (12x3): green #2ca02c

Data is read entirely from macs.json — no PPL values used here.

Produces:
  fig3.png
  fig3.pdf

Usage:
  python scripts/plot_fig3.py --macs <macs.json> --output-dir <run_N/figs/>
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
# label text; the MACs ylabel is wrapped onto two lines for the same reason.
LABEL_FONTSIZE = 16
XTICK_FONTSIZE = 13
YTICK_FONTSIZE = 16
LEGEND_FONTSIZE = 13
CAPTION_FONTSIZE = 14  # matches scripts/plot_fig2.py (a)/(b) panel-label convention

# Canonical sequence-length axis from the schema
SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 12288]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_curve(macs_data: dict, family: str, n_layer: int, n_repeat: int) -> tuple:
    """Return (seq_lens, macs_raw) lists for the given curve config."""
    lookup = {}
    for pt in macs_data["points"]:
        if (
            pt["family"] == family
            and pt["n_layer"] == n_layer
            and pt["n_repeat"] == n_repeat
        ):
            lookup[pt["seq_len"]] = pt["macs"]

    xs, ys = [], []
    for sl in SEQ_LENGTHS:
        if sl in lookup:
            xs.append(sl)
            ys.append(lookup[sl])
    return xs, ys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Figure 3: MACs vs Sequence Length (log-log)"
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
        help="Directory where fig3.png and fig3.pdf are written",
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = YTICK_FONTSIZE

    with open(args.macs, "r") as fh:
        macs_data = json.load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    but_x, but_y = extract_curve(macs_data, family="BUT", n_layer=12, n_repeat=5)
    cot_x, cot_y = extract_curve(macs_data, family="CoTFormer", n_layer=12, n_repeat=3)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        but_x, but_y,
        color=COLOR_BUT,
        marker=MARKER,
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="Block Universal Transformer (12x5)",
        zorder=3,
    )
    ax.plot(
        cot_x, cot_y,
        color=COLOR_COT,
        marker=MARKER,
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="CoTFormer (12x3)",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Canonical x-ticks at the paper's sequence-length grid
    all_x = sorted(set(but_x) | set(cot_x))
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(v) for v in all_x], rotation=45, fontsize=XTICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)

    ax.set_xlabel("Sequence Length", fontsize=LABEL_FONTSIZE)
    # Wrapped onto two lines: rotated 90 deg, the single-line form is taller
    # than the figure canvas height at legibility-target font sizes, which
    # matplotlib's bbox_inches="tight" silently clips (see fig2 note above).
    ax.set_ylabel("Multiply-Accumulate\nOperations", fontsize=LABEL_FONTSIZE)

    # Legend placed ABOVE the axes rather than "upper left" (which matched
    # the paper screenshot at the original small font): at legibility-target
    # font size the wider legend box overlaps the CoTFormer/BUT curves near
    # their upper-right (high seq-len) end. Anchoring above the axes avoids
    # this regardless of curve shape; bbox_inches="tight" grows the saved
    # image to include it.
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(False)

    # Panel label, matching scripts/plot_fig2.py's (a)/(b) fig.text convention:
    # figure-fraction coordinates on the same base FIG_SIZE canvas as fig2,
    # centered below the axes, same fontsize, no explicit fontweight (fig2
    # doesn't set one either, so both default to normal weight).
    fig.text(
        0.5, -0.04,
        "(c)",
        ha="center",
        fontsize=CAPTION_FONTSIZE,
    )

    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"fig3.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Written: {out}")
    plt.close()


if __name__ == "__main__":
    main()
