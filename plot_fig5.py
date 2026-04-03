#!/usr/bin/env python3
"""Plot Figure 5: Router weight distribution for the last repeat at 40k vs 60k.

Reproduces Figure 5 of the CoTFormer paper (Section 5). Shows that longer
training shifts the router weight distribution for the deepest repeat towards
higher values, indicating the model learns to utilise the full depth more.

Prerequisites:
  - router_weights_40k.npy (from get_router_weights.py on ckpt_40000.pt)
  - router_weights_60k.npy (from get_router_weights.py on ckpt.pt / 60k)

Usage:
  python plot_fig5.py --weights-40k <path> --weights-60k <path>
  python plot_fig5.py --weights-40k <path> --weights-60k <path> --output fig5.png
"""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 5: Router weight distribution at 40k vs 60k",
    )
    parser.add_argument("--weights-40k", type=str, required=True,
                        help="Path to router_weights .npy from 40k checkpoint")
    parser.add_argument("--weights-60k", type=str, required=True,
                        help="Path to router_weights .npy from 60k checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: alongside 60k weights)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load router weights (list of arrays, one per repeat; index 0 = None)
    weights_40k = np.load(args.weights_40k, allow_pickle=True)
    weights_60k = np.load(args.weights_60k, allow_pickle=True)

    # Figure 5 plots the LAST repeat's weight distribution
    # router_weights[0] is None (first repeat has no router decision)
    # router_weights[-1] is the last repeat's per-token sigmoid scores
    last_40k = weights_40k[-1]
    last_60k = weights_60k[-1]

    if last_40k is None or last_60k is None:
        raise ValueError("Last-repeat router weights are None -- wrong checkpoint?")

    # -- Plot (matching paper Figure 5 style: density histogram) --
    fig, ax = plt.subplots(figsize=(5.5, 4))

    bins = np.linspace(0, 1, 41)

    # Paper colors: blue for 40k, orange for 60k
    ax.hist(last_40k, bins=bins, density=True, alpha=0.65,
            color="#1f77b4", label="CoTFormer (40k Training)", edgecolor="white", linewidth=0.5)
    ax.hist(last_60k, bins=bins, density=True, alpha=0.65,
            color="#ff7f0e", label="CoTFormer (60k Training)", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Repeat Weight Prediction", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)  # paper shows ~0.6 max; slight headroom
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if args.output:
        output = args.output
    else:
        output = os.path.join(os.path.dirname(args.weights_60k), "figure5_router_dist.png")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Figure 5 saved: {output}")
    plt.close()


if __name__ == "__main__":
    main()
