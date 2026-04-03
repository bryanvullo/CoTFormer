#!/usr/bin/env python3
"""Plot Figure 4: Perplexity vs MACs for different inference compute budgets.

Reproduces Figure 4 of the CoTFormer paper (Section 4.3). Two curves from the
same adaptive LN-CoTFormer (ADM) checkpoint:
  1. Router: vary router threshold to control per-token depth
  2. Fixed Depth: activate a fixed prefix of repeats for all tokens

Prerequisites:
  - eval_per_threshold.npy (from get_ppl_per_mac.py, router sweep)
  - eval_per_layer.npy    (from get_ppl_per_mac.py, fixed depth sweep)

Usage:
  python plot_fig4.py --checkpoint <adm_checkpoint_dir>
  python plot_fig4.py --checkpoint <dir> --output figure4.png
"""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 4: Perplexity vs MACs at inference",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="ADM checkpoint dir containing eval_per_*.npy")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: <checkpoint>/figure4_pareto.png)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt = args.checkpoint

    # Load evaluation data (saved by get_ppl_per_mac.py)
    threshold_path = os.path.join(ckpt, "eval_per_threshold.npy")
    layer_path = os.path.join(ckpt, "eval_per_layer.npy")

    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Missing {threshold_path} -- run get_ppl_per_mac.py first")
    if not os.path.exists(layer_path):
        raise FileNotFoundError(f"Missing {layer_path} -- run get_ppl_per_mac.py first")

    threshold_data = np.load(threshold_path, allow_pickle=True).item()
    layer_data = np.load(layer_path, allow_pickle=True).item()

    # Extract Router curve: threshold -> (coefs, stats, macs)
    router_macs, router_ppl = [], []
    for threshold in sorted(threshold_data.keys()):
        coefs, stats, macs = threshold_data[threshold]
        router_macs.append(macs / 1e9)
        router_ppl.append(stats["val_perplexity"])

    # Extract Fixed Depth curve: depth_index -> (coefs, stats, macs)
    fixed_macs, fixed_ppl = [], []
    for depth in sorted(layer_data.keys()):
        coefs, stats, macs = layer_data[depth]
        fixed_macs.append(macs / 1e9)
        fixed_ppl.append(stats["val_perplexity"])

    # -- Plot (matching paper Figure 4 style) --
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Paper colors: green circles for Router, orange triangles for Fixed Depth
    ax.plot(router_macs, router_ppl, "o-",
            color="#2ca02c", label="LN-CoTFormer (Router)",
            markersize=7, linewidth=1.8, zorder=3)
    ax.plot(fixed_macs, fixed_ppl, "^-",
            color="#ff7f0e", label="LN-CoTFormer (Fixed Depth)",
            markersize=7, linewidth=1.8, zorder=3)

    ax.set_xlabel(r"Multiply-Accumulate Operations ($\times 10^9$)", fontsize=11)
    ax.set_ylabel("Perplexity", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output = args.output or os.path.join(ckpt, "figure4_pareto.png")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Figure 4 saved: {output}")
    print("Note: 'LN-Block Universal (Router)' curve omitted -- requires a separate trained model.")
    plt.close()


if __name__ == "__main__":
    main()
