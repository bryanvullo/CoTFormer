#!/usr/bin/env python3
"""Plot Figure 5: Router weight distribution for the last repeat at 40k vs 60k.

Reproduces Figure 5 of the CoTFormer paper (Section 5). Shows that longer
training shifts the router weight distribution for the deepest repeat towards
higher values, indicating the model learns to utilise the full depth more.

Prerequisites:
  - router_weights_*.npy (from get_router_weights.py on each checkpoint)
  - Optional sidecar `router_weights_meta.json` alongside each .npy,
    emitted by get_router_weights.py. When present, the sidecar is the
    authoritative source for the "last repeat slot". When absent, the
    last non-empty slot in the array is detected by a tail scan.

The "last repeat slot" logic is dynamic so the same plotter handles models
with any `n_repeat` value: for n_repeat=5 the slot is 4 (standard ADM);
for a hypothetical n_repeat=4 model it would be 3; etc. No hardcoded
assumption about the trained architecture is baked into this script.

Usage:
  python plot_fig5.py --weights-40k <path> --weights-60k <path>
  python plot_fig5.py --weights-40k <path> --weights-60k <path> --output fig5.png
"""

import argparse
import json
import os

import numpy as np


def _load_sidecar(weights_path):
    """Return the metadata sidecar dict for a weights .npy, or None.

    Each extraction run writes a single `router_weights_meta.json` alongside
    the (up to three) .npy files in the same directory. We look for it next
    to whichever .npy path the user passed.
    """
    meta_path = os.path.join(
        os.path.dirname(weights_path), "router_weights_meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def _detect_last_slot(weights_array):
    """Tail scan: return the index of the last non-empty slot, or -1."""
    for slot in range(len(weights_array) - 1, -1, -1):
        entry = weights_array[slot]
        if entry is not None and len(entry) > 0:
            return slot
    return -1


def _resolve_slot(weights_array, weights_path, label):
    """Determine the Figure 5 slot for this array.

    Priority: sidecar `last_router_slot` when available and consistent with
    the array shape, else tail-scan fallback. Raises with a clear message on
    malformed input.
    """
    sidecar = _load_sidecar(weights_path)
    detected = _detect_last_slot(weights_array)
    if detected < 0:
        raise ValueError(
            f"{label}: weights array has no non-empty slots "
            f"(len={len(weights_array)}). The extraction step likely failed.")

    if sidecar is not None:
        sidecar_slot = int(sidecar.get("last_router_slot", -1))
        if 0 <= sidecar_slot < len(weights_array):
            entry = weights_array[sidecar_slot]
            if entry is not None and len(entry) > 0:
                if sidecar_slot != detected:
                    print(f"  {label}: sidecar slot={sidecar_slot}, "
                          f"tail-scan slot={detected} (using sidecar)")
                return sidecar_slot
            print(f"  {label}: sidecar slot={sidecar_slot} is empty, "
                  f"falling back to tail-scan slot={detected}")
        else:
            print(f"  {label}: sidecar slot={sidecar_slot} out of range, "
                  f"falling back to tail-scan slot={detected}")

    return detected


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
    parser.add_argument("--slot", type=int, default=None,
                        help="Manual override for the Figure 5 slot index "
                             "(default: auto-detect via sidecar + tail scan)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load router weights (object array; each slot is either None or a flat
    # 1-D float array of per-token scores from a specific router).
    weights_40k = np.load(args.weights_40k, allow_pickle=True)
    weights_60k = np.load(args.weights_60k, allow_pickle=True)

    if args.slot is not None:
        slot_40k = slot_60k = args.slot
        print(f"Using manual slot override: {args.slot}")
    else:
        slot_40k = _resolve_slot(weights_40k, args.weights_40k, "40k")
        slot_60k = _resolve_slot(weights_60k, args.weights_60k, "60k")

    if slot_40k != slot_60k:
        raise ValueError(
            f"Slot mismatch: 40k=slot {slot_40k}, 60k=slot {slot_60k}. "
            f"Both checkpoints must come from models with the same n_repeat "
            f"(else the Figure 5 comparison is meaningless). Pass --slot to "
            f"force a specific index if you are doing a cross-architecture "
            f"comparison intentionally.")

    slot = slot_60k
    print(f"Figure 5 quantity: slot {slot} (last repeat router)")

    last_40k = weights_40k[slot]
    last_60k = weights_60k[slot]

    if last_40k is None or last_60k is None or len(last_40k) == 0 or len(last_60k) == 0:
        raise ValueError(
            f"Slot {slot} is empty in at least one of 40k/60k arrays "
            f"(40k len={0 if last_40k is None else len(last_40k)}, "
            f"60k len={0 if last_60k is None else len(last_60k)}).")

    # -- Plot (matching paper Figure 5 style: density histogram) --
    fig, ax = plt.subplots(figsize=(5.5, 4))

    bins = np.linspace(0, 1, 41)
    counts_40k, _ = np.histogram(last_40k, bins=bins, density=True)
    counts_60k, _ = np.histogram(last_60k, bins=bins, density=True)

    bin_width = bins[1] - bins[0]
    bar_width = bin_width * 0.48
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Paper colors: blue for 40k, green for 60k (side-by-side bars)
    ax.bar(bin_centers - bar_width / 2, counts_40k, width=bar_width,
           color="#1f77b4", label="CoTFormer (40k Training)",
           edgecolor="white", linewidth=0.3)
    ax.bar(bin_centers + bar_width / 2, counts_60k, width=bar_width,
           color="#2ca02c", label="CoTFormer (60k Training)",
           edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Repeat Weight Prediction", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
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
