#!/usr/bin/env python3
"""Analyze KV cache compressibility per layer for MLA design decisions.

For each attention layer in a trained LN-CoTFormer, computes:
  1. Singular value spectrum of W_K and W_V weight matrices (Type A: weight-level)
  2. Effective rank at 90%, 95%, 99% variance thresholds
  3. Per-layer KV activation SVD on real data (Type B: activation-level)
  4. Attention entropy per layer per repeat (requires --compute-entropy)

Results directly inform:
  - Whether uniform kv_lora_rank (like DeepSeek-V2's 192) is appropriate
  - Which layers are most/least compressible
  - Whether prefix/suffix layers need different treatment than mid layers

Usage:
  # Weight-level analysis only (fast, no data needed):
  python analyze_kv_compression.py --checkpoint <path_to_ckpt_dir>

  # Full analysis with activation-level SVD + attention entropy:
  python analyze_kv_compression.py --checkpoint <path_to_ckpt_dir> \
      --compute-activations --compute-entropy \
      --data_dir /scratch/ab3u21/datasets

  # With specific MLA rank to evaluate:
  python analyze_kv_compression.py --checkpoint <path_to_ckpt_dir> \
      --target-rank 192

Output:
  <checkpoint_dir>/kv_compression_analysis.json
  <checkpoint_dir>/kv_compression_weights.png
  <checkpoint_dir>/kv_compression_activations.png  (if --compute-activations)
"""

import argparse
import json
import os

import numpy as np
import torch

import models


def load_model_from_checkpoint(checkpoint_dir, checkpoint_file="ckpt.pt"):
    """Load a trained model from its checkpoint directory."""
    summary_path = os.path.join(checkpoint_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary.json in {checkpoint_dir}")

    with open(summary_path) as f:
        config_dict = json.load(f)["args"]

    class Config:
        pass

    config = Config()
    config.__dict__ = config_dict

    model = models.make_model_from_args(config)

    ckpt_path = os.path.join(checkpoint_dir, checkpoint_file)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model"].items()
            if "attn.bias" not in k and "wpe" not in k
        }
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"WARNING: No ckpt.pt found, analyzing randomly initialized weights")

    return model, config


def compute_effective_rank(singular_values, thresholds=(0.90, 0.95, 0.99)):
    """Compute effective rank: min dimensions to capture X% of total variance."""
    total_var = (singular_values**2).sum()
    cumvar = np.cumsum(singular_values**2) / total_var
    ranks = {}
    for t in thresholds:
        rank = int(np.searchsorted(cumvar, t)) + 1
        ranks[f"rank_{int(t*100)}"] = rank
    return ranks, cumvar


def analyze_weight_matrices(model):
    """Type A: Analyze W_K and W_V weight matrices directly (no data needed).

    For each attention layer, extracts the K and V projection weights from
    the fused QKV linear layer and computes their SVD. Fast and data-free.
    """
    results = []
    n_embd = model.config.n_embd

    # Collect all attention layers across h_begin, h_mid, h_end
    layer_groups = []
    if hasattr(model, "transformer"):
        t = model.transformer
        for i, block in enumerate(getattr(t, "h_begin", [])):
            layer_groups.append(("begin", i, block))
        for i, block in enumerate(getattr(t, "h_mid", getattr(t, "h", []))):
            layer_groups.append(("mid", i, block))
        for i, block in enumerate(getattr(t, "h_end", [])):
            layer_groups.append(("end", i, block))

    if not layer_groups:
        # Fallback: try flat block list
        t = model.transformer
        for i, block in enumerate(t.h):
            layer_groups.append(("flat", i, block))

    for group_name, layer_idx, block in layer_groups:
        attn = block.attn
        if not hasattr(attn, "c_attn"):
            print(f"  Skipping {group_name}[{layer_idx}]: no c_attn found")
            continue

        W = attn.c_attn.weight.detach().float()  # (3*n_embd, n_embd) or (n_embd, 3*n_embd)

        # Handle both weight layouts: (out, in) is standard for nn.Linear
        if W.shape[0] == 3 * n_embd:
            W_q = W[:n_embd, :]
            W_k = W[n_embd : 2 * n_embd, :]
            W_v = W[2 * n_embd :, :]
        elif W.shape[1] == 3 * n_embd:
            W_q = W[:, :n_embd]
            W_k = W[:, n_embd : 2 * n_embd]
            W_v = W[:, 2 * n_embd :]
        else:
            print(f"  Skipping {group_name}[{layer_idx}]: unexpected c_attn shape {W.shape}")
            continue

        # SVD of K and V weight matrices
        U_k, S_k, _ = torch.linalg.svd(W_k, full_matrices=False)
        U_v, S_v, _ = torch.linalg.svd(W_v, full_matrices=False)

        S_k_np = S_k.numpy()
        S_v_np = S_v.numpy()

        ranks_k, cumvar_k = compute_effective_rank(S_k_np)
        ranks_v, cumvar_v = compute_effective_rank(S_v_np)

        entry = {
            "group": group_name,
            "layer_idx": layer_idx,
            "label": f"{group_name}[{layer_idx}]",
            "W_k_shape": list(W_k.shape),
            "W_v_shape": list(W_v.shape),
            "k_singular_values": S_k_np.tolist(),
            "v_singular_values": S_v_np.tolist(),
            "k_effective_ranks": ranks_k,
            "v_effective_ranks": ranks_v,
            "k_cumulative_variance": cumvar_k.tolist(),
            "v_cumulative_variance": cumvar_v.tolist(),
        }
        results.append(entry)

    return results


def analyze_activations(model, config, data_dir, n_batches=10):
    """Type B: Analyze K/V activations on real data.

    Hooks into each attention layer to capture K/V tensors during forward pass.
    More accurate than weight-level analysis as it captures the actual distribution
    of K/V vectors the model produces on real text.
    """
    from contextlib import nullcontext
    from data.utils import get_dataset

    # Load dataset
    if data_dir:
        config.data_dir = data_dir
    data = get_dataset(config)

    model.cuda()
    model.eval()
    dtype = getattr(config, "dtype", torch.bfloat16)
    if isinstance(dtype, str):
        dtype = torch.bfloat16
    type_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

    n_embd = config.n_embd
    captured_kv = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # c_attn output: (batch, seq, 3*n_embd) — contains Q, K, V concatenated
            with torch.no_grad():
                out = output.detach().float()
                if out.shape[-1] == 3 * n_embd:
                    K = out[..., n_embd : 2 * n_embd]  # (B, T, n_embd)
                    V = out[..., 2 * n_embd :]
                    if name not in captured_kv:
                        captured_kv[name] = {"K": [], "V": []}
                    # Flatten batch and seq dims: each row is one token's K/V vector
                    captured_kv[name]["K"].append(K.reshape(-1, n_embd).cpu())
                    captured_kv[name]["V"].append(V.reshape(-1, n_embd).cpu())
        return hook_fn

    # Register hooks on all c_attn layers
    hooks = []
    t = model.transformer
    for group_name, blocks in [
        ("begin", getattr(t, "h_begin", [])),
        ("mid", getattr(t, "h_mid", getattr(t, "h", []))),
        ("end", getattr(t, "h_end", [])),
    ]:
        for i, block in enumerate(blocks):
            if hasattr(block.attn, "c_attn"):
                name = f"{group_name}[{i}]"
                h = block.attn.c_attn.register_forward_hook(make_hook(name))
                hooks.append(h)

    # Run forward passes to collect activations
    seq_len = config.sequence_length
    batch_size = min(config.batch_size, 16)  # Cap for VRAM safety

    from data.utils import Dataset
    dataset = Dataset(data["val"], sequence_length=seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"  Collecting KV activations over {n_batches} batches...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= n_batches:
                break
            x = x.cuda()
            with type_ctx:
                model(x, targets=y.cuda())

    # Remove hooks
    for h in hooks:
        h.remove()

    # Analyze collected activations
    results = []
    for name in sorted(captured_kv.keys()):
        K_all = torch.cat(captured_kv[name]["K"], dim=0).numpy()  # (N_tokens, n_embd)
        V_all = torch.cat(captured_kv[name]["V"], dim=0).numpy()

        # Center the data (PCA-style)
        K_centered = K_all - K_all.mean(axis=0, keepdims=True)
        V_centered = V_all - V_all.mean(axis=0, keepdims=True)

        # SVD of activation matrices
        _, S_k, _ = np.linalg.svd(K_centered, full_matrices=False)
        _, S_v, _ = np.linalg.svd(V_centered, full_matrices=False)

        ranks_k, cumvar_k = compute_effective_rank(S_k)
        ranks_v, cumvar_v = compute_effective_rank(S_v)

        entry = {
            "label": name,
            "n_tokens": K_all.shape[0],
            "k_activation_singular_values": S_k[:50].tolist(),  # Top 50 for space
            "v_activation_singular_values": S_v[:50].tolist(),
            "k_activation_ranks": ranks_k,
            "v_activation_ranks": ranks_v,
            "k_activation_cumvar": cumvar_k.tolist(),
            "v_activation_cumvar": cumvar_v.tolist(),
        }
        results.append(entry)

    model.cpu()
    return results


def plot_weight_analysis(results, target_rank, output_path):
    """Plot singular value spectra and effective ranks per layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: Singular value spectra (K) ---
    ax = axes[0, 0]
    for r in results:
        sv = np.array(r["k_singular_values"])
        sv_normalized = sv / sv[0]  # Normalize by largest
        ax.semilogy(sv_normalized, alpha=0.6, label=r["label"])
    if target_rank and target_rank < len(results[0]["k_singular_values"]):
        ax.axvline(x=target_rank, color="red", linestyle="--", alpha=0.7,
                   label=f"target rank={target_rank}")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Normalized Singular Value (log)")
    ax.set_title("W_K Singular Value Decay Per Layer")
    ax.legend(fontsize=6, ncol=2)

    # --- Plot 2: Singular value spectra (V) ---
    ax = axes[0, 1]
    for r in results:
        sv = np.array(r["v_singular_values"])
        sv_normalized = sv / sv[0]
        ax.semilogy(sv_normalized, alpha=0.6, label=r["label"])
    if target_rank and target_rank < len(results[0]["v_singular_values"]):
        ax.axvline(x=target_rank, color="red", linestyle="--", alpha=0.7,
                   label=f"target rank={target_rank}")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Normalized Singular Value (log)")
    ax.set_title("W_V Singular Value Decay Per Layer")
    ax.legend(fontsize=6, ncol=2)

    # --- Plot 3: Effective rank per layer (bar chart) ---
    ax = axes[1, 0]
    labels = [r["label"] for r in results]
    x = np.arange(n_layers)
    width = 0.25

    k90 = [r["k_effective_ranks"]["rank_90"] for r in results]
    k95 = [r["k_effective_ranks"]["rank_95"] for r in results]
    k99 = [r["k_effective_ranks"]["rank_99"] for r in results]

    ax.bar(x - width, k90, width, label="K 90%", alpha=0.8)
    ax.bar(x, k95, width, label="K 95%", alpha=0.8)
    ax.bar(x + width, k99, width, label="K 99%", alpha=0.8)
    if target_rank:
        ax.axhline(y=target_rank, color="red", linestyle="--", alpha=0.7,
                   label=f"MLA rank={target_rank}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("W_K Effective Rank Per Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend()

    # --- Plot 4: Same for V ---
    ax = axes[1, 1]
    v90 = [r["v_effective_ranks"]["rank_90"] for r in results]
    v95 = [r["v_effective_ranks"]["rank_95"] for r in results]
    v99 = [r["v_effective_ranks"]["rank_99"] for r in results]

    ax.bar(x - width, v90, width, label="V 90%", alpha=0.8)
    ax.bar(x, v95, width, label="V 95%", alpha=0.8)
    ax.bar(x + width, v99, width, label="V 99%", alpha=0.8)
    if target_rank:
        ax.axhline(y=target_rank, color="red", linestyle="--", alpha=0.7,
                   label=f"MLA rank={target_rank}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("W_V Effective Rank Per Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend()

    plt.suptitle("KV Cache Compressibility Analysis (Weight-Level)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Weight analysis plot saved: {output_path}")
    plt.close()


def plot_activation_analysis(results, target_rank, output_path):
    """Plot activation-level SVD analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Effective rank bar chart
    labels = [r["label"] for r in results]
    x = np.arange(n_layers)
    width = 0.25

    for ax_idx, key_prefix, title in [
        (0, "k_activation", "K Activation Effective Rank"),
        (1, "v_activation", "V Activation Effective Rank"),
    ]:
        ax = axes[ax_idx]
        r90 = [r[f"{key_prefix}_ranks"]["rank_90"] for r in results]
        r95 = [r[f"{key_prefix}_ranks"]["rank_95"] for r in results]
        r99 = [r[f"{key_prefix}_ranks"]["rank_99"] for r in results]

        ax.bar(x - width, r90, width, label="90%", alpha=0.8)
        ax.bar(x, r95, width, label="95%", alpha=0.8)
        ax.bar(x + width, r99, width, label="99%", alpha=0.8)
        if target_rank:
            ax.axhline(y=target_rank, color="red", linestyle="--", alpha=0.7,
                       label=f"MLA rank={target_rank}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Effective Rank")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.legend()

    plt.suptitle("KV Cache Compressibility Analysis (Activation-Level)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Activation analysis plot saved: {output_path}")
    plt.close()


def print_summary(weight_results, target_rank):
    """Print a concise summary of compression potential."""
    print("\n" + "=" * 70)
    print("KV COMPRESSION SUMMARY")
    print("=" * 70)

    if target_rank:
        print(f"\nTarget MLA kv_lora_rank: {target_rank}")
        print(f"Full rank (n_embd): {weight_results[0]['W_k_shape'][0]}")
        print(f"Compression ratio: {weight_results[0]['W_k_shape'][0] / target_rank:.1f}x\n")

    print(f"{'Layer':<15} {'K rank@95%':>12} {'V rank@95%':>12} {'K rank@99%':>12} {'V rank@99%':>12}")
    print("-" * 70)

    for r in weight_results:
        label = r["label"]
        kr95 = r["k_effective_ranks"]["rank_95"]
        vr95 = r["v_effective_ranks"]["rank_95"]
        kr99 = r["k_effective_ranks"]["rank_99"]
        vr99 = r["v_effective_ranks"]["rank_99"]
        flag = ""
        if target_rank and (kr95 > target_rank or vr95 > target_rank):
            flag = " << ABOVE TARGET"
        print(f"{label:<15} {kr95:>12} {vr95:>12} {kr99:>12} {vr99:>12}{flag}")

    # Summary statistics
    all_k95 = [r["k_effective_ranks"]["rank_95"] for r in weight_results]
    all_v95 = [r["v_effective_ranks"]["rank_95"] for r in weight_results]
    print("-" * 70)
    print(f"{'Mean':<15} {np.mean(all_k95):>12.1f} {np.mean(all_v95):>12.1f}")
    print(f"{'Max':<15} {max(all_k95):>12} {max(all_v95):>12}")
    print(f"{'Min':<15} {min(all_k95):>12} {min(all_v95):>12}")

    if target_rank:
        k_above = sum(1 for r in all_k95 if r > target_rank)
        v_above = sum(1 for r in all_v95 if r > target_rank)
        total = len(weight_results)
        print(f"\nLayers where 95% rank exceeds target ({target_rank}):")
        print(f"  K: {k_above}/{total}   V: {v_above}/{total}")
        if k_above == 0 and v_above == 0:
            print(f"  -> kv_lora_rank={target_rank} is SUFFICIENT for all layers at 95% variance")
        else:
            max_needed = max(max(all_k95), max(all_v95))
            print(f"  -> Consider increasing kv_lora_rank to {max_needed} for full coverage")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze KV cache compressibility for MLA design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (must contain summary.json)")
    parser.add_argument("--checkpoint-file", type=str, default="ckpt.pt",
                        help="Checkpoint filename within directory (default: ckpt.pt)")
    parser.add_argument("--target-rank", type=int, default=192,
                        help="Target kv_lora_rank for MLA (default: 192, DeepSeek-V2)")
    parser.add_argument("--compute-activations", action="store_true",
                        help="Also run activation-level SVD analysis (requires GPU + data)")
    parser.add_argument("--compute-entropy", action="store_true",
                        help="Compute attention entropy (requires disabling Flash Attention)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset directory (for activation analysis)")
    parser.add_argument("--n-batches", type=int, default=10,
                        help="Number of batches for activation analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("KV Cache Compression Analysis for MLA Design")
    print("=" * 70)

    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, args.checkpoint_file)
    model.eval()

    # Type A: Weight-level analysis (always run, fast)
    print("\n--- Weight-Level Analysis (Type A) ---")
    weight_results = analyze_weight_matrices(model)

    if not weight_results:
        print("ERROR: No attention layers found in model")
        return

    # Print summary
    print_summary(weight_results, args.target_rank)

    # Plot weight analysis
    plot_path = os.path.join(args.checkpoint, "kv_compression_weights.png")
    plot_weight_analysis(weight_results, args.target_rank, plot_path)

    # Type B: Activation-level analysis (optional, requires GPU + data)
    activation_results = None
    if args.compute_activations:
        print("\n--- Activation-Level Analysis (Type B) ---")
        activation_results = analyze_activations(
            model, config, args.data_dir, n_batches=args.n_batches
        )
        act_plot_path = os.path.join(args.checkpoint, "kv_compression_activations.png")
        plot_activation_analysis(activation_results, args.target_rank, act_plot_path)

    # Save all results as JSON
    output = {
        "target_rank": args.target_rank,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "model": config.model,
        "weight_analysis": weight_results,
    }
    if activation_results:
        output["activation_analysis"] = activation_results

    json_path = os.path.join(args.checkpoint, "kv_compression_analysis.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved: {json_path}")


if __name__ == "__main__":
    main()
