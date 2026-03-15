# Extension Notes

> Design rationale, method selection, and architectural decisions for the
> MLA-LN-CoTFormer extension. This document explains **what** we are extending,
> **why** we chose this direction, and **how** the design maps to the original
> components.
>
> For concrete numerical divergences from the paper, see `reprod-notes.md`.
> For the implementation plan and code specs, see `plan.md`.

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Why Multi-Head Latent Attention](#2-why-multi-head-latent-attention)
- [3. Why Not Other Approaches](#3-why-not-other-approaches)
- [4. Where MLA Meets CoTFormer](#4-where-mla-meets-cotformer)
- [5. Design Decisions](#5-design-decisions)
- [6. Expected Outcomes](#6-expected-outcomes)
- [7. Experimental Plan](#7-experimental-plan)
- [References](#references)

---

## 1. Problem Statement

### The CoTFormer KV cache bottleneck

CoTFormer's core mechanism — repeating the mid-block stack R times — gives each
token multiple "thinking rounds" through the same transformer layers. During
each repeat, the attention layers accumulate keys and values from all previous
repeats into a cross-repeat KV cache, so that later thinking rounds can attend
over the representations produced by earlier ones.

This is architecturally powerful but creates a memory problem: the KV cache
grows linearly with the number of repeats.

For an LN-CoTFormer with `n_head=12`, `d_head=64`, `n_mid=21` mid-layers,
`R=5` repeats, and sequence length `T`:

```
Standard Transformer:  KV per layer = 2 × n_head × d_head × T
                       = 2 × 12 × 64 × T = 1,536T values

LN-CoTFormer:          KV per mid layer = 2 × n_head × d_head × (R × T)
                       = 2 × 12 × 64 × 5T = 7,680T values
```

Each mid-layer's attention window is 5x larger than a standard transformer's.
Across 21 mid-layers, the total cache is `21 × 7,680T = 161,280T` values,
compared to `24 × 1,536T = 36,864T` for a 24-layer standard transformer.
That is a **4.4x increase in KV cache memory** for the same model width and
sequence length.

### Why this matters

1. **Training**: The accumulated KV cache is the primary reason batch_size=16
   OOMs on L4 24 GB GPUs during backward (19.6 GiB + 786 MiB needed). We had
   to halve the micro-batch to 8 (see `reprod-notes.md` §3).

2. **Inference**: At longer sequence lengths (1024+), the cross-repeat cache
   becomes the dominant memory consumer. This limits CoTFormer's practical
   deployment at the very sequence lengths where iterative reasoning is most
   valuable.

3. **Scaling**: Increasing repeat depth R (the "thinking budget") linearly
   increases cache size. The original paper acknowledges this trade-off but
   does not address it.

---

## 2. Why Multi-Head Latent Attention

### What MLA does

Multi-Head Latent Attention (MLA), introduced in DeepSeek-V2 [1], compresses
the KV representation through a low-rank bottleneck before caching. Instead of
caching the full K and V projections (`2 × n_head × d_head` per token), MLA
caches a compact latent vector (`d_c` per token) plus a small RoPE component
(`d_R` per token). The full K and V are reconstructed from the latent on each
attention call via a learned up-projection.

For our model dimensions (d=768, n_head=12, d_head=64):

| | Per-token cache | With R=5, T=256 |
|---|---|---|
| Standard MHA | 1,536 values | 1,966,080 per mid-layer |
| MLA (d_c=192, d_R=32) | 224 values | 286,720 per mid-layer |
| **Compression** | **6.9x** | **6.9x** |

### Why this is a natural fit for CoTFormer

The compression ratio is multiplicative with the repeat count. A standard
transformer gets modest benefit from MLA (caching T tokens either way). But
CoTFormer caches `R × T` tokens in its mid-layers, so the absolute savings
scale with R:

| Repeats | MHA cache (mid) | MLA cache (mid) | Savings |
|---------|-----------------|-----------------|---------|
| R=2 | 2 × 1,536T | 2 × 224T | 65.5 KB/token |
| R=5 | 5 × 1,536T | 5 × 224T | 163.8 KB/token |
| R=10 | 10 × 1,536T | 10 × 224T | 327.6 KB/token |

The higher the thinking budget, the more MLA saves. This makes MLA
specifically valuable for CoTFormer in a way that it would not be for a
standard non-recurrent transformer.

### DeepSeek-V2's validation at scale

MLA was validated on a 236B-parameter MoE model (DeepSeek-V2) trained on 8.1T
tokens, achieving performance comparable to standard MHA despite the 6.9x
cache compression. While our model is much smaller (~200M params), the
projection mathematics are scale-independent: the low-rank structure of KV
representations is a property of the attention mechanism itself, not of model
size.

---

## 3. Why Not Other Approaches

We considered several alternatives before settling on MLA.

### 3a. Grouped-Query Attention (GQA)

GQA (Ainslie et al., 2023) [2] reduces KV cache by sharing K/V heads across
query head groups. For example, GQA with 4 KV groups on 12 query heads gives
3x compression.

**Why we rejected it**: GQA is a coarser compression — it forces groups of
query heads to share identical K/V representations. MLA preserves per-head
expressiveness through the learned up-projection `W_kvb`, which produces
distinct K/V for each head from the shared latent. For cross-repeat attention,
where different repeats may need to attend to earlier repeats in head-specific
ways, per-head expressiveness matters more than in standard single-pass
attention.

Additionally, GQA's max compression is bounded by `n_head / n_kv_groups`. With
12 heads, the maximum (single KV group = Multi-Query Attention) gives 12x
compression but significantly degrades quality at small model scales. MLA
achieves 6.9x compression while preserving full per-head diversity.

### 3b. Quantized KV Cache

Post-training KV cache quantization (INT8, INT4) can halve or quarter cache
memory at inference time.

**Why we rejected it**: (1) Quantization is an inference-time optimization
that does not help training memory — and training is our current bottleneck.
(2) It does not compose well with cross-repeat attention: quantization error
compounds across R rounds of accumulation. (3) It is orthogonal to MLA — you
could quantize the MLA latent too, stacking the savings.

### 3c. Sparse Attention / Sliding Window

Limiting the cross-repeat attention window (e.g., attending only to the
previous 2 repeats instead of all R) would reduce cache linearly.

**Why we rejected it**: The whole point of CoTFormer's cross-repeat cache is
that later thinking rounds can build on the full context of earlier rounds.
Restricting the window undermines the architectural thesis. MLA preserves the
full attention pattern while compressing the representation.

### 3d. Reducing Repeat Count

Simply using R=3 instead of R=5 reduces cache by 40%.

**Why we rejected it**: This changes the model's computational budget, not its
efficiency. The goal is to enable *more* thinking at *less* memory cost, not to
think less. MLA decouples cache size from thinking depth.

### 3e. Gradient Checkpointing

Recomputing activations during backward would reduce training memory.

**Why we rejected it for the extension (but considered for training)**: The
`InPlaceSetSlice` KV cache accumulation in the current codebase is incompatible
with `torch.utils.checkpoint` — recomputation during backward would
double-write the cache, producing incorrect gradients. A compatible
implementation would require refactoring the cache as explicit function I/O
rather than in-place mutation. This is a valid engineering task but does not
address the fundamental cache growth problem at inference time.

---

## 4. Where MLA Meets CoTFormer

### Architecture mapping

The LN-CoTFormer has three structural regions:

```
h_begin (2 prefix blocks)  →  standard CausalSelfAttention
h_mid   (21 blocks × 5 repeats)  →  CausalSelfAttention with cross-repeat KV cache
h_end   (1 suffix block)  →  standard CausalSelfAttention
```

MLA replaces the attention **only in `h_mid` blocks**, where the cross-repeat
cache creates the bottleneck. The prefix and suffix blocks use standard MHA
because:

1. They run only once (no repeat-induced cache growth).
2. They process the initial/final token representations, which benefit from
   full-rank attention for embedding quality.
3. Keeping them standard simplifies the implementation and reduces the number
   of changed variables in the experiment.

### What changes in the forward pass

**Before (standard cross-repeat attention)**:
```
For each mid-layer:
  1. Project x → Q, K, V  (3 × n_head × d_head = 2304 dims)
  2. Append K, V to cross-repeat buffer  (InPlaceSetSlice)
  3. Attend Q over full buffer  (Flash Attention)
```

**After (MLA cross-repeat attention)**:
```
For each mid-layer:
  1. Compress x → kv_latent, k_rope  (d_c + d_R = 224 dims)
  2. Append compressed kv_latent, k_rope to cross-repeat buffer
  3. Decompress buffer → K, V  (via W_kvb up-projection)
  4. Project x → Q  (via down-norm-up path)
  5. Attend Q over decompressed K, V  (Flash Attention)
```

The buffer stores 224 values per token instead of 1,536. The decompression
(`W_kvb` matmul) adds compute cost at each attention call, but the memory
savings enable larger batches or longer sequences that more than compensate.

### What does NOT change

- The repeat loop structure
- Depth embeddings (`LinearLearnedDepthPositionalEncoder`)
- Inter-repeat LayerNorm (`ln_mid`)
- The MLP blocks within each transformer block
- Prefix/suffix blocks
- The training loop, optimizer, scheduler, and data pipeline
- The ADM router mechanism (if composing MLA + ADM later)

---

## 5. Design Decisions

### DEC-EXT-001: Uniform compression rank across layers

**Decision**: Use a single `kv_lora_rank=192` for all 21 mid-layers.

**Rationale**: DeepSeek-V2 uses uniform rank across all layers in a 60-layer
model and finds no benefit from per-layer tuning. Our model has fewer layers
(21 mid), reducing the variance in per-layer compressibility. The
`analyze_kv_compression.py` tool will validate this post-training: if any
layers show effective rank > 192, we can increase the rank for those layers in
a follow-up experiment.

**Alternative considered**: Per-layer adaptive rank based on SVD analysis.
Rejected for Phase 4 due to complexity; deferred to future work.

### DEC-EXT-002: Separate RoPE branch for keys

**Decision**: The compressed latent `c_KV` contains content information only.
Positional (RoPE) information is carried in a separate `k_rope` branch that
bypasses the compression bottleneck.

**Rationale**: This is DeepSeek-V2's design, motivated by the observation that
position-dependent and position-independent attention patterns require
different representational capacity. Compressing RoPE embeddings through the
same bottleneck as content would force the low-rank space to represent both
absolute position and semantic content simultaneously, likely degrading both.

### DEC-EXT-003: RMSNorm on compressed representations

**Decision**: Apply RMSNorm after the down-projection for both the query and
KV paths (`q_norm`, `kv_norm`).

**Rationale**: The down-projection reduces dimensionality from 768 to 192-384.
Without normalisation, the compressed activations can have very different
magnitude distributions than the original hidden states, destabilising the
up-projection. DeepSeek-V2 uses RMSNorm here (not LayerNorm) because RMSNorm
has no mean-centering, preserving directional information in the compressed
space.

### DEC-EXT-004: Standard MHA in prefix/suffix blocks

**Decision**: Only mid-blocks use MLA. The 2 prefix and 1 suffix blocks retain
standard `CausalSelfAttention`.

**Rationale**: See §4 above. Prefix/suffix blocks have no repeat-induced cache
growth. Using standard MHA provides stronger baselines for the first/last
layers that set up and consolidate the repeated computation. This also means
MLA failures (if any) are isolated to the mid-blocks and do not contaminate
the embedding or prediction layers.

### DEC-EXT-005: Train from scratch (not fine-tune)

**Decision**: Train MLA-LN-CoTFormer from random initialisation for 40k steps,
matching the LN-CoTFormer training protocol.

**Rationale**: Fine-tuning from a trained LN-CoTFormer checkpoint would require
the MLA projections to learn compressed representations that are compatible
with pre-existing attention patterns. This is a harder optimisation problem than
learning both simultaneously from scratch. Additionally, training from scratch
provides a fair comparison — both models see the same data for the same number
of steps.

### DEC-EXT-006: Cache compressed latents, decompress on access

**Decision**: The cross-repeat buffer stores the compressed `c_KV` (192 dims)
and `k_rope` (32 dims) rather than the decompressed K, V (1,536 dims).

**Rationale**: The whole point of MLA is cache compression. Storing
decompressed K/V would negate the memory benefit. The trade-off is additional
compute: each attention call must run the `W_kvb` up-projection on the full
buffer (R × T tokens). For R=5, T=256, this is a matmul of shape
(batch, heads, 1280, d_c) — approximately 0.3 GFLOPs per layer per micro-step.
This is small compared to the attention computation itself.

---

## 6. Expected Outcomes

### Primary hypothesis

MLA-LN-CoTFormer achieves perplexity within +0.5 PPL of standard
LN-CoTFormer while using 6.9x less KV cache memory in the mid-blocks.

### What we expect to observe

| Metric | LN-CoTFormer | MLA-LN-CoTFormer | Direction |
|--------|-------------|------------------|-----------|
| Final PPL (40k steps) | ~24.1 (paper Table 2) | ~24.1–24.6 | Slight degradation acceptable |
| Peak VRAM (batch=8, 2 GPU) | ~15 GB | ~10–12 GB | Lower |
| Max batch_size on L4 | 8 | 12–16 | Higher |
| Training throughput (steps/sec) | baseline | ~0.85–0.95x | Slight slowdown from decompression |
| KV cache memory (R=5, T=256) | 161,280T vals | 23,520T vals | 6.9x smaller |

### What would invalidate the approach

- PPL degradation > 1.0 compared to standard LN-CoTFormer at matched training
  steps. This would indicate that `kv_lora_rank=192` is insufficient for the
  cross-repeat attention pattern.
- Training instability (loss spikes, NaN gradients) in the first 1000 steps.
  This would suggest the RMSNorm + low-rank initialisation needs adjustment.

### Fallback plan

If `kv_lora_rank=192` proves insufficient:
1. Run `analyze_kv_compression.py` on the trained LN-CoTFormer checkpoint to
   determine per-layer effective rank.
2. Increase to `kv_lora_rank=256` or `kv_lora_rank=384` (still 4.6x or 2.3x
   compression).
3. If no rank works, try hybrid: MLA for early repeats (where representations
   are less differentiated), standard MHA for later repeats.

---

## 7. Experimental Plan

### Phase 4a: Implementation (no GPU needed)

1. Create `models/mla_cotformer.py` — copy `cotformer_full_depth_lnmid_depthemb.py`,
   replace `CausalSelfAttention` with `MLACausalSelfAttention` in mid-blocks only.
2. Register in `models/__init__.py`.
3. Add MLA config args to `config/base.py` (already spec'd in `plan.md` §0.5).
4. Verify model instantiation and forward pass locally (CPU, batch=1).
5. Compare parameter counts: MLA model should have ~24% fewer attention params.

### Phase 4b: Training (GPU needed)

1. Create `iridis/mla-train/job.sh` — same template as `lncot-train/job.sh`
   with `--model mla_cotformer` and MLA dimension args.
2. Train for 40k steps, same hyperparameters as LN-CoTFormer (lr=1e-3,
   warmup=0.2, cosine schedule, seed=0).
3. Monitor: loss curve, gradient norms, VRAM usage in first 100 steps.
4. If stable, let it run to completion.

### Phase 4c: Validation

1. Run `analyze_kv_compression.py` on both the LN-CoTFormer and MLA checkpoints.
2. Compare per-layer effective rank to our chosen `kv_lora_rank=192`.
3. Evaluate final perplexity on full validation set.
4. Profile peak VRAM at batch sizes 8, 12, 16, 32 (single GPU) to quantify
   the practical memory benefit.
5. If composing with ADM: train an adaptive MLA-LN-CoTFormer for Figure 4
   comparison.

---

## References

[1] DeepSeek-AI. (2024). *DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model.* arXiv:2405.04434.

[2] Ainslie, J. et al. (2023). *GQA: Training Generalized Multi-Query
    Transformer Models from Multi-Head Checkpoints.* arXiv:2305.13245.

[3] Mohtashami, A. et al. (2025). *CoTFormer: A Chain-of-Thought Driven
    Architecture with Budget-Adaptive Computation Cost at Inference.*
    ICLR 2025.

[4] Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All
    You Need.* arXiv:1911.02150. (Original Multi-Query Attention.)

[5] Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization.*
    NeurIPS 2019.
