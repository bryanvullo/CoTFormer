# Extension Notes

> Scientific investigation and architectural extension of the CoTFormer
> (Mohtashami et al., ICLR 2025). This document records the team's
> methodology, research questions, experimental designs, and extension
> proposals across three milestones.
>
> For reproduction divergences, see `reprod-notes.md`.
> For training commands, see `experiments.md`.

## Table of Contents

- [0. Methodology](#0-methodology)
- [1. Milestone 2: Mechanistic Analysis](#1-milestone-2-mechanistic-analysis)
  - [1.1 Ontological Framework](#11-ontological-framework)
  - [1.2 Research Questions](#12-research-questions)
  - [1.3 Experimental Protocols](#13-experimental-protocols)
  - [1.4 Theoretical Predictions](#14-theoretical-predictions)
  - [1.5 Analysis Matrix](#15-analysis-matrix)
- [2. Milestone 3: Architectural Extensions](#2-milestone-3-architectural-extensions)
  - [2.1 Multi-Head Latent Attention (MLA)](#21-multi-head-latent-attention-mla)
  - [2.2 Manifold-Constrained HyperConnections (mcHC)](#22-manifold-constrained-hyperconnections-mchc)
  - [2.3 Causal Cross-Repeat Mask for ADM](#23-causal-cross-repeat-mask-for-adm)
  - [2.4 Residual Attention](#24-residual-attention)
- [Appendix A: MLA Dimensionality Reference](#appendix-a-mla-dimensionality-reference)
- [References](#references)

---

## 0. Methodology

### Three Milestones

| # | Milestone | Goal | Status |
|---|-----------|------|--------|
| 1 | Reproduce | Verify paper claims (Tables 1-2, Figures 2-5, Section 5) | Near-complete |
| 2 | Unpick and Analyze | Understand what iterative computation computes | Active |
| 3 | Extend | Improve architecture informed by Milestone 2 findings | Blocked on M2 |

The milestones are sequential: each informs the next. Milestone 2 is
self-contained -- understanding is the goal, not a stepping stone to
engineering. The fact that we use M2 insights to design M3 extensions is a
consequence of the scientific method, not the purpose of M2.

### Scientific Ethos

The authors write (Appendix F): "a thorough discussion around understanding
how these models operate is outside the scope of the current work, we hope
these results encourage such investigations." No existing work applies
mechanistic interpretability to iterative/recurrent transformers [7]. This is
the contribution gap.

Every experiment in M2 must satisfy:
1. **Specificity**: name the model, checkpoint, layer range, data split, and
   evaluation mode.
2. **Variable isolation**: one independent variable per experiment.
3. **Falsifiability**: state what would disprove the hypothesis.
4. **Ontological purpose**: what truth do we learn, independent of engineering
   utility?
5. **Confounder control**: explicitly list and mitigate confounders.

### Checkpoint Matrix

| ID | Architecture | Config | Steps | PPL | Purpose |
|----|-------------|--------|-------|-----|---------|
| C1 | CoT+Reserved | cotformer_full_depth, 2+21*5+1 | 40k | TBD | Baseline (no ln_mid, no depth_emb) |
| C2 | LN-CoT | cotformer_full_depth_lnmid_depthemb | 40k | 24.13 | Isolate ln_mid + depth_emb effect |
| C3 | LN-CoT | cotformer_full_depth_lnmid_depthemb | 60k | 23.59 | Effect of additional training |
| C4 | ADM v2 | adaptive_cotformer_..._single_final | 40k | TBD | Isolate router effect |
| C5 | ADM v2 | adaptive_cotformer_..._single_final | 60k | 24.06 | Router maturation |
| C4' | ADM v1 | Same architecture, pre-B4 RNG fix | 40k | -- | RNG divergence control |
| C5' | ADM v1 | Same architecture, pre-B4 RNG fix | 60k | -- | RNG divergence control |

Comparison C2-vs-C1 isolates ln_mid. C4-vs-C2 isolates the router.
C5-vs-C4 isolates training duration. C5'-vs-C5 isolates the B4 RNG fix
(which produces fundamentally different routing patterns -- see
`reprod-notes.md` B12).

---

## 1. Milestone 2: Mechanistic Analysis

### 1.1 Ontological Framework

The central question is: **"What does iterative computation in CoTFormer
actually compute?"**

This decomposes into five specific sub-questions, each targeting a distinct
aspect of the model's internal dynamics:

| ID | Sub-Question | What Truth We Seek |
|----|-------------|-------------------|
| SQ1 | Does the model refine predictions monotonically across repeats, or does it restructure representations non-monotonically? | Whether repeats implement iterative refinement (converging toward a fixed point) or multi-phase processing (qualitative restructuring). This determines whether the "chain-of-thought" analogy is accurate or misleading. |
| SQ2 | Do attention heads specialize by repeat, and does this specialization vary with layer depth within the mid-block? | Whether the weight-tied transformer blocks develop emergent functional specialization despite identical parameters. If yes, the cross-repeat KV cache carries structurally different information at different repeats. |
| SQ3 | Does the ADM router learn a meaningful proxy for token difficulty, or does it approximate random pruning? | Whether the budget-adaptive claim is scientifically supported. The B4 RNG divergence (v1 vs v2 routing patterns) makes this urgent -- if the router converges to different patterns under minor training perturbations, it may not learn a robust difficulty signal. |
| SQ4 | When only a fraction of tokens survive to repeat 5, do the surviving tokens lose contextual information because their sequence neighbours halted earlier? | Whether the cross-repeat KV cache (containing mixed-depth frozen representations from halted tokens) provides meaningful context, or whether surviving tokens become informationally isolated. No existing literature addresses this. |
| SQ5 | Does the absence of a proper causal mask in the ADM's cross-repeat attention degrade routing quality? | Whether the ADM's routing decisions are less informed than they could be. The current mask (`k_indices <= indices`) enforces temporal ordering but does not account for repeat-depth relationships. |

### 1.2 Research Questions

Each sub-question generates specific, falsifiable research questions with
full variable isolation.

#### RQ1: Prediction Convergence Across Repeats (SQ1)

**Statement**: During evaluation on OWT2 validation data, does the
LN-CoTFormer (C3, 60k) converge monotonically toward the correct next-token
prediction across repeats 1 to 5, as measured by top-1 accuracy of the
logit-lens projection at each repeat's `ln_mid` output?

- **IV**: Repeat index r in {1, 2, 3, 4, 5}
- **DV**: Top-1 accuracy of `lm_head(ln_mid(h_r))` vs ground-truth next token;
  KL divergence `D_KL(P_r || P_{r-1})`; entropy `H(P_r)`
- **Controls**: Same checkpoint (C3), same data (OWT2 val), same forward pass
- **Confounders**: `ln_mid` normalisation distorts the logit-lens projection
  differently at each repeat (norm varies). Mitigation: also measure WITHOUT
  `ln_mid` (project raw `h_r` through `lm_head`). Compare both.
- **H0**: Top-1 accuracy is flat across repeats. Repeats add compute, not signal.
- **Falsifies H0**: Monotonic increase (paired t-test, p < 0.01).
- **What truth**: the RATE of convergence reveals whether most useful work
  happens in repeats 1-2 (diminishing returns) or is distributed (each repeat
  adds substantial value).
- **Cross-checkpoint**: repeat on C1 (no ln_mid) and C5 (ADM). C1-vs-C3
  reveals whether ln_mid changes the convergence trajectory. C5-vs-C3 reveals
  whether the router alters which tokens converge early.

#### RQ2: Representation Structural Similarity (SQ1)

**Statement**: Across the 105 (layer, repeat) activation sites in the
LN-CoTFormer mid-block, which pairs produce structurally similar
representations, as measured by CKA?

- **IV**: Pair of (layer_i, repeat_j) activation sites
- **DV**: CKA (linear + RBF kernels) over 2048 validation tokens
- **Controls**: Same checkpoint (C3), same data batch
- **Confounders**: CKA with linear kernel saturates at high similarity for
  wide representations. Mitigation: RBF kernel check; finite-sample
  correction [9].
- **H0**: CKA > 0.9 across all cross-repeat pairs within each layer (redundant
  representations through same weights).
- **Falsifies H0**: CKA < 0.7 for any adjacent-repeat pair.
- **What truth**: if H0 holds, cross-repeat KV is redundant and highly
  compressible. If rejected, each repeat's contribution is unique. The
  layer-resolved CKA map shows WHERE the model does its most distinctive work.

#### RQ3: Attention Head Specialization (SQ2)

**Statement**: In the LN-CoTFormer (C3), do attention heads partition into
functionally distinct classes based on cross-repeat attention patterns, and
does this differ between early (layers 3-7) and late (layers 17-21) mid-layers?

- **IV**: (Layer index, head index) -- 252 heads
- **DV**: Average attention distribution of the final token at repeat 5 over
  all (token, repeat) pairs. K-means clustering (k=4-6).
- **Controls**: Same checkpoint, averaged over full validation set
- **Confounders**: Flash Attention does not return weights. Use non-flash pass.
  Verify PPL matches to 4 decimal places.
- **H0**: Head cluster assignments are uniformly distributed across layers.
- **What truth**: if early layers favour "broad" heads (all repeats) while late
  layers favour "current-repeat specialists," it reveals a division of labour.

#### RQ4: Router Validity (SQ3)

**Statement**: In the ADM (C5), do tokens halted at early repeats have
systematically lower next-token CE loss than tokens surviving to repeat 5?

- **IV**: Token halting depth (repeat at which cascaded score drops below 0.5)
- **DV**: Per-token cross-entropy loss at final output
- **Controls**: Same checkpoint (C5), same data (OWT2 val)
- **Confounders**: Position-in-sequence correlates with both loss and halting.
  Mitigation: partial correlation controlling for position.
- **H0**: No correlation (Pearson |r| < 0.1 after position control).
- **What truth**: if H0 rejected, router learns difficulty proxy. If H0 holds,
  router decisions are not meaningfully related to difficulty. C5'-vs-C5
  tests robustness to training perturbations.

#### RQ5: Context Preservation Under Adaptive Depth (SQ4)

**Statement**: In the ADM (C5), do tokens surviving to repeat 5 produce
higher-quality predictions when their context tokens also survived deep,
compared to when most context halted at repeat 1-2?

- **IV**: Fraction of context tokens surviving to same depth
- **DV**: Per-token CE loss for the target token
- **Controls**: Conditioned on target surviving to repeat 5
- **Confounders**: Sequence difficulty. Mitigation: z-score loss per sequence.
- **H0**: Target loss independent of context survival fraction.
- **What truth**: if H0 rejected, surviving tokens suffer "contextual
  isolation" -- a fundamental limitation of adaptive depth.

#### RQ6: Effective Dimensionality (SQ1)

**Statement**: Does intrinsic dimensionality decrease across repeats 1 to 5
in the LN-CoTFormer (C3)?

- **IV**: Repeat index r in {1, 2, 3, 4, 5}
- **DV**: Participation ratio with finite-sample correction [9] at each
  (layer, repeat) site
- **Controls**: Same checkpoint, same 2048 tokens
- **H0**: d_eff constant (+/- 5%) across repeats within each layer.
- **What truth**: if d_eff decreases, later repeats operate in
  lower-dimensional subspaces, directly determining MLA rank allocation.

### 1.3 Experimental Protocols

| Protocol | Serves | Implementation | Compute |
|----------|--------|---------------|---------|
| A: Logit Lens | RQ1 | `analyze_model.py --logit-lens` | ~10 min/ckpt (GPU) |
| B: CKA Matrix | RQ2 | `analyze_model.py --cka` | ~20 min total (CPU) |
| C: Attention Taxonomy | RQ3 | `analyze_model.py --attention-taxonomy` | ~30 min/ckpt (GPU, non-flash) |
| D: Router-Loss Correlation | RQ4, RQ5 | `analyze_model.py --router-analysis` | ~15 min/ADM ckpt (GPU) |
| E: Residual Diagnostics | SQ1 | `analyze_model.py --residual-diagnostics` | Piggybacks on A |
| F: Effective Dimensionality | RQ6 | `analyze_model.py --effective-dim` | ~5 min (CPU) |
| G: KV Compressibility | M3 design | `analyze_model.py --kv-compressibility` | ~15 min/ckpt (GPU) |

All protocols share a unified analysis framework (`analyze_model.py`) with
modular backends. Protocols A-F share the same forward pass to minimize GPU
time.

### 1.4 Theoretical Predictions

**Prediction 1** (from [11]): Weight decay (wd=0.1) on weight-tied QK/OV
matrices drives them to low-rank solutions. The 5x repetition compounds this.
*Test*: Protocol G; measure 95% effective rank of K/V activations. Confirmed
if rank < 40 (< 63% of nominal d_head=64).

**Prediction 2** (from [12]): Repetition accelerates sparse attention
emergence. *Test*: Protocol C; measure per-head attention entropy across
repeats. Confirmed if entropy decreases monotonically (Spearman rho < -0.8).

### 1.5 Analysis Matrix

| Protocol | C1 | C2 | C3 | C4 | C5 | C4' | C5' | GPU? |
|----------|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:----:|
| A: Logit Lens | X | X | X | X | X | -- | -- | Yes |
| B: CKA | X | X | X | X | X | -- | -- | No |
| C: Attn Taxonomy | X | X | -- | X | -- | -- | -- | Yes |
| D: Router-Loss | -- | -- | -- | X | X | X | X | Yes |
| E: Residual | X | X | -- | X | -- | -- | -- | No |
| F: Eff. Dim | X | X | X | X | X | -- | -- | No |
| G: KV Compress. | -- | -- | X | -- | X | -- | -- | Yes |

Total GPU: ~3h on L4.

---

## 2. Milestone 3: Architectural Extensions

Each extension is motivated by specific M2 findings. Implementation details
are deliberately brief -- they will be expanded once M2 results inform design.

### 2.1 Multi-Head Latent Attention (MLA)

#### Motivation

CoTFormer's cross-repeat KV cache grows as `R*T` per mid-layer (4.4x overhead
at R=5). MLA [1] compresses KV through a low-rank latent bottleneck: 224
values/token instead of 1536 (6.9x compression). Savings scale with R.

Alternatives (GQA [2], KV quantization, sparse attention, fewer repeats) are
either less expressive, orthogonal, or change the compute budget rather than
efficiency. MLA preserves full per-head diversity and the complete attention
pattern while compressing the cache.

#### Design Decisions (provisional, pending M2)

| ID | Decision | Contingent On |
|----|----------|---------------|
| DEC-EXT-001 | Rank allocation: uniform kv_lora_rank=192 | RQ6. If d_eff varies >2x, adopt CARE [14]. |
| DEC-EXT-002 | Separate RoPE branch (d_R=32) | MHA2MLA [15] analysis |
| DEC-EXT-003 | RMSNorm on compressed latents | Standard [5] |
| DEC-EXT-004 | Mid-blocks only (prefix/suffix keep MHA) | No cache growth there |
| DEC-EXT-005 | Train from scratch, 40k steps, matched hyperparams | Fair comparison |
| DEC-EXT-006 | Cache compressed latents, decompress on access | Core MLA [1] |

Target: PPL < +0.5 vs LN-CoT, 6.9x mid-block KV compression, peak VRAM < 12 GB.

See Appendix A for projection architecture and dimensionality reference.

### 2.2 Manifold-Constrained HyperConnections (mcHC)

mcHC [6] generalises residual connections via multi-stream routing (`alpha`
streams, learned `C in R^{alpha x alpha}` per layer, Stiefel-constrained).
For CoTFormer: different subspaces routed through different repeats.

**Decision gate** (from M2): strong RQ4 correlation = mcHC as complement;
weak = mcHC as router replacement. CKA collapse (RQ2) = strong motivation.

Design: alpha=2, mid-layers only. Memory: 2x hidden state. Params: 8/layer.

### 2.3 Causal Cross-Repeat Mask for ADM

The ADM uses `is_causal=False` with an index-based mask (`k_indices <= indices`)
that enforces temporal ordering but ignores repeat-depth structure. A token at
repeat 3 attends uniformly to ALL previous tokens at ALL repeats.

**Research question** (SQ5): does adding repeat-depth-aware causal constraints
improve the perplexity-compute trade-off? Ablation: same ADM architecture,
modified mask, identical hyperparameters.

Design pending M2 Protocol C (attention taxonomy) and literature findings.

### 2.4 Residual Attention

Instead of compressing the full accumulated KV cache (grows linearly with R),
attend directly to per-position residual deltas across repeats. The "residual
cache" has size `R * d_model` per position (depth dimension), not `R * T *
d_model` (sequence times depth), avoiding linear growth entirely.

**Research question**: does attending to residual deltas preserve cross-repeat
information quality while eliminating the KV growth problem?

Design pending targeted literature review on differential/residual attention.

---

## Appendix A: MLA Dimensionality Reference

### Projection Architecture

```python
class MLACausalSelfAttention(nn.Module):
    def __init__(self, config):
        d = config.n_embd          # 768
        n_h = config.n_head        # 12
        d_c = config.kv_lora_rank  # 192
        d_qc = config.q_lora_rank  # 384
        d_R = config.qk_rope_head_dim  # 32
        d_nope = config.qk_nope_head_dim  # 64
        d_v = config.v_head_dim    # 64

        # Query: down -> norm -> up + RoPE branch
        self.wq_a = nn.Linear(d, d_qc, bias=False)
        self.q_norm = RMSNorm(d_qc)
        self.wq_b = nn.Linear(d_qc, n_h * d_nope, bias=False)
        self.wq_rope = nn.Linear(d_qc, n_h * d_R, bias=False)

        # KV: joint down -> norm -> up
        self.wkv_a = nn.Linear(d, d_c + d_R, bias=False)
        self.kv_norm = RMSNorm(d_c)
        self.wkv_b = nn.Linear(d_c, n_h * (d_nope + d_v), bias=False)

        # Output
        self.c_proj = nn.Linear(n_h * d_v, d, bias=False)
```

### RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

### Dimensionality Map

```
Standard MHA:  Cache/token = 2 * 12 * 64 = 1536 floats
MLA:           Cache/token = 192 + 32 = 224 floats
Compression:   6.9x
```

---

## References

[1] DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model.* arXiv:2405.04434, 2024.

[2] J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints.* arXiv:2305.13245, 2023.

[3] A. Mohtashami et al. *CoTFormer: A Chain-of-Thought Driven Architecture
    with Budget-Adaptive Computation Cost at Inference.* In ICLR, 2025.

[4] N. Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.*
    arXiv:1911.02150, 2019.

[5] B. Zhang and R. Sennrich. *Root Mean Square Layer Normalization.* In
    NeurIPS, 2019.

[6] DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024.

[7] Z. Zheng et al. *Attention Heads of Large Language Models: A Survey.* In
    Patterns (Cell Press), 2025. arXiv:2409.03752.

[8] J. Chen et al. *KV-CoRE: Benchmarking Data-Dependent Low-Rank
    Compressibility of KV-Caches in LLMs.* arXiv:2602.05929, 2026.

[9] C. Chun et al. *Estimating Dimensionality of Neural Representations from
    Finite Samples.* arXiv:2509.26560, 2025.

[10] T. Nait Saada et al. *Mind the Gap: A Spectral Analysis of Rank Collapse
     and Signal Propagation in Attention Layers.* arXiv:2410.07799, 2025.

[11] S. Kobayashi et al. *Weight Decay Induces Low-Rank Attention Layers.*
     arXiv:2410.23819, 2024.

[12] N. Zucchet et al. *The Emergence of Sparse Attention: Impact of Data
     Distribution and Benefits of Repetition.* arXiv:2505.17863, 2025.

[13] Y. Xu et al. *Tracking the Feature Dynamics in LLM Training: A
     Mechanistic Study.* arXiv:2412.17626, 2024.

[14] Z. Zhou et al. *CARE: Covariance-Aware and Rank-Enhanced Decomposition
     for Enabling Multi-Head Latent Attention.* arXiv:2603.17946, 2026.

[15] T. Ji et al. *MHA2MLA: Towards Economical Inference -- Enabling
     DeepSeek's Multi-Head Latent Attention in Any Transformer.* In ACL, 2025.
     arXiv:2502.14837.

[16] J. Geiping et al. *Scaling up Test-Time Compute with Latent Reasoning: A
     Recurrent Depth Approach.* arXiv:2502.05171, 2025.

[17] W. Merrill and A. Sabharwal. *The Expressive Power of Transformers with
     Chain of Thought.* In ICLR, 2024. arXiv:2310.07923.

[18] S. Hao et al. *Training Large Language Models to Reason in a Continuous
     Latent Space (Coconut).* arXiv:2412.06769, 2024.

[19] B. Wang et al. *Grokked Transformers are Implicit Reasoners.* In
     NeurIPS, 2024. arXiv:2405.15071.

[20] H. Yao et al. *Thin Keys, Full Values: Reducing KV Cache via
     Low-Dimensional Attention Selection.* arXiv:2603.04427, 2026.

[21] J. Sok et al. *Garbage Attention in LLMs: BOS Sink Heads and Sink-aware
     Pruning.* arXiv:2601.06787, 2026.

[22] D. Lesens et al. *KQ-SVD: Compressing the KV Cache with Provable
     Guarantees on Attention Fidelity.* arXiv:2512.05916, 2025.

[23] X. Chen et al. *Reasoning Beyond Language: A Comprehensive Survey on
     Latent Chain-of-Thought Reasoning.* arXiv:2505.16782, 2025.
