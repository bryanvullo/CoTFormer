# COMP6258 RCoTFormer -- Surgical Implementation Plan

> **Goal**: Reproduce CoTFormer paper claims (Tables 1-2, Figures 2-4) and extend with MLA-LN-CoTFormer + mcHC.
> **Hardware**: NVIDIA L4 (24 GB VRAM), Iridis X cluster. 2x L4 DDP for long runs.
> **Training regime**: 24-layer models, 40k steps (non-adaptive, Table 2) / 60k steps (ADM + LN-CoT Section 5), batch 128 (effective), seq_len 256, OpenWebText2.
> **Repo**: `/home/totob/projects/CoTFormer/`
> **Focus**: CoTFormer variants only (Standard and Block Universal Transformer experiments excluded per team decision 2026-03-23).

---

## Table of Contents

1. [Phase 0: Codebase Amendments](#phase-0-codebase-amendments)
2. [Phase 1: Container & Data Setup](#phase-1-container--data-setup)
3. [Phase 2: Reproduce CoTFormer Baselines (Table 1)](#phase-2-reproduce-cotformer-baselines-table-1)
4. [Phase 3: 24-Layer Training & ADM Evaluation (Table 2, Figure 4)](#phase-3-24-layer-training--adm-evaluation-table-2-figure-4)
5. [Phase 4: MLA-LN-CoTFormer Extension](#phase-4-mla-ln-cotformer-extension)
6. [Phase 5: mcHC Extension (Manifold-Constrained HyperConnections)](#phase-5-mchc-extension-manifold-constrained-hyperconnections)
7. [Phase 6: Evaluation & Plotting](#phase-6-evaluation--plotting)
8. [Appendix A: Config Cheat Sheet](#appendix-a-config-cheat-sheet)
9. [Appendix B: L4 VRAM Budget](#appendix-b-l4-vram-budget)
10. [Appendix C: MLA Dimensionality Reference](#appendix-c-mla-dimensionality-reference)

---

## Phase 0: Codebase Amendments

This is a fork of the original CoTFormer repo. Python code stays at root (bare imports
in `main.py` require it). Iridis deployment config lives under `iridis/`.

### 0.1 Files to Amend

| File | Change |
|------|--------|
| `models/__init__.py` | Trim unused entries (pondernet, halting). Keep 7 models we need. |
| `data/utils.py` | Add `"openwebtext2"` alias alongside `"owt2"` |
| `data/openwebtext2.py` | `num_proc=40` -> `min(os.cpu_count() or 1, 16)` |
| `models/utils.py` | Add `RMSNorm` class (needed for MLA) |
| `config/base.py` | Add MLA args: `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim` |

### 0.2 Files to Create

| File | Purpose | Phase |
|------|---------|-------|
| `models/mla_cotformer.py` | MLA-LN-CoTFormer model | Phase 4 |
| `analyze_kv_compression.py` | KV cache SVD analysis for MLA rank selection | Phase 4 |
| `plot_pareto.py` | Figure 2 reproduction | Phase 6 |
| `plot_scaling.py` | Figure 3 reproduction | Phase 6 |
| `plot_adm_pareto.py` | Figure 4 reproduction | Phase 6 |

### 0.3 `models/__init__.py` -- Trimmed Registry

```python
from . import base
from . import but_full_depth
from . import cotformer_full_depth
from . import cotformer_full_depth_lnmid_depthemb
from . import but_mod_efficient_sigmoid_lnmid_depthemb_random_factor
from . import but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute
from . import adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final

MODELS = {
    "base": base.GPTBase,
    "but_full_depth": but_full_depth.GPTBase,
    "cotformer_full_depth": cotformer_full_depth.GPTBase,
    "cotformer_full_depth_lnmid_depthemb": cotformer_full_depth_lnmid_depthemb.GPTBase,
    "but_mod_efficient_sigmoid_lnmid_depthemb_random_factor": but_mod_efficient_sigmoid_lnmid_depthemb_random_factor.GPTBase,
    "but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute": but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute.GPTBase,
    "adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final": adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.GPTBase,
}
# Later: MODELS["mla_cotformer"] = mla_cotformer.GPTBase
```

### 0.4 `models/utils.py` -- RMSNorm

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

### 0.5 `config/base.py` -- MLA Arguments

```python
parser.add_argument("--kv_lora_rank", default=192, type=int)
parser.add_argument("--q_lora_rank", default=384, type=int)
parser.add_argument("--qk_rope_head_dim", default=32, type=int)
parser.add_argument("--qk_nope_head_dim", default=64, type=int)
parser.add_argument("--v_head_dim", default=64, type=int)
```

---

## Phase 1: Container & Data Setup

### 1.1 Conda Environment (Shared)

Team shares a single conda environment on scratch (`/scratch/ab3u21/cotformer-env`).
All paths centralised in `iridis/env.sh`.

### 1.2 Pre-tokenize OpenWebText2

Two-step pipeline:
1. `bash iridis/download-dataset/job.sh` (login node, downloads raw tarball)
2. `bash iridis/extract-tokenize-dataset/job.sh` (CPU job, produces `train.bin` + `val.bin`)

Output: `$DATA_DIR/openwebtext2/{train,val}.bin` (~8 GB + ~4 MB memmap).
All subsequent training jobs skip the download.

### 1.3 Job Script Pattern

All packages use the self-submitting `job.sh` pattern (template: `iridis/job.example`):
- `bash job.sh` from login node: creates `run_N/`, calls `sbatch`
- Auto-resume via `--use_pretrained auto` (scans checkpoint dir for `ckpt_*.pt`)
- SLURM logs in `iridis/<package>/run_N/slurm_*.{out,err}`
- Experiment outputs on scratch: `/scratch/ab3u21/exps/`

---

## Phase 2: Reproduce CoTFormer Baselines (Table 1)

Reproduce the CoTFormer rows of Table 1. Standard Transformer and Block Universal Transformer rows are excluded (not the paper's novelty). Paper-reported values for those models serve as reference points.

### 2.1 Experiment Matrix

All: `n_embd=768, n_head=12, seq_len=256, lr=1e-3, wd=0.1, warmup=0.2, 40k steps, owt2, seed=0`.
No reserved layers: `n_layer_begin=0, n_layer_end=0`.

| # | Model | `--model` | `--n_layer` | `--n_repeat` | `--batch_size` | `--acc_steps` | Paper PPL |
|---|-------|-----------|-------------|-------------|----------------|---------------|-----------|
| 1 | CoTFormer 12x2 | `cotformer_full_depth` | 12 | 2 | 32 | 4 | 27.55(0.02) |
| 2 | CoTFormer 12x3 | `cotformer_full_depth` | 12 | 3 | 32 | 4 | 27.07(0.01) |
| 3 | CoTFormer 12x5 | `cotformer_full_depth` | 12 | 5 | 32 | 4 | 26.64(0.04) |
| 4 | CoTFormer 24x2 | `cotformer_full_depth` | 24 | 2 | 8 | 16 | 25.28(0.00) |
| 5 | CoTFormer 24x3 | `cotformer_full_depth` | 24 | 3 | 8 | 16 | 24.85(0.04) |
| 6 | CoTFormer 24x5 | `cotformer_full_depth` | 24 | 5 | 8 | 16 | 24.48(0.03) |

Experiments 1-6 add:
```
--depth_random_method uniform_random_range
--n_layer_begin 0 --n_layer_end 0
--min_repeat <same as n_repeat>
```

### 2.2 Validation

Compare reproduced perplexity against Table 1 CoTFormer rows. Target: within +/-0.3 PPL.
Paper values for Standard and BUT models are cited as baselines (not reproduced).

---

## Phase 3: 24-Layer Training & ADM Evaluation (Table 2, Figure 4)

### 3.1 Training Runs

Three training runs reproduce Table 2 and Section 5 of the paper. All use the 24-layer architecture with reserved layers: 2 prefix + 21 mid (repeated 5x) + 1 suffix = 108 effective layers.

| Run | Package | Model | Steps | Paper Target | Status |
|-----|---------|-------|-------|-------------|--------|
| CoT + Reserved | `cot-res-train` | `cotformer_full_depth` | 40k | PPL 24.51 | Pending |
| LN-CoTFormer | `lncot-train` | `cotformer_full_depth_lnmid_depthemb` | 60k | PPL 24.11 (40k) / 23.19 (60k) | 40k DONE (A1), 60k pending |
| ADM | `adm-train` | `adaptive_cotformer_..._single_final` | 60k | PPL 23.83 | Retraining (v2, RNG fix) |

**ADM retrain rationale**: The v1 run completed 60k steps but had RNG state discontinuities at resume boundaries (B4 in reprod-notes). The v2 run uses the fixed per-rank GPU RNG checkpointing. Fresh start with exp_name `adm_v2_*`.

**LN-CoTFormer 60k**: The existing 40k checkpoint (matching Table 2) is extended to 60k via auto-resume to enable the direct Section 5 comparison.

### 3.2 ADM Architecture

The ADM model is an LN-CoTFormer with an additional Mixture-of-Repeats router (Section 4.1 of the paper). The router adds 4 embedding vectors of 768 dims (~0.01M params).

```
--model adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final
--n_layer 24 --n_repeat 5
--depth_embedding linear_learned
--n_layer_begin 2 --n_layer_end 1
--batch_size 8 --acc_steps 16
```

### 3.3 Figure 4 Evaluation

The `lncot-exp4` package evaluates the trained ADM model at multiple compute budgets:

1. **Phase 1**: Extract router weights (`get_router_weights.py`)
2. **Phase 2**: Evaluate perplexity vs MACs (`get_ppl_per_mac.py`)
   - Router-derived thresholds: "Router" curve in Figure 4
   - Fixed prefix depths: "Fixed Depth" curve in Figure 4

Both curves come from the **same** adaptive model checkpoint.

### 3.4 KV Compression Analysis

Before Phase 4 implementation, run `analyze_kv_compression.py` via the `kv-analysis` package on all trained checkpoints. This informs MLA rank selection by measuring:
- Per-layer effective rank of W_K and W_V (Type A: weight-level)
- Per-layer KV activation compressibility on real data (Type B: activation-level)
- Prefix vs mid vs suffix layer differences

Analysis matrix: LN-CoT 40k, LN-CoT 60k, ADM v2 40k, ADM v2 60k, CoT+Reserved 40k.

---

## Phase 4: MLA-LN-CoTFormer Extension

### 4.1 Architecture Overview

Integrate DeepSeek-V2's MLA into LN-CoTFormer's cross-repeat attention. Goal: compress the KV cache that grows with `n_repeat x seq_len`.

**What changes**: `CausalSelfAttention` in `h_mid` blocks -> `MLACausalSelfAttention`.
**What stays**: Router, depth embeddings, `ln_mid`, repeat loop, token pruning, `h_begin`/`h_end` blocks.

### 4.2 Dimensionality Parameters

For d=768, n_h=12, d_h=64:

| Parameter | Symbol | Value |
|-----------|--------|-------|
| KV compression dim | d_c | 192 |
| Query compression dim | d'_c | 384 |
| RoPE head dim | d_R | 32 |
| Content QK head dim | d_nope | 64 |
| Value head dim | d_v | 64 |

KV cache per token: `d_c + d_R = 224` floats vs MHA's `2 x n_h x d_h = 1536` -> **6.9x compression**.

**Note**: The kv_lora_rank of 192 is DeepSeek-V2's default for a 236B model. Our 208M model may need a smaller rank. The `analyze_kv_compression.py` results (Phase 3.4) will inform this decision.

### 4.3 `MLACausalSelfAttention` Projections

```python
class MLACausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
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

### 4.4 Cross-Repeat Cache Strategy

Cache the **compressed** `kv_latent` (d_c=192) and `k_rope` (d_R=32) instead of full K/V (1536). Trade-off: must re-decompress via `wkv_b` on each attention call, but memory savings are massive.

Over 5 repeats at seq_len=256: standard caches `1,966,080` values; MLA caches `286,720` values.

### 4.5 RoPE Handling

Existing `rotary.py` already supports arbitrary head dims via `adapt_queries`/`adapt_keys`. We pass the smaller d_R-dimensional tensors directly. **No changes to `rotary.py` needed.**

### 4.6 New File: `models/mla_cotformer.py`

Copy `adaptive_cotformer_...single_final.py` and make these surgical changes:
1. Replace `CausalSelfAttention` with `MLACausalSelfAttention`
2. Replace `InPlaceSetSlice` K/V buffers with compressed `kv_latent` + `k_rope` buffers
3. `h_begin`/`h_end` keep standard MHA
4. Everything else identical: MoDBlock, depth embeddings, ln_mid, repeat loop

### 4.7 Register in `models/__init__.py`

```python
from . import mla_cotformer
MODELS["mla_cotformer"] = mla_cotformer.GPTBase
```

### 4.8 Parameter Count

MLA attention is ~24% fewer params per layer than standard MHA. Report parameter counts alongside perplexity for fair comparison.

---

## Phase 5: mcHC Extension (Manifold-Constrained HyperConnections)

### 5.1 Background

Manifold-Constrained HyperConnections (mcHC) generalise the standard residual connection `y = x + f(x)` by replacing it with a learned linear transformation that routes information between layers through multiple "streams." Introduced by Wang et al. (2024) and adopted in DeepSeek-V3, mcHC addresses representation collapse in very deep networks by allowing each layer to selectively read from and write to different subspaces of the hidden representation.

In the standard formulation, the hidden state h is expanded into alpha > 1 streams of dimension d each. Each layer reads a weighted combination of streams, processes it, and writes its output back as a weighted combination. The connection weights form a matrix C in R^{alpha x alpha} per layer, constrained to lie on a smooth manifold (e.g., the Stiefel manifold of orthonormal frames) to prevent degeneration during training.

### 5.2 Motivation for CoTFormer

CoTFormer's cross-repeat architecture creates a natural multi-pass information flow where the same transformer block processes tokens n_repeat times. The standard residual connections within and between repeats treat all information uniformly: every dimension of the residual stream receives equal additive contribution from each layer.

Three properties make mcHC particularly attractive for CoTFormer:

1. **Repeat-aware information routing**: Different repeats serve different purposes (early repeats: broad context gathering; late repeats: refinement). mcHC could learn to route different subspaces of the hidden state through different repeats, effectively specialising the repeat loop without explicit gating.

2. **Complementary to MLA**: MLA (Phase 4) compresses the KV cache, an inference-time memory optimisation. mcHC modifies the residual stream structure, a capacity optimisation that changes what the model can learn. They operate on orthogonal axes and can be combined.

3. **Potential router replacement**: The ADM router decides per-token whether to continue processing. mcHC's multi-stream structure could subsume this: if a stream's write-weight goes to zero for a given layer, that stream is effectively "halted." This is a softer, differentiable alternative to the hard top-k selection in the current router.

### 5.3 Architectural Sketch

For d=768 and alpha=2 streams (minimal configuration):

```
Hidden state: h in R^{2 x 768} (2 streams of 768 dims)

Per-layer processing:
  h_read = C_read @ h           C_read in R^{1 x 2}  (combine streams for layer input)
  h_out  = TransformerBlock(h_read)
  h      = h + C_write @ h_out  C_write in R^{2 x 1}  (distribute output across streams)
```

The connection matrices C_read and C_write are learned per layer. The manifold constraint ensures they remain well-conditioned during training (prevents one stream from dominating).

**Memory overhead**: 2x hidden state size during forward pass (streams are concatenated). No additional KV cache cost: mcHC operates on the residual stream, not on attention keys/values.

**Parameter overhead**: 2 x alpha x alpha scalars per layer = 8 parameters per layer for alpha=2. Negligible relative to the ~8.7M parameters per transformer layer.

### 5.4 Design Questions (To Resolve Before Implementation)

| Question | Options | Decision Criteria |
|----------|---------|-------------------|
| Number of streams (alpha) | 2, 4, 8 | Trade-off: expressiveness vs memory. Start with alpha=2 |
| Manifold constraint | Stiefel (orthogonal), unconstrained + regularisation | Stiefel if training instability observed; unconstrained otherwise |
| Scope of mcHC | Mid layers only, all layers, cross-repeat only | KV analysis results (Phase 3.4) will reveal which layers benefit most |
| Interaction with MLA | Shared latent space, independent | Independent first; explore shared compression after baseline ablation |
| Interaction with ADM router | Co-exist (mcHC + router), replace router | Ablation required: mcHC-only vs router-only vs combined |

### 5.5 Dependencies and Scope

This phase will be expanded after Phase 4 (MLA) completes. Prerequisites:
- Trained MLA model with measured perplexity and KV cache savings
- KV compression analysis results (`analyze_kv_compression.py`) to identify which layers have the most/least compressible representations
- Attention entropy analysis to understand per-layer information flow patterns

**Scope boundary**: Phase 5 implementation begins only after Phase 4 training is complete and results are validated. The mcHC module will be developed as a standalone component that can be inserted into any CoTFormer variant (MLA or standard MHA).

---

## Phase 6: Evaluation & Plotting

### 6.1 Figure 2: Complexity-Perplexity Pareto Frontier

`plot_pareto.py`: Read `summary.json` from each experiment, scatter plot params vs PPL with Pareto frontier. Uses paper-reported values for Standard and BUT models as reference.

### 6.2 Figure 3: Compute Scaling vs Sequence Length

Evaluate trained models at seq_len in {256, 512, 1024, 2048, 4096, 8192}. Compute MACs via `ptflops`. MLA should show sublinear cache growth.

`plot_scaling.py`: Line plot of MACs vs seq_len per model.

### 6.3 Figure 4: ADM Pareto (MACs vs PPL)

`plot_adm_pareto.py`: Read evaluation outputs from `lncot-exp4`. Plot curves for:
1. LN-CoTFormer + ADM (standard attention)
2. MLA-LN-CoTFormer + ADM (compressed attention)

### 6.4 Memory Profiling

Peak GPU memory vs sequence length per model variant:
```python
torch.cuda.reset_peak_memory_stats()
# ... forward pass ...
peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
```

### 6.5 KV Compression Summary

Compile `analyze_kv_compression.py` results into a comparative table:
- Per-layer effective rank at 95% variance (K and V)
- Activation-level vs weight-level rank comparison
- ADM vs non-ADM compressibility differences
- Recommendation for per-layer vs uniform kv_lora_rank

---

## Appendix A: Config Cheat Sheet

### CoTFormer 12xN (Table 1, no reserved layers)

```bash
--model cotformer_full_depth --n_layer 12 --n_repeat N \
--n_embd 768 --n_head 12 \
--batch_size 32 --acc_steps 4 --sequence_length 256 \
--iterations 40000 --dataset owt2 --lr 1e-3 \
--weight_decay 0.1 --warmup_percent 0.2 --dropout 0.0 \
--depth_random_method uniform_random_range \
--n_layer_begin 0 --n_layer_end 0 --min_repeat N \
--eval_freq 100 --seed 0
```

### CoTFormer 24xN (Table 1, no reserved layers)

```bash
--model cotformer_full_depth --n_layer 24 --n_repeat N \
--n_embd 768 --n_head 12 \
--batch_size 8 --acc_steps 16 --sequence_length 256 \
--iterations 40000 --dataset owt2 --lr 1e-3 \
--weight_decay 0.1 --warmup_percent 0.2 --dropout 0.0 \
--depth_random_method uniform_random_range \
--n_layer_begin 0 --n_layer_end 0 --min_repeat N \
--eval_freq 100 --seed 0
```

### CoTFormer + Reserved Layers (Table 2, row 1)

```bash
--model cotformer_full_depth --n_layer 24 --n_repeat 5 \
--n_embd 768 --n_head 12 \
--batch_size 8 --acc_steps 16 --sequence_length 256 \
--iterations 40000 --dataset owt2 --lr 1e-3 \
--weight_decay 0.1 --warmup_percent 0.2 --dropout 0.0 \
--n_layer_begin 2 --n_layer_end 1 \
--eval_freq 100 --seed 0
```

### LN-CoTFormer (Table 2, row 2 / Section 5)

```bash
--model cotformer_full_depth_lnmid_depthemb --n_layer 24 --n_repeat 5 \
--n_embd 768 --n_head 12 \
--batch_size 8 --acc_steps 16 --sequence_length 256 \
--iterations 60000 --dataset owt2 --lr 1e-3 \
--weight_decay 0.1 --warmup_percent 0.2 --dropout 0.0 \
--depth_embedding linear_learned \
--n_layer_begin 2 --n_layer_end 1 --min_repeat 5 \
--eval_freq 100 --seed 0
```

### Adaptive LN-CoTFormer (ADM, Figure 4)

```bash
--model adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final \
--n_layer 24 --n_repeat 5 \
--n_embd 768 --n_head 12 \
--batch_size 8 --acc_steps 16 --sequence_length 256 \
--iterations 60000 --dataset owt2 --lr 1e-3 \
--weight_decay 0.1 --warmup_percent 0.2 --dropout 0.0 \
--depth_embedding linear_learned \
--n_layer_begin 2 --n_layer_end 1 \
--eval_freq 100 --seed 0
```

### MLA-LN-CoTFormer (Our Extension)

```bash
--model mla_cotformer --n_layer 24 --n_repeat 5 \
--n_embd 768 --n_head 12 \
--batch_size 8 --acc_steps 16 --sequence_length 256 \
--iterations 60000 --dataset owt2 --lr 1e-3 \
--weight_decay 0.1 --warmup_percent 0.2 --dropout 0.0 \
--depth_embedding linear_learned \
--n_layer_begin 2 --n_layer_end 1 \
--kv_lora_rank 192 --q_lora_rank 384 \
--qk_rope_head_dim 32 --qk_nope_head_dim 64 --v_head_dim 64 \
--eval_freq 100 --seed 0
```

---

## Appendix B: L4 VRAM Budget

NVIDIA L4: 24 GB GDDR6. Estimates assume bf16 model + activations, fp32 optimizer states.

### 24-Layer Models: ~208M params = 416 MB (bf16)

### Optimizer (AdamW): 2 x 208M x 4B ~ 1.6 GB

### Total Budget

| Experiment | Model | Optim | Activations | Total | Fits L4? |
|-----------|-------|-------|-------------|-------|----------|
| CoTFormer 12x5 (bs=32) | 0.25 | 1.0 | 9.0 | 10.3 GB | Yes |
| CoTFormer 24x5 (bs=8) | 0.42 | 1.6 | 15.0 | 17.0 GB | Yes (tight) |
| LN-CoTFormer 24x5 (bs=8) | 0.42 | 1.6 | 15.0 | 17.0 GB | Yes (tight) |
| Adaptive 24x5 (bs=8) | 0.42 | 1.6 | 15.0 | 17.0 GB | Yes (tight) |
| MLA 24x5 (bs=8) | 0.40 | 1.5 | 10.0 | 11.9 GB | Yes |

Max safe batch_size=8 on L4 24 GB for 24-layer models with 108 effective layers.
Use `ACC_STEPS=16` to keep effective BS=128.

---

## Appendix C: MLA Dimensionality Reference

```
STANDARD MHA (per layer):
  Q = W_Q . h_t    W_Q in R^(768x768)
  K = W_K . h_t    W_K in R^(768x768)
  V = W_V . h_t    W_V in R^(768x768)
  Cache per token: K + V = 1536 floats

MLA (per layer):
  Query path:
    c_Q = RMSNorm(W_qa . h_t)           W_qa in R^(384x768)  -> c_Q in R^384
    q_nope = W_qb . c_Q                 W_qb in R^(768x384)  -> 12 heads x 64
    q_rope = RoPE(W_qr . c_Q)           W_qr in R^(384x384)  -> 12 heads x 32
    q = [q_nope ; q_rope]  per head: R^96

  KV path:
    [c_KV ; k_pe] = W_kva . h_t         W_kva in R^(224x768) -> c_KV in R^192, k_pe in R^32
    c_KV = RMSNorm(c_KV)
    [k_nope ; v] = W_kvb . c_KV         W_kvb in R^(1536x192)
    k_rope = RoPE(k_pe)                 shared across heads
    k = [k_nope ; k_rope]  per head: R^96

  Cache per token: c_KV + k_pe = 224 floats  (6.9x compression)
```

### Weight Absorption (Inference-Only)

At inference, absorb `W_kvb` into the attention dot product to avoid materializing full K/V. Not needed during training -- deferred to post-training optimization.

---

## Task Assignment

| Phase | Owner | Dependencies | Est. GPU-hours |
|-------|-------|-------------|----------------|
| Phase 0-1: Amendments + Data | -- | -- | 0 (DONE) |
| Phase 2: CoTFormer 12L baselines (3 runs) | 1 person | Phase 1 | 3 x ~12h = 36h |
| Phase 2: CoTFormer 24L baselines (3 runs) | 1 person | Phase 1 | 3 x ~24h = 72h |
| Phase 3: CoT+Reserved training (40k) | 1 person | Phase 1 | ~24h |
| Phase 3: LN-CoTFormer 60k extension | 1 person | A1 checkpoint | ~12h (20k remaining) |
| Phase 3: ADM v2 retrain (60k) | 1 person | Phase 1 | ~52h (3 submissions) |
| Phase 3: Figure 4 eval | 1 person | ADM v2 done | ~6h |
| Phase 3: KV analysis | 1 person | All Phase 3 training | ~2h |
| Phase 4: MLA implementation | 1 person | KV analysis results | 0 (code only) |
| Phase 4: MLA training | 1 person | Phase 4 code | ~52h |
| Phase 5: mcHC implementation | 1 person | Phase 4 results | 0 (code only) |
| Phase 5: mcHC training | 1 person | Phase 5 code | ~52h |
| Phase 6: Plotting | 1 person | Phases 2-5 results | 0 (CPU only) |

**Critical path**: Phase 3 training (LN-CoT 60k + ADM v2 retrain) -> KV analysis -> Phase 4 (MLA) -> Phase 5 (mcHC) -> Phase 6 (plots).
Phase 2 (12/24-layer baselines) runs in parallel with Phase 3.
