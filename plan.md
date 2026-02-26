# COMP6258 RCoTFormer -- Surgical Implementation Plan

> **Goal**: Reproduce CoTFormer paper claims (Table 1, Figures 2-4) and extend with MLA-LN-CoTFormer.
> **Hardware**: NVIDIA L4 (24 GB VRAM), Iridis X cluster via Apptainer.
> **Training regime**: 12-layer models, 40k steps, batch 128 (effective), seq_len 256, OpenWebText2.

---

## Table of Contents

1. [Phase 0 -- Codebase Amendments](#phase-0--codebase-amendments)
2. [Phase 1 -- Container & Data Setup](#phase-1--container--data-setup)
3. [Phase 2 -- Reproduce Baselines (Table 1)](#phase-2--reproduce-baselines-table-1)
4. [Phase 3 -- ADM Router (Mixture of Repeats)](#phase-3--adm-router-mixture-of-repeats)
5. [Phase 4 -- MLA-LN-CoTFormer Extension](#phase-4--mla-ln-cotformer-extension)
6. [Phase 5 -- Evaluation & Plotting](#phase-5--evaluation--plotting)
7. [Appendix A -- Config Cheat Sheet](#appendix-a--config-cheat-sheet)
8. [Appendix B -- L4 VRAM Budget](#appendix-b--l4-vram-budget)
9. [Appendix C -- MLA Dimensionality Reference](#appendix-c--mla-dimensionality-reference)

---

## Phase 0 -- Codebase Amendments

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
| `eval_adm.py` | ADM evaluation across MAC budgets | Phase 3 |
| `plot_pareto.py` | Figure 2 reproduction | Phase 5 |
| `plot_scaling.py` | Figure 3 reproduction | Phase 5 |
| `plot_adm_pareto.py` | Figure 4 reproduction | Phase 5 |
| `iridis/jobs/train.slurm` | Training job template | Phase 1 |
| `iridis/jobs/data_prep.slurm` | Data prep CPU job | Phase 1 |

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

## Phase 1 -- Container & Data Setup

### 1.1 Update `iridis/rcotformer.def`

Current container only has torch + numpy. Add all dependencies:

```singularity
%post
    pip3 install --no-cache-dir \
        numpy tiktoken tqdm transformers wandb datasets zstandard ptflops
```

### 1.2 Build and Upload

```bash
sudo apptainer build iridis/rcotformer.sif iridis/rcotformer.def
scp iridis/rcotformer.sif iridis-x:~/dpdl/iridis/
rsync -avz --exclude='*.sif' --exclude='slurm_*' --exclude='data/datasets' \
    ./ iridis-x:~/dpdl/
```

### 1.3 Pre-tokenize OpenWebText2

Run as a separate CPU job before any training:

```bash
apptainer exec --bind "$PWD" iridis/rcotformer.sif \
    python3 -c "
from data.utils import prepare_dataset
import argparse
args = argparse.Namespace(dataset='owt2')
prepare_dataset(args)
"
```

This creates `data/datasets/openwebtext2/{train,val}.bin` (~20 GB memmap). All
subsequent training jobs skip the download.

### 1.4 Training Slurm Template (`iridis/jobs/train.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=rcot_train
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=iridis/jobs/slurm_%j.out
#SBATCH --error=iridis/jobs/slurm_%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/dpdl}"
module load apptainer 2>/dev/null || module load singularity 2>/dev/null
cd "$PROJECT_DIR"

apptainer exec --nv --unsquash --bind "$PWD" \
    iridis/rcotformer.sif \
    python3 main.py \
    --config_format base \
    --model base \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 64 --sequence_length 256 --acc_steps 2 \
    --iterations 40000 --dataset owt2 --lr 1e-3 \
    --weight_decay 0.1 --warmup_percent 0.2 \
    --eval_freq 100 --seed 0 --dropout 0.0 \
    --results_base_folder exps \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    "$@"
```

`$@` enables per-experiment overrides: `sbatch iridis/jobs/train.slurm --model cotformer_full_depth --n_repeat 2`.

---

## Phase 2 -- Reproduce Baselines (Table 1)

### 2.1 Experiment Matrix

All: `n_embd=768, n_head=12, n_layer=12, seq_len=256, lr=1e-3, wd=0.1, warmup=0.2, 40k steps, owt2, seed=0`.

| # | Model | `--model` | `--n_repeat` | `--batch_size` | `--acc_steps` | Eff. BS |
|---|-------|-----------|-------------|----------------|---------------|---------|
| 1 | Standard 12L | `base` | -- | 64 | 2 | 128 |
| 2 | BUT 12x2 | `but_full_depth` | 2 | 32 | 4 | 128 |
| 3 | BUT 12x3 | `but_full_depth` | 3 | 32 | 4 | 128 |
| 4 | BUT 12x5 | `but_full_depth` | 5 | 32 | 4 | 128 |
| 5 | CoTFormer 12x2 | `cotformer_full_depth` | 2 | 32 | 4 | 128 |
| 6 | CoTFormer 12x3 | `cotformer_full_depth` | 3 | 32 | 4 | 128 |
| 7 | CoTFormer 12x5 | `cotformer_full_depth` | 5 | 32 | 4 | 128 |
| 8 | LN-CoTFormer 12x5 | `cotformer_full_depth_lnmid_depthemb` | 5 | 32 | 4 | 128 |

Experiments 2-8 add:
```
--depth_random_method uniform_random_range
--n_layer_begin 0 --n_layer_end 0
--min_repeat <same as n_repeat>
```

LN-CoTFormer (exp 8) also adds: `--depth_embedding linear_learned`.

### 2.2 Validation

Compare reproduced perplexity against Table 1. Target: within +/-0.3 PPL.

### 2.3 Reserved-Layer Ablation (if time permits)

```
--n_layer_begin 2 --n_layer_end 1  # 2 prefix + 9 mid (repeated) + 1 suffix
```

---

## Phase 3 -- ADM Router (Mixture of Repeats)

### 3.1 Existing Router Code

Two implementations already in the codebase:
- **Variant A** (`but_mod_efficient_sigmoid_lnmid_depthemb_random_factor.py`): active-mask, standard causal attention
- **Variant B** (`adaptive_cotformer_...single_final.py`): prune-and-cache, custom cross-repeat mask

We use Variant B as-is for ADM experiments.

### 3.2 Training the Adaptive Model

```
--model adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final
--n_layer 12 --n_repeat 5
--depth_embedding linear_learned
--depth_random_method uniform_random_range
--n_layer_begin 0 --n_layer_end 0
--batch_size 16 --acc_steps 8
```

### 3.3 New File: `eval_adm.py`

Evaluates the adaptive model across MAC budgets:
1. Load model + checkpoint
2. Load router weights (from `get_router_weights.py`)
3. For each threshold in `[0.0, 0.1, ..., 1.0]`:
   - Compute `eval_length_factor` by thresholding router weights
   - Compute MACs via `ptflops`
   - Run eval, record PPL
4. Save `adm_pareto.json`

### 3.4 Router Weight Extraction

```bash
python3 get_router_weights.py --checkpoint exps/owt2/adaptive_cotformer_.../
```

---

## Phase 4 -- MLA-LN-CoTFormer Extension

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

## Phase 5 -- Evaluation & Plotting

### 5.1 Figure 2: Complexity-Perplexity Pareto Frontier

`plot_pareto.py`: Read `summary.json` from each experiment, scatter plot params vs PPL with Pareto frontier.

### 5.2 Figure 3: Compute Scaling vs Sequence Length

Evaluate trained models at seq_len in {256, 512, 1024, 2048, 4096, 8192}. Compute MACs via `ptflops`. MLA should show sublinear cache growth.

`plot_scaling.py`: Line plot of MACs vs seq_len per model.

### 5.3 Figure 4: ADM Pareto (MACs vs PPL)

`plot_adm_pareto.py`: Read `adm_pareto.json` from `eval_adm.py`. Plot curves for:
1. LN-CoTFormer + ADM (standard attention)
2. MLA-LN-CoTFormer + ADM (compressed attention)

### 5.4 Memory Profiling

Peak GPU memory vs sequence length per model variant:
```python
torch.cuda.reset_peak_memory_stats()
# ... forward pass ...
peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
```

---

## Appendix A -- Config Cheat Sheet

### Base 12-Layer Standard Transformer

```bash
--model base --n_layer 12 --n_embd 768 --n_head 12 \
--batch_size 64 --acc_steps 2 --sequence_length 256 \
--iterations 40000 --dataset owt2 --lr 1e-3 \
--weight_decay 0.1 --warmup_percent 0.2 --dropout 0.0 \
--eval_freq 100 --seed 0
```

### CoTFormer 12xN

```bash
--model cotformer_full_depth --n_layer 12 --n_repeat N \
--batch_size 32 --acc_steps 4 --sequence_length 256 \
--depth_random_method uniform_random_range \
--n_layer_begin 0 --n_layer_end 0 --min_repeat N
```

### LN-CoTFormer 12xN

```bash
--model cotformer_full_depth_lnmid_depthemb --n_layer 12 --n_repeat N \
--depth_embedding linear_learned \
--batch_size 32 --acc_steps 4
```

### Adaptive LN-CoTFormer (Variant B + Router)

```bash
--model adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final \
--n_layer 12 --n_repeat 5 \
--depth_embedding linear_learned \
--depth_random_method uniform_random_range \
--batch_size 16 --acc_steps 8
```

### MLA-LN-CoTFormer (Our Extension)

```bash
--model mla_cotformer --n_layer 12 --n_repeat 5 \
--depth_embedding linear_learned \
--depth_random_method uniform_random_range \
--kv_lora_rank 192 --q_lora_rank 384 \
--qk_rope_head_dim 32 --qk_nope_head_dim 64 --v_head_dim 64 \
--batch_size 16 --acc_steps 8
```

---

## Appendix B -- L4 VRAM Budget

NVIDIA L4: 24 GB GDDR6. Estimates assume bf16 model + activations, fp32 optimizer states.

### Model: ~124M params = 248 MB (bf16)

### Optimizer (AdamW): 2 x 124M x 4B ~ 1 GB

### Total Budget

| Experiment | Model | Optim | Activations | Total | Fits L4? |
|-----------|-------|-------|-------------|-------|----------|
| Standard 12L (bs=64) | 0.25 | 1.0 | 4.0 | 5.3 GB | Yes |
| CoTFormer 12x2 (bs=32) | 0.25 | 1.0 | 4.0 | 5.3 GB | Yes |
| CoTFormer 12x5 (bs=32) | 0.25 | 1.0 | 9.0 | 10.3 GB | Yes |
| Adaptive 12x5 (bs=16) | 0.25 | 1.0 | 6.0 | 7.3 GB | Yes |
| MLA 12x5 (bs=16) | 0.24 | 0.95 | 4.0 | 5.2 GB | Yes |

---

## Appendix C -- MLA Dimensionality Reference

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

## Task Assignment Suggestion

| Phase | Owner | Dependencies | Est. GPU-hours |
|-------|-------|-------------|----------------|
| Phase 0-1: Amendments + Container | Any 1 person | None | 0 (CPU only) |
| Phase 2: Baselines (8 runs) | All 3 in parallel | Phase 1 | 8 x ~12h = 96h |
| Phase 3: ADM router eval | 1 person | Phase 2 adaptive run | ~24h |
| Phase 4: MLA implementation | 1 person (strongest PyTorch) | Phase 0 | 0 (code only) |
| Phase 4: MLA training | 1 person | Phase 4 code + Phase 1 | ~24h |
| Phase 5: Plotting | 1 person | Phases 2-4 results | 0 (CPU only) |

**Critical path**: Phase 0 -> Phase 1 -> Phase 2 (longest, parallelizable across team).
Phase 4 MLA code can be developed in parallel with Phase 2 baseline training.
