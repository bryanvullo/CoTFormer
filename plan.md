# COMP6258 RCoTFormer — Surgical Implementation Plan

> **Goal**: Reproduce CoTFormer paper claims (Table 1, Figures 2-4) and extend with MLA-LN-CoTFormer.
> **Hardware**: NVIDIA L4 (24 GB VRAM), Iridis X cluster via Apptainer.
> **Training regime**: 12-layer models, 40k steps, batch 128 (effective), seq_len 256, OpenWebText2.

---

## Table of Contents

1. [Phase 0 — Codebase Transfer](#phase-0--codebase-transfer)
2. [Phase 1 — Container & Dependency Setup](#phase-1--container--dependency-setup)
3. [Phase 2 — Data Pipeline](#phase-2--data-pipeline)
4. [Phase 3 — Reproduce Baselines (Table 1)](#phase-3--reproduce-baselines-table-1)
5. [Phase 4 — ADM Router (Mixture of Repeats)](#phase-4--adm-router-mixture-of-repeats)
6. [Phase 5 — MLA-LN-CoTFormer Extension](#phase-5--mla-ln-cotformer-extension)
7. [Phase 6 — Evaluation & Plotting](#phase-6--evaluation--plotting)
8. [Appendix A — File Transfer Manifest](#appendix-a--file-transfer-manifest)
9. [Appendix B — Config Cheat Sheet](#appendix-b--config-cheat-sheet)
10. [Appendix C — L4 VRAM Budget](#appendix-c--l4-vram-budget)
11. [Appendix D — MLA Dimensionality Reference](#appendix-d--mla-dimensionality-reference)

---

## Phase 0 — Codebase Transfer

### 0.1 Files to Copy Verbatim

Copy the following from `CoTFormer/` into a new package directory
`COMP6258-RCoTFormer/rcotformer/` (all paths relative to `CoTFormer/`):

```
rcotformer/                         ← new package root
├── main.py
├── eval.py
├── get_ppl_per_mac.py
├── get_router_weights.py
├── get_router_weights_but.py
├── config/
│   ├── __init__.py
│   └── base.py
├── data/
│   ├── __init__.py               ← create empty (doesn't exist upstream)
│   ├── utils.py
│   └── openwebtext2.py
├── optim/
│   ├── __init__.py               ← create empty
│   ├── base.py
│   ├── utils.py
│   └── adafactor.py
├── distributed/
│   ├── __init__.py
│   ├── backend.py
│   ├── single.py
│   └── ddp.py
└── models/
    ├── __init__.py
    ├── utils.py
    ├── base.py
    ├── cotformer_full_depth.py
    ├── cotformer_full_depth_lnmid_depthemb.py
    ├── adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py
    ├── but_mod_efficient_sigmoid_lnmid_depthemb_random_factor.py
    ├── but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute.py
    ├── but_full_depth.py
    ├── positional_encoders/
    │   ├── __init__.py
    │   ├── encoder.py
    │   ├── rotary.py
    │   └── rotary_utils.py
    └── caches/
        ├── __init__.py
        └── cache.py
```

**Omit** (not needed for reproduction scope):
- `models/pondernet.py` — PonderNet baseline, not in our scope
- `models/but_halting_freeze_input_on_stop.py` — halting variant, not in scope
- `models/depth_predictor/` — auxiliary module unused in target experiments

### 0.2 Amendments to Transferred Files

#### 0.2.1 `models/__init__.py` — Trim Registry

Remove unused model entries. Keep only the six models we need:

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
# ... keep make_model_from_args and registered_models as-is
```

New models (ADM, MLA) will be registered here in later phases.

#### 0.2.2 `data/utils.py` line 7 — Add Dataset Alias

The upstream code registers only `"owt2"`. Add an alias so `--dataset openwebtext2`
also works:

```python
PREPARE_GET_DATASET_MAP = {
    "owt2": (prepare_openwebtext2_data, get_openwebtext2_data),
    "openwebtext2": (prepare_openwebtext2_data, get_openwebtext2_data),  # alias
}
```

#### 0.2.3 `data/openwebtext2.py` — `num_proc` Adjustment

Line with `num_proc=40`: L4 nodes have limited CPU cores. Change to:

```python
num_proc = min(os.cpu_count() or 1, 16)
```

This avoids HuggingFace tokenization crashing on nodes with fewer cores.

#### 0.2.4 `config/base.py` — No Functional Changes

The defaults (`n_layer=24`, `sequence_length=512`, `iterations=25000`, etc.) are
overridden via CLI flags per experiment. No code changes needed — all tuning
happens at invocation time. See [Appendix B](#appendix-b--config-cheat-sheet).

#### 0.2.5 `optim/utils.py` — Perplexity Constant

The code uses `2.71828` instead of `math.e`. This is a minor precision issue
(6 decimal places vs 15). Optionally fix:

```python
import math
val_perplexity = math.exp(val_loss)  # instead of 2.71828 ** val_loss
```

#### 0.2.6 `requirements.txt` — Update for CUDA 12.1

The upstream pins `torch==2.0.0+cu118`. Our container uses CUDA 12.1. Replace:

```
tiktoken
torch>=2.1.0
tqdm
transformers
wandb
datasets
zstandard
ptflops
```

Torch installation will come from the Apptainer container (already has PyTorch +
CUDA 12.1). The `requirements.txt` covers pip extras installed on top.

---

## Phase 1 — Container & Dependency Setup

### 1.1 Update `rcotformer.def`

The current container only has `torch` and `numpy`. Add all Python dependencies:

```singularity
%post
    apt-get update && apt-get install -y \
        python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

    pip3 install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu121

    pip3 install --no-cache-dir \
        numpy tiktoken tqdm transformers wandb datasets zstandard ptflops
```

### 1.2 Rebuild and Upload

```bash
# Local (WSL)
sudo apptainer build rcotformer.sif rcotformer.def
scp rcotformer.sif iridis-x:~/dpdl/

# Upload package
rsync -avz --exclude='*.sif' --exclude='slurm_*' --exclude='data/datasets' \
    rcotformer/ iridis-x:~/dpdl/rcotformer/
```

### 1.3 Slurm Template for Training

Create `rcotformer/job.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=rcot_train
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=rcotformer/slurm_%j.out
#SBATCH --error=rcotformer/slurm_%j.err

PKG_NAME="rcotformer"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/dpdl}"
module load apptainer 2>/dev/null || module load singularity 2>/dev/null
cd "$PROJECT_DIR"

apptainer exec --nv --unsquash --bind "$PWD" \
    rcotformer.sif \
    python3 "${PKG_NAME}/main.py" \
    --config_format base \
    --model base \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 64 --sequence_length 256 --acc_steps 2 \
    --iterations 40000 --dataset owt2 --lr 1e-3 \
    --weight_decay 0.1 --warmup_percent 0.2 \
    --eval_freq 100 --seed 0 --dropout 0.0 \
    --results_base_folder "${PKG_NAME}/exps" \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    "$@"  # Allow CLI overrides
```

`$@` at the end enables per-experiment overrides: e.g.
`sbatch rcotformer/job.slurm --model cotformer_full_depth --n_repeat 2`.

---

## Phase 2 — Data Pipeline

### 2.1 Pre-tokenize OpenWebText2

The first `main.py` run downloads OpenWebText2 from HuggingFace and tokenizes it
into `data/datasets/openwebtext2/{train,val}.bin` (uint16 memmap files). This
takes significant time and network I/O.

**Recommended**: Run data prep as a separate CPU-only job first:

```bash
#SBATCH --partition=ecsstudents_l4    # or a CPU partition
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

apptainer exec --bind "$PWD" rcotformer.sif \
    python3 -c "
from data.utils import prepare_dataset
import argparse
args = argparse.Namespace(dataset='owt2')
prepare_dataset(args)
"
```

After this, `rcotformer/data/datasets/openwebtext2/{train,val}.bin` exist and all
training jobs skip the download step.

### 2.2 Data Loading Path

The `data/openwebtext2.py` script uses a path relative to its own location:
```python
OWT2_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/openwebtext2/")
```

This means the memmap files must live at `rcotformer/data/datasets/openwebtext2/`.
Because these files are large (~20 GB), add to `.gitignore`:

```gitignore
# Training data (downloaded at runtime)
rcotformer/data/datasets/
```

---

## Phase 3 — Reproduce Baselines (Table 1)

### 3.1 Experiment Matrix

All experiments use: `n_embd=768, n_head=12, n_layer=12, seq_len=256,
lr=1e-3, weight_decay=0.1, warmup_percent=0.2, 40k steps, owt2, seed=0`.

| # | Model | `--model` | `--n_repeat` | `--batch_size` | `--acc_steps` | Effective BS |
|---|-------|-----------|-------------|----------------|---------------|-------------|
| 1 | Standard 12L | `base` | — | 64 | 2 | 128 |
| 2 | BUT 12×2 | `but_full_depth` | 2 | 32 | 4 | 128 |
| 3 | BUT 12×3 | `but_full_depth` | 3 | 32 | 4 | 128 |
| 4 | BUT 12×5 | `but_full_depth` | 5 | 32 | 4 | 128 |
| 5 | CoTFormer 12×2 | `cotformer_full_depth` | 2 | 32 | 4 | 128 |
| 6 | CoTFormer 12×3 | `cotformer_full_depth` | 3 | 32 | 4 | 128 |
| 7 | CoTFormer 12×5 | `cotformer_full_depth` | 5 | 32 | 4 | 128 |
| 8 | LN-CoTFormer 12×5 | `cotformer_full_depth_lnmid_depthemb` | 5 | 32 | 4 | 128 |

Additional flags for experiments 2-8:
```
--depth_random_method uniform_random_range
--n_layer_begin 0 --n_layer_end 0
--min_repeat <same as n_repeat>
```

For LN-CoTFormer (exp 8), also add:
```
--depth_embedding linear_learned
```

### 3.2 Validation

After each training run, evaluate:

```bash
apptainer exec --nv --bind "$PWD" rcotformer.sif \
    python3 rcotformer/eval.py \
    --checkpoint rcotformer/exps/owt2/<model>/<exp_name>/
```

Expected outputs: `val_loss`, `val_perplexity`, `val_acc`.

Compare perplexity values against Table 1 in the paper. Target: reproduced
values within ±0.3 PPL of reported numbers.

### 3.3 L4 VRAM Feasibility

See [Appendix C](#appendix-c--l4-vram-budget). Summary: all 12-layer experiments
at seq_len=256 with batch_size=32 fit comfortably in 24 GB. The 12×5 CoTFormer
(cross-repeat KV cache) is the tightest at ~14 GB peak.

### 3.4 Reserved-Layer Ablation (if time permits)

For the LN-CoTFormer, test reserved layers:

```
--n_layer_begin 2 --n_layer_end 1  # 2 prefix + 9 mid (repeated) + 1 suffix
```

This matches the paper's LN-CoTFormer that achieves 24.11 PPL.

---

## Phase 4 — ADM Router (Mixture of Repeats)

### 4.1 What the Paper Already Provides

The upstream codebase has **two** router implementations:
- **Variant A** (`but_mod_efficient_sigmoid_lnmid_depthemb_random_factor.py`): active-mask approach, standard causal attention
- **Variant B** (`adaptive_cotformer_...single_final.py`): prune-and-cache approach, custom cross-repeat attention mask, physically removes tokens

Per the paper scope, we reuse the **Variant B** model (the adaptive CoTFormer)
as-is for the ADM experiments. It is already transferred in Phase 0.

### 4.2 What to Write from Scratch: Router Weight Extraction & Evaluation Hooks

The upstream `get_router_weights.py` and `get_router_weights_but.py` contain
**monkey-patched forward functions** that are tightly coupled to the model
internals. They work but are fragile. We need to:

1. **Verify they work** with the transferred code (test with a small checkpoint).
2. **Write a unified evaluation hook** that:
   - Runs the adaptive model with varying `eval_length_factor` thresholds
   - Computes MACs via `ptflops` for each threshold
   - Logs perplexity for each threshold
   - Outputs a JSON/NPY file mapping `(threshold → PPL, MACs)` for Figure 4

#### 4.2.1 New File: `rcotformer/eval_adm.py`

This script wraps `get_ppl_per_mac.py` logic but in a cleaner interface:

```python
"""Evaluate adaptive depth model across MAC budgets.

Usage:
    python eval_adm.py --checkpoint <path> --thresholds 0.0 0.1 ... 1.0

Outputs:
    <checkpoint_dir>/adm_pareto.json
    Fields per threshold: {length_factors, ppl, macs, avg_depth}
"""
```

**Implementation steps:**
1. Load model and checkpoint (reuse `eval.py` arg reconstruction)
2. Load `router_weights.npy` (generated by `get_router_weights.py`)
3. For each threshold `t` in `[0.0, 0.1, ..., 1.0]`:
   a. Compute `eval_length_factor` by thresholding router weights at `t`
   b. Set `config.eval_length_factor = computed_factors`
   c. Compute MACs via `ptflops.get_model_complexity_info()`
      - Swap model to `_for_mac_compute` variant
      - Disable Flash Attention for tracing
   d. Run eval loop, record PPL
4. Save results to `adm_pareto.json`

### 4.3 Training the Adaptive Model

```
--model adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final
--n_layer 12 --n_repeat 5
--depth_embedding linear_learned
--depth_random_method uniform_random_range
--n_layer_begin 0 --n_layer_end 0
--batch_size 16 --acc_steps 8
```

Effective BS = 128. Reduced per-GPU batch to 16 because the cross-repeat KV
cache + token pruning + custom mask consumes more memory.

### 4.4 Extracting Router Weights Post-Training

```bash
python3 rcotformer/get_router_weights.py \
    --checkpoint rcotformer/exps/owt2/adaptive_cotformer_.../
```

This produces `router_weights.npy` in the checkpoint directory.

---

## Phase 5 — MLA-LN-CoTFormer Extension

### 5.1 Architecture Overview

We integrate DeepSeek-V2's Multi-Head Latent Attention (MLA) into the
LN-CoTFormer's cross-repeat attention mechanism. The goal: compress the KV cache
that grows linearly with `n_repeat × seq_len` by caching a low-rank latent
vector instead of full K/V tensors.

**What changes:**
- `CausalSelfAttention` in the `h_mid` blocks is replaced with `MLACausalSelfAttention`
- `h_begin` and `h_end` blocks retain standard MHA (they don't participate in the repeat loop)
- The router (`MoDBlock`) is unchanged
- The repeat loop, `ln_mid`, and depth embeddings are unchanged

**What does NOT change:**
- `Block` wrapper (pre-LN structure)
- `MLP`
- `GPTBase.__init__` layout (`h_begin/h_mid/h_end`)
- `GPTBase.forward` repeat logic, token pruning, index tracking
- Router, depth embeddings, `ln_mid`

### 5.2 MLA Attention — Detailed Design

#### 5.2.1 Dimensionality Parameters

For a 12-layer, 768-dim, 12-head model (`d=768, n_h=12, d_h=64`):

| Parameter | Symbol | Value | Rationale |
|-----------|--------|-------|-----------|
| Model dim | d | 768 | Match paper |
| Heads | n_h | 12 | Match paper |
| Head dim | d_h | 64 | d / n_h |
| KV compression dim | d_c | 192 | 3× head dim (compression ratio ≈ 8) |
| Query compression dim | d'_c | 384 | 2× d_c (matches DeepSeek ratio) |
| RoPE head dim | d_R | 32 | d_h / 2 |
| Content QK head dim | d_nope | 64 | Same as d_h |
| Value head dim | d_v | 64 | Same as d_h |

These give a **KV cache per token per layer** of `d_c + d_R = 224` floats
vs standard MHA's `2 × n_h × d_h = 1536` floats — a **6.9× compression**.

Over 5 repeats at seq_len=256: standard caches `5 × 256 × 1536 = 1,966,080`
values; MLA caches `5 × 256 × 224 = 286,720` values — same 6.9× saving.

#### 5.2.2 Projection Layers

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

        # Query path: down → norm → up + RoPE branch
        self.wq_a = nn.Linear(d, d_qc, bias=False)          # (768 → 384)
        self.q_norm = RMSNorm(d_qc)
        self.wq_b = nn.Linear(d_qc, n_h * d_nope, bias=False)  # (384 → 768)
        self.wq_rope = nn.Linear(d_qc, n_h * d_R, bias=False)  # (384 → 384)

        # KV path: joint down → norm → up
        self.wkv_a = nn.Linear(d, d_c + d_R, bias=False)    # (768 → 224)
        self.kv_norm = RMSNorm(d_c)
        self.wkv_b = nn.Linear(d_c, n_h * (d_nope + d_v), bias=False)  # (192 → 1536)

        # Output projection
        self.c_proj = nn.Linear(n_h * d_v, d, bias=False)   # (768 → 768)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
```

#### 5.2.3 Forward Pass (Training Mode)

```python
def forward(self, x, pos_emb_closure, cache_context, start_index, indices=None):
    B, T, C = x.size()

    # --- Query path ---
    q_compressed = self.q_norm(self.wq_a(x))                  # (B, T, d_qc)
    q_nope = self.wq_b(q_compressed)                          # (B, T, n_h * d_nope)
    q_rope = self.wq_rope(q_compressed)                       # (B, T, n_h * d_R)
    # Reshape to per-head: (B, n_h, T, dim)
    q_nope = q_nope.view(B, T, n_h, d_nope).transpose(1, 2)
    q_rope = q_rope.view(B, T, n_h, d_R).transpose(1, 2)
    # Apply RoPE to q_rope only
    q_rope = pos_emb_closure.adapt_queries(q_rope, start_index=start_index, indices=indices)
    # Concatenate: q = [q_nope ; q_rope]  → (B, n_h, T, d_nope + d_R)
    q = torch.cat([q_nope, q_rope], dim=-1)

    # --- KV path ---
    kv_out = self.wkv_a(x)                                    # (B, T, d_c + d_R)
    kv_latent, k_rope = kv_out.split([d_c, d_R], dim=-1)
    kv_latent = self.kv_norm(kv_latent)                       # (B, T, d_c)
    # Decompress: get content keys and values
    kv_decompressed = self.wkv_b(kv_latent)                   # (B, T, n_h*(d_nope+d_v))
    kv_decompressed = kv_decompressed.view(B, T, n_h, d_nope + d_v).transpose(1, 2)
    k_nope, v = kv_decompressed.split([d_nope, d_v], dim=-1)
    # Apply RoPE to k_rope (shared across heads, MQA-style)
    k_rope = k_rope.unsqueeze(1).expand(-1, n_h, -1, -1)     # (B, n_h, T, d_R)
    k_rope = pos_emb_closure.adapt_keys(k_rope, start_index=start_index, indices=indices)
    # Concatenate: k = [k_nope ; k_rope]  → (B, n_h, T, d_nope + d_R)
    k = torch.cat([k_nope, k_rope], dim=-1)

    # --- Cross-repeat cache accumulation ---
    # Cache the COMPRESSED representations, not full K/V
    # self.kv_latent_cache: (full_buffer, current_slice) using InPlaceSetSlice
    # self.k_rope_cache: same pattern
    if self.kv_latent_cache is not None:
        self.kv_latent_cache = apply_inplace_set(
            self.kv_latent_cache,
            kv_latent.unsqueeze(1),  # add head dim for uniform shape
            dim=2
        )
        self.k_rope_cache = apply_inplace_set(
            self.k_rope_cache, k_rope, dim=2
        )
        # Reconstruct full K from cached compressed representations
        cached_kv_latent = self.kv_latent_cache[1].squeeze(1)  # (B, acc_T, d_c)
        cached_kv = self.wkv_b(cached_kv_latent)               # decompress
        cached_kv = cached_kv.view(B, -1, n_h, d_nope + d_v).transpose(1, 2)
        k_nope_all, v = cached_kv.split([d_nope, d_v], dim=-1)
        k_rope_all = self.k_rope_cache[1]                      # (B, n_h, acc_T, d_R)
        k = torch.cat([k_nope_all, k_rope_all], dim=-1)

    # --- Attention ---
    # attn_mask from indices (same cross-repeat mask as Variant B)
    scale = 1.0 / math.sqrt(d_nope + d_R)
    att = (q @ k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        att = att.masked_fill(~attn_mask, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v                                                # (B, n_h, T, d_v)

    # --- Output ---
    y = y.transpose(1, 2).contiguous().view(B, T, n_h * d_v)
    y = self.resid_dropout(self.c_proj(y))
    return y
```

#### 5.2.4 Cross-Repeat Cache: Compress, Then Cache

**Critical design decision**: We cache the compressed latent `kv_latent` (d_c=192
dims) and the RoPE key `k_rope` (d_R=32 dims per head, but shared so 32 total)
instead of the full decompressed K and V (n_h × d_h = 768 each).

The trade-off: we must **re-decompress** (`wkv_b` forward pass) on every
attention computation to reconstruct K and V from the cached latent. This adds
compute but massively reduces memory.

**Cache init/drop** follows the same pattern as Variant B:

```python
def init_cache(self, expected_total_length):
    self._lazy_init_cache_length = expected_total_length

def drop_cache(self):
    self.kv_latent_cache = None
    self.k_rope_cache = None
    self._lazy_init_cache_length = None
```

Lazy initialization in forward (first call after `init_cache`):

```python
if self._lazy_init_cache_length is not None:
    L = self._lazy_init_cache_length
    self.kv_latent_cache = (
        kv_latent.new_empty((B, 1, L, d_c)),  # head dim=1 (shared)
        None
    )
    self.k_rope_cache = (
        k_rope.new_empty((B, n_h, L, d_R)),
        None
    )
    self._lazy_init_cache_length = None
```

#### 5.2.5 RoPE Handling — The Decoupled Approach

The existing `rotary.py` applies RoPE via `adapt_queries(q, indices=...)` and
`adapt_keys(k, indices=...)`. These methods call `adapt_vector_for_indices` which
takes the last `d//2` pairs and rotates them.

For MLA, we apply RoPE **only to the d_R-dimensional components**:
- `q_rope ∈ (B, n_h, T, d_R)` — multi-head query positional component
- `k_rope ∈ (B, n_h, T, d_R)` — shared-head key positional component

The existing `adapt_queries`/`adapt_keys` already support arbitrary last-dim
sizes (they compute `freqs` based on the input's `hs` dimension). So **the
existing RoPE code works without modification** — we just pass the smaller
tensors to it.

However, `rotary.py` currently ties `pos_size = k.shape[-1] // 2` to the head
dimension. Since `d_R = 32` (half of the original `d_h = 64`), the RoPE
frequency vector is already the right size. **No changes to `rotary.py` needed.**

### 5.3 New File: `rcotformer/models/mla_cotformer.py`

Create this file by copying
`adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py`
and making these surgical changes:

1. **Replace `CausalSelfAttention`** with `MLACausalSelfAttention` (Section 5.2).
2. **Replace `InPlaceSetSlice` usage**: Instead of accumulating full K/V buffers,
   accumulate compressed `kv_latent` and `k_rope` buffers.
3. **Update `Block.forward`**: No signature change needed — MLA attention has the
   same external interface `(x, pos_emb_closure, cache_context, start_index, indices)`.
4. **Update `GPTBase.__init__`**: Add MLA config parameters, construct `h_mid`
   blocks with MLA attention. `h_begin` and `h_end` keep standard MHA.
5. **Keep everything else identical**: MoDBlock, depth embeddings, ln_mid,
   repeat loop, token pruning, output reassembly.

### 5.4 Config Extensions

Add to `config/base.py`:

```python
# MLA parameters (only used when model is mla_cotformer)
parser.add_argument("--kv_lora_rank", default=192, type=int,
    help="KV compression dimension for MLA")
parser.add_argument("--q_lora_rank", default=384, type=int,
    help="Query compression dimension for MLA")
parser.add_argument("--qk_rope_head_dim", default=32, type=int,
    help="RoPE key/query dimension per head for MLA")
parser.add_argument("--qk_nope_head_dim", default=64, type=int,
    help="Content key/query dimension per head for MLA")
parser.add_argument("--v_head_dim", default=64, type=int,
    help="Value dimension per head for MLA")
```

### 5.5 Register in `models/__init__.py`

```python
from . import mla_cotformer

MODELS["mla_cotformer"] = mla_cotformer.GPTBase
```

### 5.6 RMSNorm Implementation

MLA requires RMSNorm (not LayerNorm). Add to `models/utils.py`:

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

### 5.7 Parameter Count Parity

MLA changes the parameter count. For a single attention layer:

| Component | Standard MHA | MLA |
|-----------|-------------|-----|
| Q projection | 768×768 = 590K | 768×384 + 384×768 + 384×384 = 737K |
| K projection | 768×768 = 590K | Part of wkv_a: 768×224 = 172K |
| V projection | 768×768 = 590K | Part of wkv_b: 192×1536 = 295K |
| Output projection | 768×768 = 590K | 768×768 = 590K |
| Norms | — | 2 × (192 + 384) = 1.2K |
| **Total** | **2.36M** | **1.80M** |

MLA is ~24% fewer attention parameters per layer. To ensure a fair comparison
for the paper, we should report parameter counts alongside perplexity. The
reduced parameter count actually strengthens the argument if MLA matches or
beats standard MHA at lower memory.

### 5.8 Implementation Subtlety: Decompression Inside the Cache Loop

A key implementation detail: when we accumulate the compressed latent across
repeats, we need to **decompress all accumulated latents** (not just the current
repeat's) to form the full K and V for attention. This means:

```python
# After accumulating kv_latent into cache:
cached_kv_latent = self.kv_latent_cache[1]  # (B, 1, accumulated_T, d_c)
# Decompress ALL cached latents
full_kv = self.wkv_b(cached_kv_latent.squeeze(1))  # (B, accumulated_T, n_h*(d_nope+d_v))
```

This decompression is O(accumulated_T) and happens every forward call. The
memory savings come from **caching** d_c instead of n_h × d_h, but the compute
cost of decompression is present. For seq_len=256 and 5 repeats, this is
manageable (max 1280 tokens to decompress).

**Alternative (optimization, lower priority)**: Absorb `wkv_b` into the attention
computation to avoid materializing full K/V. This requires restructuring the
attention math (see Appendix D). Only pursue if memory is a bottleneck.

---

## Phase 6 — Evaluation & Plotting

### 6.1 Figure 2: Complexity-Perplexity Pareto Frontier

**What to plot**: X-axis = total parameters (or MACs), Y-axis = validation PPL.
**Data points**: All 8 baseline experiments from Phase 3 + MLA variants.

**Script**: `rcotformer/plot_pareto.py`

Reads `summary.json` from each experiment directory, extracts `val_perplexity`
and parameter count. Generates a scatter plot with Pareto frontier overlay.

### 6.2 Figure 3: Compute Scaling vs Sequence Length

**What to plot**: X-axis = sequence length, Y-axis = MACs (or wall-clock time).
**Experiments**: Evaluate trained models at seq_len ∈ {256, 512, 1024, 2048, 4096, 8192}.

**Implementation**: Modify `eval.py` to accept `--override_sequence_length` and
compute MACs via `ptflops` at each length. MLA models should show sublinear
cache growth.

**Script**: `rcotformer/plot_scaling.py`

### 6.3 Figure 4: ADM Pareto (MACs vs PPL at Different Budgets)

**What to plot**: X-axis = MACs, Y-axis = PPL. Each point = a different
`eval_length_factor` threshold.

**Data source**: Output of `eval_adm.py` (Phase 4.2).

**Script**: `rcotformer/plot_adm_pareto.py`

Plot curves for:
1. LN-CoTFormer + ADM (standard attention)
2. MLA-LN-CoTFormer + ADM (compressed attention)

### 6.4 Memory Profiling

**New evaluation**: Peak GPU memory vs sequence length for each model variant.

```python
torch.cuda.reset_peak_memory_stats()
# ... run forward pass ...
peak_mem = torch.cuda.max_memory_allocated() / (1024**3)  # GB
```

Plot this alongside Figure 3 to demonstrate MLA's memory advantage.

---

## Appendix A — File Transfer Manifest

### Files to Copy (25 files)

| Source (relative to CoTFormer/) | Destination (relative to rcotformer/) | Changes |
|---|---|---|
| `main.py` | `main.py` | None |
| `eval.py` | `eval.py` | None |
| `get_ppl_per_mac.py` | `get_ppl_per_mac.py` | None |
| `get_router_weights.py` | `get_router_weights.py` | None |
| `get_router_weights_but.py` | `get_router_weights_but.py` | None |
| `config/__init__.py` | `config/__init__.py` | None |
| `config/base.py` | `config/base.py` | Add MLA args (Phase 5.4) |
| `data/utils.py` | `data/utils.py` | Add alias (Phase 0.2.2) |
| `data/openwebtext2.py` | `data/openwebtext2.py` | Fix num_proc (Phase 0.2.3) |
| `optim/base.py` | `optim/base.py` | None |
| `optim/utils.py` | `optim/utils.py` | Optional: fix e constant |
| `optim/adafactor.py` | `optim/adafactor.py` | None |
| `distributed/__init__.py` | `distributed/__init__.py` | None |
| `distributed/backend.py` | `distributed/backend.py` | None |
| `distributed/single.py` | `distributed/single.py` | None |
| `distributed/ddp.py` | `distributed/ddp.py` | None |
| `models/__init__.py` | `models/__init__.py` | Trim + add MLA (Phase 0.2.1, 5.5) |
| `models/utils.py` | `models/utils.py` | Add RMSNorm (Phase 5.6) |
| `models/base.py` | `models/base.py` | None |
| `models/but_full_depth.py` | `models/but_full_depth.py` | None |
| `models/cotformer_full_depth.py` | `models/cotformer_full_depth.py` | None |
| `models/cotformer_full_depth_lnmid_depthemb.py` | `models/cotformer_full_depth_lnmid_depthemb.py` | None |
| `models/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor.py` | `models/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor.py` | None |
| `models/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute.py` | `models/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute.py` | None |
| `models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py` | `models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py` | None |
| `models/positional_encoders/__init__.py` | `models/positional_encoders/__init__.py` | None |
| `models/positional_encoders/encoder.py` | `models/positional_encoders/encoder.py` | None |
| `models/positional_encoders/rotary.py` | `models/positional_encoders/rotary.py` | None |
| `models/positional_encoders/rotary_utils.py` | `models/positional_encoders/rotary_utils.py` | None |
| `models/caches/__init__.py` | `models/caches/__init__.py` | None |
| `models/caches/cache.py` | `models/caches/cache.py` | None |

### Files to Write from Scratch (5 files)

| File | Purpose | Phase |
|---|---|---|
| `models/mla_cotformer.py` | MLA-LN-CoTFormer model | Phase 5 |
| `eval_adm.py` | ADM evaluation across MAC budgets | Phase 4.2 |
| `plot_pareto.py` | Figure 2 reproduction | Phase 6.1 |
| `plot_scaling.py` | Figure 3 reproduction | Phase 6.2 |
| `plot_adm_pareto.py` | Figure 4 reproduction | Phase 6.3 |

---

## Appendix B — Config Cheat Sheet

### Base 12-Layer Standard Transformer

```bash
--model base --n_layer 12 --n_embd 768 --n_head 12 \
--batch_size 64 --acc_steps 2 --sequence_length 256 \
--iterations 40000 --dataset owt2 --lr 1e-3 \
--weight_decay 0.1 --warmup_percent 0.2 --dropout 0.0 \
--eval_freq 100 --seed 0
```

### CoTFormer 12×N

```bash
--model cotformer_full_depth --n_layer 12 --n_repeat N \
--batch_size 32 --acc_steps 4 --sequence_length 256 \
--depth_random_method uniform_random_range \
--n_layer_begin 0 --n_layer_end 0 --min_repeat N \
# ... same lr/wd/warmup/iterations as above
```

### LN-CoTFormer 12×N

```bash
--model cotformer_full_depth_lnmid_depthemb --n_layer 12 --n_repeat N \
--depth_embedding linear_learned \
--batch_size 32 --acc_steps 4 \
# ... rest same as CoTFormer
```

### Adaptive LN-CoTFormer (Variant B + Router)

```bash
--model adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final \
--n_layer 12 --n_repeat 5 \
--depth_embedding linear_learned \
--depth_random_method uniform_random_range \
--batch_size 16 --acc_steps 8 \
# ... rest same
```

### MLA-LN-CoTFormer (Our Extension)

```bash
--model mla_cotformer --n_layer 12 --n_repeat 5 \
--depth_embedding linear_learned \
--depth_random_method uniform_random_range \
--kv_lora_rank 192 --q_lora_rank 384 \
--qk_rope_head_dim 32 --qk_nope_head_dim 64 --v_head_dim 64 \
--batch_size 16 --acc_steps 8 \
# ... rest same
```

---

## Appendix C — L4 VRAM Budget

NVIDIA L4: 24 GB GDDR6. All estimates assume bfloat16 (2 bytes/param) for
model + activations, float32 (4 bytes/param) for optimizer states.

### Model Parameters (12-layer, 768-dim)

| Component | Params | bf16 Size |
|-----------|--------|-----------|
| Token embedding (50304×768) | 38.6M | 77 MB |
| 12 × (Attn + MLP + 2×LN) | ~85M | 170 MB |
| LM head (tied with embedding) | 0 | 0 |
| **Total model** | ~124M | **248 MB** |

### Optimizer States (AdamW)

AdamW stores 2 fp32 states per parameter: `2 × 124M × 4B = 992 MB ≈ 1 GB`.

### Activation Memory (per micro-batch)

For batch_size=32, seq_len=256, n_layer=12:

| Model | Peak Activations (estimate) |
|-------|---------------------------|
| Standard 12L | ~2 GB |
| CoTFormer 12×2 | ~4 GB (2× for repeats + KV cache) |
| CoTFormer 12×5 | ~9 GB (5× repeats + 5×256 KV cache) |
| Adaptive 12×5 (batch=16) | ~6 GB (token pruning reduces later repeats) |
| MLA 12×5 (batch=16) | ~4 GB (compressed cache + decompression) |

### Total VRAM Budget

| Experiment | Model | Optim | Activations | **Total** | Fits L4? |
|-----------|-------|-------|-------------|-----------|----------|
| Standard 12L (bs=64) | 0.25 | 1.0 | 4.0 | **5.3 GB** | Yes |
| CoTFormer 12×2 (bs=32) | 0.25 | 1.0 | 4.0 | **5.3 GB** | Yes |
| CoTFormer 12×5 (bs=32) | 0.25 | 1.0 | 9.0 | **10.3 GB** | Yes |
| Adaptive 12×5 (bs=16) | 0.25 | 1.0 | 6.0 | **7.3 GB** | Yes |
| MLA 12×5 (bs=16) | 0.24 | 0.95 | 4.0 | **5.2 GB** | Yes |

All experiments fit within L4's 24 GB with significant headroom. If memory
becomes tight at longer sequence lengths (Phase 6.2 scaling experiments), reduce
batch size further or use gradient checkpointing.

---

## Appendix D — MLA Dimensionality Reference

### Standard MHA vs MLA Side-by-Side

```
STANDARD MHA (per layer):
  Input: h_t ∈ R^768

  Q = W_Q · h_t          W_Q ∈ R^(768×768)    → Q ∈ R^768
  K = W_K · h_t          W_K ∈ R^(768×768)    → K ∈ R^768
  V = W_V · h_t          W_V ∈ R^(768×768)    → V ∈ R^768

  Cache per token: K + V = 1536 floats

  Attention: softmax(Q·K^T / √64) · V
  Output: W_O · concat(heads)


MLA (per layer):
  Input: h_t ∈ R^768

  Query path:
    c_Q = RMSNorm(W_qa · h_t)              W_qa ∈ R^(384×768)   → c_Q ∈ R^384
    q_nope = W_qb · c_Q                    W_qb ∈ R^(768×384)   → q_nope ∈ R^768 (12 heads × 64)
    q_rope = RoPE(W_qr · c_Q)              W_qr ∈ R^(384×384)   → q_rope ∈ R^384 (12 heads × 32)
    q = [q_nope ; q_rope]  per head: R^96

  KV path:
    [c_KV ; k_pe] = W_kva · h_t            W_kva ∈ R^(224×768)  → c_KV ∈ R^192, k_pe ∈ R^32
    c_KV = RMSNorm(c_KV)
    [k_nope ; v] = W_kvb · c_KV            W_kvb ∈ R^(1536×192) → per head: k_nope ∈ R^64, v ∈ R^64
    k_rope = RoPE(k_pe)                    shared across heads    → k_rope ∈ R^32
    k = [k_nope ; k_rope]  per head: R^96

  Cache per token: c_KV + k_pe = 192 + 32 = 224 floats  (6.9× compression)

  Attention: softmax(q·k^T / √96) · v
  Output: W_O · concat(heads)
```

### Weight Absorption (Inference-Only Optimization)

At inference time, since there's no nonlinearity between `W_kvb` and the
attention dot product, we can absorb the decompression weight:

```
score = q_nope^T · k_nope = q_nope^T · (W_kvb_k · c_KV)
      = (W_kvb_k^T · q_nope)^T · c_KV

→ Pre-compute q_absorbed = W_kvb_k^T · q_nope, then dot with cached c_KV directly
```

This avoids materializing the full K tensor during generation. Same trick applies
to the value path. **This optimization is NOT needed for training** (where we
materialize full K/V for gradient flow) and can be deferred to a post-training
optimization pass.

---

## Task Assignment Suggestion

| Phase | Owner | Dependencies | Est. GPU-hours |
|-------|-------|-------------|----------------|
| Phase 0-1: Transfer + Container | Any 1 person | None | 0 (CPU only) |
| Phase 2: Data prep | Same person | Phase 1 | 0 (CPU only) |
| Phase 3: Baselines (8 runs) | All 3 in parallel | Phase 2 | 8 × ~12h = 96h |
| Phase 4: ADM router eval | 1 person | Phase 3 adaptive run | ~24h |
| Phase 5: MLA implementation | 1 person (strongest PyTorch) | Phase 0 | 0 (code only) |
| Phase 5: MLA training | 1 person | Phase 5 code + Phase 2 | ~24h |
| Phase 6: Plotting | 1 person | Phases 3-5 results | 0 (CPU only) |

**Critical path**: Phase 0 → Phase 2 → Phase 3 (longest, parallelizable across team).
Phase 5 MLA code can be developed in parallel with Phase 3 baseline training.
