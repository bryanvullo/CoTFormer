# COMP 6258 Differentiable Programming and Deep Learning

## Table of Contents

- [References](#references)
- [Repo Structure](#repo-structure)
- [QuickStart](#quickstart)
  - [SSH Access](#ssh-access)
  - [Environment Setup](#environment-setup)
  - [Uploading and Downloading](#uploading-and-downloading)
  - [GPU Partitions](#gpu-partitions)
  - [Submitting and Monitoring](#submitting-and-monitoring)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
  - [SLURM Job Configuration](#slurm-job-configuration)
  - [Parameter Reference](#parameter-reference)
  - [Model Variants](#model-variants)
- [Acknowledgements](#acknowledgements)

## References

| Paper | We use | Link |
|-------|--------|------|
| **CoTFormer** (Mohtashami et al., ICLR 2025) | Base architecture, Variant B adaptive depth router | [OpenReview](https://openreview.net/forum?id=7igPXQFupX), [GitHub](https://github.com/epfml/CoTFormer) |
| **DeepSeek-V2** (DeepSeek-AI, 2024) | Multi-Head Latent Attention (MLA) | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| **RoFormer** (Su et al., 2024) | Rotary Position Embeddings (RoPE) | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| **The Pile / OpenWebText2** (Gao et al., 2020) | Training dataset | [arXiv:2101.00027](https://arxiv.org/abs/2101.00027) |
| **Pause Tokens** (Goyal et al., 2023) | Stretch goal -- explicit pause tokens | [arXiv:2310.02226](https://arxiv.org/abs/2310.02226) |
| **Pre-LN Transformer** (Xiong et al., 2020) | LN-CoTFormer variant uses pre-layer-norm placement | [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) |

## Repo Structure

```text
.
├── main.py                      ← Training entry point
├── eval.py                      ← Evaluation
├── get_ppl_per_mac.py           ← MACs-aware PPL evaluation (Figure 4)
├── get_router_weights.py        ← Extract ADM router weights from checkpoint
├── analyze_kv_compression.py    ← KV cache SVD analysis for MLA design
├── config/                      ← Argument parsing and defaults
├── data/                        ← Dataset loaders (OpenWebText2)
├── models/                      ← All model definitions
├── optim/                       ← Training loop and optimizers
├── distributed/                 ← DDP / single-GPU backends
│
└── iridis/                      ← Iridis X cluster operations
    ├── env.sh                         ← Shared scratch paths (source this)
    ├── download-dataset/              ← Step 1: Download OWT2 (login node)
    │   └── job.sh
    ├── extract-tokenize-dataset/      ← Step 2: Tokenize (compute node)
    │   └── job.sh
    ├── lncot-train/                   ← Train LN-CoTFormer (Table 2)
    │   └── job.sh
    ├── adm-train/                     ← Train Adaptive LN-CoTFormer (Figure 4)
    │   └── job.sh
    └── lncot-exp4/                    ← Evaluate Figure 4 (post-training)
        └── job.sh
```

Each operation on Iridis lives in its own package directory under `iridis/`. SLURM `.out`/`.err` files land inside the package dir, keeping the source tree clean. Python code lives at the repo root (required by bare imports in `main.py`).

## QuickStart

### SSH Access

Log into Iridis X directly using:

```bash
ssh <username>@loginx001.iridis.soton.ac.uk
```

For easier access, add this block to your local `~/.ssh/config`:

```text
Host iridis-x
    HostName loginX002.iridis.soton.ac.uk
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
```

### Environment Setup

The team shares a single conda environment, dataset, and cache directory under `/scratch/ab3u21/`. All paths are defined in `iridis/env.sh` -- this is the single source of truth.

> **WARNING:** The setup steps below create or rebuild the **shared** environment. Running them affects all team members. Only do this to bootstrap a fresh environment or after a deliberate reset.

**Configure shell for interactive sessions.** Create `~/.bash_aliases`:

```bash
# Source shared env config (all scratch paths + helpers)
source ~/CoTFormer/iridis/env.sh

# Convenience aliases
alias cds="cd /scratch/ab3u21"
alias cdp="cd ~/CoTFormer"
alias act="module load conda && conda activate $CONDA_ENV_PREFIX"
```

**Link to bashrc.** Append this to `~/.bashrc` (if not already present):

```bash
if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi
```

**Build the shared environment** (one-time, affects all team members):

```bash
source ~/.bashrc
module load conda
conda config --add channels defaults
conda config --add channels conda-forge
conda create --prefix /scratch/ab3u21/cotformer-env python=3.11 -y
act
pip install -r ~/CoTFormer/requirements.txt
```

**Update packages** (after `requirements.txt` changes):

```bash
act
pip install -r ~/CoTFormer/requirements.txt
```

### Uploading and Downloading

**First time push:**

```bash
rsync -avz --exclude='slurm_*' ./ iridis-x:~/CoTFormer/
```

**Subsequent pushes (code only):**

```bash
rsync -avz --exclude='slurm_*' --exclude='data/datasets' ./ iridis-x:~/CoTFormer/
```

**Downloading Results:**

```bash
rsync -avz iridis-x:/scratch/ab3u21/job-outputs/ ./job-outputs/
```

### GPU Partitions

| Partition | GPU | Max time | Access |
|-----------|-----|----------|--------|
| `ecsstudents_l4` | L4 (24GB) | 1 day | ECS undergrads (guaranteed) |
| `a100` | A100 (80GB) | 2d12h | May require approval |
| `scavenger_l4` | L4 (24GB) | 12h | Preemptible, open |
| `scavenger_4a100` | A100 (80GB) | 12h | Preemptible, open |

### Submitting and Monitoring

All jobs are launched with `bash` from the repo root. SLURM jobs self-submit via a wrapper that creates a `run_N/` directory for logs before calling `sbatch`.

```bash
ssh iridis-x
cd ~/CoTFormer
bash iridis/<package>/job.sh        # Submit (or run on login node)
squeue -u $(whoami)                 # Check job status
scancel <job_id>                    # Cancel job
seff <job_id>                       # View post-run efficiency
```

SLURM `.out`/`.err` files land in `iridis/<package>/run_N/`, numbered incrementally per run. Non-SLURM outputs (checkpoints, metrics) go to `/scratch/ab3u21/job-outputs/<USER>-<JOB_ID>-<JOB_NAME>/`.

**Email notifications:** SLURM jobs email `<username>@soton.ac.uk` on completion or failure. To override, set `NOTIFY_EMAIL` in `~/.bash_aliases` before sourcing `env.sh`.

## Dataset Setup

The dataset is [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest/) (17.1M documents, ~28 GB compressed) from EleutherAI's The Pile. The original HuggingFace dataset is defunct; we use an [exact mirror on HuggingFace](https://huggingface.co/datasets/segyges/OpenWebText2). The dataset is stored once under `/scratch/ab3u21/datasets/` and shared by all team members.

The pipeline is split into two jobs because only login nodes have internet access, while compute nodes have the memory and CPU needed for extraction and tokenization.

### Step 1: Download (login node)

```bash
ssh iridis-x
cd ~/CoTFormer
bash iridis/download-dataset/job.sh            # interactive prompt
bash iridis/download-dataset/job.sh --bpe      # cache tiktoken BPE vocab only
bash iridis/download-dataset/job.sh --reset    # delete everything and re-download
```

**Fallback** -- if the login node cannot reach `huggingface.co`:

```bash
wget -c "https://huggingface.co/datasets/segyges/OpenWebText2/resolve/main/openwebtext2.jsonl.zst.tar" \
    -P /tmp/
rsync -avz --progress /tmp/openwebtext2.jsonl.zst.tar \
    iridis-x:/scratch/ab3u21/datasets/openwebtext2/
```

### Step 2: Extract and tokenize (compute node)

```bash
cd ~/CoTFormer
bash iridis/extract-tokenize-dataset/job.sh
```

Self-submits to `amd_student` (40 GB RAM, 16 CPUs, 3h). Produces `train.bin` (~8 GB) and `val.bin` (~4 MB) under `/scratch/ab3u21/datasets/openwebtext2/`.

### Verify

```bash
ls -lh /scratch/ab3u21/datasets/openwebtext2/
# Expected: train.bin ~8GB, val.bin ~4MB
```

## Training

All training runs execute `main.py` from the repo root. On Iridis, use the self-submitting job packages under `iridis/`. Each package handles conda activation, GPU allocation, WandB offline mode, and email notifications automatically.

### Training Packages

| Package | Model | Steps | Purpose |
|---------|-------|-------|---------|
| `iridis/lncot-train/` | LN-CoTFormer (24L, 5 repeats) | 40k | Table 2 perplexity baseline |
| `iridis/adm-train/` | Adaptive LN-CoTFormer + Router | 60k | Figure 4 (MACs vs perplexity) |
| `iridis/lncot-exp4/` | Evaluation only | -- | Reproduce Figure 4 curves |

Submit from the repo root on Iridis:

```bash
cd ~/CoTFormer
bash iridis/lncot-train/job.sh     # LN-CoTFormer (Table 2)
bash iridis/adm-train/job.sh       # Adaptive model (Figure 4)
bash iridis/lncot-exp4/job.sh      # Evaluate Figure 4 (after adm-train completes)
```

Both training packages use 2x L4 GPUs with DDP. Adjust `N_GPUS` and `#SBATCH --gres=gpu:N` at the top of each `job.sh` if needed.

### Checkpointing and Job Chaining

Training on L4 GPUs may not finish in a single 24-hour job. The packages are designed for **seamless multi-job chaining**:

1. **Automatic checkpoints** every 2000 steps to `/scratch/ab3u21/exps/` (off home quota)
2. **Auto-resume**: `--use_pretrained auto` finds the latest `ckpt_N.pt` on restart
3. **To continue**: just resubmit the same job — `bash iridis/<package>/job.sh`
4. **Completion signal**: when training finishes, `summary.json` is written to the checkpoint directory

Checkpoint path: `/scratch/ab3u21/exps/owt2/<model_name>_<hyperparams>/`

Each checkpoint is ~1.5 GB (model + AdamW states + RNG states). With `save_checkpoint_freq=2000` over 40k steps, expect ~30 GB of checkpoints per run. Old intermediary checkpoints can be deleted manually after training completes — only `ckpt.pt` (the final checkpoint) is needed for evaluation.

### SLURM Job Configuration

Both training packages have a configuration block at the top of `job.sh`:

```bash
N_GPUS=2          # Must match --gres=gpu:N in SBATCH directives
N_LAYER=24        # 24 for paper reproduction, 12 for faster experiments
N_REPEAT=5        # Number of block repeats
ITERATIONS=40000  # Training steps (60000 for adm-train)
BATCH_SIZE=8      # Per-GPU micro-batch size (max 8 on L4 24GB for 24L×5)
ACC_STEPS=16      # Gradient accumulation steps (DDP halves this)
CKPT_FREQ=2000    # Checkpoint frequency
```

Effective batch size = `BATCH_SIZE * ACC_STEPS` = 128 (matching the paper). DDP divides `ACC_STEPS` by the world size, not `BATCH_SIZE` — each GPU processes the full micro-batch.

> **GPU memory allocator:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in `iridis/env.sh` and applies to all GPU jobs. It reduces VRAM fragmentation on L4 24 GB GPUs. No action needed — it is sourced automatically.

### Manual Training (without job packages)

For local development or custom experiments:

```bash
source iridis/env.sh
python main.py \
    --data_dir "$DATA_DIR" \
    --results_base_folder /scratch/ab3u21/exps \
    --model base --n_layer 12 --n_embd 768 --n_head 12 \
    --batch_size 64 --acc_steps 2 --sequence_length 256 \
    --iterations 40000 --dataset owt2 --lr 1e-3 \
    --eval_freq 100 --seed 0
```

Without `--data_dir`, data defaults to `data/datasets/` relative to the repo. See `experiments.md` for the full experiment matrix.

### Parameter Reference

All parameters are defined in `config/base.py` and passed to `main.py` via command line. Parameters not listed in `experiments.md` or job scripts use the defaults below.

#### Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_size` | int | 32 | Per-GPU micro-batch size. DDP does NOT divide this — each GPU forward-passes the full batch. On L4 (24 GB), max 8 for 24-layer × 5-repeat models; max 16 for 12-layer models. |
| `--acc_steps` | int | 4 | Gradient accumulation steps. DDP halves this (divides by `gcd(acc_steps, world_size)`). Effective batch size = `batch_size × acc_steps` (before DDP adjustment). |
| `--iterations` | int | 25000 | Total optimizer steps (not epochs). Paper uses 40k for LN-CoTFormer (Table 2), 60k for ADM (Section 4.2). |
| `--seed` | int | 2 | Random seed for PyTorch, numpy, and Python `random`. Each DDP rank adds `local_rank` to this. |
| `--data_seed` | int | None | Separate seed for data loading/shuffling. Defaults to `--seed` if not set. |
| `--device` | str | `cuda:0` | Training device. DDP overrides this to `cuda:<local_rank>`. |

#### Optimizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--opt` | str | `adamw` | Optimizer. Choices: `adamw`, `sgd`, `adafactor`. AdamW uses fused CUDA kernel when available (PyTorch 2.0+). |
| `--lr` | float | 1e-3 | Peak learning rate. The scheduler ramps up from `lr/100` during warmup, then decays to `lr × final_div_factor`. |
| `--beta1` | float | 0.9 | AdamW first moment decay. |
| `--beta2` | float | 0.95 | AdamW second moment decay. Note: PyTorch default is 0.999; 0.95 is standard for transformer pretraining. |
| `--weight_decay` | float | 0.1 | Applied to Linear weights only. LayerNorm, Embedding, and bias parameters are excluded (handled in `get_parameter_group_specs()`). |
| `--grad_clip` | float | 0.0 | Max gradient L2 norm. 0.0 = no clipping (gradients are still measured for diagnostics). NanoGPT uses 1.0; the CoTFormer paper does not clip. |
| `--scheduler` | str | `cos` | LR schedule. Choices: `cos` (cosine annealing), `linear`, `none`. Both `cos` and `linear` use `OneCycleLR` with `cycle_momentum=False`. |
| `--warmup_percent` | float | 0.05 | Fraction of `--iterations` spent in linear warmup. Paper uses 0.2 (8000 steps at 40k iterations). |
| `--final_div_factor` | float | 0.05 | Final LR = `lr × final_div_factor`. At default 0.05, the LR decays to 5% of peak. |

#### Dataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | str | `slimpajama` | Dataset name. Choices: `slimpajama`, `wikitext`, `pg19`, `shakespeare-char`, `arxivmath`, `owt2`, `arxiv2000`, `arxiv+wiki`, `openwebtext2`. Use `owt2` for our OpenWebText2 pipeline (same as `openwebtext2` but shorter alias). |
| `--data_dir` | str | None | Root directory for datasets. Our pipeline stores data at `/scratch/ab3u21/datasets/`. If None, defaults to `data/datasets/` relative to the repo. |
| `--data_in_ram` | flag | off | Load the full tokenized dataset (`train.bin`, `val.bin`) into RAM as numpy arrays before training. Faster data loading but uses ~16 GB RAM (×2 for DDP). Recommended on HPC; request `--mem=128G` in SLURM. |
| `--vocab_size` | int | 50304 | Vocabulary size. 50304 = GPT-2's 50257 rounded up to the nearest multiple of 64 for tensor core efficiency. Must match the tokenizer. |
| `--add_additional_space` | flag | off | Unused in current codebase. Legacy parameter. |

#### Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | `base` | Model variant. See [Model Variants](#model-variants) below. |
| `--n_layer` | int | 24 | Number of transformer blocks (total, including prefix/suffix). Paper uses 24 for Table 2 / Figure 4, 12 for Table 1 baselines. |
| `--n_embd` | int | 768 | Hidden dimension (embedding size). All models use 768 to match GPT-2 small. |
| `--n_head` | int | 12 | Number of attention heads. Head dimension = `n_embd / n_head` = 64. |
| `--sequence_length` | int | 512 | Max input sequence length. Paper trains at 256 tokens. Evaluation can use a different length via `--eval_seq_length`. |
| `--dtype` | str | `torch.bfloat16` | Training precision. Choices: `torch.bfloat16`, `torch.float16`. Used inside `torch.amp.autocast()`. Model weights remain in float32; activations are cast to this dtype during forward/backward. |
| `--bias` | bool | False | Whether Linear and LayerNorm layers include bias terms. False follows modern practice (GPT-NeoX, LLaMA). |
| `--dropout` | float | 0.0 | Dropout rate applied after attention, MLP, and embedding. 0.0 for pretraining (paper setting). |
| `--compile` | flag | off | Compile the model with `torch.compile()` (PyTorch 2.0+). Can improve throughput by 10-30% but increases startup time and has known issues with weight tying. |
| `--rmsnorm_eps` | float | 1e-5 | Epsilon for RMSNorm layers (used in some model variants). |
| `--multiple_of` | int | 256 | SwiGLU hidden layer size is rounded up to this multiple. Only applies to model variants that use SwiGLU MLP (not used in standard CoTFormer models, which use GELU MLP). |

#### CoTFormer-Specific

These parameters control the block-repeat (Chain-of-Thought) mechanism that is the core of CoTFormer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_repeat` | int | 2 | Number of times the mid-block stack is repeated. The "thinking depth." Paper uses 5 for 24-layer models (Table 2). Total effective layers = `n_layer_begin + (n_layer - n_layer_begin - n_layer_end) × n_repeat + n_layer_end`. |
| `--n_layer_begin` | int | 0 | Number of prefix layers (not repeated). Paper uses 2 for LN-CoTFormer. These run once before the repeated mid-block. |
| `--n_layer_end` | int | 0 | Number of suffix layers (not repeated). Paper uses 1. These run once after all repeats. |
| `--min_repeat` | int | 1 | Minimum repeat depth during training (for stochastic depth). Used with `--depth_random_method`. For non-adaptive models, set to `n_repeat` (fixed depth). |
| `--depth_random_method` | str | `uniform` | How to sample per-token repeat depth during training. Choices: `uniform` (each token gets a random depth in [`min_repeat`, `n_repeat`]), `uniform_random_range` (sample a random range [a, b] and all tokens use depths in that range), `more_chance_for_zero_too`, `exponential_decay`, `pick_mid_and_recurse`. Paper uses `uniform_random_range` for LN-CoTFormer. |
| `--depth_embedding` | str | None | Depth-aware positional embedding added at each repeat. Choices: `linear_learned` (single learned vector scaled by repeat index — paper default for LN-CoTFormer), `learned` (separate learned embedding per repeat index), None (no depth embedding). |
| `--disable_ln_mid` | flag | off | Replace the inter-repeat LayerNorm (`ln_mid`) with an identity. The "LN" in "LN-CoTFormer" refers to this LayerNorm being **enabled**. Disabling it gives the vanilla CoTFormer variant. |
| `--mod_capacity_factor` | float | 0.6 | Router capacity factor for adaptive (Mixture-of-Depths) models. Controls what fraction of tokens are allowed to continue to the next repeat. Only used by `but_mod_*` model variants. |

#### Adaptive Depth / Universal Transformer

These parameters only apply to specific model variants (PonderNet, adaptive CoTFormer).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--halt_threshold` | float | 0.999 | Halting probability threshold for PonderNet-style models. When cumulative halt probability exceeds this, the token stops iterating. |
| `--depth_predictor_loss_coef` | float | None | Coefficient for the depth predictor auxiliary loss. If None, defaults to 1.0 with `act_penalty=0.03` for backward compatibility. If set explicitly, `act_penalty` equals this value. Currently commented out in most model variants. |
| `--eval_length_factor` | float[] | None | Per-repeat length factors for evaluation (used by `get_ppl_per_mac.py` to generate Figure 4 curves). A list of `n_repeat - 1` floats controlling what fraction of tokens continue at each repeat boundary. Set automatically by the evaluation script. |
| `--training_length_factor` | float[] | None | Same as `eval_length_factor` but applied during training. Rarely used directly. |

#### Positional Encoding & Cache

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--positional_encoder` | str | `rotary` | Position embedding type. Choices: `rotary` (RoPE — default, used by all models), `none`. |
| `--lm_cache` | str | `none` | KV cache type for inference. Currently only `none` is implemented. The cache infrastructure exists but is unused during training. |
| `--attention_window_length` | int | None | Sliding window attention size. None = full causal attention (default). Incompatible with Flash Attention. |
| `--mem_cache_size` | int | None | Size of the external memory cache (unused — `lm_cache=none`). |
| `--cache_topk` | int | 1 | Top-k retrieval for cache (unused). |
| `--cache_selection_method` | str | `per_token_and_head` | Cache selection strategy (unused). |
| `--cache_indexing_scheme` | str | `by_total` | Cache indexing strategy (unused). |
| `--track_cache_attention` | flag | off | Log cache attention patterns (unused). |
| `--allow_cache_during_training` | flag | off | Normally raises an error if cache is active during training. This flag overrides that safety check. |

#### Evaluation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--eval_freq` | int | 200 | Evaluate every N optimizer steps. Logs loss, perplexity, accuracy, and gradient norms. Uses 24 val batches for intermediate evals, full val set at the final iteration. |
| `--eval_seq_prefix` | str | `Once upon a time` | Prompt prefix for text generation samples logged to WandB (currently disabled in code — the `if False:` guard at `optim/base.py:183`). |
| `--eval_seq_length` | int | None | Override sequence length at evaluation time (used by `eval.py`). Defaults to `--sequence_length`. |
| `--eval_sample_size` | int | None | Total number of tokens to evaluate (used by `eval.py`). None = use entire val set. |

#### Logging & Checkpoints

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--wandb` | flag | off | Enable Weights & Biases logging. On HPC, combine with `WANDB_MODE=offline` (set automatically in job scripts). |
| `--wandb_project` | str | `my-project` | WandB project name. Our job scripts use `rcotformer`. |
| `--wandb_entity` | str | None | WandB team/entity name. None = personal account. |
| `--results_base_folder` | str | `./exps` | Root directory for checkpoints and experiment outputs. On Iridis, use `/scratch/ab3u21/exps` (off home quota). |
| `--save_checkpoint_freq` | int | None | Save intermediate checkpoints every N steps. None = only save final checkpoint. Our job scripts use 2000. Each checkpoint is ~2 GB (model + AdamW states + RNG). |
| `--remove_intermediary_checkpoints_at_end` | flag | off | Delete intermediate `ckpt_N.pt` files after training completes, keeping only `ckpt.pt`. |
| `--use_pretrained` | str | `auto` | Resume from checkpoint. `auto` finds the latest `ckpt_N.pt` in the output directory (enables seamless job chaining). Can also be a specific filename or None (train from scratch). |
| `--exp_name` | str | None | Override the auto-generated experiment directory name. If None, a name is built from the model, lr, batch size, sequence length, and all non-default parameters. |
| `--run_prefix` | str | None | Prepended to the auto-generated experiment name. Useful for grouping related runs. |

#### Distributed Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--distributed_backend` | str | None | Distributed backend. Choices: `nccl` (multi-GPU DDP), None (single GPU). Set automatically by `torchrun` in job scripts. |
| `--config_format` | str | `base` | Config parser format. Only `base` is implemented. |

#### Derived / Internal

These are computed at runtime, not set directly:

| Behaviour | How it works |
|-----------|-------------|
| **Effective batch size** | `batch_size × acc_steps` (before DDP). DDP divides `acc_steps` by `gcd(acc_steps, world_size)`, keeping per-GPU `batch_size` unchanged. |
| **Experiment directory** | `<results_base_folder>/<dataset>/<model>/<exp_name>/`. Auto-resume and checkpoint saving use this path. |
| **Auto-skip completed** | If `summary.json` exists in the experiment directory, `main.py` exits immediately (prevents re-running finished experiments). |
| **TF32 matmul** | Always enabled (`torch.backends.cuda.matmul.allow_tf32 = True`). Provides ~2x speedup on Ampere+ GPUs with negligible precision loss. |

### Model Variants

The `--model` parameter selects which architecture to train. Each variant is a separate file in `models/`.

| `--model` value | Paper equivalent | Key features |
|-----------------|-----------------|--------------|
| `base` | Standard Transformer | No block repetition. Baseline for Table 1. |
| `but_full_depth` | BUT (Block-Universal Transformer) | Repeats all layers, per-token stochastic depth. Table 1. |
| `cotformer_full_depth` | CoTFormer (vanilla) | Repeats mid-blocks with cross-repeat KV accumulation. No inter-repeat LN. |
| `cotformer_full_depth_lnmid_depthemb` | **LN-CoTFormer** | + inter-repeat LayerNorm + learned depth embedding. Table 2. Our primary model. |
| `but_mod_efficient_sigmoid_lnmid_depthemb_random_factor` | Adaptive BUT | Mixture-of-Depths router with sigmoid gating. |
| `adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final` | **Adaptive LN-CoTFormer (ADM)** | LN-CoTFormer + per-repeat router. Figure 4. Train for 60k steps. |
| `but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute` | ADM (MAC evaluation) | Same as ADM but with MAC computation hooks for `get_ppl_per_mac.py`. |
| `pondernet` | PonderNet | Halting-based adaptive depth (Banino et al.). Uses `--halt_threshold`. |
| `but_halting_freeze_input_on_stop` | PonderNet + freeze | PonderNet variant that freezes token representations after halting. |

### Training Diagnostics

The training loop logs gradient norms alongside loss/perplexity metrics. These help diagnose the "benign spikes" documented in the paper for LN-CoTFormer (Section 3.3).

**Logged every `eval_freq` steps** (printed to SLURM stdout + WandB):
- `grad_norm`: current gradient L2 norm
- `max_grad_norm`: maximum gradient norm since last eval
- `mean_grad_norm`: mean gradient norm since last eval
- Spike warnings when `max_grad_norm > 5x mean`

**WandB sync** (after job completes, from login node):

```bash
wandb sync /scratch/ab3u21/.cache/wandb/<offline-run-*>
```

### KV Compression Analysis (for MLA Extension)

After training the LN-CoTFormer, run the KV compression analysis to inform MLA design:

```bash
# Weight-level analysis (fast, no GPU needed):
python analyze_kv_compression.py --checkpoint /scratch/ab3u21/exps/owt2/<model_dir>/ \
    --target-rank 192

# Full analysis with activation-level SVD (needs GPU + data):
python analyze_kv_compression.py --checkpoint /scratch/ab3u21/exps/owt2/<model_dir>/ \
    --compute-activations --data_dir /scratch/ab3u21/datasets --target-rank 192
```

Outputs per-layer effective rank plots and determines whether the planned `kv_lora_rank=192` is sufficient for each layer.

## Acknowledgements

We gratefully acknowledge the University of Southampton's ECS faculty for granting access to the Iridis X high-performance computing cluster and its GPU resources used for training and evaluation in this project.
