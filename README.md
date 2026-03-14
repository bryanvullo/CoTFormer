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

### Configuration

Both training packages have a configuration block at the top of `job.sh`:

```bash
N_GPUS=2          # Must match --gres=gpu:N in SBATCH directives
N_LAYER=24        # 24 for paper reproduction, 12 for faster experiments
N_REPEAT=5        # Number of block repeats
ITERATIONS=40000  # Training steps (60000 for adm-train)
BATCH_SIZE=32     # Per-GPU micro-batch size
ACC_STEPS=4       # Gradient accumulation steps
CKPT_FREQ=2000    # Checkpoint frequency
```

Effective batch size = `BATCH_SIZE * ACC_STEPS` = 128 (matching the paper).

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

Without `--data_dir`, data defaults to `data/datasets/` relative to the repo.

See `experiments.md` for the full experiment matrix.

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
