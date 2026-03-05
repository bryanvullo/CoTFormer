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
├── main.py              ← Training entry point
├── eval.py              ← Evaluation
├── get_ppl_per_mac.py   ← MACs-aware PPL evaluation
├── config/              ← Argument parsing and defaults
├── data/                ← Dataset loaders (OpenWebText2)
├── models/              ← All model definitions
├── optim/               ← Training loop and optimizers
├── distributed/         ← DDP / single-GPU backends
│
└── iridis/              ← Iridis X cluster operations
    ├── env.sh           ← Per-user scratch paths (source this)
    ├── get_dataset/     ← Download and tokenize OpenWebText2
    │   └── job.sh
    └── gpu_test/        ← GPU smoke test
        ├── job.sh
        └── test_gpu.py
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

We utilize `/scratch/<username>` to store environments and datasets to avoid strict home directory quotas. Choose one of the two setup methods below.

#### Method 1: Iridis Conda Module (Recommended)

This modular approach uses the cluster's pre-installed Conda module while rerouting massive machine learning caches to your scratch space.

**Configure Aliases and Caches:** Create `~/.bash_aliases` and add the following:

```bash
# Reroute ML caches to scratch
export PIP_CACHE_DIR=/scratch/<username>/.cache/pip
export HF_HOME=/scratch/<username>/.cache/huggingface
export WANDB_DIR=/scratch/<username>/.cache/wandb
export CONDA_PKGS_DIRS=/scratch/<username>/.conda/pkgs

# Navigation and Activation Aliases
alias cds="cd /scratch/<username>"
alias cdp="cd ~/CoTFormer"
alias act="module load conda && conda activate /scratch/<username>/cotformer-env"
```

**Link to bashrc:** Append this to your `~/.bashrc`:

```bash
if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi
```

**Initialize Space:** Run `source ~/.bashrc` and create the cache directories in scratch (`mkdir -p ...`).

**Build Environment:**

```bash
module load conda
conda config --add channels defaults
conda config --add channels conda-forge
conda create --prefix /scratch/<username>/cotformer-env python=3.11 -y
```

**Install Packages:** Run `act` to activate the environment, then `pip install -r ~/CoTFormer/requirements.txt`.

#### Method 2: Standalone Miniforge (Wget)

Use this method if you prefer to manage a localized installation of Miniforge entirely within your scratch directory.

**Download and Install:**

```bash
cd /scratch/<username>
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ./mambaforge
```

**Initialize:**

```bash
echo 'source /scratch/<username>/mambaforge/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
```

**Build Environment:**

```bash
conda create -n cotformer-env python=3.11 -y
conda activate cotformer-env
```

**Install Packages:** `pip install -r ~/CoTFormer/requirements.txt`.

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
rsync -avz iridis-x:/scratch/<username>/cotformer/exps/ ./exps/
```

### GPU Partitions

| Partition | GPU | Max time | Access |
|-----------|-----|----------|--------|
| `ecsstudents_l4` | L4 (24GB) | 1 day | ECS undergrads (guaranteed) |
| `a100` | A100 (80GB) | 2d12h | May require approval |
| `scavenger_l4` | L4 (24GB) | 12h | Preemptible, open |
| `scavenger_4a100` | A100 (80GB) | 12h | Preemptible, open |

### Submitting and Monitoring

```bash
ssh iridis-x
cd ~/CoTFormer
sbatch iridis/gpu_test/job.sh      # Submit job
squeue -u $(whoami)                 # Check job status
scancel <job_id>                    # Cancel job
seff <job_id>                       # View post-run efficiency
```

## Dataset Setup

The dataset is [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest/) (17.1M documents, ~28 GB compressed) from EleutherAI's The Pile. Each team member stores data under their own `/scratch/$USER/` directory.

Iridis login nodes **block outbound HTTPS to external hosts**, so the tarball must be downloaded on your local machine and transferred via rsync. The Python tokenizer (`data/openwebtext2.py`) then handles extraction, tokenization, and cleanup on Iridis.

### Step 1: Download locally

On your local machine (WSL / laptop):

```bash
wget -c https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar \
    -P /tmp/
```

This is ~28 GB and supports resume (`-c`) if interrupted.

### Step 2: Transfer to Iridis

```bash
rsync -avz --progress /tmp/openwebtext2.jsonl.zst.tar \
    iridis-x:/scratch/<username>/datasets/openwebtext2/
```

### Step 3: Extract and tokenize on Iridis

```bash
ssh iridis-x
cd ~/CoTFormer
bash iridis/get_dataset/job.sh
```

The job script sources `iridis/env.sh`, activates conda, and calls the Python tokenizer which:

1. **Extracts** the tarball into `$DATA_DIR/openwebtext2/raw/` (`.jsonl.zst` files)
2. **Loads** the raw files via `datasets.load_dataset("json", ...)`
3. **Splits** with `train_test_split(test_size=0.0005, seed=2357)` (matches original CoTFormer)
4. **Tokenizes** with GPT-2 BPE via tiktoken (`num_proc=40`)
5. **Writes** `train.bin` (~8 GB) and `val.bin` (~4 MB) as uint16 memmap files
6. **Cleans up** raw files and tarball automatically (~94 GB freed)

### Step 4: Verify

```bash
ls -lh /scratch/$USER/datasets/openwebtext2/
# Expected: train.bin ~8GB, val.bin ~4MB
```

Once both `.bin` files exist, training jobs can run without internet access.

## Training

All training runs execute `main.py` from the repo root. On Iridis, source `iridis/env.sh` first and pass `--data_dir` and `--results_base_folder` to direct I/O to scratch:

```bash
source iridis/env.sh
python main.py \
    --data_dir "$DATA_DIR" \
    --results_base_folder "$RESULTS_DIR" \
    --model base --n_layer 12 --n_embd 768 --n_head 12 \
    --batch_size 64 --acc_steps 2 --sequence_length 256 \
    --iterations 40000 --dataset owt2 --lr 1e-3 \
    --eval_freq 100 --seed 0
```

For local development (without `--data_dir`), data defaults to `data/datasets/` relative to the repo.

See `experiments.md` for the full experiment matrix.

## Acknowledgements

We gratefully acknowledge the University of Southampton's ECS faculty for granting access to the Iridis X high-performance computing cluster and its GPU resources used for training and evaluation in this project.
