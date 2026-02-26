# COMP 6258 Differentiable Programming and Deep Learning

## Table of Contents

- [References](#references)
- [Repo Structure](#repo-structure)
- [QuickStart](#quickstart)
  - [Building the Container](#building-the-container)
  - [Iridis X](#iridis-x)
  - [Adding a New Iridis Package](#adding-a-new-iridis-package)
- [Training](#training)
- [Acknowledgements](#acknowledgements)

## References

| Paper | We use | Link |
|-------|--------|------|
| **CoTFormer** (Mohtashami et al., ICLR 2025) | Base architecture, Variant B adaptive depth router (Mixture of Repeats with cross-repeat KV cache), evaluation scripts (`get_ppl_per_mac.py`) | [OpenReview](https://openreview.net/forum?id=7igPXQFupX), [GitHub](https://github.com/epfml/CoTFormer) |
| **DeepSeek-V2** (DeepSeek-AI, 2024) | Multi-Head Latent Attention (MLA) -- KV down/up-projection for compressing the cross-repeat cache | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| **RoFormer** (Su et al., 2024) | Rotary Position Embeddings (RoPE), applied post-decompression in MLA integration | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| **The Pile / OpenWebText2** (Gao et al., 2020) | Training dataset (`the_pile_openwebtext2` via HuggingFace) | [arXiv:2101.00027](https://arxiv.org/abs/2101.00027) |
| **Pause Tokens** (Goyal et al., 2023) | Stretch goal -- explicit pause tokens as a baseline against internalized recurrence | [arXiv:2310.02226](https://arxiv.org/abs/2310.02226) |
| **Pre-LN Transformer** (Xiong et al., 2020) | LN-CoTFormer variant uses pre-layer-norm placement studied here | [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) |

## Repo Structure

```
.
├── main.py                      ← Training entry point
├── eval.py                      ← Evaluation
├── get_ppl_per_mac.py           ← MACs-aware PPL evaluation
├── config/                      ← Argument parsing and defaults
├── data/                        ← Dataset loaders (OpenWebText2)
├── models/                      ← All model definitions
├── optim/                       ← Training loop and optimizers
├── distributed/                 ← DDP / single-GPU backends
│
└── iridis/                      ← Cluster deployment config
    ├── rcotformer.def           ← Container recipe (PyTorch + CUDA 12.1)
    ├── rcotformer.def.example   ← Annotated container recipe template
    ├── job.slurm.example        ← Slurm template for new packages
    └── gpu_test/                ← GPU smoke test package
        ├── job.slurm
        └── test_gpu.py
```

Python code lives at the repo root (required by bare imports in `main.py`). All Iridis cluster config lives under `iridis/`. The `.sif` container image is built into `iridis/` and gitignored.

## QuickStart

### Building the Container

Requires WSL2 with Apptainer (`sudo apt install apptainer`). Build once locally:

```bash
sudo apptainer build iridis/rcotformer.sif iridis/rcotformer.def
```

To add Python packages, edit `iridis/rcotformer.def` (see `iridis/rcotformer.def.example` for guidance) and rebuild.

**Local validation** (WSL with an NVIDIA GPU):

```bash
apptainer exec --nv \
    --bind /usr/lib/wsl:/usr/lib/wsl \
    --env LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    iridis/rcotformer.sif python3 iridis/gpu_test/test_gpu.py
```

> **WSL2 note:** The extra `--bind` and `LD_LIBRARY_PATH` are needed because WSL2's `libcuda` shim depends on D3D12/dxcore libraries that `--nv` alone does not inject. This is WSL-only -- Iridis has native NVIDIA drivers.

### Iridis X

#### SSH Access

Add to `~/.ssh/config`:

```
Host iridis-x
    HostName loginX002.iridis.soton.ac.uk
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
```

| Login node | Hardware |
|------------|----------|
| LoginX001 | GPU (4x L4, 64 cores, 1TB RAM) |
| LoginX002 | CPU only (64 cores, 512GB RAM) |
| LoginX003 | GPU-enabled |

Any login node can submit jobs to any partition.

#### Uploading

First time (includes the large `.sif`):

```bash
scp iridis/rcotformer.sif iridis-x:~/dpdl/iridis/
rsync -avz --exclude='*.sif' --exclude='slurm_*' ./ iridis-x:~/dpdl/
```

Subsequent pushes (code only):

```bash
rsync -avz --exclude='*.sif' --exclude='slurm_*' --exclude='data/datasets' \
    ./ iridis-x:~/dpdl/
```

#### Downloading Results

```bash
scp iridis-x:~/dpdl/iridis/gpu_test/slurm_<job_id>.{out,err} iridis/gpu_test/
```

Or sync everything:

```bash
rsync -avz --exclude='*.sif' iridis-x:~/dpdl/iridis/ iridis/
```

#### GPU Partitions

| Partition | GPU | Max time | Access |
|-----------|-----|----------|--------|
| `ecsstudents_l4` | L4 (24GB) | 1 day | ECS undergrads (guaranteed) |
| `a100` | A100 (80GB) | 2d12h | May require approval |
| `scavenger_l4` | L4 (24GB) | 12h | Preemptible, open |
| `scavenger_4a100` | A100 (80GB) | 12h | Preemptible, open |

#### Submitting and Monitoring

```bash
ssh iridis-x
cd ~/dpdl
sbatch iridis/gpu_test/job.slurm      # Submit
squeue -u $(whoami)                    # Job status
scancel <job_id>                       # Cancel
seff <job_id>                          # Post-run efficiency
```

### Adding a New Iridis Package

```bash
mkdir -p iridis/<your_package>
cp iridis/job.slurm.example iridis/<your_package>/job.slurm
```

1. Edit the `job.slurm` -- replace every `<PKG_NAME>` with your directory name, pick a partition, adjust resources.

2. Add your scripts to `iridis/<your_package>/`.

3. Upload and submit:
   ```bash
   rsync -avz iridis/<your_package>/ iridis-x:~/dpdl/iridis/<your_package>/
   ssh iridis-x "cd ~/dpdl && sbatch iridis/<your_package>/job.slurm"
   ```

## Training

All training runs execute `main.py` from the repo root inside the container:

```bash
apptainer exec --nv --unsquash --bind "$PWD" \
    iridis/rcotformer.sif \
    python3 main.py \
    --model base --n_layer 12 --n_embd 768 --n_head 12 \
    --batch_size 64 --acc_steps 2 --sequence_length 256 \
    --iterations 40000 --dataset owt2 --lr 1e-3 \
    --eval_freq 100 --seed 0
```

See `plan.md` for the full experiment matrix and config cheat sheet.

## Acknowledgements

We gratefully acknowledge the University of Southampton's ECS faculty for granting access to the Iridis X high-performance computing cluster and its GPU resources used for training and evaluation in this project.
