# COMP 6258 Differentiable Programming and Deep Learning

## Table of Contents

- [References](#references)
- [Repo Structure](#repo-structure)
- [QuickStart](#quickstart)
  - [Building the Container](#building-the-container)
  - [Iridis X](#iridis-x)
  - [Adding a New Package](#adding-a-new-package)
- [Use Guide](#use-guide)
- [Further Contributions](#further-contributions)
- [Acknowledgements](#acknowledgements)

## References

| Paper | We use | Link |
|-------|--------|------|
| **CoTFormer** (Mohtashami et al., ICLR 2025) | Base architecture, Variant B adaptive depth router (Mixture of Repeats with cross-repeat KV cache), evaluation scripts (`get_ppl_per_mac.py`) | [OpenReview](https://openreview.net/forum?id=7igPXQFupX), [GitHub](https://github.com/epfml/CoTFormer) |
| **DeepSeek-V2** (DeepSeek-AI, 2024) | Multi-Head Latent Attention (MLA) — KV down/up-projection for compressing the cross-repeat cache | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| **RoFormer** (Su et al., 2024) | Rotary Position Embeddings (RoPE), applied post-decompression in MLA integration | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| **The Pile / OpenWebText2** (Gao et al., 2020) | Training dataset (`the_pile_openwebtext2` via HuggingFace) | [arXiv:2101.00027](https://arxiv.org/abs/2101.00027) |
| **Pause Tokens** (Goyal et al., 2023) | Stretch goal — explicit pause tokens as a baseline against internalized recurrence | [arXiv:2310.02226](https://arxiv.org/abs/2310.02226) |
| **Pre-LN Transformer** (Xiong et al., 2020) | LN-CoTFormer variant uses pre-layer-norm placement studied here | [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) |

## Repo Structure

```
.
├── rcotformer.def               ← Container recipe (PyTorch + CUDA 12.1)
├── rcotformer.sif               ← Container image (gitignored)
├── job.slurm.example            ← Slurm template for new packages
├── rcotformer.def.example       ← Annotated container recipe template
│
└── iridis_gpu_test/             ← Package: GPU smoke test
    ├── job.slurm
    └── test_gpu.py
```

Each package is a self-contained directory with its own `job.slurm` and scripts. The `.sif` container image is shared at the project root. All jobs are submitted from `~/dpdl/` on Iridis.

## QuickStart

### Building the Container

Requires WSL2 with Apptainer (`sudo apt install apptainer`). Build once locally:

```bash
sudo apptainer build rcotformer.sif rcotformer.def
```

To add Python packages, edit `rcotformer.def` (see `rcotformer.def.example` for guidance) and rebuild.

**Local validation** (WSL with an NVIDIA GPU):

```bash
apptainer exec --nv \
    --bind /usr/lib/wsl:/usr/lib/wsl \
    --env LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    rcotformer.sif python3 iridis_gpu_test/test_gpu.py
```

> **WSL2 note:** The extra `--bind` and `LD_LIBRARY_PATH` are needed because WSL2's `libcuda` shim depends on D3D12/dxcore libraries that `--nv` alone does not inject. This is WSL-only — Iridis has native NVIDIA drivers.

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

Any login node can submit jobs to any partition — pick whichever is available.

#### Uploading Packages

First time (includes the large `.sif`):

```bash
scp rcotformer.sif iridis-x:~/dpdl/
scp iridis_gpu_test/job.slurm iridis_gpu_test/test_gpu.py iridis-x:~/dpdl/iridis_gpu_test/
```

Subsequent pushes (scripts only — use `rsync` to skip unchanged files):

```bash
rsync -avz --exclude='*.sif' --exclude='slurm_*' \
    iridis_gpu_test/ iridis-x:~/dpdl/iridis_gpu_test/
```

Or with `scp` for individual files:

```bash
scp iridis_gpu_test/job.slurm iridis_gpu_test/test_gpu.py iridis-x:~/dpdl/iridis_gpu_test/
```

#### Downloading Results

Pull Slurm logs for a specific job:

```bash
scp iridis-x:~/dpdl/iridis_gpu_test/slurm_<job_id>.{out,err} iridis_gpu_test/
```

Sync an entire package directory (excludes the `.sif`):

```bash
rsync -avz --exclude='*.sif' iridis-x:~/dpdl/iridis_gpu_test/ iridis_gpu_test/
```

#### GPU Partitions

Run `sinfo` on Iridis to see current state. Key partitions (as of Feb 2026):

| Partition | GPU | Max time | Access |
|-----------|-----|----------|--------|
| `ecsstudents_l4` | L4 (24GB) | 1 day | ECS undergrads (guaranteed) |
| `a100` | A100 (80GB) | 2d12h | May require approval |
| `scavenger_l4` | L4 (24GB) | 12h | Preemptible, open |
| `scavenger_4a100` | A100 (80GB) | 12h | Preemptible, open |

Useful discovery commands:

```bash
sinfo                                    # All partitions and node states
sinfo -p ecsstudents_l4 --Node --long    # Detailed view of a partition
scontrol show partition ecsstudents_l4   # Partition limits
sacctmgr show assoc user=$USER           # Your account name
```

#### Submitting and Monitoring

```bash
ssh iridis-x
cd ~/dpdl
sbatch iridis_gpu_test/job.slurm        # Submit
squeue -u $(whoami)                      # Job status
cat iridis_gpu_test/slurm_<job_id>.out   # Output
scancel <job_id>                         # Cancel
seff <job_id>                            # Post-run efficiency
```

### Adding a New Package

```bash
mkdir -p <your_package>
cp job.slurm.example <your_package>/job.slurm
```

1. Edit `<your_package>/job.slurm` — replace every `<PKG_NAME>` with your directory name, pick a partition, adjust memory and wall time. Use `--gres=gpu:N` without a type specifier (the partition determines the GPU type).

2. Add your Python scripts to `<your_package>/`.

3. Upload and submit:
   ```bash
   rsync -avz <your_package>/ iridis-x:~/dpdl/<your_package>/
   ssh iridis-x "cd ~/dpdl && sbatch <your_package>/job.slurm"
   ```

## Use Guide

## Further Contributions

## Acknowledgements

We gratefully acknowledge the University of Southampton's ECS faculty for granting access to the Iridis X high-performance computing cluster and its GPU resources used for training and evaluation in this project.
