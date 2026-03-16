# Reproducibility Notes

> Known divergences from [CoTFormer (Mohtashami et al., ICLR 2025)][cotformer-paper]
> in our COMP 6258 reproduction. Only items that **differ** from the original are
> documented here — aligned details are omitted.

## Table of Contents

- [1. Train/Val Split Divergence](#1-trainval-split-divergence)
- [2. Token Ordering Within Bins](#2-token-ordering-within-bins)
- [3. Micro-Batch Size (Hardware-Constrained)](#3-micro-batch-size-hardware-constrained)
- [4. RNG State Restoration on DDP Resume (ADM)](#4-rng-state-restoration-on-ddp-resume-adm)
- [5. Base Model changes](#5-base-discrepancy)
- [References](#references)

---

## 1. Train/Val Split Divergence

**Impact:** Negligible (within noise floor of stochastic training)

The original codebase uses a pre-built HuggingFace Arrow dataset
(`the_pile_openwebtext2`, now permanently defunct) with
`train_test_split(test_size=0.0005, seed=2357)`. Our pipeline uses the
identical raw tarball ([`segyges/OpenWebText2`][segyges-mirror]) but
diverges in three independent ways:

### 1a. Document ordering

The pre-built Arrow dataset had a fixed row ordering that determined which
documents landed in train vs. val. We read raw `.jsonl.zst` files in sorted
filename order, which is not guaranteed to match the original Arrow row
ordering. Since the split partitions by row index, different ordering means
different documents in each split.

### 1b. Split RNG implementation

The original uses HF datasets' `train_test_split()`, which internally
generates a permutation via numpy. The exact backend depends on the
`datasets` library version (`RandomState` vs `default_rng`), which is
unrecorded in the original codebase. Our pipeline uses:

```python
rng = np.random.default_rng(2357)
perm = rng.permutation(n_docs)
val_set = set(perm[:n_val].tolist())
```

Even with identical document ordering, the split would differ unless the
exact `datasets` version were matched.

### 1c. Validation set size rounding

The original uses `test_size=0.0005` with HF's internal rounding. We use
`n_val = max(1, round(n_docs * 0.0005))`, which may differ by ±1 document.

### Why the original split is unrecoverable

The pre-built Arrow artifact was permanently disabled on HuggingFace
(PII/safety flags, copyright takedowns). No mirror preserves the Arrow row
ordering — only raw tarball copies exist.

### Impact assessment

The split ratio is 99.95% train / 0.05% val across ~3.4M documents. At
this ratio, any two random partitions produce statistically equivalent
validation sets. Negligible impact on reported perplexity.

---

## 2. Token Ordering Within Bins

**Impact:** None

The original pipeline writes tokens in shuffled order (via
`train_test_split(shuffle=True)` reindexing + 1024 sharded batches). Our
pipeline writes in file-encounter order (sorted filenames, sequential
within each file).

No effect on training — the training loop (`optim/base.py`) samples random
offsets into the memmap file, making physical token ordering irrelevant.

---

## 3. Micro-Batch Size (Hardware-Constrained)

**Impact:** Negligible (statistically equivalent gradients)

The original codebase was developed on A100 80 GB GPUs. Our training runs on
2x NVIDIA L4 (24 GB each) via DDP, which constrains the per-GPU micro-batch
size.

### What changed

The effective batch size is preserved at 128 tokens/step, but the micro-batch
/ accumulation split differs:

| Setting | Paper (assumed A100) | Ours (2x L4 24 GB) |
|---------|---------------------|---------------------|
| `batch_size` (per GPU) | Unspecified (likely 32+) | 8 |
| `acc_steps` (per GPU) | Unspecified | 8 (16 before DDP halving) |
| Effective batch size | 128 | 128 |

### Why it diverges numerically

With loss normalisation `loss = outputs['loss'] / acc_steps`, gradient
accumulation is mathematically equivalent regardless of micro-batch size.
However, floating-point addition is non-associative: accumulating 8
micro-batches of 8 samples produces different rounding than 4 micro-batches
of 16 samples (or any other decomposition). Under bfloat16 autocast, these
rounding differences are amplified compared to float32.

### Impact assessment

The expected gradient is identical. Only the accumulation noise floor changes,
and it is dwarfed by stochastic minibatch sampling noise. No measurable effect
on final perplexity.

### Additional hardware-driven settings

| Setting | Value | Reason |
|---------|-------|--------|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Enabled | Reduces VRAM fragmentation on 24 GB GPUs |
| `find_unused_parameters` (DDP) | `False` | No unused params; avoids extra autograd traversal |
| `--mem` (SLURM) | 128 GB | `--data_in_ram` with 2 DDP workers needs >96 GB system RAM |

---

## 4. RNG State Restoration on DDP Resume (ADM)

**Impact:** None for LN-CoTFormer; potential divergence for ADM training

**Status:** Open — to be resolved before ADM training begins.

### Background

Checkpoint saving (`optim/base.py:207-210`) records four RNG states:
`cpu_rng_state`, `gpu_rng_state`, `numpy_rng_state`, `py_rng_state`.
On resume, these are loaded (`main.py:152-162`) but **never restored** —
the calls in `optim/base.py:63-67` are commented out.

### Why it doesn't affect LN-CoTFormer

The LN-CoTFormer (`cotformer_full_depth_lnmid_depthemb`) has no runtime
randomness: `dropout=0.0` (all `nn.Dropout` are no-ops) and a fixed
repeat loop (`range(1, n_repeat+1)`). The `depth_random_method` arg is
unused by this model variant. Training is fully deterministic modulo cuDNN
kernel selection.

### Why it will affect ADM training

The ADM model (`but_full_depth` / adaptive variants) uses `torch.randint`
for depth sampling during forward passes (via `_predict_depth`), and will
likely train with `dropout > 0`. Both consume CUDA RNG state, so a resume
without RNG restoration produces different dropout masks and depth
schedules than a continuous run.

### DDP bug preventing naive fix

Simply uncommenting lines 63-67 is **incorrect** under DDP. The checkpoint
is saved only by rank 0 (`if distributed_backend.is_master_process()`), so
the stored `gpu_rng_state` belongs to GPU 0. On resume, all ranks load the
same file — rank 1 would receive rank 0's CUDA state, giving both GPUs
identical dropout masks and breaking statistical independence.

### Fix required (before ADM training)

Save per-rank RNG states by writing one checkpoint per rank, or embedding
a dict keyed by rank:

```python
# Save (rank-aware)
rng_states = {f"gpu_rng_state_{rank}": torch.cuda.get_rng_state(rank) for rank in range(world_size)}
# Restore (rank-aware)
torch.cuda.set_rng_state(checkpoint[f"gpu_rng_state_{local_rank}"])
```

Until this is implemented, ADM job chaining across SLURM timeouts will
have non-reproducible dropout/depth sequences at resume boundaries.

## 5. Base Model Changes

discrepancy found between the paper and `experiments.md` file. Paper states they use a batch size of 128 where as the `md` file states 64. I opted to use a size of 64 due to GPU memory constraints. 

---

## References

[cotformer-paper]: https://openreview.net/forum?id=7igPXQFupX
[segyges-mirror]: https://huggingface.co/datasets/segyges/OpenWebText2

1. Mohtashami, A. et al. (2025). *CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference.* [ICLR 2025](https://openreview.net/forum?id=7igPXQFupX)
