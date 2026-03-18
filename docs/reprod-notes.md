# Reproducibility Notes

> Reproduction fidelity log for [CoTFormer (Mohtashami et al., ICLR 2025)][cotformer-paper]
> in our COMP 6258 reproduction. Covers both successful replications and known
> divergences.

## Table of Contents

- [Part A: Successful Replications](#part-a-successful-replications)
  - [A1. LN-CoTFormer Perplexity (Table 2)](#a1-ln-cotformer-perplexity-table-2)
  - [A2. Architecture Depth](#a2-architecture-depth)
  - [A3. Training Stability](#a3-training-stability)
  - [A4. Effective Batch Size](#a4-effective-batch-size)
- [Part B: Known Divergences](#part-b-known-divergences)
  - [B1. Train/Val Split Divergence](#b1-trainval-split-divergence)
  - [B2. Token Ordering Within Bins](#b2-token-ordering-within-bins)
  - [B3. Micro-Batch Size (Hardware-Constrained)](#b3-micro-batch-size-hardware-constrained)
  - [B4. RNG State Restoration on DDP Resume (ADM)](#b4-rng-state-restoration-on-ddp-resume-adm)
  - [B5. Base Model Batch Size](#b5-base-model-batch-size)
- [References](#references)

---

# Part A: Successful Replications

## A1. LN-CoTFormer Perplexity (Table 2)

**Result:** Reproduced within +0.02 PPL of the paper.

| Metric | Paper (Table 2) | Ours | Delta |
|--------|----------------|------|-------|
| Val perplexity | ~24.11 | 24.13 | +0.02 |
| Val accuracy | — | 0.4129 | — |
| Val loss | — | 3.183 | — |

Trained for 40,000 steps on 2x L4 24 GB (DDP), OWT2 dataset, cosine LR
schedule (peak 1e-3, final ~2e-4). Wall time: 23h 26m across 2 SLURM
submissions with auto-resume from checkpoint.

This confirms that our data pipeline, architecture, and training loop
faithfully reproduce the paper's non-adaptive LN-CoTFormer result, despite
the divergences documented in Part B.

---

## A2. Architecture Depth

**Result:** Exact match.

The paper's 24-layer LN-CoTFormer uses 2 prefix + 21 mid (repeated 5x) + 1
suffix = 108 effective layers. Our implementation reports `avg_depth=108.000`
consistently across all 40,000 training steps, confirming the block structure
and repeat loop are correct.

---

## A3. Training Stability

**Result:** No gradient spikes observed.

The paper (Section 3.3) documents "benign spikes" during LN-CoTFormer
training. Our run with gradient norm tracking shows:

| Signal | Last 1000 steps |
|--------|----------------|
| Gradient norm range | 0.49 -- 0.54 |
| Max gradient norm | 0.83 (isolated, at step 39300) |
| Spike threshold (5x mean) | Never triggered |

The cosine learning rate decayed smoothly from 1e-3 to ~2e-4. Loss curves
show stable convergence with normal per-eval variance (PPL 22.4 -- 25.7 in
final 1000 steps; this is expected from the 24-batch eval sample size).

---

## A4. Effective Batch Size

**Result:** Exact match at 128.

Despite using smaller micro-batches than the original (8 vs likely 32+), the
effective batch size of 128 is preserved via gradient accumulation. DDP
correctly halves `acc_steps` (16 -> 8 per GPU) while keeping `batch_size` (8)
per GPU: 8 x 8 x 2 = 128.

---

# Part B: Known Divergences

## B1. Train/Val Split Divergence

**Impact:** Negligible (within noise floor of stochastic training)

The original codebase uses a pre-built HuggingFace Arrow dataset
(`the_pile_openwebtext2`, now permanently defunct) with
`train_test_split(test_size=0.0005, seed=2357)`. Our pipeline uses the
identical raw tarball ([`segyges/OpenWebText2`][segyges-mirror]) but
diverges in three independent ways:

### B1a. Document ordering

The pre-built Arrow dataset had a fixed row ordering that determined which
documents landed in train vs. val. We read raw `.jsonl.zst` files in sorted
filename order, which is not guaranteed to match the original Arrow row
ordering. Since the split partitions by row index, different ordering means
different documents in each split.

### B1b. Split RNG implementation

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

### B1c. Validation set size rounding

The original uses `test_size=0.0005` with HF's internal rounding. We use
`n_val = max(1, round(n_docs * 0.0005))`, which may differ by +/-1 document.

### Why the original split is unrecoverable

The pre-built Arrow artifact was permanently disabled on HuggingFace
(PII/safety flags, copyright takedowns). No mirror preserves the Arrow row
ordering — only raw tarball copies exist.

### Impact assessment

The split ratio is 99.95% train / 0.05% val across ~3.4M documents. At
this ratio, any two random partitions produce statistically equivalent
validation sets. Negligible impact on reported perplexity.

---

## B2. Token Ordering Within Bins

**Impact:** None

The original pipeline writes tokens in shuffled order (via
`train_test_split(shuffle=True)` reindexing + 1024 sharded batches). Our
pipeline writes in file-encounter order (sorted filenames, sequential
within each file).

No effect on training — the training loop (`optim/base.py`) samples random
offsets into the memmap file, making physical token ordering irrelevant.

---

## B3. Micro-Batch Size (Hardware-Constrained)

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

## B4. RNG State Restoration on DDP Resume (ADM)

**Impact:** None for LN-CoTFormer; minor divergence for ADM training at
resume boundaries

### Background

Checkpoint saving (`optim/base.py:207-210`) records four RNG states:
`cpu_rng_state`, `gpu_rng_state`, `numpy_rng_state`, `py_rng_state`.
On resume, these are loaded (`main.py:152-162`) but **never restored** —
the calls in `optim/base.py:63-67` are commented out.

### Why it doesn't affect LN-CoTFormer

The LN-CoTFormer (`cotformer_full_depth_lnmid_depthemb`) has no runtime
randomness: `dropout=0.0` (all `nn.Dropout` are no-ops) and a fixed
repeat loop (`range(1, n_repeat+1)`). Training is fully deterministic modulo
cuDNN kernel selection.

### Why it affects ADM training

The ADM model (`adaptive_cotformer_..._single_final`) generates random
`length_factors` via `torch.rand()` at every training step (forward pass,
lines 400-431 of the model file). These factors control per-repeat capacity
and are consumed by the router to determine token routing. Although
`dropout=0.0` in our config, the `torch.rand()` calls consume CUDA RNG state,
so a resume without RNG restoration produces different capacity factor
sequences than a continuous run.

Note: the original reprod-notes incorrectly referenced `_predict_depth` and
`torch.randint` — our specific model variant uses `torch.rand()` for
`length_factors`, not `torch.randint` for depth prediction.

### DDP complication

Simply uncommenting lines 63-67 is **incorrect** under DDP. The checkpoint
is saved only by rank 0 (`if distributed_backend.is_master_process()`), so
the stored `gpu_rng_state` belongs to GPU 0. On resume, all ranks load the
same file — rank 1 would receive rank 0's CUDA state, giving both GPUs
identical random sequences and breaking statistical independence.

### Practical impact

At 60k steps with ~24h SLURM wall limit, the ADM requires 2-3 submissions.
Each resume boundary produces a different `length_factors` sequence than a
continuous run would. Since `length_factors` are random (uniform, sorted
descending) and the router learns against them, the effect is equivalent to a
small random perturbation in the training curriculum. Unlikely to measurably
affect final model quality but means exact bit-for-bit reproducibility across
job boundaries is not achievable.

---

## B5. Base Model Batch Size

**Impact:** Minor

Discrepancy found between the paper and `experiments.md` file. Paper states
they use a batch size of 128 whereas the `md` file states 64. We opted to
use a size of 64 due to GPU memory constraints.

---

## References

[cotformer-paper]: https://openreview.net/forum?id=7igPXQFupX
[segyges-mirror]: https://huggingface.co/datasets/segyges/OpenWebText2

1. Mohtashami, A. et al. (2025). *CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference.* [ICLR 2025](https://openreview.net/forum?id=7igPXQFupX)
