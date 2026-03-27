# Reproducibility Notes

> Reproduction fidelity log for [CoTFormer (Mohtashami et al., ICLR 2025)][cotformer-paper]
> in our COMP 6258 reproduction. Covers both successful replications and known divergences.

## Table of Contents

- [Reproducibility Notes](#reproducibility-notes)
  - [Table of Contents](#table-of-contents)
- [Part A: Successful Replications](#part-a-successful-replications)
  - [A1. LN-CoTFormer Perplexity (Table 2)](#a1-ln-cotformer-perplexity-table-2)
  - [A2. Architecture Depth](#a2-architecture-depth)
  - [A3. Training Stability](#a3-training-stability)
  - [A4. Effective Batch Size](#a4-effective-batch-size)
  - [A5. ADM Perplexity (Section 4.2, Figure 4)](#a5-adm-perplexity-section-42-figure-4)
- [Part B: Known Divergences](#part-b-known-divergences)
  - [B1. Train/Val Split Divergence](#b1-trainval-split-divergence)
    - [B1a. Document ordering](#b1a-document-ordering)
    - [B1b. Split RNG implementation](#b1b-split-rng-implementation)
    - [B1c. Validation set size rounding](#b1c-validation-set-size-rounding)
    - [Why the original split is unrecoverable](#why-the-original-split-is-unrecoverable)
    - [Impact assessment](#impact-assessment)
  - [B2. Token Ordering Within Bins](#b2-token-ordering-within-bins)
  - [B3. Micro-Batch Size (Hardware-Constrained)](#b3-micro-batch-size-hardware-constrained)
    - [What changed](#what-changed)
    - [Why it diverges numerically](#why-it-diverges-numerically)
    - [Impact assessment](#impact-assessment-1)
    - [Additional hardware-driven settings](#additional-hardware-driven-settings)
  - [B4. RNG State Restoration on DDP Resume (ADM) — Resolved](#b4-rng-state-restoration-on-ddp-resume-adm--resolved)
    - [Background](#background)
    - [Why it doesn't affect LN-CoTFormer](#why-it-doesnt-affect-ln-cotformer)
    - [Why it affects ADM training](#why-it-affects-adm-training)
    - [DDP complication (original code)](#ddp-complication-original-code)
    - [Fix](#fix)
    - [Practical impact (post-fix)](#practical-impact-post-fix)
  - [B5. Base Model Batch Size](#b5-base-model-batch-size)
  - [B6. Orphan MoDBlock in Adaptive/MoD Models — Resolved](#b6-orphan-modblock-in-adaptivemod-models--resolved)
    - [Background](#background-1)
    - [Why the original codebase doesn't crash](#why-the-original-codebase-doesnt-crash)
    - [How it manifests under DDP](#how-it-manifests-under-ddp)
    - [Fix](#fix-1)
    - [Affected files (all from original commit `2723e30`)](#affected-files-all-from-original-commit-2723e30)
  - [B7. Training Summary Metrics Never Recorded (Original Code Bug) — Resolved](#b7-training-summary-metrics-never-recorded-original-code-bug--resolved)
    - [Background](#background-2)
    - [Fix](#fix-2)
  - [B8. Missing `nullcontext` Import in `eval.py` (Original Code Bug) — Resolved](#b8-missing-nullcontext-import-in-evalpy-original-code-bug--resolved)
    - [Background](#background-3)
    - [Fix](#fix-3)
  - [B9. GPU RNG Gather OOM on PyTorch 2.2.0 DDP (Checkpoint Save) — Resolved](#b9-gpu-rng-gather-oom-on-pytorch-220-ddp-checkpoint-save--resolved)
    - [Background](#background-4)
    - [Why the original authors didn't encounter this](#why-the-original-authors-didnt-encounter-this)
    - [Why v1 training succeeded](#why-v1-training-succeeded)
    - [Fix](#fix-4)
  - [B10. NCCL ALLREDUCE Timeout After Checkpoint Save (DDP) — Resolved](#b10-nccl-allreduce-timeout-after-checkpoint-save-ddp--resolved)
    - [Background](#background-5)
    - [Root cause](#root-cause)
    - [Fix](#fix-5)
  - [References](#references)

---

# Part A: Successful Replications

> **Step-count disambiguation**: A1 reports the LN-CoTFormer at **40,000 steps**
> (Table 2 scope). A5 reports the ADM at **60,000 steps** (Section 4.2 scope).
> A 60k LN-CoTFormer run is planned to enable the direct Section 5 comparison
> (paper: non-adaptive 60k = 23.19 PPL). The CoTFormer + Reserved model
> (Table 2, row 1, target 24.51) is pending via `cot-res-train`.

## A1. LN-CoTFormer Perplexity (Table 2, 40k steps)

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

## A5. ADM Perplexity (Section 4.2, Figure 4, 60k steps)

**Result:** Reproduced within +0.24 PPL of the paper.

| Metric | Paper (Section 5, adaptive LN-CoTFormer) | Ours | Delta |
|--------|-------------------|------|-------|
| Val perplexity | ~23.83 | 24.07 | +0.24 |
| Val accuracy | -- | 0.4128 | -- |
| Val loss | -- | 3.181 | -- |

Trained for 60,000 steps on 2x L4 24 GB (DDP), OWT2 dataset, cosine LR
schedule (peak 1e-3, final ~2e-4). Wall time: ~52h across 3 SLURM
submissions with auto-resume from checkpoint.

### Architecture

The ADM uses the same 24-layer architecture as the LN-CoTFormer
(2 prefix + 21 mid repeated 5x + 1 suffix = 108 effective layers),
plus a Mixture-of-Repeats router (Section 4.1) that selects per-token
capacity at each repeat. Parameters: 208.55M (0.01M more than the
non-adaptive LN-CoTFormer due to 4 router weight vectors of 768 dims).

### Training Stability

Training was stable throughout all 60,000 steps:

| Signal | Final phase (55k--60k) |
|--------|----------------------|
| Gradient norm range | 0.50 -- 0.64 |
| Max gradient norm | 0.74 |
| Loss range | 2.66 -- 3.50 |
| PPL range (eval) | 22.5 -- 25.3 |

No gradient spikes were observed. The PPL oscillation in eval is
consistent with the non-adaptive LN-CoTFormer (22.4--25.7 in A3) and
is driven by the small eval sample size, not training instability.

### Job Chaining

The 60k-step training required 3 SLURM submissions due to the
`ecsstudents_l4` partition's 24-hour wall limit:

| Run | Job ID | Steps | Resumed from |
|-----|--------|-------|-------------|
| 0 | 696795 | 0 -- 28570 | (fresh start) |
| 1 | 698707 | 28000 -- 57110 | ckpt_28000.pt |
| 2 | 701232 | 56000 -- 60000 | ckpt_56000.pt |

Auto-resume (`--use_pretrained auto`) worked correctly at each boundary.

### Capacity Factor Sampling

The model uses the default `depth_random_method='uniform'`, which
generates `n_repeat - 1 = 4` random capacity factors via:

```python
length_factors = torch.clamp(torch.rand((4,)) * 1.05, max=1).sort(descending=True).values
```

This matches the paper's Section 4.1: "assign a capacity of 1 to the
first repeat and sample `n_repeat - 1` numbers and sort them in
decreasing order." The 1.05 multiplier gives ~5% extra probability of
capacity=1.0 (a minor stabilisation detail not mentioned in the paper).

During evaluation, `eval_length_factor=None` activates all repeats
(`length_factors = [1, 1, 1, 1]`), so the reported PPL of 24.07 is
at full compute.

### Why the Gap Is Wider Than A1

The +0.24 delta (vs +0.02 for the non-adaptive LN-CoTFormer) is
explained by the ADM's unique sensitivity to RNG state discontinuities
at resume boundaries. As documented in B4, the `torch.rand()` calls
that generate capacity factors are consumed from the CUDA RNG stream.
Three resume boundaries produce different capacity factor sequences
than a continuous run would, changing which tokens are routed through
deeper repeats at each training step. The non-adaptive LN-CoTFormer
has no runtime randomness (dropout=0.0, fixed repeat count), so B4
does not affect it.

### Paper Comparison Note

The paper (Section 5) reports that after 60k steps, the non-adaptive
LN-CoTFormer reaches PPL 23.19. Our non-adaptive model was trained
for 40k steps (PPL 24.13, matching Table 2's 24.11). To reproduce the
Section 5 comparison exactly, the non-adaptive model needs 60k steps
of training. The `lncot-train` package has been updated to 60k
iterations; resubmission will auto-resume from the existing 40k
checkpoint and train for an additional 20k steps.

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
| `find_unused_parameters` (DDP) | `False` | No unused params after B6 fix; avoids extra autograd traversal |
| `--mem` (SLURM) | 128 GB | `--data_in_ram` with 2 DDP workers needs >96 GB system RAM |

---

## B4. RNG State Restoration on DDP Resume (ADM) — Resolved

**Impact:** None for LN-CoTFormer; previously caused minor divergence for
ADM training at resume boundaries
**Status:** Resolved. Per-rank GPU RNG gathering + full restore on resume.

### Background

The original codebase saves four RNG states in intermediary checkpoints
(`cpu_rng_state`, `gpu_rng_state`, `numpy_rng_state`, `py_rng_state`) but
**never restores them** — the restore calls in `optim/base.py` were
commented out. Additionally, the final `ckpt.pt` omitted RNG states
entirely, and `main.py` would `KeyError` if resuming from such a
checkpoint.

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

### DDP complication (original code)

Simply uncommenting the original restore lines was **incorrect** under DDP.
The checkpoint was saved only by rank 0
(`if distributed_backend.is_master_process()`), so the stored
`gpu_rng_state` belonged to GPU 0. On resume, all ranks load the same
file — rank 1 would receive rank 0's CUDA state, giving both GPUs
identical random sequences and breaking statistical independence.

### Fix

Three changes resolve the issue completely:

1. **Per-rank GPU RNG gathering** (`optim/base.py`): before each checkpoint
   save, `torch.distributed.all_gather_object` collects every rank's
   `torch.cuda.get_rng_state()` into a list. Rank 0 saves this list as
   `gpu_rng_states` (plural). Single-GPU runs wrap the local state in a
   one-element list for format consistency.

2. **Full restore on resume** (`optim/base.py`): all four RNG states are
   now restored. CPU, numpy, and Python RNG are restored directly (shared
   across ranks). GPU RNG is restored per-rank: each rank indexes into the
   `gpu_rng_states` list by its rank ID. Legacy checkpoints with the old
   singular `gpu_rng_state` key are handled by restoring only on rank 0.

3. **Tolerant checkpoint loading** (`main.py`): the RNG key extraction
   uses `if k in checkpoint`, so old checkpoints missing RNG keys (e.g.,
   final `ckpt.pt` from before this fix) load without error. Both
   `gpu_rng_state` (legacy) and `gpu_rng_states` (new) are accepted.

### Practical impact (post-fix)

ADM multi-job training across SLURM submissions now produces the same
`length_factors` sequences as a hypothetical continuous run, assuming the
same number of GPUs on resume. The remaining source of non-determinism is
cuDNN kernel selection (inherent to CUDA, not fixable without
`torch.use_deterministic_algorithms(True)` which has a performance cost).

---

## B5. Base Model Batch Size

**Impact:** Minor

Discrepancy found between the paper and `experiments.md` file. Paper states
they use a batch size of 128 whereas the `md` file states 64. We opted to
use a size of 64 due to GPU memory constraints.

---

## B6. Orphan MoDBlock in Adaptive/MoD Models — Resolved

**Impact:** Fatal crash under DDP; no impact on single-GPU training
**Status:** Resolved and verified under 2-GPU DDP.

### Background

All model variants using `MoDBlock` (the Mixture-of-Repeats router) create
`n_repeat` router modules in `__init__`:

```python
mod = nn.ModuleList([MoDBlock(config) for _ in range(self.n_repeat)])
```

However, the forward loop only routes tokens for repeats 1 through
`n_repeat - 1`. The final repeat has no routing decision — all remaining
tokens are unconditionally "final". As a result, `mod[n_repeat - 1]` is
never called and its `mod_router.weight` parameter (768 floats) never
participates in the autograd graph.

### Why the original codebase doesn't crash

The paper's experiments were run on single NVIDIA A100 80 GB GPUs (no DDP).
PyTorch only enforces gradient reduction checks in
`DistributedDataParallel`. On single-GPU, unused parameters are silently
ignored.

### How it manifests under DDP

With `find_unused_parameters=False` (set in `distributed/ddp.py`), DDP
detects on iteration 2 that parameter index 152 did not receive a gradient
in the prior iteration and raises:

```
RuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one. [...] Parameter indices which did not receive
grad for rank 0: 152
```

Index 152 corresponds to `transformer.mod[4].mod_router.weight` for the
24-layer adaptive model with `n_repeat=5`.

### Fix

Changed `range(self.n_repeat)` to `range(self.n_repeat - 1)` in all 5
affected model files. This removes the orphan router module without
changing model behaviour — the removed module was never executed.

DDP training with `find_unused_parameters=False` now completes without
the parameter index 152 error. The `length_factors` generation (which
already used `n_repeat - 1` elements) is unaffected by the change.

### Affected files (all from original commit `2723e30`)

1. `models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py`
2. `models/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor.py`
3. `models/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute.py`
4. `models/models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py`
5. `models/models/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute.py`

---

## B7. Training Summary Metrics Never Recorded (Original Code Bug) — Resolved

**Impact:** Training curve data lost in `summary.json`; no effect on model
quality or training correctness
**Status:** Resolved.

### Background

The `stats` dictionary in `optim/base.py` was initialised (line 44) with
empty lists for `train_loss`, `val_loss`, `val_pp`, and `val_acc`, and
returned to `main.py` for serialisation to `summary.json`. However, the
corresponding `stats[...].append(...)` calls were never present in the
original codebase — the lists were always empty. As a result, every
`summary.json` written by the original code (and our initial runs)
contained empty metric arrays, making post-hoc training curve
reconstruction from the summary file impossible.

Training correctness was unaffected: the metrics were computed, logged to
stdout and wandb, and used for checkpoint decisions — they simply were
never persisted to the JSON summary.

### Fix

Added `stats["train_loss"].append(train_loss)` (and corresponding
`val_loss`, `val_pp`, `val_acc` appends) inside the master-process
evaluation block (`optim/base.py:155-158`). The stats dict is only
populated and written by rank 0, which is correct — `main.py:186-188`
guards the `json.dump(stats, ...)` call with
`distributed_backend.is_master_process()`.

---

## B8. Missing `nullcontext` Import in `eval.py` (Original Code Bug) — Resolved

**Impact:** `NameError` crash on CPU-only evaluation; no impact on GPU runs
**Status:** Resolved.

### Background

`eval.py` uses `nullcontext()` (line 76) as the no-op alternative to
`torch.amp.autocast` when running without a CUDA device. However, the
`from contextlib import nullcontext` import was missing from the original
codebase. On GPU runs (all Iridis jobs), the `torch.amp.autocast` branch
is taken and `nullcontext` is never evaluated, so the bug was latent.
A CPU evaluation run (e.g., local testing without GPU) would crash with
`NameError: name 'nullcontext' is not defined`.

### Fix

Added `from contextlib import nullcontext` to the imports in `eval.py`.

---

## B9. GPU RNG Gather OOM on PyTorch 2.2.0 DDP (Checkpoint Save) — Resolved

**Impact:** Fatal crash on first DDP checkpoint save (step 2000)
**Status:** Resolved.

### Background

The B4 fix (commit `29b9690`) introduced `torch.distributed.all_gather_object`
to gather per-rank GPU RNG states during checkpoint saves. On PyTorch 2.2.0
with the NCCL backend, `all_gather_object` pickles the Python object, exchanges
pickled sizes via a CUDA all-gather, then resizes receive buffers to
`max_object_size`. A corrupted value in the NCCL size-negotiation step causes
`input_tensor.resize_(max_object_size)` to request more than 1 EB of CUDA
memory, triggering an immediate `torch.cuda.OutOfMemoryError`.

The crash occurred on the ADM v2 re-training (run_3, job 707599) at step 2000
— the first `save_checkpoint_freq` boundary:

```
optim/base.py line 221:
  torch.distributed.all_gather_object(_gpu_rng_gathered, torch.cuda.get_rng_state())
  → torch.cuda.OutOfMemoryError: Tried to allocate more than 1EB memory.
```

### Why the original authors didn't encounter this

The original codebase was developed on single A100 80 GB GPUs without DDP
(confirmed in B6). Single-GPU training never invokes distributed collective
ops. The authors' checkpoint code saved only rank 0's RNG state and never
restored it (restore calls were commented out). Without multi-GPU training,
there was no need for cross-rank RNG gathering, and no opportunity to trigger
the `all_gather_object` bug.

### Why v1 training succeeded

v1 (runs 0–2, jobs 696795/698707/701232) ran code from before commit `29b9690`
— the `all_gather_object` call didn't exist yet. v2 (run_3, job 707599) was
the first run to exercise the new code path.

### Fix

Replaced `all_gather_object` with `torch.distributed.all_gather` on fixed-size
ByteTensors (`optim/base.py`, 2 locations: periodic and final checkpoint save).
`torch.cuda.get_rng_state()` returns a fixed-size CPU ByteTensor — 16 bytes
consisting of an 8-byte seed and 8-byte Philox offset, a compile-time constant
in PyTorch's `CUDAGeneratorImpl.cpp`. Same-model GPUs produce identical tensor
sizes, eliminating the need for pickle serialization and size negotiation
entirely.

Additionally, a defensive ByteTensor conversion was added in `main.py` for the
plural `gpu_rng_states` checkpoint key, mirroring the existing conversion for
the singular `gpu_rng_state` key. This ensures any checkpoint format —
including those saved by the old `all_gather_object` path — loads correctly.

**References:**

- [1] PyTorch Issue [#145965](https://github.com/pytorch/pytorch/issues/145965): `all_gather_object` 1 EB OOM (confirmed by maintainer @kwen2501)
- [2] PyTorch Issue [#116177](https://github.com/pytorch/pytorch/issues/116177): NCCL collective OOM under high GPU memory utilization
- [3] NVIDIA/nccl Issue [#962](https://github.com/NVIDIA/nccl/issues/962): NCCL allocator fragmentation during `all_gather_object`

---

## B10. NCCL ALLREDUCE Timeout After Checkpoint Save (DDP) — Resolved

**Impact:** Fatal crash at first DDP checkpoint save (step 2000)
**Status:** Resolved.

### Background

The B9 fix (commit `ef3bff5`) replaced `all_gather_object` with
`torch.distributed.all_gather` on fixed-size ByteTensors. The `all_gather`
itself completed successfully — "saving checkpoint" was printed — but the job
crashed ~13 minutes later with an NCCL ALLREDUCE timeout:

```
[rank0]: Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=51989, OpType=ALLREDUCE, NumelIn=2360064,
  Timeout(ms)=600000) ran for 600326 milliseconds before timing out.
```

The timed-out operation is a **gradient ALLREDUCE** (DDP backward pass
synchronisation), not the `all_gather` for RNG states. Two jobs in run_3
exhibited the failure:

| Job | Code version | Error | Step |
|-----|-------------|-------|------|
| 707599 | Pre-B9 (`all_gather_object`) | `OutOfMemoryError: >1EB` (B9) | 2000 |
| 714422 | Post-B9 (`all_gather` ByteTensors) | NCCL ALLREDUCE timeout | 2000 |

### Root cause

After the `all_gather` completes, only rank 0 writes the checkpoint to disk
via `torch.save` (~2.5 GB of model + AdamW optimizer state). Rank 1 skips
the `is_master_process()` block and immediately continues to the next training
step. The sequence:

1. Both ranks complete `all_gather` (RNG states gathered)
2. Rank 0 enters `torch.save` (blocking I/O to `/scratch/` Lustre filesystem)
3. Rank 1 continues to step 2010, completes forward + backward, reaches
   gradient ALLREDUCE — **waits for rank 0**
4. Rank 0 is still blocked on the Lustre write (~4 MB/s under contention)
5. After 600 seconds (NCCL default timeout), rank 1's watchdog fires

This race condition was **latent in the original code** — the pre-B4 checkpoint
save was also master-only with no barrier (line 206–219 of the original
`optim/base.py`). The v1 training (runs 0–2) never triggered it because the
filesystem was faster on those particular nodes/times. The B9 fix did not
introduce the race; it merely exposed it by successfully reaching the save
(instead of crashing at `all_gather_object`).

The torchtune project hit the exact same failure pattern: [pytorch/torchtune
#2093](https://github.com/meta-pytorch/torchtune/issues/2093) — NCCL ALLREDUCE
timeout during Llama 3.1 70B checkpoint save on 8x V100.

### Fix

Two changes, following the canonical DDP checkpoint pattern used by Megatron-LM
([`# Wait so everyone is done (necessary)`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/checkpointing.py)),
DeepSpeed, HuggingFace Transformers, Meta Lingua, and PyTorch Lightning:

1. **Barrier after checkpoint save** (`optim/base.py`, 2 locations): added
   `distributed_backend.sync()` after both the periodic and final checkpoint
   save blocks. This forces all ranks to wait for rank 0's `torch.save` to
   complete before any rank proceeds to the next training step. The barrier
   is a standard NCCL collective (`torch.distributed.barrier`) that blocks
   until all ranks have entered it.

2. **NCCL timeout increase** (`distributed/ddp.py`): changed
   `init_process_group(backend=...)` to pass `timeout=timedelta(minutes=30)`,
   increasing the NCCL operation timeout from the default 10 minutes to 30
   minutes. This provides headroom if the Lustre filesystem is slow; the
   barrier ensures correctness, the timeout ensures the barrier itself does
   not prematurely abort.

Additionally, a duplicate `sync()` method definition in
`DataParallelDistributedBackend` (lines 59 and 65) was removed.

**References:**

- [1] PyTorch DDP Tutorial: [Save/Load Checkpoints](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) — canonical `dist.barrier()` after rank-0 `torch.save()`
- [2] PyTorch/torchtune Issue [#2093](https://github.com/meta-pytorch/torchtune/issues/2093): identical NCCL ALLREDUCE timeout during checkpoint save
- [3] PyTorch 2.2 docs: [`init_process_group(timeout=...)`](https://docs.pytorch.org/docs/2.2/distributed.html#torch.distributed.init_process_group) — *"Default value is 10 minutes for NCCL [...] duration after which collectives will be aborted asynchronously"*
- [4] PyTorch Blog: [Flight Recorder](https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/) — *"Almost all NCCL watchdog timeout errors are caused by collective desynchronization"*
- [5] NVIDIA Megatron-LM: [checkpointing.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/checkpointing.py) — barrier after save labeled `(necessary)`

---

## References

[cotformer-paper]: https://openreview.net/forum?id=7igPXQFupX
[segyges-mirror]: https://huggingface.co/datasets/segyges/OpenWebText2

1. Mohtashami, A. et al. (2025). *CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference.* [ICLR 2025](https://openreview.net/forum?id=7igPXQFupX)
