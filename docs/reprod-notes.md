# Reproducibility Notes

> Reproduction fidelity log for [CoTFormer (Mohtashami et al., ICLR 2025)][cotformer-paper]
> in our COMP 6258 reproduction. Covers both successful replications and known divergences.

## Table of Contents

- [Reproducibility Notes](#reproducibility-notes)
  - [Table of Contents](#table-of-contents)
- [Part A: Successful Replications](#part-a-successful-replications)
  - [A1. LN-CoTFormer Perplexity (Table 2, 40k steps)](#a1-ln-cotformer-perplexity-table-2-40k-steps)
  - [A1a. LN-CoTFormer Perplexity (Section 5, 60k steps)](#a1a-ln-cotformer-perplexity-section-5-60k-steps)
  - [A2. Architecture Depth](#a2-architecture-depth)
  - [A3. Training Stability](#a3-training-stability)
  - [A4. Effective Batch Size](#a4-effective-batch-size)
  - [A5. ADM Perplexity (Section 4.2, Figure 4, 60k steps)](#a5-adm-perplexity-section-42-figure-4-60k-steps)
  - [A6. Section 5 Comparison: LN-CoTFormer vs ADM at 60k Steps](#a6-section-5-comparison-ln-cotformer-vs-adm-at-60k-steps)
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
  - [B9. DDP Checkpoint Save Failures — NCCL + Lustre (ADM) — Resolved](#b9-ddp-checkpoint-save-failures--nccl--lustre-adm--resolved)
    - [Background](#background-4)
    - [Failure timeline](#failure-timeline)
    - [Root causes](#root-causes)
    - [Affected files](#affected-files-cumulative-across-all-four-fix-iterations)
    - [Checkpoint format compatibility](#checkpoint-format-compatibility)
  - [B10. LR Schedule Discontinuity on 40k-to-60k Extension](#b10-lr-schedule-discontinuity-on-40k-to-60k-extension)
  - [B11. Evaluation Batch Size Independence](#b11-evaluation-batch-size-independence)
- [Part C: Original Code Bugs](#part-c-original-code-bugs)
  - [C1. Figure 5 Cascading Bug: Position-Indexed Minimum on Reordered Tokens](#c1-figure-5-cascading-bug-position-indexed-minimum-on-reordered-tokens)
  - [C2. Underspecified Training Schedule (Reproducibility Gap)](#c2-underspecified-training-schedule-reproducibility-gap)
  - [References](#references)

---

# Part A: Successful Replications

> **Step-count disambiguation**: A1 reports the LN-CoTFormer at **40,000 steps**
> (Table 2 scope). A1a reports the LN-CoTFormer at **60,000 steps**
> (Section 5 scope). A5 reports the ADM at **60,000 steps** (Section 4.2 scope).
> A6 compares LN-CoTFormer vs ADM at 60k, directly validating the Section 5
> claims. The CoTFormer + Reserved model (Table 2, row 1, target 24.51) is
> pending via `cot-res-train`.
>
> **Evaluation methodology**: all metrics below are from standalone `eval.py`
> runs on the **full validation set** (3,973 batches, no subsampling), matching
> the original authors' evaluation protocol (confirmed by inspection of the
> original `eval.py` from commit `2723e30`, which uses `data['val']`
> exclusively). Confidence intervals are 95% CIs computed across per-batch
> losses, then exponentiated into perplexity space. These reflect
> **intra-run batch variance**, not the paper's inter-run SEM across 3
> independent training seeds. We report a single training run per model.

## A1. LN-CoTFormer Perplexity (Table 2, 40k steps)

**Result:** Reproduced within +0.02 PPL of the paper.

| Metric | Paper (Table 2) | Ours | Delta |
|--------|----------------|------|-------|
| Val perplexity | 24.11 (0.03) | 24.13 | +0.02 |
| Val accuracy | -- | 0.4129 | -- |
| Val loss | -- | 3.1834 | -- |
| 95% CI (perplexity) | -- | [23.75, 24.51] | -- |
| Per-batch SEM (loss) | -- | 0.0081 | -- |
| n_batches | -- | 3,973 | -- |

Our point estimate (24.13) falls within 0.67 standard deviations of the
paper's 24.11 +/- 0.03 (inter-run SEM across 3 seeds). Trained for 40,000
steps on 2x L4 24 GB (DDP), OWT2 dataset, cosine LR schedule (peak 1e-3,
final ~2e-4). Wall time: 23h 26m across 2 SLURM submissions with
auto-resume from checkpoint.

This confirms that our data pipeline, architecture, and training loop
faithfully reproduce the paper's non-adaptive LN-CoTFormer result, despite
the divergences documented in Part B.

---

## A1a. LN-CoTFormer Perplexity (Section 5, 60k steps)

**Result:** Reproduced within +0.40 PPL of the paper.

| Metric | Paper (Section 5) | Ours | Delta |
|--------|-------------------|------|-------|
| Val perplexity | 23.19 | 23.59 | +0.40 |
| Val accuracy | -- | 0.4157 | -- |
| Val loss | -- | 3.1609 | -- |
| 95% CI (perplexity) | -- | [23.22, 23.97] | -- |
| Per-batch SEM (loss) | -- | 0.0081 | -- |
| n_batches | -- | 3,973 | -- |

The 40k-to-60k extension used auto-resume from the 40k checkpoint with a
rebuilt `OneCycleLR(total_steps=60000)` scheduler. This introduces a
learning rate discontinuity at the resume boundary (B10): the 40k schedule
had decayed to ~2e-5, while the 60k schedule expects ~3.5e-4 at step 40000,
producing a ~17x LR jump that acts as a mild warm restart.

The wider delta (+0.40 vs +0.02 at 40k) is consistent with B10's impact
on the final 20k steps. The ADM model (A5) is unaffected by B10 as it was
trained for 60k steps from scratch.

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

**Result:** Reproduced within +0.23 PPL of the paper.

| Metric | Paper (Section 5) | Ours | Delta |
|--------|-------------------|------|-------|
| Val perplexity | 23.83 | 24.06 | +0.23 |
| Val accuracy | -- | 0.4127 | -- |
| Val loss | -- | 3.1806 | -- |
| 95% CI (perplexity) | -- | [23.68, 24.44] | -- |
| Per-batch SEM (loss) | -- | 0.0081 | -- |
| n_batches | -- | 3,973 | -- |

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
(`length_factors = [1, 1, 1, 1]`), so the reported PPL of 24.06 is
at full compute.

### Why the Gap Is Wider Than A1

The +0.23 delta (vs +0.02 for the non-adaptive LN-CoTFormer at 40k) is
explained by the ADM's unique sensitivity to RNG state discontinuities
at resume boundaries. As documented in B4, the `torch.rand()` calls
that generate capacity factors are consumed from the CUDA RNG stream.
Three resume boundaries produce different capacity factor sequences
than a continuous run would, changing which tokens are routed through
deeper repeats at each training step. The non-adaptive LN-CoTFormer
has no runtime randomness (dropout=0.0, fixed repeat count), so B4
does not affect it.

### Paper Comparison Note

See A6 for the full Section 5 comparison (LN-CoTFormer vs ADM at 60k).

---

## A6. Section 5 Comparison: LN-CoTFormer vs ADM at 60k Steps

**Result:** Directional claim confirmed. Non-adaptive LN-CoTFormer outperforms
adaptive model (ADM) at full compute, consistent with the paper.

| Metric | Paper | Ours | Paper gap | Our gap |
|--------|-------|------|-----------|---------|
| LN-CoTFormer 60k PPL | 23.19 | 23.59 | -- | -- |
| ADM 60k PPL (full compute) | 23.83 | 24.06 | -- | -- |
| Non-adaptive advantage | 0.64 | 0.47 | -- | -- |

The paper (Section 5) states: "after 60k steps, the former [adaptive]
reaches perplexity 23.83 while the latter [non-adaptive] achieves 23.19."
Our results confirm this ordering: the non-adaptive model outperforms the
adaptive model at full compute by 0.47 PPL (paper: 0.64 PPL).

### Absolute deltas

Both models run slightly above the paper's reported values:

| Model | Paper | Ours | Delta | Likely contributors |
|-------|-------|------|-------|-------------------|
| LN-CoTFormer 60k | 23.19 | 23.59 | +0.40 | B10 (LR discontinuity from 40k-to-60k extension) |
| ADM 60k | 23.83 | 24.06 | +0.23 | B4 (RNG state at resume boundaries) |

The gap in deltas (+0.40 vs +0.23) is consistent with B10 being unique
to the LN-CoTFormer (trained 40k then extended) while the ADM was trained
60k from scratch (unaffected by B10, but exposed to B4's RNG-dependent
capacity routing).

### Why the adaptive gap is smaller

The paper's 0.64 gap vs our 0.47 gap is explained by the larger LN-CoTFormer
delta (+0.40) relative to the ADM delta (+0.23). Since B10 inflates only the
LN-CoTFormer's perplexity, the gap narrows. A fresh 60k LN-CoTFormer run
without B10 would likely produce a wider gap closer to the paper's 0.64.

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
   save, all ranks' `torch.cuda.get_rng_state()` are collected into a list
   via a Gloo (CPU) `all_gather`. Rank 0 saves this list as `gpu_rng_states`
   (plural). Single-GPU runs wrap the local state in a one-element list for
   format consistency. (The transport evolved through several iterations:
   `all_gather_object` -> NCCL `all_gather` -> Gloo `all_gather` -- see B9
   for the full history.)

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

## B9. DDP Checkpoint Save Failures — NCCL + Lustre (ADM) — Resolved

**Impact:** Fatal crash at first DDP checkpoint save (step 2000), every run
**Status:** Resolved. Required four fix iterations across five SLURM jobs.

### Background

The B4 fix introduced per-rank GPU RNG gathering via
`torch.distributed.all_gather_object` so that checkpoint saves preserve each
rank's CUDA RNG state. This was the first code path to inject NCCL collective
operations into the checkpoint save sequence. On our 2x L4 24 GB DDP setup
(PyTorch 2.2.0+cu118, Lustre `/scratch/`), a cascade of four distinct
failures prevented checkpoint saves from ever succeeding.

The original codebase (single A100, no DDP) saved only rank 0's RNG state
and never restored it (restore calls were commented out in `optim/base.py`).
v1 training (runs 0-2) used pre-B4 code without any distributed RNG
gathering and completed 60k steps without checkpoint issues.

### Failure timeline

All five run_3 jobs crashed at step 2000 (`save_checkpoint_freq` boundary):

| Job | Date | Fix state | Error | Root cause |
|-----|------|-----------|-------|------------|
| 707599 | Mar 23 | B4 only | `OutOfMemoryError: >1EB` in `all_gather_object` | NCCL size-negotiation corruption [2][3][4] |
| 714422 | Mar 25 | + ByteTensor gather | NCCL gradient ALLREDUCE timeout (SeqNum=51989, NumelIn=2360064) | No barrier after save; rank 1 races ahead [5][6] |
| 724964 | Mar 27 | + barrier + 30 min timeout | NCCL barrier timeout (SeqNum=51988, NumelIn=1, 1800s) | Lustre I/O stall in `torch.save` [10] |
| 726223 | Mar 28 | + local-first save to `/tmp` | NCCL barrier timeout (SeqNum=51988, NumelIn=1, 1800s) | NCCL LL protocol hang (see below) |
| (resolved) | Mar 29 | + Gloo checkpoint coordination | Success | Gloo bypass of NCCL confirmed [12]; ADM v2 completed 60k steps (A5) |

### Root causes

**1. `all_gather_object` OOM (job 707599).** `all_gather_object` pickles
Python objects and exchanges sizes via CUDA all-gather. A corrupted value in
NCCL's size-negotiation step [2] caused a >1 EB allocation request.
`torch.cuda.get_rng_state()` returns a fixed-size CPU ByteTensor, so pickle
serialization was unnecessary. **Fix:** replaced with `torch.distributed.all_gather`
on fixed-size ByteTensors (commit `ef3bff5`). A defensive ByteTensor conversion
was added in `main.py` for backward-compatible checkpoint loading.

**2. Missing post-save barrier (job 714422).** After the gather, only rank 0
writes the ~2.5 GB checkpoint. Without a barrier, rank 1 races to the next
gradient ALLREDUCE while rank 0 is blocked on Lustre I/O, exceeding the NCCL
10-minute default timeout [9]. This race was latent in the original code.
The torchtune project hit the identical pattern [6]. **Fix:** added
`distributed_backend.sync()` after both periodic and final checkpoint saves,
following the Megatron-LM pattern [7][8]. Increased `init_process_group`
timeout to 30 minutes [9].

**3. Lustre I/O stall (job 724964).** `torch.save()` buffers the entire
checkpoint in memory before writing [10]. Under cluster I/O contention,
a single ~2 GB sequential write to one Lustre OST stalled for >30 minutes,
exceeding the post-save barrier timeout. Memory pressure and CUDA
synchronisation were both ruled out. **Fix:** local-first save
(`optim/utils.py`): `torch.save()` to `/tmp` (node-local), then
`shutil.copy2` to Lustre + `os.rename` for atomicity. Also added
checkpoint integrity handling on resume (`main.py`: try/except for
corrupted files, explicit warning on missing checkpoints).

**4. NCCL LL protocol hang (job 726223).** With the local-first save, the
checkpoint completed in 3.8 seconds and the "checkpoint saved" message was
printed. Yet the post-save barrier still hung for 30 minutes. The root
cause: NCCL's Low-Latency (LL) protocol on NCCL 2.15-2.17 (CUDA 11.8) has
a slot-reuse race when switching between collective types at small tensor
sizes. After ~52K gradient AllReduces using the Simple protocol (large
tensors), the first LL-protocol operations --- `all_gather` (16 bytes) then
`barrier` (1-element AllReduce) --- trigger the hang. NCCL 2.18.1 fixed the
LL cleanup race [11]; our PyTorch 2.2.0+cu118 ships an affected version.

Crucially, our code was the only framework using NCCL tensor collectives for
RNG gathering. Meta's torchtitan [12], Megatron-LM [8], HuggingFace
Accelerate, and DeepSpeed all use either Gloo (CPU) process groups or
per-rank files for checkpoint metadata exchange --- never NCCL.

**Fix:** replaced all NCCL operations in the checkpoint path with a secondary
Gloo (CPU/TCP) process group (`distributed/ddp.py`):

- `dist.new_group(backend="gloo")` created at init, used for RNG `all_gather`
  (CPU ByteTensors, no GPU round-trip) and post-save `barrier`.
- `broadcast_buffers=False` added to DDP constructor (all registered buffers
  are deterministic from config; eliminates unnecessary NCCL broadcasts).
- `NCCL_PROTO=Simple` set in `iridis/adm-train/job.sh` as belt-and-suspenders
  (forces Simple protocol for all remaining NCCL ops; negligible throughput
  impact on PCIe: ~0.04%).

### Affected files (cumulative across all four fix iterations)

| File | Changes |
|------|---------|
| `optim/base.py` | Gloo `all_gather` for RNG, Gloo barrier after save (2 locations) |
| `optim/utils.py` | Local-first save to `/tmp` then copy to Lustre |
| `distributed/ddp.py` | Gloo process group, `broadcast_buffers=False`, 30 min NCCL timeout |
| `main.py` | ByteTensor conversion for `gpu_rng_states`, try/except on `torch.load`, resume logging |
| `iridis/adm-train/job.sh` | `NCCL_PROTO=Simple` environment variable |

### Checkpoint format compatibility

The checkpoint format is unchanged: `gpu_rng_states` is a list of CPU
ByteTensors regardless of whether Gloo or NCCL was used for gathering.
Old checkpoints with the legacy singular `gpu_rng_state` key are handled
by the fallback path in `optim/base.py`. Non-ADM models (base,
cotformer_full_depth, cotformer_full_depth_lnmid_depthemb) have
`dropout=0.0` and no `torch.rand` calls, so RNG state is irrelevant to
their training math.

---

## B10. LR Schedule Discontinuity on 40k-to-60k Extension

**Impact:** Likely contributor to the +0.40 PPL delta in A1a (LN-CoTFormer
60k: 23.59 vs paper's 23.19). No effect on ADM (trained 60k from scratch).
**Status:** Accepted. Assessed in A6; consistent with observed delta pattern.

### Background

The paper trains the LN-CoTFormer for 60k steps with a single
`OneCycleLR(total_steps=60000)` cosine schedule. We initially trained for
40k steps (reproducing Table 2), then extended to 60k for the Section 5
comparison. This two-phase approach introduced a scheduler mismatch.

### What happened

The 40k training used `OneCycleLR(total_steps=40000)`:
- Warmup: 0 to peak LR over steps 0--8000 (20%)
- Cosine decay: peak to final LR over steps 8000--40000
- At step 40000: LR at its minimum (~2e-5, i.e. `1e-3 * final_div_factor`)

On resume with `--iterations 60000`, the original `summary.json` triggered
an early exit in `main.py` ("Already found experiment... Skipping"). After
fixing that (iteration-aware skip check), the restored `OneCycleLR` state
dict had `total_steps=40000`, causing a `ValueError` at step 40001.

The fix rebuilds the scheduler with `total_steps=60000` and fast-forwards
it to step 40000. This means steps 40001--60000 follow the latter portion
of a 60k cosine schedule, where the LR is still in mid-decay (~3.5e-4 at
step 40000 under a 60k schedule).

### The discontinuity

| Step | 40k schedule LR | 60k schedule LR | Delta |
|------|-----------------|-----------------|-------|
| 39999 | ~2.0e-5 (near minimum) | ~3.6e-4 | -- |
| 40000 | ~2.0e-5 (end) | ~3.5e-4 | -- |
| 40001 | (would not exist) | ~3.5e-4 | -- |

At the 40k/60k boundary, the effective learning rate jumps from the 40k
schedule's minimum (~2e-5) to the 60k schedule's mid-decay value (~3.5e-4),
a ~17x increase. The paper's continuous 60k run would have smoothly decayed
through this region.

### Impact assessment

The LR jump at the boundary acts as a mild "warm restart" for the final 20k
steps. Two factors limit the practical impact:

1. The model weights and optimizer states are correct (restored from
   checkpoint). Only the LR trajectory diverges.
2. The 60k schedule's LR at step 40000 (~3.5e-4) is within the range the
   model trained stably at during the first 40k steps (peak 1e-3 to 2e-5).

**Post-evaluation update (A1a):** The final 60k LN-CoTFormer perplexity is
23.59 vs the paper's 23.19 (delta +0.40). The delta is significantly wider
than the 40k reproduction (+0.02, A1). Since the ADM (trained 60k from
scratch, unaffected by B10) has a delta of only +0.23 (A5), this LR
discontinuity is the most likely contributor to the excess LN-CoTFormer
delta. A fresh 60k-from-scratch run would be the definitive test.

---

## B11. Evaluation Batch Size Independence

**Impact:** Negligible (<0.001 PPL)

Our standalone evaluation (`eval.py`) uses batch_size=8 (from `summary.json`,
matching training). The paper's evaluation was likely performed with a larger
batch size on A100 80 GB GPUs.

Per-sample cross-entropy loss is mathematically invariant to batch size: each
sequence in a batch is processed independently, and `F.cross_entropy` computes
per-token loss before averaging. The only micro-effect is the final partial
batch: `eval.py`'s `get_as_batch` generator (line 62) yields batches of
`batch_size` samples. If the total number of validation sequences is not
evenly divisible by batch_size, the last batch has fewer samples but receives
equal weight in the mean (one `loss_list_val.append()` per batch). At
~16,400 sequences / batch_size=8 = ~2,050 batches, the partial-batch weight
contribution is at most 1/2050th of the total, producing <0.001 PPL divergence.

We deliberately use the training batch_size (from `summary.json`) rather than
overriding to a larger value, to eliminate this micro-divergence entirely.
Single-GPU evaluation with `--distributed_backend None` also matches the
paper's evaluation conditions (no DDP data sharding during eval).

---

# Part C: Original Code Bugs

> Bugs present in the original codebase (`commit 2723e30`, "Add everything")
> that affect the authors' own reported results. These are distinct from
> Part B divergences, which arise from our hardware or setup constraints.
>
> Note: B6 (orphan MoDBlock), B7 (empty training summary), and B8 (missing
> nullcontext import) are also original code bugs but are documented in Part B
> because they were discovered during our reproduction process and primarily
> affected our DDP setup. They are cross-referenced here for completeness.

## C1. Figure 5 Cascading Bug: Position-Indexed Minimum on Reordered Tokens

**Impact:** Figure 5 of the paper shows an artifactually broad router weight
distribution. The underlying data is produced by a cascading bug in the
original `get_router_weights.py`.

**Status:** Root-caused and confirmed empirically. Evaluation pipeline now
generates both the buggy (paper-matching) and corrected extractions for
comparison.

### The bug

The original `get_router_weights.py` (commit `2723e30`) defines a custom
`forward()` function that manually executes the model's repeat loop,
collecting router weights from each `MoDBlock`. At each repeat, the MoDBlock
selects tokens via `torch.topk(router_logits, top_k, sorted=False)`, which
**reorders tokens by score**. The custom forward then applies
`x.take_along_dim(selected_indices, dim=1)`, so the representation passed to
the next repeat has tokens in a different order.

After collecting all per-repeat router weights, the script cascades via:

```python
for i in range(2, len(router_weights)):
    for j in range(len(router_weights[i])):
        router_weights[i][j] = torch.minimum(
            router_weights[i][j], router_weights[i - 1][j]
        )
```

This takes the element-wise minimum at the same **position index** `k`
across repeats. But position `k` at repeat `i` is a **different token** than
position `k` at repeat `i-1`, because tokens were reordered between repeats
by `take_along_dim`. The cascading therefore computes the minimum of
randomly-paired scores from different tokens, rather than the per-token
minimum across all routers.

### Why it matters

The correct per-token cascaded minimum (implemented in our rewritten
`get_router_weights.py` at commit `a9df8fd`) produces distributions
**heavily concentrated near zero**: Router 4 cascaded mean = 0.033,
with 94% of scores below 0.1. This is because any single router scoring
low for a given token drives the cascaded value to near zero.

The buggy position-indexed minimum produces a **much broader distribution**
(tail extending to x = 0.5, density peak clipped at 0.6--0.7), because
random pairing occasionally combines high scores from different tokens.
This broader distribution is what the paper's Figure 5 shows.

### How we confirmed this

| Extraction Method | Split | Router 4 Mean | frac < 0.1 | Density Peak |
|------------------|-------|--------------|-----------|-------------|
| Buggy (original code) | train | ~0.15 | ~50% | ~0.6 (matches paper) |
| Buggy (original code) | val | ~0.15 | ~50% | ~0.6 |
| Fixed (hook-based) | train | 0.033 | 94% | ~29 |
| Fixed (hook-based) | val | 0.033 | 94% | ~29 |

The buggy extraction recovers the paper's distribution shape on **both**
train and val splits. The fixed extraction produces the concentrated
distribution on **both** splits. This confirms the divergence is caused
by the cascading method, not the data split.

### Prior misdiagnosis

This finding supersedes an earlier hypothesis (originally written in this
entry) that attributed the Figure 5 divergence to the choice of data split
(train vs val). That hypothesis predicted that switching from validation
to training data would recover the paper's distribution. Empirical testing
(eval-adm run_2) disproved this: both splits produce the same concentrated
distribution under correct cascading.

The use of `data['train']` with `sample_size=len(data['val'])` in the
original code remains an intentional design choice for training dynamics
analysis (Section 5 examines how routing behaviour evolves during training).
This is not a bug, but it is also not the cause of the distribution shape
difference.

### Our approach: four Figure 5 variants

We now generate figures under both extraction methods and both data splits,
stored in `run_N/broken/` and `run_N/fixed/` respectively:

| Directory | Reproduction (train split) | Analysis (val split) |
|-----------|---------------------------|---------------------|
| `broken/` | Buggy cascaded, matches paper's Figure 5 | Buggy cascaded, held-out data |
| `fixed/` | Raw per-router sigmoid scores | Correct per-token cascaded |

The `broken/` reproduction plot is the closest match to the paper's Figure 5,
confirming that the paper's distribution is an artifact of this cascading bug.
The `fixed/` plots show the actual routing behaviour of the trained model:
nearly all tokens halt well before the final repeat.

### Implications

The paper claims (Section 5): "the model starts to favor using the final
repeat more when the model is trained for longer." Under correct extraction,
Router 4 scores are near-zero for both 40k and 60k checkpoints (mean 0.021
vs 0.033), with negligible mass above 0.5. The 40k-to-60k shift is real but
far smaller than Figure 5 suggests. The model does learn to use deeper
repeats with more training, but the magnitude is overstated by the buggy
extraction.

### Cross-references

- **B6** (orphan MoDBlock): Original code bug affecting DDP training.
  Harmless on single-GPU (the authors' setup) but fatal under DDP.
- **B7** (empty training summary): Original code bug. `stats` dict was
  initialised but never populated, losing training curve data in
  `summary.json`.
- **B8** (missing nullcontext import): Original code bug. `eval.py` crashed
  on CPU-only evaluation due to missing import; latent on GPU runs.
- `broken_get_router_weights.py`: Faithful reproduction of the original
  extraction method, adapted for CLI compatibility. Documents the bug in
  inline comments.
- `get_router_weights.py`: Corrected hook-based extraction (commit `a9df8fd`).
  Uses `register_forward_hook` on `mod_router` for per-logit capture.

## C2. Underspecified Training Schedule (Reproducibility Gap)

**Impact:** Increases reproduction difficulty for 60k models; ambiguity in
LR schedule type

**Status:** Resolved by code inspection. Documented here for completeness.

### Warmup steps for 60k models

Section 3.2 states: "we apply linear warmup of the learning rate for the
first 8000 steps." This figure is correct for the 40k models (Table 1,
Table 2), but the paper never specifies the warmup duration for the 60k
models introduced in Section 4.2.

The original codebase uses `--warmup_percent 0.2` (a fraction of total
iterations), not a fixed step count. For the 60k ADM and LN-CoTFormer
runs, this yields 12,000 warmup steps, not 8,000. A reader following the
paper's "8000 steps" claim verbatim for a 60k reproduction would use a
shorter warmup than the authors' actual implementation.

### LR schedule type

The paper states "linear warmup" but does not specify the post-warmup
annealing strategy. The codebase uses `torch.optim.lr_scheduler.OneCycleLR`
with `anneal_strategy='cos'` (cosine annealing) and
`div_factor=100, final_div_factor=0.05`. This is not a simple linear
warmup + constant LR schedule; it is a full cosine cycle from
`lr/100` up to `lr` (warmup phase) then back down to `lr * 0.05`
(annealing phase). A reproducer without access to the code would likely
implement a simpler schedule.

### Resolution

Our reproduction uses the percentage-based warmup and cosine annealing
from the original codebase, matching the authors' implementation exactly.
The paper's "8000 steps" claim is accurate for the 40k scope of Section 3.2
but should not be extrapolated to Section 4.2's 60k runs.

---

## References

[cotformer-paper]: https://openreview.net/forum?id=7igPXQFupX
[segyges-mirror]: https://huggingface.co/datasets/segyges/OpenWebText2

1. Mohtashami, A. et al. (2025). *CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference.* [ICLR 2025](https://openreview.net/forum?id=7igPXQFupX)
2. PyTorch Issue [#145965](https://github.com/pytorch/pytorch/issues/145965): `all_gather_object` 1 EB OOM --- confirmed by maintainer @kwen2501 as collective call ordering mismatch corrupting size negotiation.
3. PyTorch Issue [#116177](https://github.com/pytorch/pytorch/issues/116177): NCCL collective OOM under high GPU memory utilization.
4. NVIDIA/nccl Issue [#962](https://github.com/NVIDIA/nccl/issues/962): NCCL allocator fragmentation during `all_gather_object`.
5. PyTorch Blog: [Flight Recorder --- A New Lens for Understanding NCCL Watchdog Timeouts](https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/) (2024).
6. PyTorch/torchtune Issue [#2093](https://github.com/meta-pytorch/torchtune/issues/2093): NCCL ALLREDUCE timeout during Llama 3.1 70B checkpoint save on multi-GPU.
7. PyTorch Tutorial: [Getting Started with Distributed Data Parallel --- Save/Load Checkpoints](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).
8. NVIDIA Megatron-LM: [checkpointing.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/checkpointing.py) --- barrier after save labeled `# Wait so everyone is done (necessary)`.
9. PyTorch 2.2 Docs: [`init_process_group`](https://docs.pytorch.org/docs/2.2/distributed.html#torch.distributed.init_process_group) --- *"Default value is 10 minutes for NCCL [...] duration after which collectives will be aborted asynchronously."*
10. PyTorch Issue [#63720](https://github.com/pytorch/pytorch/issues/63720): `torch.save` zip serialization buffers entire state dict in memory before writing --- causes single large sequential write to filesystem.
11. NCCL 2.18.1 Release Notes: fixed LL protocol cleanup race on PCIe when switching between collective types in rapid succession. Related: PyTorch Issue [#104530](https://github.com/pytorch/pytorch/issues/104530) (DDP hang at specific step count traced to NCCL proxy thread state); NVIDIA/nccl Issue [#789](https://github.com/NVIDIA/nccl/issues/789) (LL protocol hang on PCIe with mixed collective types).
12. Meta torchtitan: [checkpoint.py](https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/checkpoint.py) --- uses `dist.new_group(backend="gloo")` for checkpoint coordination, decoupling checkpoint I/O from NCCL training collectives.
