# Extension Technical Notes

> Implementation-level companion to `docs/extend-notes.md`. Records engineering
> decisions, package layouts, environment specifications, HPC deployment
> recipes, and branch-discipline protocol. Scientific framing, research
> questions, experimental protocols, and theoretical predictions live in
> `docs/extend-notes.md`; this file records the technical substrate that
> supports them.
>
> For reproduction divergences from the original paper, see
> `docs/reprod-notes.md`.

## Scope

**This file IS**:

- Engineering specifications for the `analysis/` package and the counting
  substream training/eval pipelines.
- Config-flag plumbing and six-blocker resolution notes
  ([§3](#3-blocker-resolution-notes)).
- Environment freeze records (conda, pip, lockfile) and HPC cluster
  deployment recipes ([§4](#4-environment-and-hpc-deployment)).
- Branch discipline and worktree protocol for the experimentation stream
  ([§5](#5-branch-discipline)).
- Testing protocol: smoke tests, inter-batch CV, calibration gates
  ([§6](#6-testing-protocol)).
- Debugging artefacts and orch directive index
  ([§7](#7-debugging-and-orch-directive-index)).

**This file IS NOT**:

- A research report or a scientific framing document. Hypotheses, H0
  statements, falsification criteria, and triangulation matrices live
  in `docs/extend-notes.md`.
- A reproduction-divergence log. Paper-vs-ours numerical deltas live
  in `docs/reprod-notes.md`.
- A plan or a decision log. Plans live in `docs/extend-notes.md` §1.8;
  decisions live in `docs/extend-notes.md` §1.9.

## Table of Contents

- [1. Cross-reference index](#1-cross-reference-index)
- [2. Package layout](#2-package-layout)
- [3. BLOCKER resolution notes](#3-blocker-resolution-notes)
- [4. Environment and HPC deployment](#4-environment-and-hpc-deployment)
- [5. Branch discipline](#5-branch-discipline)
- [6. Testing protocol](#6-testing-protocol)
- [7. Debugging and orch directive index](#7-debugging-and-orch-directive-index)
- [8. Analysis engineering consequences](#8-analysis-engineering-consequences)

---

## 1. Cross-reference index

Every section in this file corresponds to one or more sections in
`docs/extend-notes.md`. The cross-reference index below is the
authoritative mapping; future edits must preserve it.

| Technical section (this file) | Scientific section (extend-notes.md) | Relationship |
|---|---|---|
| §2 Package layout | §1.7 Code and Infrastructure Strategy | §2 is the engineering detail of the scientific layout proposed in §1.7 |
| §3 BLOCKER resolution notes | §1.2 RQ9 "Code changes required" BLOCKERs 1-6 | §3 documents the concrete code changes; §1.2 specifies WHAT must change and WHY |
| §4 Environment and HPC deployment | §1.7 HPC packages, §1.8 Compute Budget | §4 records the conda env, SLURM configs, and scratch layout; §1.7/§1.8 specify the compute requirement |
| §5 Branch discipline | §1.7 Branch strategy | §5 codifies the abeprobes-specific workflow; §1.7 codifies the main-vs-tak strategy decided earlier |
| §6 Testing protocol | §1.6 Reliability Discipline, §1.3 Protocol D-calibration validation ladder | §6 records the smoke-test and gate-check procedures; §1.6/§1.3 specify the scientific content |
| §7 Orch directive index | §1.8 Sequencing, §1.9 Decision Log | §7 records which orch completed which wave; §1.8/§1.9 specify the plan and decisions |
| §8 Analysis engineering consequences | §1.3 Protocol B / C / D implementation notes, §1.4 Prediction 1 + Prediction 2, §1.8 Compute Budget | §8 records the Wave-2 / Wave-2c engineering consequences surfaced during protocol implementation; §1.3 / §1.4 / §1.8 specify the scientific constraint each engineering note serves. §8.6 (attention-taxonomy flash fallback) + §8.7 (router-analysis ADM-only execution) cover the Protocol C and Protocol D engineering consequences respectively. |

Every cross-reference is bidirectional: a `docs/extend-notes.md`
anchor cited here must cite back to this file from the corresponding
section. The bidirectional audit is part of every doc orch's closing
checklist.

---

## 2. Package layout

The `analysis/` package at the repo root contains the mechanistic-analysis
code. The package layout mirrors the protocol decomposition defined in
`docs/extend-notes.md` §1.7.

```
analysis/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── loader.py              # load_model_from_checkpoint (raw | argparse)
│   ├── sites.py               # ActivationSite enum + enumerate_sites()
│   ├── collector.py           # ActivationCollector (one forward per ckpt)
│   ├── data.py                # iterate_owt2_val (replaces duplicated get_as_batch)
│   ├── spectral.py            # compute_effective_rank, participation_ratio, kv_core_rank
│   ├── stats.py               # partial_correlation, paired_t_test, z_score_per_sequence
│   └── plotting.py            # setup_figure, site_label, palette_for_repeats
├── logit_lens.py              # Protocol A -- RQ1 unweighted lens
├── tuned_lens.py              # Protocol A-ext -- RQ1 canonical Belrose 2023 Tuned Lens
├── cka.py                     # Protocol B -- RQ2 debiased HSIC U-statistic CKA
├── attention_taxonomy.py      # Protocol C -- RQ3 head clustering + Gini + top-k mass
├── router_analysis.py         # Protocol D -- RQ4 + RQ5 (two JSON outputs per call)
├── residual_diagnostics.py    # Protocol E -- SQ1 pre-check
├── effective_dim.py           # Protocol F -- RQ6 participation ratio
├── kv_rank.py                 # Protocol G -- KV compressibility (architectural-extension stream MLA design)
├── interpolation_validity.py  # Protocol H -- RQ7 clone-set entropy
├── depth_emb_freeze.py        # Protocol I -- RQ8 8-condition freeze ablation
├── synthesis.py               # Cross-checkpoint plotting
└── calibration/
    ├── __init__.py
    ├── entropy_calibration.py # Protocol D-calibration main entry point
    ├── synth_sequences.py     # Condition A/B deterministic generators
    └── preregister.md         # Frozen pre-registration thresholds (Spearman, classifier, sink)
```

### Shared collector contract

`analysis/common/collector.py` provides a single `ActivationCollector` that
registers forward hooks on each `ActivationSite` the caller requests and
runs ONE forward pass per checkpoint to populate the per-site caches. The
collector is responsible for:

- Registering hooks at the correct module addresses for both standard
  `model.transformer.h` and CoTFormer `model.transformer.h_mid`
  module paths (the `--module-path` argument of the common loader --
  `extend-notes.md` §1.7 "Common loader hook for module-path
  generality").
- Maintaining a per-forward repeat counter via the `ln_mid` hook (which
  fires once per repeat; exists as `nn.Identity` in the C1 pre-`ln_mid`
  variant). Each `h_mid[k]` hook reads this counter to label its capture
  with `repeat=1..n_repeat`.
- Persisting per-site tensors to a workspace directory on `/scratch`
  (see [§4.3 Scratch workspace layout](#43-scratch-workspace-layout)).

Every protocol script (A through I) consumes the workspace cache produced
by a single collector run. Protocols B, E, F, G-weight, and synthesis are
CPU-only post-processors: they never allocate a new forward pass. The
shared forward pass is the single largest compute saving in the
mechanistic-analysis sub-stream.

### Data flow

```
    iridis/analyze-lncot+adm/job.sh
                │
                ├── Stage 1: flash-enabled collector (one per ckpt, 5 ckpts)
                │       → /scratch/$USER/analysis_workspace/<tag>/residual_*.npy
                │       → /scratch/$USER/analysis_workspace/<tag>/kv_*.npy
                │
                ├── Stage 2: non-flash collector (Protocol C only; 3 ckpts)
                │       → /scratch/$USER/analysis_workspace/<tag>/attn_weights_*.npy
                │
                ├── Stage 3: protocol dispatch (per ckpt × per protocol)
                │       → run_N/<tag>/protocol_<name>_results.json
                │       → run_N/<tag>/protocol_<name>_figure.png
                │
                └── Stage 4: synthesis (cross-checkpoint)
                        → run_N/synthesis_*.png
                        → run_N/synthesis_report.md
```

---

## 3. BLOCKER resolution notes

Six BLOCKERs must be resolved before any counting training run can start.
Their full specifications live in `docs/extend-notes.md` §1.2 RQ9
"Code changes required"; the engineering notes below record the
implementation path, test expectation, and any non-obvious consequences.

### 3.1 BLOCKER 1 -- `--activation` flag

- **Extend-notes anchor**: §1.2 RQ9 point 1.
- **Files touched**: `config/base.py`, every `MLP.__init__` in `models/*.py`.
- **Default**: `gelu` (preserves existing behaviour across all non-counting
  jobs).
- **Counting**: job scripts pass `--activation relu` to match Chang and
  Bisk [25] reference codebase `config.py:27`.
- **Test expectation**: `python main.py --activation relu --dataset
  counting --model but_full_depth --iterations 10` runs without
  AttributeError; loss decreases over the 10-step smoke test.
- **Non-obvious consequence**: the MLP init path in every model variant
  (`models/*.py`) must be audited; there is no shared `MLP` class.

### 3.2 BLOCKER 2 -- `--tie_word_embeddings` flag

- **Extend-notes anchor**: §1.2 RQ9 point 2.
- **Files touched**: `config/base.py`, `models/but_full_depth.py:216`
  (the hardcoded `wte.weight = lm_head.weight`),
  `models/but_full_depth.py:get_parameter_group_specs` (the hardcoded
  `decay.remove('lm_head.weight')`).
- **Default**: `True` (preserves existing behaviour).
- **Counting 4L baseline**: `--tie_word_embeddings False` matches Chang
  and Bisk codebase.
- **Test expectation**: parameter count at `--tie_word_embeddings False`
  exceeds the `True` count by `vocab_size × n_embd` (the newly-separate
  `lm_head.weight`).
- **Non-obvious consequence**: the `decay.remove` branch must be skipped
  when tying is disabled; otherwise `lm_head.weight` is excluded from
  weight decay on a per-parameter basis even though it is a regular
  linear layer.

### 3.3 BLOCKER 3 -- `--scale_attn_by_inverse_layer_idx` flag

- **Extend-notes anchor**: §1.2 RQ9 point 3.
- **Files touched**: `config/base.py`, `models/but_full_depth.py:Block.__init__`
  (thread `layer_idx` argument), `CausalSelfAttention.forward`
  (apply scale via SDPA `scale` kwarg when flash disabled, or
  pre-multiply `q` when flash enabled).
- **Default**: `False`.
- **Counting**: enabled **only in Pilot 1** (Chang and Bisk-exact regime)
  per the decision log in `docs/extend-notes.md` §1.9. The main sweep
  uses the CoTFormer-native regime where this flag is `False`.
- **Test expectation**: at `--scale_attn_by_inverse_layer_idx True`,
  attention logits at layer 0 are divided by 1 (no-op), at layer 1 by
  2, etc.
- **Non-obvious consequence for CoTFormer**: in weight-tied recurrent
  blocks (`h_mid`), the same layer is re-used across `n_repeat` passes.
  `layer_idx` in the classical sense is ambiguous: it could be the
  module index (always 0 for a 1L-4R config) or the repeat index
  (varies 0-4). The flag is enabled ONLY in Pilot 1 (standard 4L
  transformer with distinct layers), avoiding the ambiguity for the
  mechanistic-analysis sub-stream. If a future experiment needs
  `scale_attn_by_inverse_layer_idx` on CoTFormer, the semantics must
  be pre-registered first.

### 3.4 BLOCKER 4 -- `data/counting.py` + dispatcher refactor

- **Extend-notes anchor**: §1.2 RQ9 point 4 (five-file change).
- **Files touched** (5 + 1 defensive):
  1. `data/counting.py` NEW -- `CountingDataset` class with
     `__getitem__`/`__len__` protocol matching `optim/base.py`; generates
     `[start_int, "a"*L]` input format and `["-1", start+1, ..., start+L]`
     output format. Seeds per-rank RNG via the project's `seed_list`
     convention (train seed, val seed = train + 1000).
  2. `config/base.py` -- add `"counting"` to the `--dataset` choice
     list (plus `"constant_with_warmup"` to `--scheduler` choices).
  3. `data/utils.py` -- extend `PREPARE_GET_DATASET_MAP` dispatcher
     with the `counting` tuple `(prepare_counting_dataset,
     get_counting_data)`; `get_dataloader` gains a defensive 3-line
     branch to pass-through an already-wrapped
     `torch.utils.data.Dataset` (see §3.4 "Non-obvious consequence").
  4. `optim/base.py` -- branch training loop on `args.dataset ==
     "counting"` to skip OWT2-specific numpy memmap handling and
     consume the counting dataset's 4-tuple output directly. Control-row
     3 label-side masking hook: per-position CE is scattered over
     `loss_mask` which excludes the start-int position and every pad
     position; the scatter is the single override point for the
     cumulative-product-of-counts control-task selectivity baseline.
  5. `main.py` -- register the `constant_with_warmup` LR schedule
     (linear warmup to `--lr` across `--warmup_percent` fraction of
     iterations, then constant). Matches Chang and Bisk [25]
     `trainer.py:72` (their paper prose said cosine but the code uses
     `get_constant_schedule_with_warmup`).
- **Test expectation**: `python main.py --dataset counting --model
  but_full_depth --n_layer 4 --n_embd 128 --iterations 100` runs
  without KeyError and training loss decreases over 100 steps.
- **Non-obvious consequence**: the existing `(x, y)` tensor assumption
  in `optim/base.py` is generalised via a `is_counting` branch that
  consumes the counting dataset's 4-tuple output
  (`x, y, pad_mask, loss_mask`) and pipes `pad_mask` into the model
  as `attention_mask` (BLOCKER 6). The OWT2-specific branch is
  preserved unchanged. A defensive three-line branch in
  `data/utils.py:get_dataloader` passes an already-wrapped
  `torch.utils.data.Dataset` through without re-wrapping it in the
  1-D-stream `Dataset` shim; without this branch, main.py would
  require a counting-specific detour around `get_dataloader` which
  the directive explicitly forbids.

### 3.5 BLOCKER 5 -- `indices` threading for shifted PEs

- **Extend-notes anchor**: §1.2 RQ9 point 5.
- **Files touched**: `models/but_full_depth.py:CausalSelfAttention.forward`
  (now accepts `indices` kwarg and forwards to
  `pos_emb_closure.adapt_queries` / `adapt_keys`, both of which
  already supported `indices` at the closure level),
  `Block.forward` (forwards `indices` to `self.attn(...)`),
  `GPTBase.forward` (accepts `position_ids=None`; threads to every
  block call in `h_begin`, the `n_repeat` `h_mid` iterations, and
  `h_end`).
- **Specifics**: the skeleton resolves the hostile-review FIN-B3
  evidence that `Block.forward` already declared `indices=None` in
  its signature but never passed it to `self.attn(...)`; the fix is
  a one-line `indices=indices` keyword at the `self.attn(...)` call.
  `CausalSelfAttention.forward` gained the kwarg and forwards it to
  the PE closure (no change to the closure itself, which already
  supported the per-sample override).
- **Test expectation**: with `shift_value=100` and `seq_length=256`,
  the attention indices use `[100, 101, ..., 355]` rather than the
  default `[0, 1, ..., 255]`. Numerical equivalence check: the output
  at `shift_value=0` must match the pre-patch model to 1e-6 absolute.
- **Non-obvious consequence**: the same `indices` threading must work
  across CoTFormer's 5 variants too (if any is later used for counting).
  Audit required for `cotformer_full_depth_lnmid_depthemb` and
  `adaptive_cotformer_*_single_final`; the Phase 1 skeleton threads
  only through `but_full_depth.py` per the DIR-001 scope.

### 3.6 BLOCKER 6 -- Pad-mask-aware attention

- **Extend-notes anchor**: §1.2 RQ9 point 6.
- **Files touched**: `models/but_full_depth.py`
  (`CausalSelfAttention.forward` and `Block.forward` accept
  `attention_mask` kwarg; `GPTBase.forward` accepts and threads to
  every block); `optim/utils.py` (new `get_counting_batch`,
  `eval_counting`); `optim/base.py` (loss scatter over `loss_mask`,
  eval dispatches to `eval_counting` when `args.dataset ==
  "counting"`).
- **SDPA additive bias**: `-1e9 * (1 - attention_mask)` constructed
  from the per-sample `[B, T]` mask (1 for valid, 0 for pad), broadcast
  to `[B, 1, 1, T]`. The Flash-disabled path adds this bias to `att`
  before the existing `self.bias` causal masked_fill; the Flash-enabled
  path combines it with a `[1, 1, T, T]` causal bias (triu of -1e9
  above the diagonal) and dispatches SDPA with `is_causal=False`
  (Flash cannot be used with a variable-length attn_mask, so
  PyTorch's SDPA router selects the math backend).
- **Loss handling**: cross-entropy is computed with
  ``reduction="none"`` and ``ignore_index=-1``, then multiplied by
  ``loss_mask`` and divided by ``loss_mask.sum()`` to yield a masked
  mean. This is algebraically equivalent to the default ignore-index
  mean under the `CountingDataset` convention (targets = -1 at every
  non-count position) but keeps the scatter explicit for the
  control-row 3 override point.
- **Test expectation**: a batch with mixed sequence lengths produces
  the same per-valid-position loss as running each sequence
  individually at its true length. Difference should be within 1e-5
  per-position.
- **Non-obvious consequence**: the mask is created by the
  `CountingDataset` (BLOCKER 4), not on-the-fly in the training loop.
  The loop's responsibility is to forward the mask, not compute it.
  At `attention_mask=None` (OWT2 default), the forward path is
  bit-identical to pre-change -- the Flash fast path is preserved
  and no additive bias is constructed.

---

## 4. Environment and HPC deployment

### 4.1 Conda environment freeze

The shared conda environment at `$CONDA_ENV_PREFIX =
/scratch/ab3u21/cotformer-env` is the single source of truth for the
Python runtime on Iridis. Dependencies are pinned in two files:

- `environment.yml` -- human-readable conda specification.
- `environment.lock.yml` -- resolved lockfile for bit-exact reproduction.

The lockfile is rebuilt manually on the Iridis login node before any new
HPC submission wave. Agents do NOT auto-regenerate the lockfile; Abe
triggers the rebuild and commits the result manually.

**Pinned versions** (extend-notes.md §1.7 HPC packages row):

| Package | Version | Purpose |
|---|---|---|
| `pytorch` | `2.2.*+cu118` | Project-standard (paper-reproduction baseline) |
| `numpy` | `<2` | `torch.from_numpy` crash on numpy 2.x with PyTorch 2.2 |
| `scikit-learn` | `>=1.3` | Protocol D calibration classifier, Ridge probes |
| `pingouin` | `>=0.5` | Mixed-effects models for Prediction P2 |
| `statsmodels` | `>=0.14` | ANOVA (RQ9b), MixedLM |
| `muon-optimizer` | github.com/KellerJordan/Muon | Tuned Lens optimiser fallback |

### 4.2 iridis/env.sh single source of truth

Per `docs/extend-notes.md` §1.7 "env.sh consolidation", the following
HPC environment variables live EXCLUSIVELY in `iridis/env.sh`:

- `WANDB_MODE=offline`
- `NCCL_PROTO=Simple`
- `EXPS_DIR=$SHARED_SCRATCH/exps`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `TIKTOKEN_CACHE_DIR=$SHARED_SCRATCH/.cache/tiktoken`

Every `iridis/*/job.sh` sources this file once at the top, before any
other configuration. No `job.sh` re-assigns these variables inline.
Audit procedure: `grep -r -l "WANDB_MODE\|NCCL_PROTO\|TIKTOKEN_CACHE_DIR"
iridis/` should return only `env.sh` (apart from documentation comments).

### 4.3 Scratch workspace layout

Large intermediate files live on `/scratch` under a per-experiment,
per-checkpoint workspace directory. The layout mirrors the `iridis/eval-adm/`
precedent.

```
/scratch/$USER/analysis_workspace/
├── cotres_40k/
│   ├── meta.json
│   ├── residual_h_begin.npy
│   ├── residual_h_mid_r<R>_l<L>.npy
│   ├── residual_ln_mid_r<R>.npy
│   ├── kv_h_mid_l<L>.npy           # repeat-averaged (Protocol G weight consumer)
│   ├── kv_h_mid_r<R>_l<L>.npy      # per-repeat (Protocol G activation consumer)
│   └── attn_weights_h_mid_l<L>.npy # only when ATTN_WEIGHTS site requested
├── lncot_40k/
├── lncot_60k/
├── adm_v2_40k/
└── adm_v2_60k/
```

**Output triage rule** (inherited from `iridis/eval-adm/job.sh`): large
regeneratable `.npy` files never leave `/scratch`; only small meaningful
outputs (JSON summaries, PNG plots, scalar sweeps) are copied to
`iridis/<package>/run_N/`. The workspace is treated as transient; the
scratch purge schedule determines its lifetime.

### 4.4 SLURM deployment

Each protocol group has its own `iridis/` package with a self-submitting
`job.sh`. The Python invocation pattern is `python -m analysis.XXX` so
the repo root is the module root; no `sys.path` manipulation occurs.
Protocol-level failure isolation uses `set +e` around each dispatch: a
crash in one protocol does not cascade to others in the same run.

Current HPC packages:

| Package | Purpose | Queue |
|---|---|---|
| `iridis/analyze-lncot+adm/job-gpu.sh` | GPU protocols (A, C, D, D-cal, G-activation, H, I) | 1 GPU, 8h walltime |
| `iridis/analyze-lncot+adm/job-cpu.sh` | CPU protocols + Tuned Lens training + D-cal analysis | 0 GPU, 8h walltime |
| `iridis/counting-train/job.sh` | Pilot 1 + main sweep training | 2 GPUs, 24h walltime, chainable via auto-resume |
| `iridis/counting-eval/job.sh` | RQ9 + RQ9b OOD evaluation + DV-3 + DV-4 | 1 GPU, 12h walltime |

---

## 5. Branch discipline

### 5.1 Branch roles

Two persistent branches govern the project's codebase:

- **`main`** -- the stable reproduction branch. Contains the code that
  reproduces the original CoTFormer paper's Table 2, Figure 4, and
  Section 5 results. No experimental variants, no mechanistic analysis
  scripts, no counting substream. Changes to `main` are rare and
  require hand-review by Abe.
- **`abeprobes`** -- the experimentation branch. All mechanistic
  analysis code (`analysis/` package), counting substream
  (`data/counting.py`, `iridis/counting-*`), and architectural
  extension scaffolding (MLA, mcHC) live here. Branches off `main`.

The two branches are deliberately separated so that anyone reading the
paper-reproduction code on `main` is not confused by in-progress
experimental work, and vice versa.

### 5.2 Agent constraints

All agents work EXCLUSIVELY on `abeprobes`. Verification is a mandatory
first step for every orch directive:

```bash
git -C /home/totob/projects/comp6258/CoTFormer branch --show-current
# Expected output: abeprobes
# If any other branch, ABORT and escalate.
```

Agents do NOT:
- Commit (`git commit`) or push (`git push`). Abe commits manually.
- Switch branches (`git checkout`, `git switch`) without explicit
  instruction in the directive.
- Create new branches (`git checkout -b`, `git switch -c`) without
  explicit instruction.
- Merge branches (`git merge`) without explicit instruction.
- Cherry-pick from `main` without explicit instruction.

### 5.3 Cherry-pick protocol (if ever needed)

If `main` advances during experimentation (rare; e.g., a reproduction
bug-fix backport), the cherry-pick onto `abeprobes` is case-by-case:

1. Abe identifies the main commit and requests the cherry-pick
   explicitly in a directive.
2. The designated agent performs `git cherry-pick <sha>` on `abeprobes`
   only (never the reverse direction).
3. Conflicts are resolved with the `w-merger` worker; resolution is
   not auto-accepted.
4. The resulting commit is left UNCOMMITTED for Abe to review and sign.

### 5.4 Worktree protocol

For edits to the `main` branch README or other one-off `main`-side
maintenance, the preferred pattern is a temporary worktree at
`/tmp/claude/cotformer-main-edit` rather than switching the main
working tree's branch:

```bash
git worktree add /tmp/claude/cotformer-main-edit main
# Edit files in /tmp/claude/cotformer-main-edit/
# Leave uncommitted; report path to Abe for his manual commit.
# After Abe commits: git worktree remove /tmp/claude/cotformer-main-edit
```

This preserves the `abeprobes` working state and avoids accidental
commits on `main` from an agent context.

---

## 6. Testing protocol

### 6.1 Smoke tests (per BLOCKER)

Every BLOCKER's implementation carries a smoke-test expectation (see
[§3](#3-blocker-resolution-notes)). The smoke test is the minimum
evidence required to mark a BLOCKER as resolved. Full integration
tests come later.

Smoke tests run under:

```bash
cd /home/totob/projects/comp6258/CoTFormer
source iridis/env.sh  # locally harmless -- just sets env vars
module load conda 2>/dev/null || true
conda activate $CONDA_ENV_PREFIX 2>/dev/null || source .venv/bin/activate
# Then run the smoke-test command documented in §3.X
```

### 6.2 Inter-batch coefficient-of-variation gate

Per `docs/extend-notes.md` §1.2 RQ1 "Inter-batch robustness", all
activation-based protocols run on **4 independent validation batches**
and report inter-batch CV. If CV < 5% for a metric, the single-batch
point estimate is validated. Compute overhead: 4x on forward-pass-dependent
protocols.

The gate is applied per-protocol, per-metric. Metrics failing the CV
gate are flagged in the per-protocol JSON output rather than
silently averaged away.

### 6.3 Calibration-ladder gates (Protocol D-calibration)

The four-gate validation ladder for Protocol D-calibration is
documented in `docs/extend-notes.md` §1.3 "Protocol D-calibration".
Each gate has a hard threshold that must be passed in order; failure at
any gate triggers a fallback rather than continuing the ladder. The
thresholds are frozen at the Phase 1 skeleton commit in
`analysis/calibration/preregister.md` and cannot be adjusted post-hoc.

### 6.4 Pilot 1 reproduction gate (counting)

Per `docs/extend-notes.md` §1.2 RQ9 "Twin-pilot design" and
§1.8 "Empirical step-time validation gate", Pilot 1 reproduces Chang
and Bisk [25] at `n_embd=1024`. Two hard gates apply:

- **IND accuracy >= 80%** at the `te200` variant on the train OOD
  boundary (length 50). Failure triggers the whole RQ9 substream to
  shed to the extension stream (§0 scope-shedding trigger 1).
- **Per-step wall time within 50% of projection**. Failure triggers
  a budget revision before the main sweep is submitted.

---

## 7. Debugging and orch directive index

### 7.1 Orch directive index

Each orch that touches the abeprobes branch records its completion in
the index below. The index is append-only; entries are never removed.

| Orch | Directive | Status | Completed on | Summary |
|---|---|---|---|---|
| o-cotformer-skel-1 | DIR-001 | DONE (uncommitted) | 2026-04-22 | Phase 1 skeleton + 6 BLOCKERs + environment + pre-registration freeze |
| o-cotformer-mvp-a | DIR-001 | COMPLETE (uncommitted) | 2026-04-22 | Phase 2a common/ infrastructure + Protocols A, A-ext, E + HPC packaging |
| o-cotformer-mvp-b | DIR-001 | COMPLETE (uncommitted) | 2026-04-22 | Phase 2b Protocols B (debiased CKA), F (effective dim), G (KV rank + KV-CoRE NER) + HPC script extensions; analyze_kv_compression.py monolith removed |
| Phase-2D meta-direct + 5 w-refactorers | -- | COMPLETE (uncommitted) | 2026-04-28 | 23-fix hostile-review pass: Type II SS via statsmodels + bootstrap CI on η²_AB (DEC-028); cosine subsidiary local-centring (Protocol B); verdict triple-gate composition with direction-aware Spearman + fit-error guard + raw-vs-sink-corrected gate1 split (Protocol C); FP32 lens path with deepcopy + n_tokens guard + epoch-shuffled sampler (Protocol A / A-ext); per-layer monotonicity n_measured denominator (Protocol A); cosine-IO boundary-cell NaN + JSON-safe NaN→None (Protocol E); DV-4 ablation hook registration order; VERSIONS array `_build_versions()` deferred to post-`source env.sh`; `_workspace_has_kv` DRY refactor; `discover_workspace_sites` SoT helper (Items 4 + 8). All 13 Python parses + 3 bash syntaxes + 3 built-in smoke tests + 8 custom unit tests pass; 0 firewall hits. |

### 7.2 Debugging log

Debugging artefacts that are worth preserving across sessions (beyond
what git history captures) are recorded below. Each entry carries:
(a) the symptom observed, (b) the root cause, (c) the fix applied,
(d) the audit trail (commit hash or file change).

*(no entries yet)*

---

## 8. Analysis engineering consequences

Engineering-side notes surfaced during the Wave-2 analysis-protocol
implementation. Each subsection records a consequence whose scope is
operational (compute budget, cache-layout coupling, module interface
details) rather than scientific (the scientific statements stay in
`docs/extend-notes.md`).

### 8.1 Debiased HSIC U-statistic compute budget

Protocol B (`analysis/cka.py`) implements the Song et al. 2012 /
Murphy et al. 2024 unbiased HSIC U-statistic estimator, NOT the
biased estimator. The unbiased form is required because biased CKA
inflates similarity at the low-``N`` high-``d`` regime that applies
here (2048 tokens sampled from OWT2, 105 sites at ``n_embd=768``).

Compute-cost accounting for a single C3 checkpoint:

- 105 residual sites -> ``C(105, 2) = 5460`` unordered pairs.
- Per pair: one observed debiased CKA evaluation plus
  ``n_permutations`` permutation-null evaluations.
- At ``n_permutations = 1000``, 5460 x 1000 = 5.46 M debiased CKA
  evaluations per checkpoint.
- Each evaluation is O(``n_tokens^2``) in the element-wise
  Gram-product path; at ``n_tokens = 2048`` this is ~4 M FLOPs per
  evaluation, giving a single-core workload of ~ 2.2e13 FLOPs.
- On the SLURM CPU node with 16 cores the wall time is ~15 h
  worst case at the native NumPy BLAS throughput, falling to
  ~2 h with ``multiprocessing.Pool(processes=12)``. The
  ``CKA_WORKERS`` knob in `iridis/analyze-lncot+adm/job-cpu.sh`
  selects the parallelism; the default of 12 leaves a modest
  reserve against BLAS-thread oversubscription.

Implementation strategy:

- Gram matrices ``K_tilde = X X^T`` (diagonal zeroed) are
  pre-computed once per site before the pair loop. Memory cost is
  105 x 32 MB = ~3.3 GB at ``float64``, well within the 64 GB
  SBATCH allocation.
- ``HSIC_u(Y, Y)`` is permutation-invariant because the trace,
  row-sum dot product, and total-sum terms are all unchanged when
  Y's row/column indices are permuted by the same permutation
  (``P K_tilde_Y P^T`` has the same trace as ``K_tilde_Y``, and
  ``1^T P K_tilde_Y P^T 1 = 1^T K_tilde_Y 1``). Only
  ``HSIC_u(X, Y_perm)`` varies across null samples; the CKA
  denominator is therefore precomputed once per pair, saving two
  thirds of the in-loop cost.
- Permutation is implemented as a two-axis fancy index
  ``K_tilde_Y[p[:, None], p[None, :]]`` rather than regenerating
  the Gram matrix from a permuted data matrix. The former is
  ~``n_tokens^2`` memory traffic per draw; the latter would be
  ``n_tokens^2 * n_embd`` per draw (400x more work).

Escalation path: if profiling on the target node shows > 30 h CPU
per checkpoint the compute budget is blown up (§1.8 subtotal is
~2 h CPU across B / F / G). Halve ``n_permutations`` only as a
last resort and document the zone-threshold widening in the RPT.

### 8.2 KV capture inside `kv_rank.py` (documented exception to §8.5)

Protocol A (`analysis/logit_lens.py`) captures residual sites
only; the fused ``c_attn`` site required by Protocol G Type B is
not part of its collector configuration. Rather than extending
Protocol A's scope (which would entangle its CV-gate accounting
with an unrelated site), Protocol G carries a ``--capture-kv``
flag that runs a second minimal collector pass at the
``ActivationSite.KV_C_ATTN`` site when the workspace lacks
``kv_mid_l<L>_r<R>.npy``. The pass reuses
`analysis/common/data.py`'s ``iterate_owt2_val``, the shared
autocast context, and the `analysis.common.collector` state-
management contract, so the per-repeat counter hooks remain the
single source of truth.

This is a **documented exception** to the collector-only principle
stated in §8.5: the CV-gate accounting constraint makes combining
the two captures costly. Any future exception must be raised the
same way -- a script-local collector pass using the shared
``ActivationCollector`` class, NOT a bespoke capture path.

Re-runs are idempotent: `kv_rank.py` checks
``workspace_dir`` for any ``kv_mid_l*_r*.npy`` file before
triggering the capture. When files are present, `--capture-kv`
is a no-op and only Type A (weight-level) and Type B
(activation-level from the existing cache) run.

### 8.3 Workspace file-naming contract

The canonical file names produced by `analysis.common.collector`
for every site consumed by Protocols A, B, C, D, E, F, G are:

| Site | File name | Shape |
|---|---|---|
| `RESIDUAL_POST_MID` at layer ``L``, repeat ``R`` | `residual_mid_l<L>_r<R>.npy` | `(N_tokens, n_embd)` |
| `RESIDUAL_POST_LN_MID` at repeat ``R`` | `residual_ln_mid_r<R>.npy` | `(N_tokens, n_embd)` |
| `RESIDUAL_PRE_LN_F` (one per forward; input to ``ln_f``) | `residual_pre_ln_f.npy` | `(N_tokens, n_embd)` |
| `KV_C_ATTN` at layer ``L``, repeat ``R`` | `kv_mid_l<L>_r<R>.npy` | `(N_tokens, 3 * n_embd)` |
| `ATTN_WEIGHTS` at layer ``L``, repeat ``R`` (Protocol C) | `attn_weights_mid_l<L>_r<R>.npy` | `(N_tokens, n_head, T_k)` |
| `ROUTER_LOGITS` at router index ``K`` (Protocol D; ADM only) | `router_mod<K>.npy` | `(N_tokens, 1)` |

The `kv_mid_*.npy` files store the fused ``[Q, K, V]`` output of
``c_attn``; downstream consumers slice K as
``[:, n_embd : 2 * n_embd]`` and V as ``[:, 2 * n_embd : 3 * n_embd]``.
`kv_rank.py` honours this layout in its
``_analyse_activation_one`` helper.

The `attn_weights_mid_*.npy` files store the post-softmax
per-head causal attention distribution, one row per (batch, query)
position. ``N_tokens = B * T_q`` matches the token budget; ``T_k``
is the sequence length (keys live at positions ``0..T_k - 1``).
Rows are ordered as the flattened ``(B, T_q)`` indexing of the
raw attention matrix ``(B, n_head, T_q, T_k)`` after
``permute(0, 2, 1, 3).contiguous().reshape(B * T_q, n_head, T_k)``;
query positions within a single batch occupy consecutive rows,
allowing the Protocol C previous-token detector to derive query
indices as ``row_index mod T_k``. The files are written ONLY
when the caller passes ``non_flash=True`` to the collector; see
[§8.6](#86-attention-taxonomy-flash-fallback) for the flash-backend
limitation rationale and the monkey-patch implementation.

The `router_mod<K>.npy` files store the raw per-token router
logit from ``mod[K].mod_router`` (one per router, ``K`` in
``0..n_repeat - 2``). No sigmoid is applied at capture time;
downstream consumers in `router_analysis.py` apply
``sigmoid(logit)`` and threshold at ``0.5`` to recover the
continue / halt decision. Non-ADM checkpoints (C1 / C2 / C3) do
not instantiate ``transformer.mod`` and the files are never
written on those tags; Protocol D short-circuits with a
``not_applicable`` verdict; see
[§8.7](#87-router-analysis-adm-only-execution).

The key-naming is produced deterministically by the collector's
``_make_site_hook`` via
``key.replace('[', '_l').replace(']', '')`` on the site key
produced by `enumerate_sites`. The §4.3 workspace-layout diagram
in this document was written against an earlier stub convention
(``residual_h_mid_r<R>_l<L>.npy``) and does not match the
collector's live output; treat the actual collector key format
as canonical until a future doc-wave resolves the text.

### 8.4 Spectral module dependency chain

The Protocols B / F / G bodies all import from
`analysis/common/spectral.py`:

- Protocol B (cka.py) uses the HSIC U-statistic machinery
  defined in its own module; it does NOT consume spectral.py.
- Protocol F (effective_dim.py) imports
  ``participation_ratio`` and ``kv_core_rank`` for the Chun
  gamma_row estimator and the Chen NER cross-check, plus
  `analysis/common/plotting.py` helpers.
- Protocol G (kv_rank.py) imports ``compute_effective_rank``
  (cumulative-variance rank search) and ``kv_core_rank`` (NER).

The `spectral.py` module stays NumPy-only; none of its helpers
take a ``torch.Tensor``. When a protocol needs to apply a
spectral helper to a GPU-resident tensor, it detaches to CPU and
passes the ``numpy`` view -- matching the collector's per-site
hook contract (CPU-resident float32 by default).

### 8.5 Collector-only principle

``analysis/common/collector.py`` is the single place that OWNS
activation capture. Protocol scripts extract from the workspace
cache the collector produces; they do NOT install their own
forward hooks, monkey-patch ``CausalSelfAttention``, or materialise
activations via a bespoke forward pass.

Rationale:

- **DRY**: one hook-management contract, one per-repeat counter,
  one CPU-resident-float32 capture convention, one workspace
  file-naming scheme. Protocol scripts need only know the
  workspace file-naming table (§8.3).
- **Reproducibility**: all captures share the same ``torch.no_grad``
  envelope, the same ``type_ctx`` autocast, the same
  ``_flatten_bt`` convention. A second capture path would drift
  from these guarantees and contaminate cross-protocol
  triangulation rows in §1.6.
- **Test surface**: a single collector is testable against
  ``sites_per_forward_count``; bespoke capture paths fragment the
  test surface and make per-repeat bookkeeping bugs
  protocol-specific rather than reproducible via smoke tests.

**Exceptions** are only permissible when an EVIDENT computational
constraint makes combining captures infeasible (e.g. the Protocol
A CV-gate accounting in §8.2). Every exception must:

1. be documented in this section with a pointer to the justifying
   constraint,
2. use the shared ``ActivationCollector`` class (no bespoke
   hooks), and
3. check the workspace for existing outputs before triggering a
   re-capture (idempotency).

Future protocol scripts that wish to derogate from this rule must
raise the exception via the same mechanism and document it here;
silent one-off capture paths are a defect even when they work.

### 8.6 Attention-taxonomy flash fallback

Protocol C requires per-(query, head) attention weights; the
flash-kernel SDPA backend
(``torch.nn.functional.scaled_dot_product_attention`` with
``is_causal=True``) does NOT return them. `ActivationCollector`
therefore treats ``ATTN_WEIGHTS`` as a site-requiring-non-flash
capture and enforces two coupled toggles on the run:

1. ``non_flash=True`` is a constructor precondition. Requesting
   ``ATTN_WEIGHTS`` without ``non_flash=True`` raises
   ``ValueError`` with an explicit pointer back to this section.
   The flag is validated in ``__init__`` so the mis-config fails
   before any forward pass.
2. ``_apply_non_flash`` (called from ``register_hooks`` when
   ``non_flash=True``) flips ``attn.flash = False`` on every
   ``CausalSelfAttention`` module in ``transformer.h_begin``,
   ``h_mid``, ``h_end``, and ``h``. This routes the
   ``CausalSelfAttention.forward`` math path at
   ``models/base.py:85-119`` to the non-flash branch.

Checkpoint interaction: ``CausalSelfAttention.__init__`` only
registers the ``bias`` causal-mask buffer in its non-flash branch
(``models/base.py:52-62``); checkpoints trained under PyTorch 2.0+
(i.e. every checkpoint in the Analysis Matrix) have ``flash=True``
at init and carry no ``bias`` buffer. The non-flash forward path at
``models/base.py:94`` uses ``self.bias[:, :, :T, :T]``; without a
buffer this crashes with AttributeError when the collector later
flips ``flash=False``. `_apply_non_flash` detects the missing
buffer and installs a causal mask on the fly via plain-attribute
assignment (NOT ``register_buffer``, so the collector-installed
mask cannot pollute any subsequent ``state_dict``). A sibling flag
``_analysis_nonflash_bias = True`` records which modules the
collector patched; ``_restore_flash`` removes only those (leaving
any checkpoint-native ``bias`` buffer intact).

Attention-weights capture mechanism: because
``CausalSelfAttention.forward`` returns only ``y`` (the residual-
projected output) and not ``(y, att)``, a standard
``register_forward_hook`` on the attention module cannot observe
``att``. `ActivationCollector._install_attn_monkey_patch` replaces
``attn.forward`` per-module with a wrapper that:

1. Calls the original forward to produce ``y`` (no change to the
   residual stream; downstream modules remain unaffected).
2. Re-runs the Q/K math path on the same input to recompute
   ``att`` post-softmax. The re-computation mirrors the
   ``models/base.py:92-99`` math, intentionally omitting the
   attention-dropout and cache-prefix branches because the
   collector runs in ``torch.no_grad() + model.eval()`` where
   ``attn_dropout`` is an identity and ``cache_context`` is None.
3. Stashes ``att`` on the module as ``_analysis_att_stash``
   (detached, float32, CPU) so the forward-post-hook can read it.

Restoration is symmetric: `_restore_attn_monkey_patch` deletes
the instance-level ``forward`` override so
``CausalSelfAttention.forward`` (the class-level descriptor) is
visible again. Both the monkey-patch and the on-the-fly bias
buffer are cleaned up inside the ``with collector:`` context
manager exit. Users who invoke ``register_hooks`` / ``remove_hooks``
manually must pair the calls inside a ``try / finally`` so a
mid-run crash cannot leave the model in a modified state.

Compute cost: the non-flash math path is roughly 4x the flash-
kernel per-block cost; the monkey-patch's Q/K re-computation adds
another 2x on top. Combined, Protocol C forwards run at ~8x the
flash-backend baseline. ``docs/extend-notes.md`` §1.3 Protocol C
compute row quotes ~9 min on L4 per checkpoint at
``max_tokens = 2048``, ``seq_length = 256``; this matches the ~8x
multiplier over the ~1 min flash-backend forward on the same
budget.

This is the second pre-approved derogation of the collector-only
principle ([§8.5](#85-collector-only-principle)) -- the first is
the [§8.2](#82-kv-capture-inside-kv_rankpy-documented-exception-to-85)
KV-capture-inside-kv_rank.py exception. Unlike §8.2 (which
installs a bespoke hook path outside the collector), the
attention-taxonomy exception lives INSIDE ``collector.py``; the
monkey-patch is scoped to ``register_hooks`` / ``remove_hooks``
and cannot leak past a properly-used context manager.

### 8.7 Router-analysis ADM-only execution

Protocol D consumes the ``ROUTER_LOGITS`` site which hooks
``model.transformer.mod[K].mod_router`` for ``K`` in
``0..n_repeat - 2``. Non-ADM variants (C1
``cotformer_full_depth``, C2 / C3 ``cotformer_full_depth_lnmid_depthemb``)
do not instantiate ``transformer.mod`` -- ``getattr(transformer,
"mod", None)`` returns ``None`` on those checkpoints.
`router_analysis.py` short-circuits in two places:

1. The capture stage (``_run_capture_stage``) checks
   ``getattr(model.transformer, "mod", None)`` before constructing
   the collector. When the attribute is absent or empty it returns
   ``{"short_circuit": True, "reason": "non-ADM checkpoint, no
   mod[] router layers"}`` and skips both the collector and the
   analysis stages.
2. The analysis stage (``_analyse_routers``) scans the workspace
   for ``router_mod<K>.npy`` entries and returns ``{"verdict":
   "not_applicable"}`` when none are found. This guard handles the
   case where a prior run saved a workspace meta.json with the
   short-circuit flag but no router files (i.e. the GPU stage
   short-circuited and the CPU stage re-invoked on the same
   workspace).

Short-circuit semantics preserve the per-tag output directory
contract: even on non-ADM checkpoints a
``router_analysis_results.json`` is written, carrying the
``verdict: not_applicable`` payload. The HPC `job-cpu.sh` stage
thereby produces a per-tag JSON for every checkpoint, keeping the
run_N/<tag>/ tree uniform and making cross-checkpoint synthesis
scripts (``analysis.synthesis`` in a later wave) simpler to
write.

Cross-version comparison semantics: Protocol D's
``--compare-against`` flag accepts a prior
``router_analysis_results.json`` and rebuilds the per-router
continue masks when the prior workspace path is recorded in the
JSON meta block (enable via ``--record-workspace-path`` on the
prior run; the `job-cpu.sh` template sets the flag by default).
When the prior workspace is unavailable the fallback
``fingerprint_only`` probe compares SHA-1 fingerprints of the
continue masks for strict equality (no partial-overlap resolution
possible). The DIR-001 T3 cross-version H0 "router decisions
identical at the same repeat index" rejects at
``min(jaccard) < 0.7`` across routers; the ``rejects_identical_null``
flag in the ``cross_version`` sub-block records the verdict. C5
v2 against C4 v1 at matched 60k steps is the primary comparison
in the Analysis Matrix.

Why Protocol D lives on the CPU stage (job-cpu.sh): the forward
pass on a 24L x 5 model at ``max_tokens = 2048, seq_length = 256``
takes ~5 min on a 16-core CPU, an acceptable cost relative to
the Tuned Lens ~5 h CPU budget per checkpoint that dominates the
CPU stage. Running Protocol D on the GPU would compete with
Protocol C's ~9 min GPU slot per checkpoint and the Protocol G
Type B ~25 min per checkpoint; the CPU placement keeps the GPU
job tighter and still honours the §1.8 compute budget's "~10 min
GPU" row (Protocol D's per-checkpoint budget is accepted as CPU-
resident without loss of scientific fidelity).

### 8.8 Counter-registration order in collector + DV-4 ablation

PyTorch fires forward post-hooks in registration order. Two scripts
install per-repeat counter-increment hooks alongside per-block site
or ablation hooks: ``analysis.common.collector`` (Protocols A through
G observational caches) and ``analysis.counting_dv4_causal``
(DV-4 zero-ablation). Both have a "last block doubles as the
counter target" failure mode for variants without ``ln_mid``
(Arm A's ``but_full_depth`` and the bare ``cotformer_full_depth``
class C1 + counting V1 / V2): the increment hook lands on
``h_mid[-1]`` which is also a site / ablation target. With the
increment registered FIRST (the legacy order), the per-block hook
on ``h_mid[-1]`` reads the BUMPED counter at every repeat;
the n_repeat clamp masks the off-by-one only at the FINAL repeat,
so every earlier repeat's capture / ablation activates one repeat
late.

Resolution: in both files the counter-increment hook is registered
AFTER all per-block site / ablation hooks. The per-block hooks
fire first, read the pre-bump counter, label the capture or
activate the ablation at the correct repeat; the bump then runs
once per repeat as designed. V3 / V4 (with ``ln_mid``) are
unaffected by the order because the counter increment lands on a
different module (``ln_mid``) than the h_mid blocks; either order
yields the same captured indexing.

The fix is mirror-applied: the same registration-order rule lives
in both ``analysis.common.collector`` (`register_hooks`) and
``analysis.counting_dv4_causal`` (`ablation_hooks` context
manager); a future site-or-ablation script with the same shape
must follow the same convention or document a deliberate
deviation.

### 8.9 Workspace key-prefix module-path awareness

The collector emits per-site cache files keyed by the trailing
group name in ``module_path``: for ``model.transformer.h_mid``
the prefix is ``residual_mid_l<L>_r<R>.npy``; for the
Protocol D-calibration Tier 1 GPT-2-large substrate at
``model.transformer.h`` the prefix is ``residual_flat_l<L>_r<R>.npy``.
Five protocol scripts (`logit_lens`, `tuned_lens`, `cka`,
`effective_dim`, `residual_diagnostics`) consume the residual
cache and were originally hardcoded to ``residual_mid_l*``,
silently missing every key when the calibration tier-1 substrate
runs.

Resolution: ``analysis.common.sites.residual_key_prefix(module_path)``
is the single source of truth for the prefix string (returns
``"residual_mid"`` for ``*.h_mid``, ``"residual_flat"`` otherwise).
Every consumer derives ``residual_prefix = residual_key_prefix(args.module_path)``
near the top of ``main()`` and substitutes it into both the
buffer-key f-strings AND the file-scan ``startswith`` / slice
logic. The DRY consolidation extends further:
``analysis.common.sites.discover_workspace_sites(workspace_dir, prefix)``
is the SoT for the workspace site-discovery routine; CKA,
effective-dim, KV-rank, and the kv-existence pre-flight check
all delegate to it (the per-protocol ``_discover_*_sites``
functions are now thin wrappers that adapt the canonical
3-tuple ``(layer, repeat, path)`` return shape to each
caller's expected shape).

The workspace file-naming table in [§8.3](#83-workspace-file-naming-contract)
shows the ``residual_mid_l<L>_r<R>.npy`` form because the OWT2
analyse-lncot+adm pipeline always uses ``module_path = h_mid``;
when the calibration substrate runs at ``module_path = h`` the
files written are named ``residual_flat_l<L>_r<R>.npy`` and
all consumers read them transparently via the prefix helper.

---

*End of extend-technical.md.*
