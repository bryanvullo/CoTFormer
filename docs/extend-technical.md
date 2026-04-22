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

### 7.2 Debugging log

Debugging artefacts that are worth preserving across sessions (beyond
what git history captures) are recorded below. Each entry carries:
(a) the symptom observed, (b) the root cause, (c) the fix applied,
(d) the audit trail (commit hash or file change).

*(no entries yet)*

---

*End of extend-technical.md.*
