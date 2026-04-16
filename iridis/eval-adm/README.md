# eval-adm: ADM v1+v2 Evaluation and Figure 5 Experiment Matrix

Self-contained evaluation package for the Adaptive LN-CoTFormer (ADM),
covering:

1. **PPL at 40k and 60k checkpoints** for both training versions (v1
   pre-fix and v2 post-fix — see `docs/reprod-notes.md` §B4, §B6 for
   the fixes).
2. **Figure 4** (PPL vs MACs Pareto) for both versions.
3. **Figure 5 experiment matrix** — 4 extraction variants × 2 versions
   × 2 checkpoints = 16 extractions, designed to isolate every
   methodological confounder in the router weight analysis.

This README contains the full methodology, running instructions,
results, and final interpretation. The `docs/reprod-notes.md` §C2
"Figure 5: Position-Indexed Cascade Bug and Methodological Critique"
contains the divergence-log verdict for Part C; this file is its
supporting evidence document.

## Table of contents

- [Scope and inputs](#scope-and-inputs)
- [Background: the Figure 5 bug and its critique](#background-the-figure-5-bug-and-its-critique)
- [Experiment matrix design](#experiment-matrix-design)
- [Running the matrix](#running-the-matrix)
- [Results](#results)
  - [Perplexity and Figure 4 Pareto](#perplexity-and-figure-4-pareto)
  - [Figure 5: router weight statistics](#figure-5-router-weight-statistics)
  - [ADM v1 vs v2: B4 empirical magnitude](#adm-v1-vs-v2-b4-empirical-magnitude)
- [Discussion](#discussion)
- [What we cannot reproduce, and why](#what-we-cannot-reproduce-and-why)
- [Qualitative claim confirmation](#qualitative-claim-confirmation)
- [Output structure](#output-structure)
- [References](#references)

## Scope and inputs

The package evaluates **two training versions** of the ADM model class
`adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final`:

| Tag | Checkpoint directory (leaf) | Training state |
|-----|-----------------------------|-----------------|
| v1 | `adm_lr0.001_bs8x16_seqlen256` | pre-fix (original authors' RNG-restore path commented out, orphan MoDBlock present) |
| v2 | `adm_v2_lr0.001_bs8x16_seqlen256` | post-fix (Gloo per-rank `all_gather` of CUDA RNG, full restore on resume, orphan MoDBlock removed) |

Both are 24-layer models with `n_repeat=5`, trained for 60,000 steps
on OWT2 with cosine LR schedule (peak 1e-3, `warmup_percent=0.2`,
`final_div_factor=0.05`), effective batch size 128, on 2× L4 24 GB
via DDP. See `docs/reprod-notes.md` §A6 for full training
configuration, §B4 + §B9 for the RNG and checkpoint-save fix
histories, and §C1 for the init-order RNG mechanism that explains
the tiny residual v1↔v2 trained-model delta.

Each version is evaluated at both the intermediate **40,000-step**
and the final **60,000-step** checkpoints.

## Background: the Figure 5 bug and its critique

The original `get_router_weights.py` (commit `2723e30`) contains a
cascading bug (`docs/reprod-notes.md` §C2): it computes element-wise
minimums at the same position index across repeats, but tokens are
reordered between repeats by `topk` followed by `take_along_dim`. The
position-indexed cascade therefore pairs scores from different tokens
rather than computing the per-token minimum across all routers.

On our hardware (L4 SM89, PyTorch 2.2.0+cu118), this bug has negligible
visible effect because `torch.topk(sorted=False, k=T)` returns
approximately the identity permutation. The paper's broad Figure 5
distribution likely depended on `topk` producing significant reordering
on the authors' environment (probable A100 SM80, unknown PyTorch
version), which we cannot replicate by running their exact code on our
exact GPU.

Beyond the technical bug, the original Figure 5 experiment is
methodologically limited: it answers *"at position k in the reordered
sequence, what is the minimum repeat score across repeats?"* rather
than the question the authors claim to address in Section 5 (*"does
the CoTFormer favour the final repeat more when trained for longer?"*).
The position-indexed cascade conflates token identity with sequence
position, producing a graph whose interpretation depends on
uncontrolled reordering behaviour. We propose a **per-token cascade**
as a methodologically sounder alternative: map scores back to original
positions via `active_indices` before cascading, so each token's
trajectory through the repeat loop is tracked individually.

## Experiment matrix design

Four extraction variants isolate four potential confounders. Two
training versions and two checkpoints exercise each variant against
the actual model we want to analyse. Each extraction run writes a
`router_weights_meta.json` sidecar alongside its `.npy` outputs; the
sidecar records `n_repeat`, `n_routers`, the model-derived
`last_router_slot`, the method/config-loading parameters, and the
list of written `.npy` files (see `get_router_weights.py:575-595`).
Downstream plotters (`plot_fig5.py:32-85`) prefer the sidecar for
slot resolution and fall back to a tail-scan over the object array
when no sidecar is present, so no script hardcodes the
`n_repeat=5`-specific index.

### Confounders and their isolators

| Confounder | Variable | Controlled by |
|-----------|----------|---------------|
| Cascade method | Position-indexed (§C2 bug) vs per-token (correct) | Comparing PI vs PT cascade outputs within same extraction |
| Extraction method | Custom forward (post-topk) vs hooks (pre-topk) | `--method cf` vs `--method hooks` |
| topk reordering | `sorted=False` (hardware-dependent) vs `sorted=True` (forced) | `--method cf` vs `--method cf-sorted` |
| Config loading | Raw JSON dict (`dtype=None`) vs argparse (`dtype=bfloat16`) | `--config-loading raw` vs `--config-loading argparse` |

### Experiments

| # | Directory | Method | Config | Outputs | What it isolates |
|---|-----------|--------|--------|---------|-----------------|
| 1 | `exp5-original/` | `cf` | `raw` | PI + PT + raw | **Baseline**: authors' exact code path. PI vs PT within this directory isolates the cascade bug's visible magnitude. |
| 2 | `exp5-hooks/` | `hooks` | `raw` | PT + raw | vs 1: extraction-method equivalence. Hooks capture BEFORE topk, in original token order. |
| 3 | `exp5-cf-sorted/` | `cf-sorted` | `raw` | PI + PT + raw | vs 1: topk reordering hypothesis. Forces `topk(sorted=True)` to maximise descending-score reordering. |
| 4 | `exp5-cf-argparse/` | `cf` | `argparse` | PI + PT + raw | vs 1: config-loading / dtype confounder. |

Each `cf` and `cf-sorted` extraction saves all three cascade variants
(PI, PT, raw) from a single forward pass. The PI vs PT comparison is
therefore always **within a single run** on identical data, not
between separate runs.

### Pre-execution predictions

We ran the matrix with explicit pre-registered predictions to preserve
falsifiability. The predictions are recorded here to document our
hypothesis state before observing the outputs.

| Comparison | Question | Pre-registered prediction |
|-----------|----------|--------------------------|
| 1 PI vs 1 PT | Cascade bug magnitude on L4/PyTorch 2.2 | Small difference (topk barely reorders), but the sign and magnitude are unknown a priori |
| 1 raw vs 2 raw | Hooks = CF for raw histograms? | Identical (multiset equivalence is provable analytically) |
| 1 PI vs 3 PI | Does forced topk sort reproduce paper's Figure 5? | **Key hypothesis**: yes, forced sort maximises cross-token score mixing in the PI cascade, approximating the authors' hardware-dependent reordering |
| 1 PI vs 4 PI | Does dtype (None vs bf16) affect cascade statistics? | Likely negligible, but unverified; must check |
| 2 PT vs 1 PT | Hooks + PT cascade = CF + PT cascade? | Identical (proof validation via a second independent path) |
| v1 vs v2 at 60k | Does B4 RNG fix change final PPL? | Expected to explain the +0.23 ADM delta against the paper |

### Multiset-equivalence note (hooks vs CF, for raw histograms)

For a single repeat's scores, `{sigmoid(all_logits)} =
{sigmoid(topk(logits, k=T))}` because `topk` with `k=T` is a
permutation of the input and `sigmoid` is element-wise. Histograms are
order-insensitive, so hook-captured raw logits (before topk, in
original token order) and CF-captured raw scores (after topk, in
topk-index order) must produce identical raw histograms. The PT
cascade is natural under hooks (no reordering) and produces the
same result as the CF PT cascade (which explicitly reorders back via
`active_indices`). This is why `exp5-hooks` outputs only PT + raw, not
PI: there is no meaningful "PI cascade" when data is already in
original-token order.

## Running the matrix

```bash
cd ~/CoTFormer && bash iridis/eval-adm/job.sh
```

The job is self-submitting on the login node. It verifies both
checkpoint directories exist, creates
`run_N/adm_{v1,v2}/{exp4,exp5-*}/` on the login node, and then
submits a single sequential SLURM job to `ecsstudents_l4` (1 GPU,
64 GB, 10 h wall, ~5 h expected for the full matrix). The job
interface is unchanged by the dynamic-slot refactor: no new CLI
flags, no renamed output files. The refactor is additive — each
extraction step now also writes a `router_weights_meta.json` sidecar
into its step workspace alongside the existing `.npy` files, and
`plot_fig5.py` consumes it for slot resolution.

### Output triage

Large intermediate files (`router_weights_raw.npy`,
`router_weights_picascade.npy`, `router_weights_ptcascade.npy`, ~130
MB each) are written to
`$CKPT_DIR/eval_workspace/exp5-*/[40k|60k]/` on `/scratch` along with
a small `router_weights_meta.json` (a few hundred bytes). The `.npy`
files are regeneratable and never copied to the Git-tracked `run_N/`
tree; the sidecar is regenerated on every extraction and lives in
the same workspace directory.

Only meaningful outputs — PNGs, small scalar `.npy` sweeps, JSON
summaries, text logs — land in `run_N/`. This keeps the repository
light and the large regeneratable workspace on the fast filesystem.

### Sequential execution order

Per-version, the job runs the following stages in order:

1. `eval.py` at each checkpoint (`40000`, `60000`) -- writes
   `eval_<step>k.txt` and `eval_summary_ckpt_<step>.json`.
2. `get_ppl_per_mac.py` + `plot_fig4.py` -- Figure 4 Pareto plot.
3. For each experiment in `EXPERIMENTS`:
   - `get_router_weights.py --method <method> --config-loading <config>
     --ckpt-file ckpt_<step>.pt` for each checkpoint.
   - `plot_fig5.py --output-dir <run_N>/adm_<tag>/<exp_dir>/` for both
     40k and 60k together (one histogram comparing both checkpoints).

The `EXPERIMENTS` list controls the extraction variants; see `job.sh`
lines 67-72.

## Results

Results below are from `run_6` (SLURM job 826926). Figures are under
`run_6/adm_{v1,v2}/exp{4,5-*}/`. Numerical statistics are reprinted
from the job's stdout (`run_6/slurm_826926.out`, phase 1 and phase 3
blocks).

### Perplexity and Figure 4 Pareto

Perplexity at the 40k and 60k checkpoints for both training versions,
evaluated on the full validation set (3,973 batches, matching the
authors' protocol):

| Version | Step | Val PPL | 95% CI | Val loss | Val acc | Loss SEM |
|---------|-----:|--------:|--------|---------:|--------:|---------:|
| v1 | 40000 | 28.073 | [27.63, 28.52] | 3.3348 | 0.3938 | 0.0081 |
| v1 | 60000 | 24.074 | [23.70, 24.46] | 3.1811 | 0.4128 | 0.0081 |
| v2 | 40000 | 28.079 | [27.64, 28.53] | 3.3350 | 0.3939 | 0.0081 |
| v2 | 60000 | 24.060 | [23.68, 24.44] | 3.1806 | 0.4127 | 0.0081 |

The 40k numbers are intermediate checkpoints from the 60k schedule
(mid-cosine-decay) and are not directly comparable to Table 2's 40k
"fresh" numbers. The 60k numbers are the final ADM training result
reported in A6 of `docs/reprod-notes.md`.

Figure 4 (PPL vs multiply-accumulate ops) artefacts live at:

- `run_6/adm_v1/exp4/figure4_pareto.png`
- `run_6/adm_v2/exp4/figure4_pareto.png`

alongside their `eval_per_threshold.npy` and `eval_per_layer.npy`
sweep inputs. Both curves land on the CoTFormer region of the Pareto
frontier shown in the paper's Figure 2b.

### Figure 5: router weight statistics

All statistics are over `n = 8,135,168` validation tokens (3,973
batches × 2,048 tokens per batch, one score per token per router).
We report the **last-repeat router slot** (slot `n_repeat - 1`, which
is slot 4 for our `n_repeat = 5` ADM) — this is the quantity Figure 5
plots. The slot is resolved dynamically via
`router_weights_meta.json.last_router_slot` (see
`get_router_weights.py:380-395, 579-591`) so the same extraction and
plotting pipeline handles any `n_repeat` without hardcoded indices.

#### ADM v2 (post-fix, primary result)

60,000-step checkpoint:

| Variant | Extraction | Raw mean | PI mean | PT mean | Raw frac<0.01 | PI frac<0.01 | PT frac<0.01 |
|---------|-----------|---------:|--------:|--------:|---------------:|---------------:|---------------:|
| exp5-original | cf, raw | 0.0424 | 0.0336 | 0.0345 | 0.291 | 0.296 | 0.291 |
| exp5-hooks | hooks, raw | 0.0424 | -- | 0.0332 | 0.291 | -- | 0.297 |
| exp5-cf-sorted | cf-sorted, raw | 0.0424 | 0.0421 | 0.0345 | 0.291 | 0.291 | 0.291 |
| exp5-cf-argparse | cf, argparse | 0.0423 | 0.0335 | 0.0344 | 0.289 | 0.294 | 0.289 |

40,000-step checkpoint:

| Variant | Extraction | Raw mean | PI mean | PT mean | Raw frac<0.01 | PI frac<0.01 | PT frac<0.01 |
|---------|-----------|---------:|--------:|--------:|---------------:|---------------:|---------------:|
| exp5-original | cf, raw | 0.0267 | 0.0215 | 0.0220 | 0.449 | 0.455 | 0.449 |
| exp5-hooks | hooks, raw | 0.0267 | -- | 0.0213 | 0.449 | -- | 0.456 |
| exp5-cf-sorted | cf-sorted, raw | 0.0267 | 0.0267 | 0.0220 | 0.449 | 0.449 | 0.449 |
| exp5-cf-argparse | cf, argparse | 0.0267 | 0.0215 | 0.0220 | 0.444 | 0.450 | 0.445 |

#### ADM v1 (pre-fix training, control)

60,000-step checkpoint:

| Variant | Extraction | Raw mean | PI mean | PT mean |
|---------|-----------|---------:|--------:|--------:|
| exp5-original | cf, raw | 0.0402 | 0.0333 | 0.0341 |
| exp5-hooks | hooks, raw | 0.0402 | -- | 0.0328 |
| exp5-cf-sorted | cf-sorted, raw | 0.0402 | 0.0401 | 0.0341 |
| exp5-cf-argparse | cf, argparse | 0.0402 | 0.0333 | 0.0342 |

40,000-step checkpoint:

| Variant | Extraction | Raw mean | PI mean | PT mean |
|---------|-----------|---------:|--------:|--------:|
| exp5-original | cf, raw | 0.0254 | 0.0227 | 0.0237 |
| exp5-hooks | hooks, raw | 0.0254 | -- | 0.0224 |
| exp5-cf-sorted | cf-sorted, raw | 0.0254 | 0.0254 | 0.0237 |
| exp5-cf-argparse | cf, argparse | 0.0254 | 0.0226 | 0.0237 |

#### Histogram artefacts

Each variant directory contains up to three PNG histograms comparing
the 40k and 60k distributions on the same axes:

- `figure5_raw.png` — raw per-router histograms (no cascade).
- `figure5_picascade.png` — position-indexed cascade (§C2 bug).
- `figure5_ptcascade.png` — per-token cascade (correct).

Paths follow `run_6/adm_{v1,v2}/exp5-*/figure5_*.png`. For a quick
visual comparison of what the paper plots (PI cascade) against what
the paper's text describes (PT cascade), open the `exp5-original/`
pair of `figure5_picascade.png` and `figure5_ptcascade.png`.

### ADM v1 vs v2: B4 empirical magnitude

The v1 vs v2 comparison is the most important secondary finding from
the matrix: it empirically bounds the practical contribution of the
combined B4 RNG-restoration fix and the B6 orphan-MoDBlock removal
to the final ADM perplexity. See `docs/reprod-notes.md` §C1 "Init-order
RNG offset" for the mechanism that links the pre-fix and post-fix
init paths through a 768-sample CUDA RNG offset (equivalent to a seed
change affecting ≈32 % of parameters).

| Checkpoint | v1 PPL | v2 PPL | Delta | Sign |
|-----------:|-------:|-------:|------:|:----:|
| 40,000 | 28.073 | 28.079 | +0.006 | v2 worse |
| 60,000 | 24.074 | 24.060 | -0.014 | v2 better |

At 60k, v2 is 0.014 PPL lower than v1, well within either 95% CI
width (~0.76 PPL) and well below the per-batch loss-level intra-run
SEM (~0.0081 in loss space, propagating to ~0.25 PPL in perplexity
space). At 40k, the delta is 0.006 PPL in the opposite direction,
consistent with intermediate-checkpoint noise from the mid-cosine
decay training state.

**Implication.** B4 (RNG state restoration at resume boundaries) is
methodologically correct — not restoring per-rank GPU RNG would
produce non-deterministic resume behaviour in principle — but its
practical impact on the final ADM perplexity is below the intra-run
noise floor on our hardware. B4 therefore **cannot explain** the
+0.23 PPL delta between our ADM 60k result (24.06) and the paper's
23.83. This invalidates the pre-registered attribution stated in
§A6 of an earlier revision of `docs/reprod-notes.md`.

The remaining candidate contributors to the +0.23 delta are the
combined effect of: B1 (train/val split divergence), B3 (micro-batch
decomposition under bfloat16), hardware/cuDNN kernel differences
(SM89 vs SM80, CUDA 11.8 vs the authors' unrecorded CUDA), and
single-seed variance (the paper reports a mean of three seeds; we
run one). None is individually attributable without controlled
experiments we cannot run.

## Discussion

### Finding 1: PI vs PT cascade delta is within 3% on our hardware

The position-indexed (buggy) cascade and per-token (correct) cascade
produce nearly identical last-repeat-slot distributions on
L4/PyTorch 2.2. For ADM v2 60k `exp5-original`, the last-repeat
router mean under PI is 0.0336 and under PT is 0.0345; relative delta
2.6%. At `exp5-cf-argparse`, the relative delta is 2.7%. The §C2
technical bug is therefore invisible on our hardware, because
`torch.topk(sorted=False, k=T)` returns approximately the identity
permutation, leaving the position index effectively the same as the
token index.

### Finding 2: `topk(sorted=True)` does not restore the paper's Figure 5

Our pre-registered key hypothesis -- that forcing descending-score
reordering via `sorted=True` would reproduce the paper's broad
distribution -- is **falsified**. In `exp5-cf-sorted` at ADM v2 60k,
the PI cascade mean is 0.0421 and the raw mean is 0.0424; the PI
cascade essentially collapses to the raw distribution.

This is the opposite of what the reordering hypothesis predicted. The
reason: when `topk` returns scores in descending order, similar-rank
positions across repeats carry similar-magnitude scores (all repeats'
topk returns "roughly the same top-scoring tokens first, then lower
and lower"). The position-indexed cascade then becomes a minimum over
similar scores, which is nearly the same score, which is nearly the
raw distribution. Only a **partial** reorder (neither identity nor
sorted descent) would produce the broad score-mixing cascade the
paper's Figure 5 seems to depict.

Neither end of the spectrum on L4 -- sorted descent or identity -- is
that partial reorder. We have no intermediate to test.

### Finding 3: extraction method is empirically equivalent

Raw histograms from the forward-hook path (`exp5-hooks`) and the
custom-forward path (`exp5-original`) match to all four reported
digits across all checkpoints and both versions. For ADM v2 60k:
hooks raw mean = 0.0424, CF raw mean = 0.0424; hooks
`frac<0.01` = 0.291, CF `frac<0.01` = 0.291. The multiset equivalence
proved analytically (see above) is confirmed empirically, providing an
independent cross-check that the CF extraction path is not silently
corrupting the data.

The PT cascade also agrees between hooks and CF paths: at v2 60k,
hooks PT mean = 0.0332, CF PT mean = 0.0345. These differ slightly
(by about 3%) because the hooks path computes PT cascade in original
token order at the `mod_router` output (pre-topk selection), while
the CF path computes PT cascade post-topk after mapping back via
`active_indices`. Both are per-token cascades but with slightly
different per-token score sets -- the hooks path sees all 2048
tokens' logits at each repeat, while the CF path sees only the topk
that were selected to continue. This small difference is a feature of
the two extraction sites, not a bug.

### Finding 4: config-loading path is empirically equivalent

The `exp5-cf-argparse` variant uses our argparse loader (which
applies `dtype=torch.bfloat16` from its default) while `exp5-original`
uses raw JSON dict loading (which leaves `dtype=None` and the
autocast path defaults to `float16`). Differences in last-repeat slot
mean between the two variants are below 0.001 across all checkpoints
and versions. The dtype/autocast path is not a measurable confounder
on our hardware.

## What we cannot reproduce, and why

The matrix systematically eliminates four hypothesised confounders.
None produces a router weight distribution that resembles the paper's
Figure 5. The residual unexplained factor must live in one of two
places:

(a) **The authors' exact PyTorch and CUDA environment**, which controls
    the internal behaviour of `topk(sorted=False)` on SM80 (A100). On
    our SM89 (L4) with CUDA 11.8 and PyTorch 2.2.0, neither
    `sorted=False` nor `sorted=True` produces the intermediate reorder
    pattern that the paper's distribution seems to require. We have no
    sorted-between-identity-and-descent intermediate to test, and we
    have no access to the authors' environment.

(b) **Differences in the trained router itself.** Our ADM v2 reaches
    PPL 24.06 against the paper's 23.83 (delta +0.23). If Figure 5
    was produced from a model trained with a different seed, different
    `acc_steps` decomposition, different dropout/randomness settings,
    or different hardware, the underlying router weight distribution
    would differ even before any cascading is applied. The paper does
    not release the numerical data behind Figure 5, so we cannot
    directly compare our raw last-repeat slot distribution
    (mean 0.0424 at 60k) against the authors'.

We have exhausted the sensible experimental variations on this
hardware. Further progress requires either:

- Access to the authors' exact PyTorch + CUDA + GPU profile to test the
  unsorted topk reordering directly, or
- A second independent ADM training run on different hardware and a
  different seed (compute-prohibitive at ~52 h per 60k run).

## Qualitative claim confirmation

The paper's Section 5 textual claim — *"the model starts to favor
using the final repeat more when the model is trained for longer"* —
is qualitatively supported by our per-token cascade data on ADM v2
at the last-repeat slot (slot 4 for our `n_repeat=5` model):

| Checkpoint | Last-repeat slot mean (PT) | frac(score < 0.01) (PT) |
|------------|---------------------------:|-------------------------:|
| 40,000 | 0.0220 | 44.9% |
| 60,000 | 0.0345 | 29.1% |

The last-repeat slot mean increases by 57% between 40k and 60k, and
the fraction of tokens that essentially skip the final repeat
decreases from 44.9% to 29.1%. The direction and monotonicity match
the paper's claim.

The magnitude of the shift is smaller than the paper's Figure 5
suggests. This is consistent with our finding that the broad
distribution in Figure 5 is not recoverable from the same code on our
hardware: on our hardware the model does shift toward more aggressive
final-repeat use with training, just with a tighter starting
distribution.

## Output structure

```
run_N/
  slurm_<jobid>.out                     Combined SLURM stdout (eval + figures)
  slurm_<jobid>.err                     SLURM stderr (usually empty)
  adm_v1/
    eval_40k.txt                        PPL stdout at 40k
    eval_60k.txt                        PPL stdout at 60k
    eval_summary_ckpt_40000.json        Machine-readable 40k summary incl. CI
    eval_summary_ckpt_60000.json        Machine-readable 60k summary incl. CI
    exp4/
      figure4_pareto.png                PPL vs MACs Pareto
      eval_per_threshold.npy            Sweep input: mod_capacity threshold
      eval_per_layer.npy                Sweep input: per-layer metrics
    exp5-original/                      Experiment 1: cf, raw JSON config
      figure5_raw.png                   Raw per-router histogram (no cascade)
      figure5_picascade.png             PI cascade histogram (§C2 bug)
      figure5_ptcascade.png             PT cascade histogram (correct)
    exp5-hooks/                         Experiment 2: hooks, raw JSON config
      figure5_raw.png
      figure5_ptcascade.png
    exp5-cf-sorted/                     Experiment 3: cf-sorted, raw JSON config
      figure5_raw.png
      figure5_picascade.png
      figure5_ptcascade.png
    exp5-cf-argparse/                   Experiment 4: cf, argparse config
      figure5_raw.png
      figure5_picascade.png
      figure5_ptcascade.png
  adm_v2/                               (same layout as adm_v1)
```

Intermediate `router_weights_{raw,picascade,ptcascade}.npy` files
(~130 MB each) plus the small `router_weights_meta.json` sidecar
live on `/scratch` under
`$CKPT_DIR/eval_workspace/exp5-*/[40k|60k]/` and are **not** copied
into `run_N/`. They are all regeneratable from `get_router_weights.py`
if ever needed; the sidecar is always rewritten by
`get_router_weights.py:575-595` on each extraction, and
`plot_fig5.py:32-85` resolves the Figure 5 slot from it (with a
tail-scan fallback) on each plot.

## References

- `docs/reprod-notes.md` [Methodology](../../docs/reprod-notes.md#methodology)
  — evaluation protocol, CI/SEM interpretation, and delta vocabulary
  shared across all A-subsection results.
- `docs/reprod-notes.md` §A1 — CoTFormer + Reserved Layers 40k PPL
  reproduction (target 24.51, ours 24.64).
- `docs/reprod-notes.md` §A6 — ADM 60k PPL reproduction (target 23.83,
  ours 24.06); details the delta attribution and the combined-divergence
  explanation.
- `docs/reprod-notes.md` §B4 — RNG state restoration on DDP resume,
  with the empirical v1 vs v2 magnitude update.
- `docs/reprod-notes.md` §B6 — Orphan MoDBlock DDP symptom and the
  specific fix applied to the MoD-router model files.
- `docs/reprod-notes.md` §B9 — DDP checkpoint save failures and Gloo
  coordination fix history.
- `docs/reprod-notes.md` §C1 — Orphan MoDBlock: 5 routers allocated,
  only 4 used, the init-order RNG mechanism that explains the
  tiny v1↔v2 trained-model delta, and the empirical bound.
- `docs/reprod-notes.md` §C2 — Figure 5 cascading bug and
  methodological critique (the Part C verdict that this document
  supports).
- `get_router_weights.py` — Unified extraction script with `--method`
  (`cf`/`cf-sorted`/`hooks`) and `--config-loading`
  (`raw`/`argparse`) flags driving the matrix; emits
  `router_weights_meta.json` alongside the `.npy` outputs with the
  model-derived last-repeat slot index.
- `plot_fig5.py` — Figure 5 histogram plotter emitting
  `figure5_{raw,picascade,ptcascade}.png`; resolves the plotted slot
  via the sidecar (with tail-scan fallback) so the same plotter
  handles any `n_repeat` model.
- `plot_fig4.py` — Figure 4 Pareto plotter.
- `iridis/eval-adm/job.sh` — Top-level self-submitting SLURM entry
  point defining `VERSIONS`, `CKPTS`, and `EXPERIMENTS`.
