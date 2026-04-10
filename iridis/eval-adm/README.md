# eval-adm: ADM v2 Evaluation and Figure 5 Experiment Matrix

Evaluation package for the Adaptive LN-CoTFormer (ADM v2, 60k steps).
Produces Figure 4 (PPL vs MACs) and a systematic Figure 5 experiment matrix
that isolates methodological confounders in the router weight analysis.

## Background

The original `get_router_weights.py` (commit `2723e30`) contains a cascading
bug (C1 in `docs/reprod-notes.md`): it takes element-wise minimums at the
same position index across repeats, but tokens are reordered between repeats
by `topk` + `take_along_dim`. The position-indexed cascade therefore compares
scores from different tokens, not per-token minimums.

On our hardware (L4, PyTorch 2.2.0+cu118), this bug has negligible visible
effect (<2% relative difference vs correct per-token cascade) because
`torch.topk(sorted=False, k=T)` returns approximately the identity
permutation. The paper's broad Figure 5 distribution likely depended on
`topk` producing significant reordering on the authors' environment (probable
A100, unknown PyTorch version), which we cannot replicate directly.

Beyond this technical bug, the original Figure 5 experiment is
methodologically limited: it answers "at position k in the reordered
sequence, what is the minimum repeat score across repeats?" rather than the
question the authors claim to address ("does the CoTFormer benefit more from
longer training?"). The position-indexed cascade conflates token identity
with sequence position, producing a graph whose interpretation depends on
uncontrolled reordering behaviour. We propose a per-token cascade as a
methodologically sounder alternative.

## Experiment Matrix

Five extraction experiments systematically isolate three confounders:

| Confounder | Variable | Controlled by |
|-----------|----------|---------------|
| Cascade method | Position-indexed (C1 bug) vs per-token (correct) | Comparing PI vs PT cascade outputs within same extraction |
| Extraction method | Custom forward vs hooks | `--method cf` vs `--method hooks` |
| topk reordering | sorted=False (hardware-dependent) vs sorted=True (forced) | `--method cf` vs `--method cf-sorted` |
| Config loading | Raw JSON dict (dtype=None) vs argparse (dtype=bfloat16) | `--config-loading raw` vs `--config-loading argparse` |

### Experiments

| # | Directory | Method | Config | Outputs | What it isolates |
|---|-----------|--------|--------|---------|-----------------|
| 1 | `exp5-original/` | cf | raw | PI + PT + raw | **Baseline**: authors' method. PI vs PT comparison within this directory isolates the cascade bug magnitude. |
| 2 | `exp5-hooks/` | hooks | raw | PT + raw | vs 1: extraction method equivalence |
| 3 | `exp5-cf-sorted/` | cf-sorted | raw | PI + PT + raw | vs 1: topk reordering hypothesis |
| 4 | `exp5-cf-argparse/` | cf | argparse | PI + PT + raw | vs 1: config loading (dtype) confounder |

Each `cf`/`cf-sorted` extraction saves all three cascade variants (PI, PT,
raw) from a single forward-pass run. The cascade bug comparison (PI vs PT)
is between files within the same directory, not between separate runs.

### Expected Results

| Comparison | Question | Prediction |
|-----------|----------|------------|
| 1 PI vs 1 PT | Cascade bug magnitude on L4/PyTorch 2.2 | <2% difference (topk barely reorders) |
| 1 raw vs 2 raw | Hooks = CF for raw histograms? | Identical (multiset equivalence, proven) |
| 1 PI vs 3 PI | Does forced topk sort reproduce paper's Fig 5? | **Key**: if yes, topk reordering is root cause |
| 1 PI vs 4 PI | Does dtype (None vs bf16) affect results? | Likely negligible, must verify empirically |
| 2 PT vs 1 PT | Hooks + PT cascade = CF + PT cascade? | Identical (proof validation) |

### Hypothesis

The paper's broad Figure 5 distribution is an artifact of
`torch.topk(sorted=False)` producing significant token reordering on the
authors' hardware. On our L4, `topk(sorted=False, k=T)` returns approximately
the identity permutation, rendering the C1 cascading bug invisible. Forcing
`sorted=True` (experiment 4) should reproduce the broad distribution by
maximising cross-token score mixing in the position-indexed cascade.

## Output Structure

```
run_N/
  eval_4k0.txt                      PPL at 40k steps
  eval_6k0.txt                      PPL at 60k steps
  eval_summary_ckpt_*.json          Full eval summaries
  exp4/                             Figure 4 (one copy)
    figure4_pareto.png
    eval_per_threshold.npy
    eval_per_layer.npy
  exp5-original/                    Experiment 1 (PI+PT+raw from one extraction)
    40k/
      router_weights_raw.npy        Per-repeat scores, no cascade
      router_weights_picascade.npy  Position-indexed cascade (C1 bug)
      router_weights_ptcascade.npy  Per-token cascade (correct)
    60k/
      (same structure)
    figure5_picascade.png           PI cascade histogram (40k vs 60k)
    figure5_ptcascade.png           PT cascade histogram
    figure5_raw.png                 Raw per-router histogram
  exp5-hooks/                       Experiment 2
  exp5-cf-sorted/                   Experiment 3
  exp5-cf-argparse/                 Experiment 4
```

## Extraction Methods

### Custom forward (`cf`, `cf-sorted`)

Replicates the authors' manual repeat loop from commit `2723e30`. Captures
router weights AFTER `topk` and `sigmoid`, in topk-index order. Also tracks
`active_indices` (position-to-original-token mapping) for per-token cascade.

With `cf-sorted`, the MoDBlock call is inlined with `torch.topk(sorted=True)`
to force deterministic descending-score reordering. This maximises the
cross-token mixing effect that the C1 cascade bug depends on.

### Hooks (`hooks`)

Registers forward hooks on `MoDBlock.mod_router` (nn.Linear) to capture raw
logits BEFORE topk selection. Applies sigmoid in post-processing. Since hooks
capture in original token order, per-token cascade is natural and
position-indexed cascade is identical to per-token (no reordering in captured
data).

Multiset equivalence (proven): for a single repeat's scores,
`{sigmoid(all_logits)} = {sigmoid(topk(logits, k=T))}` because topk with
k=T is a permutation and sigmoid is element-wise. Histograms are
order-insensitive.

### Config loading

- **raw**: Direct JSON dict loading (`config.__dict__ = json.load(f)['args']`).
  Matches the original code exactly. `dtype=null` in summary.json becomes
  `None`, causing `torch.amp.autocast` to default to float16.
- **argparse**: Our config parser. Excludes `dtype`/`device`/
  `distributed_backend` from JSON loading, applying argparse defaults instead
  (`dtype="torch.bfloat16"`). This is the config path our previous extraction
  scripts used.

## Figure 4

Produced once using hooks extraction (val split). The MAC sweep
(`get_ppl_per_mac.py`) and Pareto plot (`plot_fig4.py`) are independent of
the cascade method and need not be repeated across exp5 variants.

## References

- `docs/reprod-notes.md` C1: Cascading bug description and correction
- `docs/reprod-notes.md` C3: Orphan MoDBlock (5 allocated, 4 used)
- `get_router_weights.py`: Unified extraction script
- `plot_fig5.py`: Figure 5 histogram plotter
