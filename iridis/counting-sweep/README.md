# RQ9 Counting Sweep

HPC package for the RQ9 twin-pilot + 12-cell Arm B initial wave per
`docs/extend-notes.md` §1.2 RQ9 and DEC-033. Trains the Chang and Bisk
2024 [25] inductive-counting task on one baseline 4L standard Transformer
(Arm A, Pilot 1) and four CoTFormer variants (Arm B, V1-V4).

## Contents

| File | Role |
|---|---|
| `job.sh` | Single-cell SLURM job script. Accepts `--arm`, `--variant`, `--seed`, `--n-embd`. Self-submits on the login node; dispatches on arm and variant to build `TRAIN_ARGS`. |
| `README.md` | This file. Documents the 12+5 cell matrix, wall-time estimates, output layout, and downstream DV-3/DV-4/RQ9b analysis pipeline. |

Launchers for the full sweep live at
`scripts/rq9_launch_pilot1.sh` (Arm A, 5 seeds) and
`scripts/rq9_launch_arm_b.sh` (Arm B, 12 cells). Both call this
directory's `job.sh` once per cell via `sbatch`.

## Cell matrix

### Arm A -- Chang and Bisk-exact reproduction (Pilot 1, CONTROL)

| Cell | Model | n_embd | Seeds | Per-cell time (1x L4) | Total |
|---|---|---|---|---|---|
| baseline | `but_full_depth` (4L standard) | 1024 | 1234, 12, 123, 1234 + 1, 1234 + 2 | ~1-1.5 h | ~5-7.5 h |

Seeds follow Chang and Bisk 2024 [25]'s `[1234, 12]`-style convention
extended to 5 seeds for symmetry with the main-sweep `n_embd=1024`
panel (extend-notes §1.2 RQ9 Pilot 1 row).

Five seeds live as five individual `sbatch` submissions (one SLURM job
per seed) so a failure on any seed does not block the others.

### Arm B -- CoTFormer-native full width sweep (32 cells, DEC-035)

Per DEC-035 the DEC-033 secondary-width expansion is committed
unconditionally so the `n_embd x architecture` factorial alternative
has four non-empty cells under the DEC-028 ANOVA. Under the
initial-wave 12-cell variant roster at `n_embd=1024` only, every
candidate variant-to-factor mapping produced exactly one empty 2x2
cell (3-of-4 non-empty; see `docs/extend-notes.md §1.9 DEC-035`);
the secondary-width expansion populates the `n_embd x ln_mid`
factorial alternative with all cells non-empty.

| Variant | Model | Reserved | ln_mid | depth_emb | Widths x Seeds | Cells | Per-cell wall-time | Total |
|---|---|---|---|---|---|---|---|---|
| V1 | `cotformer_full_depth` | 0 + 0 | no | no | 1024x3 + 512x2 + 256x2 + 128x1 | 8 | 12-18 / 6-9 / 3-5 / 2-3 h | ~48-80 h |
| V2 | `cotformer_full_depth` | 2 + 1 | no | no | same | 8 | ~1.5x V1 (7L eff) | ~72-120 h |
| V3 | `cotformer_full_depth_lnmid_depthemb` | 0 + 0 | yes | yes | same | 8 | ~V1 | ~48-80 h |
| V4 | `adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final` | 0 + 0 | yes | yes | same | 8 | ~V1 (routing disabled via `min_repeat = n_repeat = 4`) | ~48-80 h |

Per-width seed rationale: `n_embd=1024` carries the primary statistical
weight of the DEC-028 factorial (3 seeds matches the DEC-031 project
convention). Smaller widths contribute secondary data on the
`n_embd x ln_mid` interaction term with progressively fewer seeds as
the variance estimates are less critical at small parameter counts.

Total Arm B cells: 4 variants x (3 + 2 + 2 + 1) width-seed = **32
cells**. Wall-clock estimate on 2x L4 parallel submission:
`~9-14 L4-days`. Arm A (Pilot 1) adds ~5-7.5 L4-hours; grand total
~9.5-14.5 L4-days.

Running a subset is supported via the launcher filters; see
`scripts/rq9_launch_arm_b.sh --help` for the `--variant / --n-embd /
--seed` flag combinations.

### Arm C -- C&B-adapted CoTFormer (deferred)

Arm C per DEC-033 is contingent on Arm B divergences and is NOT
commissioned in this DIR. A follow-up DIR will add an Arm C dispatch
branch to `job.sh` (ReLU activation, `max_grad_norm=0.3` (NOT 1.0 --
Arm C specifically uses the paper-prose value, NOT the codebase
dead-code reconciliation), CoTFormer-specific `scale_attn_by_inverse_layer_idx`)
only if Arm B results show architectural-vs-training-regime confound.

## Hyperparameter matrix (Arm A vs Arm B)

Single source of truth for the training-regime divergence is the
regime-dispatch block in `job.sh`. Seven of these differences come
from the DEC-020 reconciliation documented in
`docs/reprod-notes.md` §B12 (the seven divergences between Chang and
Bisk 2024 [25]'s paper prose and the dead code at
`inductive_counting_with_LMs/scripts/causal_transformer/{config,trainer}.py`).
The remaining three (`activation`, `tie_word_embeddings`,
`scale_attn_by_inverse_layer_idx`) are architectural bits also
documented in DEC-019 / DEC-020.

| Parameter | Arm A (C&B-exact) | Arm B (CoTFormer-native) | Divergence source |
|---|---|---|---|
| `--activation` | `relu` | `gelu` | `inductive_counting_with_LMs/.../config.py:26` |
| `--tie_word_embeddings` | `False` | `True` | `.../config.py:25`; CoTFormer hardcodes True |
| `--scale_attn_by_inverse_layer_idx` | `True` | `False` | `.../config.py:20` |
| `--grad_clip` | `1.0` | `1.0` | `.../trainer.py:231` hardcoded literal (config.py:28 value 0.3 is dead-code per `docs/reprod-notes.md` §B13) |
| `--scheduler` | `constant_with_warmup` | `cos` | `.../trainer.py:72` |
| `--weight_decay` | `0.01` | `0.1` | `.../trainer.py:70` PyTorch AdamW default `wd=0.01` |
| `--beta1` | `0.9` | `0.9` | shared |
| `--beta2` | `0.999` | `0.95` | `.../trainer.py:70` PyTorch AdamW default `beta2=0.999` |
| `--n_head` | `8` | `4` | `.../config.py:13` `num_attention_heads=8` at `hidden_size=1024` |
| `--dtype` | `torch.float16` | `torch.bfloat16` | `.../trainer.py:52` `mixed_precision="fp16"` |
| `--lr` | `1e-4` | `1e-3` | `.../config.py:40` `learning_rate=1e-4` |
| `--warmup_percent` | `0.0096` | `0.0096` | shared: `3000 / 312500 = 0.0096` matching `.../config.py:39` `warmup_steps=3000` |
| `--iterations` | `312500` | `312500` | shared (DEC-016 step-matched) |
| `--batch_size` | `32` | `32` | shared (DEC-016 step-matched) |
| `--acc_steps` | `1` | `1` | shared (counting fits on a single GPU) |

Note: `--enable_flash=False` (DEC-020 divergence 7, `.../trainer.py:211`)
is NOT exposed as a CLI flag in the current `config/base.py`; the
CoTFormer model classes set the flash-attention backend internally via
`torch.backends.cuda.sdp_kernel`. Pilot 1 inherits the backend default
rather than explicitly disabling flash. If the Pilot 1 OOD accuracy
lands outside the [25] published band, investigate enable_flash as the
first candidate divergence. See `docs/extend-technical.md` §8 if this
flag is later added.

## Output layout

Each `sbatch` submission writes:

```
iridis/counting-sweep/run_<N>/
  cell.txt                         ONE-line cell identifier (e.g.
                                   rq9_arm_b_V1_seed_2357_nembd_1024)
  slurm_<jobid>.out                stdout
  slurm_<jobid>.err                stderr
```

Training output lives at the main.py-constructed path:

```
$EXPS_DIR/counting/<model>/<exp_name>/
  summary.json                     args, step count
  ckpt_<itr>.pt                    every --save_checkpoint_freq (10K) steps
  wandb-metadata.json              (if --wandb)
```

`<exp_name>` is `rq9_arm_<a|b>_<variant>_seed_<N>_nembd_<NEMBD>` so the
13 cells land in distinctive leaves even when they share the same
`<model>` (e.g. V1 and V2 both use `cotformer_full_depth`).
Per the never-mirror-path-construction discipline documented in
`docs/reprod-notes.md` §C4, bash does NOT mirror this path construction.

## Downstream analysis

Once the 13 cells finish training, the downstream analyses consume the
checkpoint artefacts:

| Step | Script | Consumes | Per-DV estimated cost |
|---|---|---|---|
| DV-1 | `eval.py --distributed_backend None` on each cell's latest checkpoint over the OOD test set at lengths `{50, 75, 100, 125, 150, 175, 200}` | `summary.json` + latest `ckpt_<itr>.pt` | ~5 min CPU per cell |
| DV-2 | TBD (paired linear-and-non-linear probe harness; see extend-notes §1.2 RQ9 DV-2) | residual-stream captures per repeat | ~5 min per cell |
| DV-3 | `python -m analysis.counting_dv3_attention --checkpoint <ckpt_dir> --checkpoint-file <ckpt> --output-dir <path>` | checkpoint + OOD test set | ~5 min GPU per cell |
| DV-4 | `python -m analysis.counting_dv4_causal --checkpoint <ckpt_dir> --checkpoint-file <ckpt> --output-dir <path>` | checkpoint + OOD test set | ~10 min GPU per cell (4 ablations x baseline) |
| RQ9b | `python -m analysis.counting_anova_rq9b --results-dir <aggregated-DV-1-results> --output-dir <path>` | aggregated DV-1 accuracies | ~5 min CPU once |

The DV-3 / DV-4 / RQ9b pipelines are Python modules under `analysis/`;
they are invoked per-cell via a follow-up analysis job script that is
NOT part of this directory (the counting-sweep package is
training-only). See `docs/extend-notes.md` §1.3 for the downstream
analysis plan.

## Submitting the sweep

The supported entry points are the launcher scripts:

```bash
# Pilot 1 (Arm A, 5 cells)
bash scripts/rq9_launch_pilot1.sh --dry-run    # preview
bash scripts/rq9_launch_pilot1.sh              # submit (actual)

# Arm B (12 cells)
bash scripts/rq9_launch_arm_b.sh --dry-run     # preview
bash scripts/rq9_launch_arm_b.sh               # submit (actual)
```

Manual single-cell submission is also supported by calling `job.sh`
directly on the login node:

```bash
bash iridis/counting-sweep/job.sh --arm A --variant baseline \
    --seed 1234 --n-embd 1024
bash iridis/counting-sweep/job.sh --arm B --variant V3 \
    --seed 8191 --n-embd 1024
```

The self-submitting wrapper creates `run_<N>/` inside this package
directory and calls `sbatch` with the parsed arguments propagated via
`--export`. Resubmitting the same cell auto-resumes from the latest
checkpoint via `--use_pretrained auto` so 24-h-capped cells that need
more wall clock simply get resubmitted.

Use `--dry-run` on `job.sh` directly to validate the constructed
`TRAIN_ARGS` without touching SLURM:

```bash
bash iridis/counting-sweep/job.sh --arm B --variant V2 --seed 2357 \
    --n-embd 1024 --dry-run
```

## Scope-shedding trigger

Per `docs/extend-notes.md` §0 scope-shedding order and DEC-030, if
Pilot 1 (Arm A) fails to reproduce the [25] per-OOD-length pass band
(IND ~100 %, OOD@200 ~22-31 %), the entire RQ9 sub-stream sheds to
the extension. Pilot 1 is the only strict-reproduction control in
RQ9 and its result gates all Arm B interpretation. See
`docs/reprod-notes.md` §B14 for the reserved Pilot 1 results stub
that the code-wave orch populates post-execution.
