# analyze-lncot+adm: Mechanistic Analysis HPC Package

Wave-1 deliverable covering Protocols A (unweighted Logit Lens),
A-ext (canonical Belrose 2023 Tuned Lens), and E (Residual
Diagnostics) across the five checkpoint matrix rows (C1-C5) per
`docs/extend-notes.md` §1.5 Analysis Matrix. Wave-2 will extend this
package with Protocols B (debiased CKA), C (attention taxonomy),
D (router analysis), D-calibration (entropy calibration), F
(effective dimensionality), G (KV compressibility), H (interpolation
validity), and I (depth-embedding freeze).

The package is split along the canonical GPU/CPU compute boundary:

- `job-gpu.sh` runs the Stage-1 collector and the Stage-3 Protocol A
  metrics inline. The forward pass is the expensive GPU-bound step
  (~10 min per checkpoint on an L4 for 2048 tokens x 4 batches) and
  is not repeated by downstream protocols; instead it persists the
  per-(layer, repeat) residual cache plus the per-token next-token
  targets into `/scratch/$USER/analysis_workspace/<tag>/` for the
  CPU stage to consume.
- `job-cpu.sh` runs Stage-4 analyses on the pre-computed residual
  cache. Protocol A-ext (Tuned Lens) trains 105 affine translators
  per checkpoint via SGD + Nesterov at lr=1.0 for 250 steps with
  linear decay (the canonical Belrose 2023 recipe, DEC-017 frozen
  specification); Protocol E computes L2 / variance / L_inf curves
  per (mid-layer, repeat) and flags rogue-direction emergence.

## Table of contents

- [Scope and inputs](#scope-and-inputs)
- [Experiment matrix](#experiment-matrix)
- [Running the matrix](#running-the-matrix)
- [Output structure](#output-structure)
- [Protocol A interpretation guide](#protocol-a-interpretation-guide)
- [Protocol A-ext interpretation guide](#protocol-a-ext-interpretation-guide)
- [Protocol E interpretation guide](#protocol-e-interpretation-guide)
- [Triangulation decision matrix](#triangulation-decision-matrix)
- [Workspace layout and disposability](#workspace-layout-and-disposability)
- [References](#references)

## Scope and inputs

The package evaluates five checkpoints covering the three structural
variants in the analysis matrix:

| Tag | Checkpoint directory (leaf) | Model class | Notes |
|-----|-----------------------------|-------------|-------|
| c1_cotres_40k | `cotformer_full_depth_res_only_lr0.001_bs8x16_seqlen256` | `cotformer_full_depth` | CoTFormer + Reserved, 40k, no `ln_mid` |
| c2_lncot_40k | `cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256` | `cotformer_full_depth_lnmid_depthemb` | LN-CoTFormer, 40k intermediate |
| c3_lncot_60k | `cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256` | `cotformer_full_depth_lnmid_depthemb` | LN-CoTFormer, 60k (primary analytic target) |
| c4_mod_40k | `but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_lr0.001_bs8x16_seqlen256` | `but_mod_efficient_sigmoid_lnmid_depthemb_random_factor` | MoD-routing variant, 40k |
| c5_adm_v2_60k | `adm_v2_lr0.001_bs8x16_seqlen256` | `adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final` | Adaptive Depth Model v2, 60k (target for router-aware protocols) |

The checkpoint paths derive from `$EXPS_DIR` which
`iridis/env.sh` sets to the project-shared scratch root
(`/scratch/ab3u21/exps`). The VERSIONS array at the top of each
job script is the canonical list; edit it if your run history
materialises different leaf names.

Default run parameters match the `docs/extend-notes.md` §1.2 RQ1
"Inter-batch robustness" specification: 4 independent batches of
2048 tokens each, starting at offsets 0, 2048, 4096, 6144 of the
OWT2 validation stream; seed 2357.

## Experiment matrix

Per `docs/extend-notes.md` §1.5 Analysis Matrix, the three
Wave-1 protocols apply as follows:

| Protocol | C1 | C2 | C3 | C4 | C5 |
|----------|:--:|:--:|:--:|:--:|:--:|
| A: Logit Lens | Yes | Yes | Yes | Yes | Yes |
| A-ext: Tuned Lens | Yes | Yes | Yes | Yes | Yes |
| E: Residual Diagnostics | Yes | Yes | Yes | -- | -- |

Protocol E runs on C1/C2/C3 only (the observational CoTFormer /
LN-CoTFormer arms). C4 and C5 are router-gated architectures
whose residual-stream diagnostics are deferred to the router-aware
protocols (D and H) in Wave-2.

Compute budget (per `docs/extend-notes.md` §1.8):

| Stage | Wall time |
|-------|-----------|
| Stage 1 flash collector + Stage 3 Protocol A metrics (GPU) | ~10 min per checkpoint |
| Stage 4a Tuned Lens training (CPU) | ~5 h per checkpoint (105 translators x 250 steps) |
| Stage 4b Residual Diagnostics (CPU) | ~5 min per checkpoint |
| Stage 5 synthesis stub | Wave-2 deliverable |

Total GPU envelope: ~50 min for the five-checkpoint matrix.
Total CPU envelope: ~25 h for the three-checkpoint Tuned Lens
matrix plus ~15 min for Residual Diagnostics.

## Running the matrix

```bash
# Stage 1: GPU collector + Protocol A
cd ~/CoTFormer && bash iridis/analyze-lncot+adm/job-gpu.sh

# Wait for job-gpu to complete (sbatch email notification on finish),
# then:

# Stage 4: CPU Tuned Lens training + Residual Diagnostics
cd ~/CoTFormer && bash iridis/analyze-lncot+adm/job-cpu.sh
```

Each script is self-submitting: invoke as a bash script on the login
node; it pre-flights the inputs, writes a run_N/ scaffolding tree,
then re-invokes itself via sbatch with `SLURM_JOB_ID` set. On the
compute node the script sources `iridis/env.sh` and runs the main
body.

Output triage follows the same `iridis/eval-adm/` convention: large
intermediate `.npy` files live on `/scratch` and never leave it;
only meaningful small outputs (JSON summaries, PNG plots) land in
`run_N/<tag>/`.

## Output structure

```
run_N/                                                 (under iridis/analyze-lncot+adm/)
  slurm_<jobid>.out                                    SLURM stdout
  slurm_<jobid>.err                                    SLURM stderr (usually empty)
  c1_cotres_40k/
    logit_lens_results.json                            Protocol A: per-batch + aggregate
    logit_lens_trajectories.png                        top-1 and entropy curves
    tuned_lens_translators.pt                          Protocol A-ext: 105 state dicts
    tuned_lens_diagnostic.json                         identity grid + convergence verdict
    tuned_lens_triangulation.json                      Spearman rho + cross-lens KL
    tuned_lens_identity_grid.png                       diagnostic heatmap
    residual_diagnostics_results.json                  Protocol E: L2 / variance / L_inf grids
    residual_diagnostics_mean_l2.png                   mean L2 per (layer, repeat)
    residual_diagnostics_var_l2.png                    variance per (layer, repeat)
    residual_diagnostics_linf.png                      L_inf per (layer, repeat)
  c2_lncot_40k/                                        (same layout as c1)
  c3_lncot_60k/                                        (same layout as c1)
  c4_mod_40k/                                          (Protocol E omitted -- not in matrix row)
    logit_lens_results.json
    logit_lens_trajectories.png
    tuned_lens_translators.pt
    tuned_lens_diagnostic.json
    tuned_lens_triangulation.json
    tuned_lens_identity_grid.png
  c5_adm_v2_60k/                                       (same layout as c4)
```

## Protocol A interpretation guide

`logit_lens_results.json` encodes the per-(mid-layer, repeat) grid
of four metrics: top-1 accuracy via `lm_head(ln_f(h))`, top-1 via
`lm_head(h)` (raw residual lens, no ln_f), entropy via each of the
two projection paths, and adjacent-repeat KL divergence
`D_KL(P_r || P_{r-1})`.

The H0 verdict is computed in `aggregate.h0_test`:

- ``method``: paired t-test between per-layer top-1 at the first
  and last repeat.
- ``rejects_flat``: True when p < 0.01 AND mean(last) > mean(first)
  AND the across-layer mean trajectory is monotonically
  non-decreasing. This is the RQ1 falsification threshold per
  `docs/extend-notes.md` §1.2 RQ1.
- ``monotonic_non_decreasing``: True when the across-layer mean
  trajectory never decreases by more than 1e-4 between adjacent
  repeats.

The inter-batch CV gate surfaces metrics whose 4-batch coefficient
of variation exceeds 0.05. Metrics in ``aggregate.failed_cv_metrics``
are flagged as single-batch unreliable and should be cross-checked
with the Tuned-Lens curve in the triangulation table below before
interpretation.

## Protocol A-ext interpretation guide

`tuned_lens_diagnostic.json` carries the 21x5 identity-diagnostic
grid and the DEC-017 convergence verdict:

- ``verdict``: ``PASS`` when >= 95 of 105 sites have lower Tuned-Lens
  KL than unweighted-lens KL at the same site; ``PARTIAL`` when >=
  80 / 105; ``FAIL`` otherwise. PARTIAL runs trigger the Muon
  fallback re-training on failing sites automatically; a second
  evaluation runs and the final verdict reflects the post-Muon
  state.
- ``frob_identity_grid``: ``||A_l - I||_F`` for every translator.
  Values below 0.1 (with ``bias_norm_grid`` also below 0.1) place
  that site in the "identity-equivalent" band, where the Tuned Lens
  is redundant with the unweighted Logit Lens and the per-site
  triangulation should rely on the unweighted trajectory alone.
  Values in [0.1, 1.0) are "mild adjustment"; values >= 1.0 are
  "substantive translation".

`tuned_lens_triangulation.json` reports the Spearman rho between
the unweighted and tuned per-repeat top-1 trajectories per layer,
plus cross-lens KL per (layer, repeat) pair:

- ``agreement_band == "agree"``: mean Spearman rho >= 0.80 AND
  mean cross-lens KL <= 0.5 nats. The RQ1 claim is supported by
  both lenses.
- ``agreement_band == "disagree"``: mean Spearman rho < 0.60 OR
  mean cross-lens KL > 1.0 nats. The RQ1 claim downgrades to
  "single-measurement preliminary" and the synthesis script should
  cite RQ2 CKA and RQ6 effective-dim as secondary triangulation
  partners.
- ``agreement_band == "ambiguous"``: Spearman rho in [0.60, 0.80).
  Report both trajectories, draw no strong conclusion.

`tuned_lens_translators.pt` ships the 105 translator state dicts
keyed as ``l{L}_r{R}`` per (layer, repeat) pair, plus the model
dimensions and the optimiser name used for each site. Downstream
analyses that want to apply a specific translator can load a single
state dict into a fresh `nn.Linear(n_embd, n_embd, bias=True)` and
project any residual through it.

## Protocol E interpretation guide

`residual_diagnostics_results.json` holds three 21x5 grids:

- ``mean_l2_grid``: mean ``||x||_2`` per (mid-layer, repeat) site.
- ``var_l2_grid``: per-token L2 variance per site.
- ``linf_grid``: L_inf norm per site.

The H0 "residual L2 constant across repeats within each layer" is
rejected whenever the within-layer coefficient of variation exceeds
0.1 per the Protocol E operational spec; the field
``n_layers_failing_cv_gate`` counts failing layers.

``rogue_flags`` lists (layer, repeat) sites where the top-2
principal components explain more than 50 % of the centred
covariance; these sites should be cross-checked against the
top-2-PC-removed participation ratio in Protocol F (Wave-2) per
RQ6's rogue-direction sub-gate.

## Triangulation decision matrix

Per `docs/extend-notes.md` §1.6 Reliability Discipline and
Triangulation, the "representations refine across repeats" row is
supported when the following three measurements agree:

| Source | Signal | Gate |
|--------|--------|------|
| Protocol A-ext Tuned Lens (per-site identity-diagnostic-filtered) | Monotonic per-layer top-1 increase + convergence gate PASS | Tuned Lens verdict ``PASS`` in `tuned_lens_diagnostic.json` |
| Protocol A unweighted Logit Lens | Paired t-test on top-1 at r=1 vs r=n_repeat rejects flat | `logit_lens_results.json.aggregate.h0_test.rejects_flat` |
| Protocol B debiased CKA (Wave-2) | Monotonic across-repeat decrease on diagonal | Wave-2 deliverable |

The row is supported if the primary (Tuned Lens) plus at least one
secondary (unweighted or CKA) pass. The row is "preliminary" if
only the primary passes, and "contradicted" if primary and secondary
disagree; the synthesis script (Wave-2) will render the verdict
table.

## Workspace layout and disposability

Large intermediate `.npy` files live on `/scratch/$USER/analysis_workspace/<tag>/`
per `docs/extend-technical.md` §4.3 and are regeneratable by
re-running `job-gpu.sh`. They are NOT copied into `run_N/` so the
Git-tracked output tree stays small. The workspace layout per tag:

```
residual_mid_l{L}_r{R}.npy       # (N_tokens, n_embd) pre-ln_mid residual per (L, R)
residual_ln_mid_r{R}.npy         # (N_tokens, n_embd) post-ln_mid residual per R
residual_pre_ln_f.npy            # (N_tokens, n_embd) input to ln_f; canonical Belrose 2023 target residual for the Tuned Lens
targets.npy                      # (N_tokens,) ground-truth next-token ids (int64)
meta.json                        # collector metadata (sites, n_repeat, shapes)
```

The ``meta.json`` ``counter_site`` field indicates whether the
collector used ``ln_mid`` or ``h_mid[-1]`` as the per-repeat
counter-increment site; ``ln_mid`` is preferred when the model
class provides it (all C2, C3, C4, C5 checkpoints), while ``h_mid[-1]``
is the fallback for C1 (``cotformer_full_depth``) where the model
class does not declare ``ln_mid``.

## References

- `docs/extend-notes.md` §1.2 RQ1 -- Prediction Convergence Across
  Repeats (the primary falsification target of Protocol A + A-ext).
- `docs/extend-notes.md` §1.3 Protocol A / A-ext / E rows -- the
  operational specifications this package implements.
- `docs/extend-notes.md` §1.4 Prediction 1 -- the low-rank
  correlational expectation from weight decay; the Tuned Lens
  translator singular-value spectrum cross-validates RQ6 findings
  per the triangulation matrix.
- `docs/extend-notes.md` §1.5 Analysis Matrix -- the C1-C5 per-
  protocol assignment table.
- `docs/extend-notes.md` §1.6 Reliability Discipline and
  Triangulation -- the row-by-row operational-independence audit
  and the agreement-vs-disagreement verdict rules.
- `docs/extend-notes.md` §1.9 DEC-017 -- the Tuned Lens
  convergence gate and Muon fallback specification.
- `docs/extend-technical.md` §2 Package layout -- the `analysis/`
  module structure and the shared collector contract.
- `docs/extend-technical.md` §4.3 Scratch workspace layout -- the
  `.npy` naming convention and output-triage rule.
- `iridis/eval-adm/README.md` -- the precedent template this package
  follows for self-submission, pre-flight, run_N allocation, and
  workspace triage.
- Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I.,
  McKinney, L., Biderman, S., Steinhardt, J. (2023). Eliciting
  Latent Predictions from Transformers with the Tuned Lens. -- the
  canonical Tuned Lens reference whose specification DEC-017
  freezes (affine + bias, SGD + Nesterov at lr=1.0, 250 steps with
  linear decay, identity init, forward KL target).
