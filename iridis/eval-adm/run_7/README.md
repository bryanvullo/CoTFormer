# run_7 (plotting-only regeneration)

Sourced from: `run_6/`

This directory was produced by `iridis/eval-adm/replot-fig5.sh`, a
plotting-only rerun that reuses the existing extraction artefacts on
`/scratch` instead of re-running evaluation or router-weight extraction.

## What changed vs run_6/

- `adm_{v1,v2}/exp5-*/figure5_{raw,picascade,ptcascade}.png` regenerated
  under the paper-style y-axis normalisation in `plot_fig5.py`: fractions
  per bin summing to 1.0 per series (axis label "Density", matching the
  CoTFormer paper Figure 5 convention on p. 9).

## What was carried over unchanged (copied verbatim from run_6/)

- `adm_{v1,v2}/eval_{40,60}k.txt` (PPL stdout logs)
- `adm_{v1,v2}/eval_summary_ckpt_{40000,60000}.json` (machine-readable summaries)
- `adm_{v1,v2}/exp4/eval_per_{threshold,layer}.npy` (Figure 4 scalar sweeps)
- `adm_{v1,v2}/exp4/figure4_pareto.png` (regenerated from the unchanged
  sweep data for directory parity; plot logic unchanged from run_6/)

## Not in this directory

- `slurm_*.out` / `slurm_*.err`: no SLURM submission was involved; the
  replot runs entirely on the login node in ~1 minute.

See `iridis/eval-adm/README.md` for the full evaluation matrix design,
statistics, and discussion. Those numerical results apply to this run
unchanged.
