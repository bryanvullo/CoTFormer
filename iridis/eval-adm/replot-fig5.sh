#!/bin/bash
################################################################################
# replot-fig5.sh -- Regenerate Figure 5 (and Figure 4) plots from existing
# /scratch workspace .npy files, without re-running eval.py or extractions.
#
# When to use:
#   After a plotting-only change in plot_fig5.py (or plot_fig4.py) that does
#   not affect extraction or evaluation logic. The expensive stages in job.sh
#   (PPL eval, router-weight extraction x 16) each take hours on 1 L4; this
#   script skips them entirely and runs plotting directly on the workspace
#   .npy files that job.sh already wrote to /scratch.
#
# What it does:
#   1. Pre-flight: verifies all workspace .npy files still exist under
#      $CKPT_DIR/eval_workspace/ for both ADM versions (fails fast if
#      /scratch has been cleared since the last job.sh run).
#   2. Allocates the next run_N directory via next_run_dir() from env.sh.
#   3. Copies eval_*.txt + eval_summary_ckpt_*.json + exp4 scalar sweep .npy
#      from run_$SOURCE_RUN into the new run_N for snapshot completeness.
#      These artefacts are unchanged by any plotting-only fix.
#   4. Regenerates Figure 4 (unchanged y-axis, parity with run_6 layout) and
#      the full Figure 5 matrix (14 PNGs: 2 versions x 4 experiments x 3
#      variants, minus exp5-hooks/figure5_picascade.png which does not exist
#      by design -- see eval-adm README).
#   5. Writes a short run_N/README.md declaring run_N a plotting-only delta.
#
# Why not SLURM: plotting is pure numpy + matplotlib over ~130 MB .npy files
# loaded serially; total wall time is ~1 min on login-node resources. No GPU
# needed. Login-node etiquette is respected (<2 min total CPU, <300 MB peak
# resident memory).
#
# Usage (from the CoTFormer repo root on Iridis):
#   bash iridis/eval-adm/replot-fig5.sh
#
# Optional overrides:
#   SOURCE_RUN=run_6 bash iridis/eval-adm/replot-fig5.sh
#     Pull eval_*.txt and eval_summary_*.json from a run other than run_6.
#
# Output structure matches job.sh exactly, so the new run_N/ is drop-in
# compatible with any downstream tooling that reads run_N/adm_*/exp{4,5-*}/.
################################################################################

set -eo pipefail
export PYTHONUNBUFFERED=1

PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
source "$REPO_DIR/iridis/env.sh"

# ========================= CONFIGURATION (mirror of job.sh) =================
# Keep these synchronised with job.sh VERSIONS / CKPTS / EXPERIMENTS so the
# pre-flight + replot loop see the same matrix as the original extraction.

ADM_BASE="/scratch/$USER/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final"

VERSIONS=(
    "v1|$ADM_BASE/adm_lr0.001_bs8x16_seqlen256"
    "v2|$ADM_BASE/adm_v2_lr0.001_bs8x16_seqlen256"
)

CKPTS=("40000" "60000")

# "exp_dirname|space-separated variants to plot"
# exp5-hooks intentionally omits picascade: hooks capture pre-topk in
# original token order, where position-indexed cascade is undefined by
# construction (see eval-adm README line 483).
EXPERIMENTS=(
    "exp5-original|picascade ptcascade raw"
    "exp5-hooks|ptcascade raw"
    "exp5-cf-sorted|picascade ptcascade raw"
    "exp5-cf-argparse|picascade ptcascade raw"
)

# Source of eval_*.txt / eval_summary_*.json / exp4 scalar .npy sweeps.
# Override via env var if plotting off a later full-evaluation run.
SOURCE_RUN="${SOURCE_RUN:-run_6}"

# ========================= END CONFIGURATION ================================

SOURCE_RUN_DIR="$PACKAGE_DIR/$SOURCE_RUN"

# ========================= PRE-FLIGHT =======================================
echo "=== Pre-flight: verifying workspace .npy files on /scratch ==="
MISSING=0

for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag v_dir <<< "$version_entry"
    WORKSPACE="$v_dir/eval_workspace"

    # exp4 inputs: scalar sweeps live at top of CKPT_DIR (written by
    # get_ppl_per_mac.py, hardcoded reads by plot_fig4.py).
    for f in eval_per_threshold.npy eval_per_layer.npy; do
        if [ ! -f "$v_dir/$f" ]; then
            echo "  MISSING ($v_tag exp4): $v_dir/$f"
            MISSING=$((MISSING+1))
        fi
    done

    # exp5 inputs: per-experiment per-step per-variant .npy files in workspace.
    for exp_spec in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r DIRNAME VARIANTS <<< "$exp_spec"
        for step in "${CKPTS[@]}"; do
            STEP_WORKSPACE="$WORKSPACE/$DIRNAME/${step%000}k"
            for variant in $VARIANTS; do
                npy_file="$STEP_WORKSPACE/router_weights_${variant}.npy"
                if [ ! -f "$npy_file" ]; then
                    echo "  MISSING ($v_tag $DIRNAME ${step%000}k): $npy_file"
                    MISSING=$((MISSING+1))
                fi
            done
        done
    done
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING workspace .npy file(s) missing. /scratch may have"
    echo "been cleared. Re-run iridis/eval-adm/job.sh to regenerate the full"
    echo "extraction matrix before attempting a plot-only rerun."
    exit 1
fi
echo "  All workspace .npy files present."

# Verify SOURCE_RUN has the static artefacts we plan to copy.
if [ ! -d "$SOURCE_RUN_DIR" ]; then
    echo ""
    echo "ERROR: SOURCE_RUN directory not found: $SOURCE_RUN_DIR"
    echo "Set SOURCE_RUN=<run_name> to pull eval-text artefacts from a"
    echo "different run, or leave unset to default to run_6."
    exit 1
fi

# ========================= ALLOCATE run_N ===================================
RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
echo ""
echo "=== Target: $RUN_DIR (source run: $SOURCE_RUN) ==="

for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag _ <<< "$version_entry"
    mkdir -p "$RUN_DIR/adm_${v_tag}/exp4"
    for exp_spec in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r DIRNAME _ <<< "$exp_spec"
        mkdir -p "$RUN_DIR/adm_${v_tag}/$DIRNAME"
    done
done

# ========================= CONDA ============================================
# plot_fig4.py + plot_fig5.py need matplotlib + numpy. The shared team env
# has both; activate it just like job.sh does on the compute node.
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

# ========================= COPY STATIC ARTEFACTS ============================
# eval_*.txt, eval_summary_*.json: PPL numbers are deterministic from
# (checkpoint, eval.py, seed, data). A plotting-only fix cannot change them.
# Carrying them forward keeps run_N/ self-contained for downstream readers.
# exp4 scalar sweeps (eval_per_threshold.npy, eval_per_layer.npy) likewise
# unchanged -- they are get_ppl_per_mac.py outputs, not plot outputs.
echo ""
echo "=== Copying static artefacts from $SOURCE_RUN/ ==="
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag _ <<< "$version_entry"
    SRC="$SOURCE_RUN_DIR/adm_${v_tag}"
    DST="$RUN_DIR/adm_${v_tag}"

    for f in eval_40k.txt eval_60k.txt \
             eval_summary_ckpt_40000.json eval_summary_ckpt_60000.json; do
        if [ -f "$SRC/$f" ]; then
            cp "$SRC/$f" "$DST/$f"
            echo "  Copied adm_${v_tag}/$f"
        else
            echo "  SKIP (not in source): $SRC/$f"
        fi
    done

    for f in eval_per_threshold.npy eval_per_layer.npy; do
        if [ -f "$SRC/exp4/$f" ]; then
            cp "$SRC/exp4/$f" "$DST/exp4/$f"
            echo "  Copied adm_${v_tag}/exp4/$f"
        else
            echo "  SKIP (not in source): $SRC/exp4/$f"
        fi
    done
done

# ========================= REPLOT LOOP ======================================
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r VERSION CKPT_DIR <<< "$version_entry"
    V_RUN_DIR="$RUN_DIR/adm_${VERSION}"
    WORKSPACE="$CKPT_DIR/eval_workspace"

    echo ""
    echo "###################################################################"
    echo "   ADM $VERSION -- replot"
    echo "###################################################################"

    # --- Figure 4 (Pareto) ---
    # Unchanged y-axis; regenerated for directory parity with run_6 layout.
    # plot_fig4.py reads eval_per_{threshold,layer}.npy from $CKPT_DIR and
    # writes the PNG we pass via --output.
    echo ""
    echo "--- Figure 4 (Pareto) ---"
    python plot_fig4.py \
        --checkpoint "$CKPT_DIR" \
        --output "$V_RUN_DIR/exp4/figure4_pareto.png"

    # --- Figure 5 matrix ---
    echo ""
    echo "--- Figure 5 matrix ---"
    for exp_spec in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r DIRNAME VARIANTS <<< "$exp_spec"
        EXP_DIR="$V_RUN_DIR/$DIRNAME"
        WORKSPACE_EXP="$WORKSPACE/$DIRNAME"

        echo ""
        echo "  === $DIRNAME ==="

        W40="$WORKSPACE_EXP/40k"
        W60="$WORKSPACE_EXP/60k"

        for variant in $VARIANTS; do
            w40_npy="$W40/router_weights_${variant}.npy"
            w60_npy="$W60/router_weights_${variant}.npy"
            out_png="$EXP_DIR/figure5_${variant}.png"
            python plot_fig5.py \
                --weights-40k "$w40_npy" \
                --weights-60k "$w60_npy" \
                --output "$out_png"
        done
    done
done

# ========================= run_N/README.md ==================================
cat > "$RUN_DIR/README.md" <<EOF
# $(basename "$RUN_DIR") (plotting-only regeneration)

Sourced from: \`$SOURCE_RUN/\`

This directory was produced by \`iridis/eval-adm/replot-fig5.sh\`, a
plotting-only rerun that reuses the existing extraction artefacts on
\`/scratch\` instead of re-running evaluation or router-weight extraction.

## What changed vs $SOURCE_RUN/

- \`adm_{v1,v2}/exp5-*/figure5_{raw,picascade,ptcascade}.png\` regenerated
  under the paper-style y-axis normalisation in \`plot_fig5.py\`: fractions
  per bin summing to 1.0 per series (axis label "Density", matching the
  CoTFormer paper Figure 5 convention on p. 9).

## What was carried over unchanged (copied verbatim from $SOURCE_RUN/)

- \`adm_{v1,v2}/eval_{40,60}k.txt\` (PPL stdout logs)
- \`adm_{v1,v2}/eval_summary_ckpt_{40000,60000}.json\` (machine-readable summaries)
- \`adm_{v1,v2}/exp4/eval_per_{threshold,layer}.npy\` (Figure 4 scalar sweeps)
- \`adm_{v1,v2}/exp4/figure4_pareto.png\` (regenerated from the unchanged
  sweep data for directory parity; plot logic unchanged from $SOURCE_RUN/)

## Not in this directory

- \`slurm_*.out\` / \`slurm_*.err\`: no SLURM submission was involved; the
  replot runs entirely on the login node in ~1 minute.

See \`iridis/eval-adm/README.md\` for the full evaluation matrix design,
statistics, and discussion. Those numerical results apply to this run
unchanged.
EOF

# ========================= SUMMARY ==========================================
echo ""
echo "========================================="
echo " Replot complete: $(date)"
echo " Results tree:  $RUN_DIR/"
echo " Source run:    $SOURCE_RUN"
echo ""
echo " Regenerated:   $(find "$RUN_DIR" -name 'figure*.png' | wc -l) PNG files"
echo " Carried over:  eval_*.txt, eval_summary_*.json, exp4 scalar .npy"
echo "========================================="
