#!/bin/bash
#SBATCH --job-name=eval_adm
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
################################################################################
# Adaptive LN-CoTFormer (ADM v2) -- Evaluation + Figures 4 & 5
#
# Single sequential job: PPL evaluation, Figure 4 (once), and a systematic
# Figure 5 experiment matrix isolating methodological confounders.
#
# Output structure:
#   run_N/
#     eval_4k0.txt, eval_6k0.txt          PPL evaluations
#     eval_summary_ckpt_*.json             Full eval summaries
#     exp4/                                Figure 4 (PPL vs MACs), one copy
#       figure4_pareto.png
#       eval_per_threshold.npy, eval_per_layer.npy
#     exp5-original/                       CF, raw dict config (saves PI+PT+raw)
#     exp5-hooks/                          Hooks extraction, raw dict config
#     exp5-cf-sorted/                      CF + sorted topk + PI cascade
#     exp5-cf-argparse/                    CF + PI cascade, argparse config
#
# Each exp5-*/ contains per-checkpoint outputs and a Figure 5 plot.
# See README.md in this directory for the experiment matrix rationale.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/eval-adm/job.sh
################################################################################

# ========================= CONFIGURATION ====================================

ADM_BASE="/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final"
CKPT_DIR="$ADM_BASE/adm_v2_lr0.001_bs8x16_seqlen256"

CKPTS=("40000" "60000")

# Experiment matrix: method, config-loading, output-subdir
# Format: "method|config-loading|dirname"
# Note: exp5-original saves all cascade variants (PI, PT, raw) from a single
# extraction. The "bug vs fix" comparison is between PI and PT cascade files
# within this directory, not between separate runs.
EXPERIMENTS=(
    "cf|raw|exp5-original"
    "hooks|raw|exp5-hooks"
    "cf-sorted|raw|exp5-cf-sorted"
    "cf|argparse|exp5-cf-argparse"
)

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    for step in "${CKPTS[@]}"; do
        ckpt_file="$CKPT_DIR/ckpt_${step}.pt"
        if [ ! -f "$ckpt_file" ]; then
            echo "ERROR: Checkpoint not found: $ckpt_file"
            exit 1
        fi
    done
    if [ ! -f "$CKPT_DIR/summary.json" ]; then
        echo "ERROR: summary.json not found in $CKPT_DIR"
        exit 1
    fi

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")

    # Create all experiment subdirectories
    mkdir -p "$RUN_DIR/exp4"
    for exp_spec in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r _method _config _dirname <<< "$exp_spec"
        mkdir -p "$RUN_DIR/$_dirname"
    done

    echo "=== ADM v2 Evaluation + Figures 4 & 5 (experiment matrix) ==="
    echo "  Checkpoint:  $CKPT_DIR"
    echo "  Steps:       ${CKPTS[*]}"
    echo "  Experiments: ${#EXPERIMENTS[@]} exp5 variants + exp4"
    echo "  Logs:        $RUN_DIR/"
    echo ""
    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",RUN_DIR="$RUN_DIR" \
        "$0" "$@"
fi

# --- Actual job (runs on compute node) ---
set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
fi

source "$REPO_DIR/iridis/env.sh"

echo "========================================="
echo " ADM v2 -- Evaluation + Figures 4 & 5"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " GPU:           1x L4"
echo " Job ID:        $SLURM_JOB_ID"
echo " Checkpoint:    $CKPT_DIR"
echo " Started:       $(date)"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""


# ========================= PHASE 1: PPL Evaluation ==========================
echo "###################################################################"
echo " PHASE 1: PPL Evaluation"
echo "###################################################################"

for step in "${CKPTS[@]}"; do
    CKPT_FILE="$CKPT_DIR/ckpt_${step}.pt"
    OUT_FILE="$RUN_DIR/eval_${step%000}k.txt"
    echo ""
    echo "--- Evaluating ckpt_${step}.pt ---"
    python eval.py \
        --checkpoint "$CKPT_FILE" \
        --config_format base \
        --distributed_backend None \
        --output_dir "$RUN_DIR" \
        2>&1 | tee "$OUT_FILE"
done


# ========================= PHASE 2: Figure 4 (once) ========================
# Figure 4 depends on router_weights.npy in CKPT_DIR. We generate it once
# using the hooks method (raw scores), since Figure 4's MAC sweep only needs
# the weight values, not their cascade ordering.
echo ""
echo "###################################################################"
echo " PHASE 2: Figure 4 -- PPL vs MACs (single run)"
echo "###################################################################"

python get_router_weights.py \
    --method hooks --config-loading raw \
    --checkpoint "$CKPT_DIR" --split val \
    --output-dir "$RUN_DIR/exp4" \
    --distributed_backend None

# get_ppl_per_mac.py and plot_fig4.py expect router_weights.npy in CKPT_DIR
cp "$RUN_DIR/exp4/router_weights_ptcascade.npy" "$CKPT_DIR/router_weights.npy"

python get_ppl_per_mac.py --checkpoint "$CKPT_DIR"
python plot_fig4.py --checkpoint "$CKPT_DIR"
cp "$CKPT_DIR/figure4_pareto.png" "$RUN_DIR/exp4/"
cp "$CKPT_DIR/eval_per_threshold.npy" "$RUN_DIR/exp4/"
cp "$CKPT_DIR/eval_per_layer.npy" "$RUN_DIR/exp4/"

echo "  Figure 4 saved to $RUN_DIR/exp4/"


# ========================= PHASE 3: Figure 5 experiment matrix ==============
echo ""
echo "###################################################################"
echo " PHASE 3: Figure 5 -- Experiment Matrix"
echo "###################################################################"

for exp_spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r METHOD CONFIG_LOADING DIRNAME <<< "$exp_spec"
    EXP_DIR="$RUN_DIR/$DIRNAME"

    echo ""
    echo "=== $DIRNAME (method=$METHOD, config=$CONFIG_LOADING) ==="

    for step in "${CKPTS[@]}"; do
        CKPT_FILE="$CKPT_DIR/ckpt_${step}.pt"
        STEP_DIR="$EXP_DIR/${step%000}k"
        mkdir -p "$STEP_DIR"

        echo "  Extracting ckpt_${step}.pt -> $STEP_DIR/"
        python get_router_weights.py \
            --method "$METHOD" \
            --config-loading "$CONFIG_LOADING" \
            --checkpoint "$CKPT_FILE" \
            --split train \
            --output-dir "$STEP_DIR" \
            --distributed_backend None
    done

    # --- Generate Figure 5 plots ---
    # Each experiment produces raw and cascade variants. Plot the primary
    # comparison: the cascade type that defines this experiment.
    W40="$EXP_DIR/${CKPTS[0]%000}k"
    W60="$EXP_DIR/${CKPTS[1]%000}k"

    # PI cascade figure (for cf/cf-sorted/cf-argparse experiments)
    if [ -f "$W40/router_weights_picascade.npy" ]; then
        python plot_fig5.py \
            --weights-40k "$W40/router_weights_picascade.npy" \
            --weights-60k "$W60/router_weights_picascade.npy" \
            --output "$EXP_DIR/figure5_picascade.png"
    fi

    # PT cascade figure (for all experiments)
    if [ -f "$W40/router_weights_ptcascade.npy" ]; then
        python plot_fig5.py \
            --weights-40k "$W40/router_weights_ptcascade.npy" \
            --weights-60k "$W60/router_weights_ptcascade.npy" \
            --output "$EXP_DIR/figure5_ptcascade.png"
    fi

    # Raw (no cascade) figure (for all experiments)
    if [ -f "$W40/router_weights_raw.npy" ]; then
        python plot_fig5.py \
            --weights-40k "$W40/router_weights_raw.npy" \
            --weights-60k "$W60/router_weights_raw.npy" \
            --output "$EXP_DIR/figure5_raw.png"
    fi

    echo "  Figures saved to $EXP_DIR/"
done


# ========================= SUMMARY ==========================================
echo ""
echo "========================================="
echo " Evaluation complete: $(date)"
echo " Results in: $RUN_DIR/"
echo ""
echo " PPL: eval_${CKPTS[0]%000}k.txt, eval_${CKPTS[1]%000}k.txt"
echo " Figure 4: exp4/"
echo " Figure 5 experiments:"
for exp_spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r _m _c _d <<< "$exp_spec"
    echo "   $_d/ (method=$_m, config=$_c)"
done
echo "========================================="
