#!/bin/bash
#SBATCH --job-name=sec5_compare
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
################################################################################
# Section 5 Comparison — iridis/lncot+adm-sec5
#
# Reproduces the Section 5 claim: "After 60k steps, the [adaptive] reaches
# perplexity 23.83 while the [non-adaptive] achieves 23.19."
#
# Evaluates both LN-CoTFormer 60k and ADM 60k on the full validation set
# and reports the comparison table.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/lncot+adm-sec5/job.sh
#
# PREREQUISITES:
#   - LN-CoTFormer 60k checkpoint (bash iridis/lncot-train/job.sh)
#   - ADM 60k checkpoint (bash iridis/adm-train/job.sh)
################################################################################

# ========================= CONFIGURATION ====================================

LNCOT_CHECKPOINT_DIR="/scratch/ab3u21/exps/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256"

ADM_CHECKPOINT_DIR="/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256"

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    echo "=== Section 5 Comparison (lncot+adm-sec5) ==="
    echo ""

    MISSING=0
    for dir_label in "LNCot:$LNCOT_CHECKPOINT_DIR" "ADM:$ADM_CHECKPOINT_DIR"; do
        label="${dir_label%%:*}"
        dir="${dir_label#*:}"
        CKPT=$(find "$dir" -name "ckpt.pt" -type f 2>/dev/null | head -1)
        if [ -z "$CKPT" ]; then
            echo "  [$label] MISSING ckpt.pt in $dir"
            MISSING=1
        else
            echo "  [$label] OK: $(dirname "$CKPT")"
        fi
    done
    if [ "$MISSING" -eq 1 ]; then
        echo ""
        echo "ERROR: One or more checkpoints missing. Train them first."
        exit 1
    fi
    echo ""

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "  Logs: $RUN_DIR/"
    echo ""
    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",LNCOT_CHECKPOINT_DIR="$LNCOT_CHECKPOINT_DIR",ADM_CHECKPOINT_DIR="$ADM_CHECKPOINT_DIR" \
        "$0" "$@"
fi

# --- Actual job (runs on compute node) ---
set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set — falling back to $REPO_DIR"
fi

source "$REPO_DIR/iridis/env.sh"

mkdir -p "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR"

echo "========================================="
echo " Section 5 Comparison"
echo " User:    $USER"
echo " Node:    $(hostname)"
echo " Job ID:  $SLURM_JOB_ID"
echo " Started: $(date)"
echo "========================================="

# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# --- Verify dataset ---
if [ ! -f "$DATA_DIR/openwebtext2/train.bin" ]; then
    echo "ERROR: Dataset not found"
    exit 1
fi

# --- Resolve checkpoint subdirs ---
LNCOT_CKPT=$(find "$LNCOT_CHECKPOINT_DIR" -name "ckpt.pt" -type f 2>/dev/null | head -1)
LNCOT_SUBDIR=$(dirname "$LNCOT_CKPT")

ADM_CKPT=$(find "$ADM_CHECKPOINT_DIR" -name "ckpt.pt" -type f 2>/dev/null | head -1)
ADM_SUBDIR=$(dirname "$ADM_CKPT")

# ========================= PHASE 1: Evaluate LN-CoTFormer 60k =================
echo "========================================="
echo " Phase 1: LN-CoTFormer 60k evaluation"
echo " Checkpoint: $LNCOT_SUBDIR"
echo "========================================="

python eval.py \
    --checkpoint "$LNCOT_SUBDIR/ckpt.pt" \
    --distributed_backend None

echo ""

# ========================= PHASE 2: Evaluate ADM 60k ==========================
echo "========================================="
echo " Phase 2: ADM 60k evaluation"
echo " Checkpoint: $ADM_SUBDIR"
echo "========================================="

python eval.py \
    --checkpoint "$ADM_SUBDIR/ckpt.pt" \
    --distributed_backend None

echo ""

# ========================= COMPARISON =========================================
echo "========================================="
echo " Section 5 Comparison"
echo ""
echo " Paper targets:"
echo "   LN-CoTFormer 60k:  PPL 23.19"
echo "   Adaptive (ADM) 60k: PPL 23.83"
echo ""
echo " (See eval output above for our reproduced values)"
echo ""
echo " lncot+adm-sec5 complete: $(date)"
echo "========================================="

exit 0
