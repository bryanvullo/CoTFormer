#!/bin/bash
#SBATCH --job-name=adm_exp5
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
################################################################################
# Figure 5 Reproduction — iridis/adm-exp5
#
# Reproduces Figure 5 of the paper: "Distribution of router weights for the
# last repeat for different number of training steps."
#
# Compares router weight distributions at 40k and 60k training steps to show
# that longer training shifts the model towards utilising the deepest repeat.
#
# Phases:
#   1. Extract 60k router weights (reuse from adm-exp4 if available)
#   2. Extract 40k router weights (from intermediate ckpt_40000.pt)
#   3. Generate Figure 5 histogram (plot_fig5.py)
#
# Usage:
#   cd ~/CoTFormer && bash iridis/adm-exp5/job.sh
#
# PREREQUISITE: Trained ADM 60k checkpoint with ckpt.pt AND ckpt_40000.pt.
################################################################################

# ========================= CONFIGURATION ====================================

CHECKPOINT_DIR="/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256"

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    echo "=== Figure 5 Reproduction (adm-exp5) ==="
    echo ""

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
        exit 1
    fi

    CKPT_FILE=$(find "$CHECKPOINT_DIR" -name "ckpt.pt" -type f 2>/dev/null | head -1)
    if [ -z "$CKPT_FILE" ]; then
        echo "ERROR: No ckpt.pt found under $CHECKPOINT_DIR"
        exit 1
    fi
    CKPT_SUBDIR=$(dirname "$CKPT_FILE")

    # Check for 40k intermediate checkpoint
    if [ ! -f "$CKPT_SUBDIR/ckpt_40000.pt" ]; then
        echo "ERROR: No ckpt_40000.pt found at $CKPT_SUBDIR"
        echo "  Figure 5 requires both 40k and 60k router weights."
        echo "  Check if intermediate checkpoints were preserved."
        exit 1
    fi

    echo "  Checkpoint dir: $CKPT_SUBDIR"
    echo "  60k: ckpt.pt"
    echo "  40k: ckpt_40000.pt"
    echo ""

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "  Logs: $RUN_DIR/"
    echo ""
    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",CHECKPOINT_DIR="$CHECKPOINT_DIR" \
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
echo " Figure 5 Reproduction (adm-exp5)"
echo " User:       $USER"
echo " Node:       $(hostname)"
echo " Job ID:     $SLURM_JOB_ID"
echo " Checkpoint: $CHECKPOINT_DIR"
echo " Started:    $(date)"
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
    echo "ERROR: Dataset not found at $DATA_DIR/openwebtext2/train.bin"
    exit 1
fi

# --- Resolve checkpoint subdir ---
CKPT_FILE=$(find "$CHECKPOINT_DIR" -name "ckpt.pt" -type f 2>/dev/null | head -1)
CKPT_SUBDIR=$(dirname "$CKPT_FILE")
echo "Checkpoint subdir: $CKPT_SUBDIR"
echo ""

# ========================= PHASE 1: 60k Router Weights ========================
echo "========================================="
echo " Phase 1: 60k router weights"
echo "========================================="

if [ -f "$CKPT_SUBDIR/router_weights.npy" ]; then
    echo "router_weights.npy exists (from adm-exp4 or prior run)."
else
    echo "Extracting 60k router weights..."
    python get_router_weights.py --checkpoint "$CKPT_SUBDIR" --distributed_backend None
fi
cp "$CKPT_SUBDIR/router_weights.npy" "$CKPT_SUBDIR/router_weights_60k.npy"
echo "Saved: router_weights_60k.npy"
echo ""

# ========================= PHASE 2: 40k Router Weights ========================
echo "========================================="
echo " Phase 2: 40k router weights"
echo "========================================="

echo "Extracting 40k router weights from ckpt_40000.pt..."
python get_router_weights.py --checkpoint "$CKPT_SUBDIR/ckpt_40000.pt" --distributed_backend None
# get_router_weights.py saves to router_weights.npy in the same dir (overwrites 60k)
cp "$CKPT_SUBDIR/router_weights.npy" "$CKPT_SUBDIR/router_weights_40k.npy"
# Restore the 60k version as the default
cp "$CKPT_SUBDIR/router_weights_60k.npy" "$CKPT_SUBDIR/router_weights.npy"
echo "Saved: router_weights_40k.npy"
echo ""

# ========================= PHASE 3: Plot Figure 5 ============================
echo "========================================="
echo " Phase 3: Generating Figure 5"
echo "========================================="

python plot_fig5.py \
    --weights-40k "$CKPT_SUBDIR/router_weights_40k.npy" \
    --weights-60k "$CKPT_SUBDIR/router_weights_60k.npy" \
    --output "$CKPT_SUBDIR/figure5_router_dist.png"

echo ""
echo "========================================="
echo " adm-exp5 complete: $(date)"
echo ""
echo " Output files:"
echo "   $CKPT_SUBDIR/router_weights_40k.npy"
echo "   $CKPT_SUBDIR/router_weights_60k.npy"
echo "   $CKPT_SUBDIR/figure5_router_dist.png"
echo "========================================="

exit 0
