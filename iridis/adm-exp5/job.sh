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
        --export=ALL,REPO_DIR="$REPO_DIR",CHECKPOINT_DIR="$CHECKPOINT_DIR",RUN_DIR="$RUN_DIR" \
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

# ========================= PHASE 1: 60k on TRAIN split ========================
echo "========================================="
echo " Phase 1: 60k router weights (train split)"
echo "========================================="

python get_router_weights.py --checkpoint "$CKPT_SUBDIR" --split train --distributed_backend None
cp "$CKPT_SUBDIR/router_weights.npy" "$CKPT_SUBDIR/router_weights_60k_train.npy"
cp "$CKPT_SUBDIR/router_weights_raw.npy" "$CKPT_SUBDIR/router_weights_raw_60k_train.npy"
cp "$CKPT_SUBDIR/router_logits.npz" "$CKPT_SUBDIR/router_logits_60k.npz"
echo ""

# ========================= PHASE 2: 40k on TRAIN split ========================
echo "========================================="
echo " Phase 2: 40k router weights (train split)"
echo "========================================="

python get_router_weights.py --checkpoint "$CKPT_SUBDIR/ckpt_40000.pt" --split train --distributed_backend None
cp "$CKPT_SUBDIR/router_weights.npy" "$CKPT_SUBDIR/router_weights_40k_train.npy"
cp "$CKPT_SUBDIR/router_weights_raw.npy" "$CKPT_SUBDIR/router_weights_raw_40k_train.npy"
cp "$CKPT_SUBDIR/router_logits.npz" "$CKPT_SUBDIR/router_logits_40k.npz"
echo ""

# ========================= PHASE 3: 60k on VAL split ==========================
echo "========================================="
echo " Phase 3: 60k router weights (val split)"
echo "========================================="

python get_router_weights.py --checkpoint "$CKPT_SUBDIR" --split val --distributed_backend None
cp "$CKPT_SUBDIR/router_weights.npy" "$CKPT_SUBDIR/router_weights_60k_val.npy"
cp "$CKPT_SUBDIR/router_weights_raw.npy" "$CKPT_SUBDIR/router_weights_raw_60k_val.npy"
echo ""

# ========================= PHASE 4: 40k on VAL split ==========================
echo "========================================="
echo " Phase 4: 40k router weights (val split)"
echo "========================================="

python get_router_weights.py --checkpoint "$CKPT_SUBDIR/ckpt_40000.pt" --split val --distributed_backend None
cp "$CKPT_SUBDIR/router_weights.npy" "$CKPT_SUBDIR/router_weights_40k_val.npy"
cp "$CKPT_SUBDIR/router_weights_raw.npy" "$CKPT_SUBDIR/router_weights_raw_40k_val.npy"

# Restore val cascaded as default (backward compat with get_ppl_per_mac.py)
cp "$CKPT_SUBDIR/router_weights_60k_val.npy" "$CKPT_SUBDIR/router_weights.npy"
echo ""

# ========================= PHASE 5: Figure 5a — Reproduction ==================
echo "========================================="
echo " Phase 5: Figure 5 — Reproduction"
echo "   (train split, raw per-router scores)"
echo "========================================="

python plot_fig5.py \
    --weights-40k "$CKPT_SUBDIR/router_weights_raw_40k_train.npy" \
    --weights-60k "$CKPT_SUBDIR/router_weights_raw_60k_train.npy" \
    --output "$CKPT_SUBDIR/figure5_reproduction.png"
echo ""

# ========================= PHASE 6: Figure 5b — Analysis ======================
echo "========================================="
echo " Phase 6: Figure 5 — Analysis"
echo "   (val split, cascaded minimum scores)"
echo "========================================="

python plot_fig5.py \
    --weights-40k "$CKPT_SUBDIR/router_weights_40k_val.npy" \
    --weights-60k "$CKPT_SUBDIR/router_weights_60k_val.npy" \
    --output "$CKPT_SUBDIR/figure5_analysis.png"

# --- Copy outputs to run dir ---
for f in figure5_reproduction.png figure5_analysis.png; do
    cp "$CKPT_SUBDIR/$f" "$RUN_DIR/" 2>/dev/null || true
done

echo ""
echo "========================================="
echo " adm-exp5 complete: $(date)"
echo ""
echo " Output files (also copied to $RUN_DIR/):"
echo "   figure5_reproduction.png  — train split, raw scores (matches paper)"
echo "   figure5_analysis.png      — val split, cascaded scores (our analysis)"
echo "   router_weights_*_train.npy / router_weights_*_val.npy"
echo "   router_weights_raw_*_train.npy / router_weights_raw_*_val.npy"
echo "   router_logits_40k.npz / router_logits_60k.npz"
echo "========================================="

exit 0
