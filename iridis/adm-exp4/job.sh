#!/bin/bash
#SBATCH --job-name=adm_exp4
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
################################################################################
# Figure 4 Reproduction — iridis/adm-exp4
#
# Reproduces Figure 4 of the paper: "Perplexity for different amounts of
# compute budgets chosen at inference."
#
# Two curves from the SAME adaptive LN-CoTFormer (ADM) 60k checkpoint:
#   1. Router: vary router threshold → per-token depth control
#   2. Fixed Depth: activate a fixed prefix of repeats for all tokens
#
# Phases:
#   1. Extract router weights (get_router_weights.py)
#   2. Evaluate PPL at multiple compute budgets (get_ppl_per_mac.py)
#   3. Generate Figure 4 plot (plot_fig4.py)
#
# Usage:
#   cd ~/CoTFormer && bash iridis/adm-exp4/job.sh
#
# PREREQUISITE: Trained ADM 60k checkpoint (ckpt.pt + summary.json).
#   Train it with: bash iridis/adm-train/job.sh
################################################################################

# ========================= CONFIGURATION ====================================

# Path to the trained adaptive model checkpoint directory.
CHECKPOINT_DIR="/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256"

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    echo "=== Figure 4 Reproduction (adm-exp4) ==="
    echo ""

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
        echo "  Train it with: bash iridis/adm-train/job.sh"
        exit 1
    fi

    CKPT_FILE=$(find "$CHECKPOINT_DIR" -name "ckpt.pt" -type f 2>/dev/null | head -1)
    if [ -z "$CKPT_FILE" ]; then
        echo "ERROR: No ckpt.pt found under $CHECKPOINT_DIR"
        exit 1
    fi
    CKPT_SUBDIR=$(dirname "$CKPT_FILE")

    if [ ! -f "$CKPT_SUBDIR/summary.json" ]; then
        echo "ERROR: No summary.json at $CKPT_SUBDIR"
        exit 1
    fi

    echo "  Checkpoint: $CKPT_SUBDIR"
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
echo " Figure 4 Reproduction (adm-exp4)"
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

# ========================= PHASE 1: Router Weights ============================
echo "========================================="
echo " Phase 1: Extracting router weights (60k)"
echo "========================================="

if [ -f "$CKPT_SUBDIR/router_weights.npy" ]; then
    echo "router_weights.npy already exists — skipping."
else
    python get_router_weights.py --checkpoint "$CKPT_SUBDIR" --distributed_backend None
fi
echo ""

# ========================= PHASE 2: PPL vs MACs ==============================
echo "========================================="
echo " Phase 2: Evaluating PPL vs MACs"
echo "   Router thresholds + Fixed depths"
echo "========================================="

python get_ppl_per_mac.py --checkpoint "$CKPT_SUBDIR"
echo ""

# ========================= PHASE 3: Plot Figure 4 ============================
echo "========================================="
echo " Phase 3: Generating Figure 4"
echo "========================================="

python plot_fig4.py --checkpoint "$CKPT_SUBDIR"

# --- Copy outputs to run dir ---
for f in figure4_pareto.png eval_per_threshold.npy eval_per_layer.npy; do
    cp "$CKPT_SUBDIR/$f" "$RUN_DIR/" 2>/dev/null || true
done

echo ""
echo "========================================="
echo " adm-exp4 complete: $(date)"
echo ""
echo " Output files (also copied to $RUN_DIR/):"
echo "   $CKPT_SUBDIR/router_weights.npy"
echo "   $CKPT_SUBDIR/eval_per_threshold.npy"
echo "   $CKPT_SUBDIR/eval_per_layer.npy"
echo "   $CKPT_SUBDIR/figure4_pareto.png"
echo "========================================="

exit 0
