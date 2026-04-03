#!/bin/bash
#SBATCH --job-name=eval_lncot
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
################################################################################
# LN-CoTFormer Evaluation -- iridis/eval-lncot
#
# Evaluates the trained LN-CoTFormer at 40k and 60k checkpoints against the
# full OWT2 validation set (single GPU, no DDP).
#
# Paper targets:
#   40k steps (Table 2):   PPL ~24.11
#   60k steps (Section 5): PPL ~23.19
#
# Usage:
#   cd ~/CoTFormer && bash iridis/eval-lncot/job.sh
#
# Outputs (in this package's run_N/ subdir):
#   eval_40k.txt   -- stdout from eval.py at ckpt_40000.pt
#   eval_60k.txt   -- stdout from eval.py at ckpt_60000.pt
#   slurm_*.out    -- combined SLURM log
################################################################################

# ========================= CONFIGURATION ====================================

CKPT_DIR="/scratch/ab3u21/exps/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256"

# Checkpoints to evaluate (step number -> filename)
CKPTS=("40000" "60000")

# Batch size: use training value from summary.json (no override).
# Changing batch_size doesn't affect per-sample loss, but the final partial
# batch gets equal weight in the mean — keeping bs=8 (from summary.json)
# eliminates any micro-divergence from the training eval baseline.

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    # Pre-flight: verify checkpoints exist
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
    echo "=== LN-CoTFormer Evaluation ==="
    echo "  Checkpoints: ${CKPTS[*]} steps"
    echo "  Source:       $CKPT_DIR"
    echo "  Batch size:   $EVAL_BATCH_SIZE (eval, no backward)"
    echo "  Logs:         $RUN_DIR/"
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
echo " LN-CoTFormer Evaluation"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " GPU:           1x L4"
echo " Job ID:        $SLURM_JOB_ID"
echo " Model:         cotformer_full_depth_lnmid_depthemb"
echo " Checkpoints:   ${CKPTS[*]} steps"
echo " Eval BS:       $EVAL_BATCH_SIZE"
echo " Started:       $(date)"
echo "========================================="

# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# --- Evaluate each checkpoint ---
for step in "${CKPTS[@]}"; do
    CKPT_FILE="$CKPT_DIR/ckpt_${step}.pt"
    OUT_FILE="$RUN_DIR/eval_${step/000/k}.txt"

    echo ""
    echo "========================================="
    echo " Evaluating: ckpt_${step}.pt"
    echo " Output:     $OUT_FILE"
    echo "========================================="

    python eval.py \
        --checkpoint "$CKPT_FILE" \
        --config_format base \
        --distributed_backend None \
        --output_dir "$RUN_DIR" \
        2>&1 | tee "$OUT_FILE"

    echo ""
    echo "--- ckpt_${step}.pt DONE ---"
    echo ""
done

echo "========================================="
echo " All evaluations complete: $(date)"
echo " Results in: $RUN_DIR/"
echo ""
echo " Paper targets:"
echo "   40k (Table 2):   PPL ~24.11  (ours at training: 24.13)"
echo "   60k (Section 5): PPL ~23.19"
echo "========================================="
