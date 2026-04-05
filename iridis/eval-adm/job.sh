#!/bin/bash
#SBATCH --job-name=eval_adm
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
################################################################################
# Adaptive LN-CoTFormer (ADM) Evaluation -- iridis/eval-adm
#
# Evaluates the trained ADM v2 at 40k and 60k checkpoints against the full
# OWT2 validation set (single GPU, no DDP).
#
# Paper targets (Section 4.2, Section 5):
#   40k steps: intermediate (no paper target, but useful for trajectory)
#   60k steps: PPL ~23.83, with average_depth < 108 (router skipping)
#
# The ADM model also reports average_depth (effective layers used per token),
# which is critical for Section 5 analysis and Figure 4/5 reproduction.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/eval-adm/job.sh
#
# Outputs (in this package's run_N/ subdir):
#   eval_40k.txt   -- stdout from eval.py at ckpt_40000.pt
#   eval_60k.txt   -- stdout from eval.py at ckpt_60000.pt
#   slurm_*.out    -- combined SLURM log
################################################################################

# ========================= CONFIGURATION ====================================

CKPT_DIR="/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256"

# Checkpoints to evaluate (step number -> filename)
CKPTS=("40000" "60000")

# Batch size: use training value from summary.json (no override).
# See eval-lncot/job.sh for rationale.

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
    echo "=== ADM v2 Evaluation ==="
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
echo " Adaptive LN-CoTFormer (ADM v2) Evaluation"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " GPU:           1x L4"
echo " Job ID:        $SLURM_JOB_ID"
echo " Model:         adaptive_cotformer_..._single_final"
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
    echo " Evaluating: ckpt_${step}.pt (ADM v2)"
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

# ========================= PHASE 3: Figure 4 (PPL vs MACs) =====================
echo ""
echo "========================================="
echo " Phase 3: Figure 4 — PPL vs MACs sweep"
echo "========================================="

if [ ! -f "$CKPT_DIR/router_weights.npy" ]; then
    echo "Extracting router weights (needed for threshold sweep)..."
    python get_router_weights.py --checkpoint "$CKPT_DIR" --distributed_backend None
fi

echo "Running PPL vs MACs sweep (threshold + fixed depth)..."
python get_ppl_per_mac.py --checkpoint "$CKPT_DIR"

echo "Generating Figure 4..."
python plot_fig4.py --checkpoint "$CKPT_DIR"

# --- Copy outputs to run dir ---
cp "$CKPT_DIR/figure4_pareto.png" "$RUN_DIR/" 2>/dev/null || true
cp "$CKPT_DIR/eval_per_threshold.npy" "$RUN_DIR/" 2>/dev/null || true
cp "$CKPT_DIR/eval_per_layer.npy" "$RUN_DIR/" 2>/dev/null || true

echo "========================================="
echo " All phases complete: $(date)"
echo " Results in: $RUN_DIR/"
echo ""
echo " Paper targets:"
echo "   60k (Section 5): PPL ~23.83"
echo "   Also check: average_depth (should be < 108, router is skipping)"
echo ""
echo " Key outputs:"
echo "   eval_40k.txt / eval_60k.txt  -- per-checkpoint PPL"
echo "   figure4_pareto.png           -- PPL vs MACs Pareto plot"
echo "   eval_per_threshold.npy       -- router threshold sweep data"
echo "   eval_per_layer.npy           -- fixed depth sweep data"
echo "========================================="
