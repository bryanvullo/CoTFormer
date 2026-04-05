#!/bin/bash
#SBATCH --job-name=eval_adm
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
################################################################################
# Adaptive LN-CoTFormer (ADM) — Evaluation + Figures 4 & 5
#
# Single sequential job that evaluates the trained ADM v2 and reproduces
# Figures 4 and 5 from the paper. All phases share the same checkpoint
# directory; sequential execution avoids file contention on router_weights.
#
# Phases:
#   1-2. Evaluate 40k + 60k checkpoints (PPL, average_depth)
#   3.   Extract router weights (60k/40k x val/train = 4 forward passes)
#   4.   Figure 4: PPL vs MACs sweep (threshold + fixed depth)
#   5.   Figure 5: Router weight distribution (reproduction + analysis)
#
# Paper targets:
#   PPL:     60k ~23.83, average_depth < 108
#   Fig 4:   Pareto curves (Router vs Fixed Depth) from ~46 to ~228 GMACs
#   Fig 5:   Router weight distribution shift between 40k and 60k training
#
# Usage:
#   cd ~/CoTFormer && bash iridis/eval-adm/job.sh
#
# Supersedes the former adm-exp4 and adm-exp5 packages.
################################################################################

# ========================= CONFIGURATION ====================================

CKPT_DIR="/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256"

# Checkpoints to evaluate (step number -> filename)
CKPTS=("40000" "60000")

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
    echo "=== ADM v2 Evaluation + Figures 4 & 5 ==="
    echo "  Checkpoints: ${CKPTS[*]} steps"
    echo "  Source:       $CKPT_DIR"
    echo "  Phases:       eval -> router weights -> Fig 4 -> Fig 5"
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
echo " ADM v2 — Evaluation + Figures 4 & 5"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " GPU:           1x L4"
echo " Job ID:        $SLURM_JOB_ID"
echo " Checkpoint:    $CKPT_DIR"
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

# ========================= PHASE 1-2: Evaluation ==============================
for step in "${CKPTS[@]}"; do
    CKPT_FILE="$CKPT_DIR/ckpt_${step}.pt"
    OUT_FILE="$RUN_DIR/eval_${step/000/k}.txt"

    echo ""
    echo "========================================="
    echo " Phase 1-2: Evaluating ckpt_${step}.pt"
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

# ========================= PHASE 3: Router Weight Extraction ==================
# Extract on both splits (val + train) and both checkpoints (60k + 40k).
# Order: val first so the default router_weights.npy is val-based when
# Phase 4 reads it for the MAC sweep.
echo ""
echo "========================================="
echo " Phase 3: Router weight extraction"
echo "   4 runs: {60k,40k} x {val,train}"
echo "========================================="

echo ""
echo "--- 3a: 60k val ---"
python get_router_weights.py --checkpoint "$CKPT_DIR" --split val --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/router_weights_60k_val.npy"
cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/router_weights_raw_60k_val.npy"

echo ""
echo "--- 3b: 40k val ---"
python get_router_weights.py --checkpoint "$CKPT_DIR/ckpt_40000.pt" --split val --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/router_weights_40k_val.npy"
cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/router_weights_raw_40k_val.npy"

echo ""
echo "--- 3c: 60k train ---"
python get_router_weights.py --checkpoint "$CKPT_DIR" --split train --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/router_weights_60k_train.npy"
cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/router_weights_raw_60k_train.npy"
cp "$CKPT_DIR/router_logits.npz" "$CKPT_DIR/router_logits_60k.npz"

echo ""
echo "--- 3d: 40k train ---"
python get_router_weights.py --checkpoint "$CKPT_DIR/ckpt_40000.pt" --split train --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/router_weights_40k_train.npy"
cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/router_weights_raw_40k_train.npy"
cp "$CKPT_DIR/router_logits.npz" "$CKPT_DIR/router_logits_40k.npz"

# Restore val cascaded as the default (get_ppl_per_mac.py reads router_weights.npy)
cp "$CKPT_DIR/router_weights_60k_val.npy" "$CKPT_DIR/router_weights.npy"
echo ""

# ========================= PHASE 4: Figure 4 (PPL vs MACs) ====================
echo "========================================="
echo " Phase 4: Figure 4 — PPL vs MACs sweep"
echo "========================================="

python get_ppl_per_mac.py --checkpoint "$CKPT_DIR"
python plot_fig4.py --checkpoint "$CKPT_DIR"
echo ""

# ========================= PHASE 5: Figure 5 (Router Distributions) ===========
echo "========================================="
echo " Phase 5a: Figure 5 — Reproduction"
echo "   (train split, raw per-router scores)"
echo "========================================="

python plot_fig5.py \
    --weights-40k "$CKPT_DIR/router_weights_raw_40k_train.npy" \
    --weights-60k "$CKPT_DIR/router_weights_raw_60k_train.npy" \
    --output "$CKPT_DIR/figure5_reproduction.png"

echo ""
echo "========================================="
echo " Phase 5b: Figure 5 — Analysis"
echo "   (val split, cascaded minimum scores)"
echo "========================================="

python plot_fig5.py \
    --weights-40k "$CKPT_DIR/router_weights_40k_val.npy" \
    --weights-60k "$CKPT_DIR/router_weights_60k_val.npy" \
    --output "$CKPT_DIR/figure5_analysis.png"

# ========================= OUTPUT COPY ========================================
echo ""
echo "Copying outputs to $RUN_DIR/ ..."
for f in figure4_pareto.png eval_per_threshold.npy eval_per_layer.npy \
         figure5_reproduction.png figure5_analysis.png; do
    cp "$CKPT_DIR/$f" "$RUN_DIR/" 2>/dev/null || true
done

echo ""
echo "========================================="
echo " All phases complete: $(date)"
echo " Results in: $RUN_DIR/"
echo ""
echo " Paper targets:"
echo "   PPL 60k (Section 5): ~23.83"
echo "   average_depth:       < 108 (router is skipping)"
echo ""
echo " Outputs:"
echo "   eval_40k.txt / eval_60k.txt    — per-checkpoint PPL"
echo "   figure4_pareto.png             — PPL vs MACs (Router + Fixed Depth)"
echo "   figure5_reproduction.png       — train split, raw scores (matches paper)"
echo "   figure5_analysis.png           — val split, cascaded (supplementary)"
echo "========================================="
