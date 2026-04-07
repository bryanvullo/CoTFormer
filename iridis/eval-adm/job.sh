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

ADM_BASE="/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final"

# 2x2 matrix: evaluate BOTH v1 (pre-B4 RNG fix) and v2 (post-B4 RNG fix)
# to characterise the training dynamics divergence (reprod-notes B12).
declare -A ADM_DIRS=(
    [adm_v1]="$ADM_BASE/adm_lr0.001_bs8x16_seqlen256"
    [adm_v2]="$ADM_BASE/adm_v2_lr0.001_bs8x16_seqlen256"
)

# Checkpoints to evaluate (step number -> filename)
CKPTS=("40000" "60000")

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    # Pre-flight: verify checkpoints exist for ALL ADM versions
    for adm_label in "${!ADM_DIRS[@]}"; do
        adm_dir="${ADM_DIRS[$adm_label]}"
        for step in "${CKPTS[@]}"; do
            ckpt_file="$adm_dir/ckpt_${step}.pt"
            if [ ! -f "$ckpt_file" ]; then
                echo "ERROR: Checkpoint not found: $ckpt_file"
                exit 1
            fi
        done
        if [ ! -f "$adm_dir/summary.json" ]; then
            echo "ERROR: summary.json not found in $adm_dir"
            exit 1
        fi
    done

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== ADM v1+v2 Evaluation + Figures 4 & 5 ==="
    echo "  Versions:    ${!ADM_DIRS[*]}"
    echo "  Checkpoints: ${CKPTS[*]} steps"
    echo "  Phases:      eval -> router weights -> Fig 4 -> Fig 5 (per version)"
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
echo " ADM v1+v2 — Evaluation + Figures 4 & 5"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " GPU:           1x L4"
echo " Job ID:        $SLURM_JOB_ID"
echo " Versions:      ${!ADM_DIRS[*]}"
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

# ========================= MAIN LOOP: iterate over ADM versions ===============
for ADM_LABEL in "${!ADM_DIRS[@]}"; do
    CKPT_DIR="${ADM_DIRS[$ADM_LABEL]}"
    OUT_SUBDIR="$RUN_DIR/$ADM_LABEL"
    mkdir -p "$OUT_SUBDIR"

    echo ""
    echo "###################################################################"
    echo " === $ADM_LABEL === ($CKPT_DIR)"
    echo "###################################################################"

    # ========================= PHASE 1-2: Evaluation ==========================
    for step in "${CKPTS[@]}"; do
        CKPT_FILE="$CKPT_DIR/ckpt_${step}.pt"
        OUT_FILE="$OUT_SUBDIR/eval_${step/000/k}.txt"

        echo ""
        echo "--- $ADM_LABEL Phase 1-2: Evaluating ckpt_${step}.pt ---"

        python eval.py \
            --checkpoint "$CKPT_FILE" \
            --config_format base \
            --distributed_backend None \
            --output_dir "$OUT_SUBDIR" \
            2>&1 | tee "$OUT_FILE"
    done

    # ========================= PHASE 3: Router Weight Extraction ==============
    echo ""
    echo "--- $ADM_LABEL Phase 3: Router weight extraction (4 runs) ---"

    echo "  3a: 60k val"
    python get_router_weights.py --checkpoint "$CKPT_DIR" --split val --distributed_backend None
    cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/router_weights_60k_val.npy"
    cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/router_weights_raw_60k_val.npy"

    echo "  3b: 40k val"
    python get_router_weights.py --checkpoint "$CKPT_DIR/ckpt_40000.pt" --split val --distributed_backend None
    cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/router_weights_40k_val.npy"
    cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/router_weights_raw_40k_val.npy"

    echo "  3c: 60k train"
    python get_router_weights.py --checkpoint "$CKPT_DIR" --split train --distributed_backend None
    cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/router_weights_60k_train.npy"
    cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/router_weights_raw_60k_train.npy"
    cp "$CKPT_DIR/router_logits.npz" "$CKPT_DIR/router_logits_60k.npz"

    echo "  3d: 40k train"
    python get_router_weights.py --checkpoint "$CKPT_DIR/ckpt_40000.pt" --split train --distributed_backend None
    cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/router_weights_40k_train.npy"
    cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/router_weights_raw_40k_train.npy"
    cp "$CKPT_DIR/router_logits.npz" "$CKPT_DIR/router_logits_40k.npz"

    # Restore val cascaded as default (get_ppl_per_mac.py reads router_weights.npy)
    cp "$CKPT_DIR/router_weights_60k_val.npy" "$CKPT_DIR/router_weights.npy"

    # ========================= PHASE 4: Figure 4 (PPL vs MACs) ================
    echo ""
    echo "--- $ADM_LABEL Phase 4: Figure 4 — PPL vs MACs sweep ---"

    python get_ppl_per_mac.py --checkpoint "$CKPT_DIR"
    python plot_fig4.py --checkpoint "$CKPT_DIR"

    # ========================= PHASE 5: Figure 5 (Router Distributions) =======
    echo ""
    echo "--- $ADM_LABEL Phase 5a: Figure 5 — Reproduction (train/raw) ---"

    python plot_fig5.py \
        --weights-40k "$CKPT_DIR/router_weights_raw_40k_train.npy" \
        --weights-60k "$CKPT_DIR/router_weights_raw_60k_train.npy" \
        --output "$CKPT_DIR/figure5_reproduction.png"

    echo "--- $ADM_LABEL Phase 5b: Figure 5 — Analysis (val/cascaded) ---"

    python plot_fig5.py \
        --weights-40k "$CKPT_DIR/router_weights_40k_val.npy" \
        --weights-60k "$CKPT_DIR/router_weights_60k_val.npy" \
        --output "$CKPT_DIR/figure5_analysis.png"

    # ========================= OUTPUT COPY ====================================
    echo ""
    echo "Copying $ADM_LABEL outputs to $OUT_SUBDIR/ ..."
    for f in figure4_pareto.png eval_per_threshold.npy eval_per_layer.npy \
             figure5_reproduction.png figure5_analysis.png; do
        cp "$CKPT_DIR/$f" "$OUT_SUBDIR/" 2>/dev/null || true
    done

    echo ""
    echo "=== $ADM_LABEL COMPLETE ==="

done  # end ADM version loop

echo ""
echo "========================================="
echo " All versions complete: $(date)"
echo " Results in: $RUN_DIR/"
echo ""
echo " Outputs per version (adm_v1/, adm_v2/):"
echo "   eval_40k.txt / eval_60k.txt    — per-checkpoint PPL"
echo "   figure4_pareto.png             — PPL vs MACs (Router + Fixed Depth)"
echo "   figure5_reproduction.png       — train split, raw scores (matches paper)"
echo "   figure5_analysis.png           — val split, cascaded (supplementary)"
echo "========================================="
