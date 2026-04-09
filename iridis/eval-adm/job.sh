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
# Single sequential job that evaluates the trained ADM v2 and produces both
# BROKEN (original authors' code) and FIXED (hook-based) router weight
# extractions. This isolates the cascading bug discovered in the original
# get_router_weights.py (see reprod-notes.md C1).
#
# The original code takes element-wise minimums at the same POSITION INDEX
# across repeats, but after top-k reordering, positions correspond to
# DIFFERENT tokens at each repeat. The result is a minimum of randomly-paired
# scores, producing the broader distribution shown in the paper's Figure 5.
# Our fixed version hooks into mod_router directly and cascades per-logit,
# yielding distributions concentrated near zero (the correct result).
#
# Output structure:
#   run_N/
#     eval_4k0.txt, eval_6k0.txt        -- PPL evaluations
#     eval_summary_ckpt_*.json           -- full eval summaries
#     broken/                            -- original authors' extraction method
#       figure4_pareto.png               -- PPL vs MACs (buggy router weights)
#       figure5_reproduction.png         -- train split, buggy cascaded (matches paper)
#       figure5_analysis.png             -- val split, buggy cascaded
#       eval_per_threshold.npy, eval_per_layer.npy
#     fixed/                             -- corrected hook-based extraction
#       figure4_pareto.png               -- PPL vs MACs (correct router weights)
#       figure5_reproduction.png         -- train split, raw per-router scores
#       figure5_analysis.png             -- val split, correct cascaded
#       eval_per_threshold.npy, eval_per_layer.npy
#
# Paper targets:
#   PPL:     60k ~23.83, average_depth < 108
#   Fig 4:   Pareto curves (Router vs Fixed Depth) from ~46 to ~228 GMACs
#   Fig 5:   Router weight distribution shift between 40k and 60k training
#
# Usage:
#   cd ~/CoTFormer && bash iridis/eval-adm/job.sh
################################################################################

# ========================= CONFIGURATION ====================================

ADM_BASE="/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final"
CKPT_DIR="$ADM_BASE/adm_v2_lr0.001_bs8x16_seqlen256"

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
    echo "=== ADM v2 Evaluation + Figures 4 & 5 (broken + fixed) ==="
    echo "  Checkpoint:  $CKPT_DIR"
    echo "  Steps:       ${CKPTS[*]}"
    echo "  Phases:      eval -> broken extraction -> fixed extraction -> Figs"
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

# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# Create output subdirectories
mkdir -p "$RUN_DIR/broken" "$RUN_DIR/fixed"


# ========================= PHASE 1-2: PPL Evaluation =========================

for step in "${CKPTS[@]}"; do
    CKPT_FILE="$CKPT_DIR/ckpt_${step}.pt"
    OUT_FILE="$RUN_DIR/eval_${step/000/k}.txt"

    echo ""
    echo "--- Phase 1-2: Evaluating ckpt_${step}.pt ---"

    python eval.py \
        --checkpoint "$CKPT_FILE" \
        --config_format base \
        --distributed_backend None \
        --output_dir "$RUN_DIR" \
        2>&1 | tee "$OUT_FILE"
done


# ========================= PHASE 3: BROKEN Router Extraction =================
# Uses the original authors' custom forward method with the cascading bug.
# Position-indexed min across reordered repeats = mismatched token scores.
echo ""
echo "###################################################################"
echo " PHASE 3: BROKEN extraction (original authors' method)"
echo "###################################################################"

echo "  3a: 60k val (broken)"
python broken_get_router_weights.py --checkpoint "$CKPT_DIR" --split val --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/broken_router_weights_60k_val.npy"

echo "  3b: 40k val (broken)"
python broken_get_router_weights.py --checkpoint "$CKPT_DIR/ckpt_40000.pt" --split val --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/broken_router_weights_40k_val.npy"

echo "  3c: 60k train (broken)"
python broken_get_router_weights.py --checkpoint "$CKPT_DIR" --split train --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/broken_router_weights_60k_train.npy"

echo "  3d: 40k train (broken)"
python broken_get_router_weights.py --checkpoint "$CKPT_DIR/ckpt_40000.pt" --split train --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/broken_router_weights_40k_train.npy"

# Set broken val cascaded as default for Figure 4 (broken)
cp "$CKPT_DIR/broken_router_weights_60k_val.npy" "$CKPT_DIR/router_weights.npy"

echo ""
echo "--- BROKEN Phase 4: Figure 4 -- PPL vs MACs sweep ---"
python get_ppl_per_mac.py --checkpoint "$CKPT_DIR"
python plot_fig4.py --checkpoint "$CKPT_DIR"
cp "$CKPT_DIR/figure4_pareto.png" "$RUN_DIR/broken/"
cp "$CKPT_DIR/eval_per_threshold.npy" "$RUN_DIR/broken/"
cp "$CKPT_DIR/eval_per_layer.npy" "$RUN_DIR/broken/"

echo ""
echo "--- BROKEN Phase 5a: Figure 5 -- Reproduction (train, buggy cascaded) ---"
python plot_fig5.py \
    --weights-40k "$CKPT_DIR/broken_router_weights_40k_train.npy" \
    --weights-60k "$CKPT_DIR/broken_router_weights_60k_train.npy" \
    --output "$RUN_DIR/broken/figure5_reproduction.png"

echo "--- BROKEN Phase 5b: Figure 5 -- Analysis (val, buggy cascaded) ---"
python plot_fig5.py \
    --weights-40k "$CKPT_DIR/broken_router_weights_40k_val.npy" \
    --weights-60k "$CKPT_DIR/broken_router_weights_60k_val.npy" \
    --output "$RUN_DIR/broken/figure5_analysis.png"


# ========================= PHASE 6: FIXED Router Extraction ==================
# Uses hook-based extraction with correct per-logit cascading.
echo ""
echo "###################################################################"
echo " PHASE 6: FIXED extraction (hook-based, correct cascading)"
echo "###################################################################"

echo "  6a: 60k val (fixed)"
python get_router_weights.py --checkpoint "$CKPT_DIR" --split val --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/fixed_router_weights_60k_val.npy"
cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/fixed_router_weights_raw_60k_val.npy"

echo "  6b: 40k val (fixed)"
python get_router_weights.py --checkpoint "$CKPT_DIR/ckpt_40000.pt" --split val --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/fixed_router_weights_40k_val.npy"
cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/fixed_router_weights_raw_40k_val.npy"

echo "  6c: 60k train (fixed)"
python get_router_weights.py --checkpoint "$CKPT_DIR" --split train --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/fixed_router_weights_60k_train.npy"
cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/fixed_router_weights_raw_60k_train.npy"
cp "$CKPT_DIR/router_logits.npz" "$CKPT_DIR/fixed_router_logits_60k.npz"

echo "  6d: 40k train (fixed)"
python get_router_weights.py --checkpoint "$CKPT_DIR/ckpt_40000.pt" --split train --distributed_backend None
cp "$CKPT_DIR/router_weights.npy" "$CKPT_DIR/fixed_router_weights_40k_train.npy"
cp "$CKPT_DIR/router_weights_raw.npy" "$CKPT_DIR/fixed_router_weights_raw_40k_train.npy"
cp "$CKPT_DIR/router_logits.npz" "$CKPT_DIR/fixed_router_logits_40k.npz"

# Set fixed val cascaded as default for Figure 4 (fixed)
cp "$CKPT_DIR/fixed_router_weights_60k_val.npy" "$CKPT_DIR/router_weights.npy"

echo ""
echo "--- FIXED Phase 7: Figure 4 -- PPL vs MACs sweep ---"
python get_ppl_per_mac.py --checkpoint "$CKPT_DIR"
python plot_fig4.py --checkpoint "$CKPT_DIR"
cp "$CKPT_DIR/figure4_pareto.png" "$RUN_DIR/fixed/"
cp "$CKPT_DIR/eval_per_threshold.npy" "$RUN_DIR/fixed/"
cp "$CKPT_DIR/eval_per_layer.npy" "$RUN_DIR/fixed/"

echo ""
echo "--- FIXED Phase 8a: Figure 5 -- Reproduction (train, raw per-router) ---"
python plot_fig5.py \
    --weights-40k "$CKPT_DIR/fixed_router_weights_raw_40k_train.npy" \
    --weights-60k "$CKPT_DIR/fixed_router_weights_raw_60k_train.npy" \
    --output "$RUN_DIR/fixed/figure5_reproduction.png"

echo "--- FIXED Phase 8b: Figure 5 -- Analysis (val, correct cascaded) ---"
python plot_fig5.py \
    --weights-40k "$CKPT_DIR/fixed_router_weights_40k_val.npy" \
    --weights-60k "$CKPT_DIR/fixed_router_weights_60k_val.npy" \
    --output "$RUN_DIR/fixed/figure5_analysis.png"

# Restore fixed val cascaded as default (for any downstream tools)
cp "$CKPT_DIR/fixed_router_weights_60k_val.npy" "$CKPT_DIR/router_weights.npy"


# ========================= SUMMARY ==========================================
echo ""
echo "========================================="
echo " Evaluation complete: $(date)"
echo " Results in: $RUN_DIR/"
echo ""
echo " PPL (top-level):"
echo "   eval_4k0.txt / eval_6k0.txt"
echo "   eval_summary_ckpt_*.json"
echo ""
echo " broken/ -- original authors' extraction (cascading bug):"
echo "   figure4_pareto.png"
echo "   figure5_reproduction.png   (train, buggy cascaded = matches paper)"
echo "   figure5_analysis.png       (val, buggy cascaded)"
echo ""
echo " fixed/ -- corrected hook-based extraction:"
echo "   figure4_pareto.png"
echo "   figure5_reproduction.png   (train, raw per-router scores)"
echo "   figure5_analysis.png       (val, correct cascaded)"
echo "========================================="
