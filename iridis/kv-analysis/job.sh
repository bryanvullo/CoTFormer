#!/bin/bash
#SBATCH --job-name=kv_analysis
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
################################################################################
# KV Cache Compressibility Analysis -- iridis/kv-analysis
#
# Runs analyze_kv_compression.py on one or more trained checkpoints to inform
# MLA rank selection (Phase 4) and mcHC stream design (Phase 5).
#
# Usage:
#   cd ~/CoTFormer && bash iridis/kv-analysis/job.sh
#
# WHAT THIS DOES (per checkpoint):
#   Type A (weight-level SVD): fast, no data needed.
#     - Singular value spectrum of W_K and W_V per layer
#     - Effective rank at 90%, 95%, 99% variance thresholds
#     - Output: <output_dir>/<model_tag>_kv_compression_weights.png
#
#   Type B (activation-level SVD): requires GPU + data.
#     - Hooks into forward pass, collects K/V activations on real data
#     - PCA-style SVD on centered activations
#     - Output: <output_dir>/<model_tag>_kv_compression_activations.png
#
#   Both types produce a JSON summary for downstream analysis.
#
# PREREQUISITES:
#   - Trained model checkpoints with summary.json + ckpt.pt
#   - Dataset at $DATA_DIR/openwebtext2/ (for Type B activation analysis)
#
# ANALYSIS MATRIX (update CHECKPOINTS array below):
#   - LN-CoTFormer 40k (Table 2 baseline)
#   - LN-CoTFormer 60k (Section 5 comparison)
#   - ADM v2 40k intermediate (router effect at same step count as LN-CoT)
#   - ADM v2 60k final (best adaptive model)
#   - CoTFormer + Reserved 40k (Table 2 row 1, isolates LN effect)
#
# NOTE: This job uses 1 GPU (no DDP). Analysis is single-forward-pass.
# Expected runtime: ~15-30 min per checkpoint (Type A + B combined).
################################################################################

# ========================= CONFIGURATION ======================================

# Target MLA kv_lora_rank to evaluate against (DeepSeek-V2 default: 192)
TARGET_RANK=192

# Number of batches for activation-level analysis (more = more accurate, slower)
N_BATCHES=10

# Output directory for all analysis results
OUTPUT_DIR="/scratch/ab3u21/exps/kv-analysis"

# Checkpoints to analyse. Each entry: "<tag>|<path_to_checkpoint_dir>"
# The tag is used for output filenames. The path must contain summary.json.
#
# UPDATE THESE PATHS after training completes. Use `find` on Iridis to locate:
#   find /scratch/ab3u21/exps/owt2/ -name "summary.json" -exec dirname {} \;
CHECKPOINTS=(
    "lncot_40k|/scratch/ab3u21/exps/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256"
    "lncot_60k|/scratch/ab3u21/exps/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256"
    "adm_v2_60k|/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256"
    "cot_res_40k|/scratch/ab3u21/exps/owt2/cotformer_full_depth/cotformer_full_depth_lr0.001_bs8x16_seqlen256"
)
# NOTE: lncot_40k and lncot_60k point to the same dir. The 40k analysis uses
# ckpt_40000.pt (--checkpoint-file flag, if available); the 60k uses ckpt.pt.
# For ADM 40k intermediate, add:
#   "adm_v2_40k|<same dir as adm_v2_60k>"
# and pass --checkpoint-file ckpt_40000.pt (once the script supports it).
# Currently, the script uses ckpt.pt (final checkpoint) by default.

# ========================= END CONFIGURATION ==================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== KV Cache Compressibility Analysis ==="
    echo "  Partition:    ecsstudents_l4"
    echo "  GPU:          1 (single-GPU analysis)"
    echo "  Target rank:  $TARGET_RANK"
    echo "  Checkpoints:  ${#CHECKPOINTS[@]}"
    echo "  Output:       $OUTPUT_DIR/"
    echo "  Logs:         $RUN_DIR/"
    echo ""
    for entry in "${CHECKPOINTS[@]}"; do
        tag="${entry%%|*}"
        path="${entry#*|}"
        if [ -d "$path" ]; then
            echo "  [$tag] $path"
        else
            echo "  [$tag] $path  (WARNING: not found)"
        fi
    done
    echo ""
    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR" \
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

mkdir -p "$OUTPUT_DIR" "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR"

echo "========================================="
echo " KV Cache Compressibility Analysis"
echo " User:        $USER"
echo " Node:        $(hostname)"
echo " Job ID:      $SLURM_JOB_ID"
echo " Target rank: $TARGET_RANK"
echo " N batches:   $N_BATCHES (activation analysis)"
echo " Output:      $OUTPUT_DIR/"
echo " Started:     $(date)"
echo "========================================="

# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

# --- GPU info ---
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# --- Verify dataset (needed for activation analysis) ---
if [ ! -f "$DATA_DIR/openwebtext2/train.bin" ]; then
    echo "WARNING: Dataset not found at $DATA_DIR/openwebtext2/train.bin"
    echo "  Activation-level analysis (Type B) will be skipped."
    echo "  Weight-level analysis (Type A) will still run."
    SKIP_ACTIVATIONS=1
else
    SKIP_ACTIVATIONS=0
fi

# --- Run analysis on each checkpoint ---
ANALYSED=0
SKIPPED=0

for entry in "${CHECKPOINTS[@]}"; do
    tag="${entry%%|*}"
    ckpt_dir="${entry#*|}"

    echo ""
    echo "========================================="
    echo " Analysing: $tag"
    echo " Checkpoint: $ckpt_dir"
    echo "========================================="

    if [ ! -f "$ckpt_dir/summary.json" ]; then
        echo "  SKIP: no summary.json found. Training may not have completed."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Build analysis args
    ANALYSIS_ARGS=(
        --checkpoint "$ckpt_dir"
        --target-rank "$TARGET_RANK"
    )

    if [ "$SKIP_ACTIVATIONS" -eq 0 ]; then
        ANALYSIS_ARGS+=(
            --compute-activations
            --data_dir "$DATA_DIR"
            --n-batches "$N_BATCHES"
        )
    fi

    python analyze_kv_compression.py "${ANALYSIS_ARGS[@]}"

    # Copy outputs to central output dir with tag prefix
    for f in "$ckpt_dir"/kv_compression_*.{json,png}; do
        if [ -f "$f" ]; then
            base=$(basename "$f")
            cp "$f" "$OUTPUT_DIR/${tag}_${base}"
            echo "  -> $OUTPUT_DIR/${tag}_${base}"
        fi
    done

    ANALYSED=$((ANALYSED + 1))
done

echo ""
echo "========================================="
echo " KV Analysis complete: $(date)"
echo " Analysed: $ANALYSED checkpoints"
echo " Skipped:  $SKIPPED checkpoints (missing summary.json)"
echo ""
echo " Results: $OUTPUT_DIR/"
echo "   ls $OUTPUT_DIR/*.json   # structured data"
echo "   ls $OUTPUT_DIR/*.png    # plots"
echo ""
echo " Next steps:"
echo "   1. Review effective ranks vs target ($TARGET_RANK)"
echo "   2. Check prefix/suffix vs mid layer compressibility"
echo "   3. Compare ADM vs non-ADM KV structure"
echo "   4. Use findings to set kv_lora_rank for MLA (Phase 4)"
echo "========================================="

exit 0
