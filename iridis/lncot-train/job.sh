#!/bin/bash
#SBATCH --job-name=lncot_train
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
################################################################################
# LN-CoTFormer Training — iridis/lncot-train
#
# Trains the non-adaptive LN-CoTFormer (cotformer_full_depth_lnmid_depthemb).
# This reproduces Table 2 of the paper (perplexity 24.11 target).
#
# Usage:
#   cd ~/CoTFormer && bash iridis/lncot-train/job.sh
#
# Features:
#   - Self-submitting wrapper (creates run_N dir, calls sbatch)
#   - Multi-GPU DDP via torchrun (set N_GPUS + --gres above)
#   - Auto-resume from latest checkpoint (--use_pretrained auto)
#   - Checkpoints every 2000 steps to /scratch (not home — quota safe)
#   - WandB offline mode (sync from login node after job)
#   - Email notifications on start/end/fail
#
# Job chaining (if training exceeds 24h):
#   Just resubmit: bash iridis/lncot-train/job.sh
#   Auto-resume picks up from the latest ckpt_N.pt.
#
# Architecture (paper Section 3.3, Table 2):
#   24 layers = 2 prefix (reserved) + 21 mid (repeated 5x) + 1 suffix
#   + LayerNorm between repeats + learned depth embedding
#
# NOTE: The paper trains this on A100 80GB for ~12-18h. On 2x L4 24GB
#   with DDP, expect ~18-22h. If it doesn't finish in 24h, resubmit.
################################################################################

# ========================= CONFIGURATION ====================================
# Adjust these to match your --gres and architecture choices.

N_GPUS=4          # Must match --gres=gpu:N above
N_LAYER=24        # Paper uses 24 for LN-CoTFormer (Table 2). Change to 12 if needed.
N_REPEAT=5        # Number of block repeats
ITERATIONS=40000  # Training steps
BATCH_SIZE=32     # Per-GPU batch size (DDP divides this further)
ACC_STEPS=4       # Gradient accumulation steps (DDP divides this further)
                  # Effective batch size = BATCH_SIZE * ACC_STEPS = 128
CKPT_FREQ=2000    # Save checkpoint every N steps

# Reserved layers (prefix/suffix not repeated):
N_LAYER_BEGIN=2   # 2 prefix layers
N_LAYER_END=1     # 1 suffix layer

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== LN-CoTFormer Training ==="
    echo "  Partition: ecsstudents_l4"
    echo "  GPUs:      $N_GPUS"
    echo "  Layers:    $N_LAYER (${N_LAYER_BEGIN}+mid*${N_REPEAT}+${N_LAYER_END})"
    echo "  Steps:     $ITERATIONS"
    echo "  Eff. BS:   $((BATCH_SIZE * ACC_STEPS))"
    echo "  Logs:      $RUN_DIR/"
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

# REPO_DIR passed by self-submitting wrapper via --export
if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set — falling back to $REPO_DIR"
    echo "Tip: use 'bash job.sh' instead of 'sbatch job.sh'"
fi

source "$REPO_DIR/iridis/env.sh"

# --- Scratch-based output dirs (off home quota) ---
EXPS_DIR="/scratch/ab3u21/exps"
mkdir -p "$EXPS_DIR" "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"

echo "========================================="
echo " LN-CoTFormer Training"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " CPUs:          $SLURM_CPUS_PER_TASK"
echo " GPUs:          $N_GPUS"
echo " Job ID:        $SLURM_JOB_ID"
echo " Model:         cotformer_full_depth_lnmid_depthemb"
echo " Architecture:  ${N_LAYER}L (${N_LAYER_BEGIN}→mid×${N_REPEAT}→${N_LAYER_END})"
echo " Iterations:    $ITERATIONS"
echo " Eff. BS:       $((BATCH_SIZE * ACC_STEPS))"
echo " Checkpoint:    every $CKPT_FREQ steps → $EXPS_DIR"
echo " Data dir:      $DATA_DIR"
echo " Started:       $(date)"
echo "========================================="

# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

# WandB offline (compute nodes have no internet)
export WANDB_MODE=offline

cd "$REPO_DIR"

# --- GPU diagnostics ---
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# --- Verify dataset exists ---
if [ ! -f "$DATA_DIR/openwebtext2/train.bin" ]; then
    echo "ERROR: Dataset not found at $DATA_DIR/openwebtext2/train.bin"
    echo "  Run: bash iridis/download-dataset/job.sh"
    echo "  Then: bash iridis/extract-tokenize-dataset/job.sh"
    exit 1
fi

# --- Build training args ---
TRAIN_ARGS=(
    --config_format base
    --model cotformer_full_depth_lnmid_depthemb
    --n_embd 768
    --n_head 12
    --n_layer "$N_LAYER"
    --n_repeat "$N_REPEAT"
    --batch_size "$BATCH_SIZE"
    --sequence_length 256
    --acc_steps "$ACC_STEPS"
    --dropout 0.0
    --iterations "$ITERATIONS"
    --dataset owt2
    --data_dir "$DATA_DIR"
    --data_in_ram
    --lr 1e-3
    --weight_decay 0.1
    --warmup_percent 0.2
    --eval_freq 100
    --seed 0
    --depth_random_method uniform_random_range
    --n_layer_begin "$N_LAYER_BEGIN"
    --n_layer_end "$N_LAYER_END"
    --min_repeat "$N_REPEAT"
    --depth_embedding linear_learned
    --save_checkpoint_freq "$CKPT_FREQ"
    --results_base_folder "$EXPS_DIR"
    --use_pretrained auto
    --wandb
    --wandb_project rcotformer
    "$@"
)

# --- Launch ---
if [ "$N_GPUS" -gt 1 ]; then
    export OMP_NUM_THREADS=1
    RDZV_HOST=$(hostname)
    RDZV_PORT=$(expr 10000 + $(echo -n "$SLURM_JOB_ID" | tail -c 4))

    echo "Launching DDP training: ${N_GPUS} GPUs, RDZV ${RDZV_HOST}:${RDZV_PORT}"
    torchrun \
        --nproc_per_node="$N_GPUS" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT}" \
        main.py "${TRAIN_ARGS[@]}" \
        --distributed_backend nccl
else
    echo "Launching single-GPU training"
    python main.py "${TRAIN_ARGS[@]}"
fi

EXIT_CODE=$?

echo "========================================="
echo " Training finished: $(date)"
echo " Exit code: $EXIT_CODE"
echo ""
echo " Checkpoints: $EXPS_DIR/owt2/cotformer_full_depth_lnmid_depthemb/"
echo ""
echo " If training incomplete, resubmit:"
echo "   bash iridis/lncot-train/job.sh"
echo ""
echo " After training completes, sync WandB:"
echo "   wandb sync $WANDB_DIR/<offline-run-*>"
echo "========================================="

exit $EXIT_CODE
