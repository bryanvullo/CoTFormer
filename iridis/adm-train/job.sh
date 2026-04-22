#!/bin/bash
#SBATCH --job-name=adm_train
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
################################################################################
# Adaptive LN-CoTFormer (ADM) Training — iridis/adm-train
#
# Trains the adaptive LN-CoTFormer with Mixture-of-Repeats router.
# This is the model needed to reproduce Figure 4 of the paper.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/adm-train/job.sh
#
# KEY DIFFERENCES from lncot-train:
#   - Model: adaptive_cotformer_..._single_final (has router component)
#   - Steps: 60,000 (paper Section 4.2 — 50% more than non-adaptive)
#   - Batch: bs=16 × acc=8 = 128 effective (router uses more VRAM)
#   - Produces: router weights alongside model checkpoints
#
# This model is SEPARATE from the non-adaptive LN-CoTFormer.
# Figure 4 uses ONLY this model's checkpoint:
#   - "Router" curve: eval with router-derived thresholds
#   - "Fixed Depth" curve: eval with fixed prefix activations
#   Both use the same trained adaptive checkpoint.
#
# Job chaining: will likely need 2-3 submissions for 60k steps.
#   Just resubmit — auto-resume handles continuations.
################################################################################

# ========================= CONFIGURATION ====================================

N_GPUS=2          # Must match --gres=gpu:N above
N_LAYER=24        # Paper uses 24 for adaptive model
N_REPEAT=5        # Max repeats (router decides per-token)
ITERATIONS=60000  # 60k steps (paper Section 4.2)
BATCH_SIZE=8      # Per-GPU micro-batch (L4 24GB OOMs at 16 with 108 effective layers)
ACC_STEPS=16      # Gradient accumulation (DDP halves: per-GPU=8, eff BS = 8*16 = 128)
CKPT_FREQ=2000    # Save every 2000 steps

# Reserved layers:
N_LAYER_BEGIN=2
N_LAYER_END=1

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== Adaptive LN-CoTFormer (ADM) Training ==="
    echo "  Partition: ecsstudents_l4"
    echo "  GPUs:      $N_GPUS"
    echo "  Layers:    $N_LAYER (${N_LAYER_BEGIN}+mid*${N_REPEAT}+${N_LAYER_END})"
    echo "  Steps:     $ITERATIONS (60k — needs 2-3 job submissions)"
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

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set — falling back to $REPO_DIR"
fi

source "$REPO_DIR/iridis/env.sh"

# Every shared variable (checkpoint root, tiktoken cache, CUDA allocator
# config, wandb mode, NCCL protocol) is exported by iridis/env.sh (single
# source of truth).
mkdir -p "$EXPS_DIR" "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"

echo "========================================="
echo " Adaptive LN-CoTFormer (ADM) Training"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " CPUs:          $SLURM_CPUS_PER_TASK"
echo " GPUs:          $N_GPUS"
echo " Job ID:        $SLURM_JOB_ID"
echo " Model:         adaptive_cotformer_..._single_final"
echo " Architecture:  ${N_LAYER}L (${N_LAYER_BEGIN}->mid*${N_REPEAT}->${N_LAYER_END}) + Router"
echo " Iterations:    $ITERATIONS (60k)"
echo " Eff. BS:       $((BATCH_SIZE * ACC_STEPS))"
echo " Checkpoint:    every $CKPT_FREQ steps -> $EXPS_DIR"
echo " Data dir:      $DATA_DIR"
echo " Started:       $(date)"
echo "========================================="

# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

# Wandb offline-mode and the NCCL Simple protocol are exported by
# iridis/env.sh (single source of truth). NCCL Simple avoids the
# LL-protocol hang observed on NCCL 2.15-2.17 / CUDA 11.8 with mixed
# small/large tensor ops (~0.04 per cent throughput cost on PCIe L4; see
# docs/reprod-notes.md §B9).

cd "$REPO_DIR"

# --- GPU diagnostics ---
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# --- Verify dataset ---
if [ ! -f "$DATA_DIR/openwebtext2/train.bin" ]; then
    echo "ERROR: Dataset not found at $DATA_DIR/openwebtext2/train.bin"
    echo "  Run: bash iridis/download-dataset/job.sh"
    echo "  Then: bash iridis/extract-tokenize-dataset/job.sh"
    exit 1
fi

# --- Build training args ---
TRAIN_ARGS=(
    --config_format base
    --model adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final
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
    --n_layer_begin "$N_LAYER_BEGIN"
    --n_layer_end "$N_LAYER_END"
    --depth_embedding linear_learned
    --save_checkpoint_freq "$CKPT_FREQ"
    --results_base_folder "$EXPS_DIR"
    --use_pretrained auto
    --wandb
    --wandb_project rcotformer
    # v2: fresh start with fixed per-rank GPU RNG checkpointing (B4 fix,
    # commits 657de47+29b9690). The v1 run (adm_lr0.001_bs8x16_seqlen256)
    # completed 60k steps but had RNG state discontinuities at resume
    # boundaries, producing +0.22 excess delta vs paper. Preserved on
    # scratch for comparison.
    --exp_name "adm_v2_lr0.001_bs${BATCH_SIZE}x${ACC_STEPS}_seqlen256"
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
echo " ADM Training finished: $(date)"
echo " Exit code: $EXIT_CODE"
echo ""
echo " Checkpoints: $EXPS_DIR/owt2/adaptive_cotformer_*/"
echo ""
echo " If training incomplete (likely for 60k steps), resubmit:"
echo "   bash iridis/adm-train/job.sh"
echo ""
echo " After training completes:"
echo "   1. Sync WandB: wandb sync $WANDB_DIR/<offline-run-*>"
echo "   2. Run Figure 4 eval: bash iridis/lncot-exp4/job.sh"
echo "========================================="

exit $EXIT_CODE
