#!/bin/bash
#SBATCH --job-name=base_cot
#SBATCH --partition=ecsstudents_l4   # Bryan: you may use -p l4 if your account has access
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
################################################################################
# CoTFormer Base Training -- Table 1 Reproduction (iridis/base-train)
#
# Trains the plain CoTFormer (cotformer_full_depth) for Table 1 of the paper
# (Section 3.2). This is the BASE architecture: no LayerNorm between repeats,
# no depth embedding, no reserved layers. Those tweaks come in Section 3.3
# (see iridis/lncot-train for that model).
#
# QUICK START:
#   1. Set N_LAYER and N_REPEAT in the CONFIGURATION section below
#   2. Run from the repo root (runs directly, or copy to job.sh first):
#        cd ~/CoTFormer && bash iridis/base-train/job-template.sh
#   3. The script auto-submits via sbatch. Do NOT run with 'sbatch' directly.
#
# TABLE 1 CONFIGURATIONS (paper target perplexities):
#
#   N_LAYER  N_REPEAT  Eff. Layers  Target PPL  Est. Time (2x L4)
#   -------  --------  -----------  ----------  -----------------
#   12       2         24           27.55       ~4h
#   12       3         36           27.07       ~6h
#   12       5         60           26.64       ~10h
#   24       2         48           25.28       ~8h
#   24       3         72           24.85       ~13h
#   24       5         120          24.48       ~20h
#
#   All configs: 40k steps, eff. batch size 128, lr 0.001, 8000 warmup steps.
#   Paper reports mean perplexity of 3 runs (seeds 0, 1, 2).
#
# BATCH SIZE GUIDE (L4 24GB VRAM, sequence_length=256):
#
#   Eff. Layers  Max Safe BS  Recommended (BS x ACC = 128)
#   -----------  -----------  ----------------------------
#   24  (12x2)   32           8 x 16  (or 16 x 8 for speed)
#   36  (12x3)   16           8 x 16  (or 16 x 8)
#   60  (12x5)   8            8 x 16
#   48  (24x2)   16           8 x 16  (or 16 x 8)
#   72  (24x3)   8            8 x 16
#   120 (24x5)   8            8 x 16
#
#   Default BATCH_SIZE=8, ACC_STEPS=16 is safe for ALL configs.
#   Larger BATCH_SIZE reduces gradient accumulation steps (faster) but risks
#   OOM on deeper configs. Start with 8; increase only after a successful run.
#
# CHECKPOINT RESUMPTION:
#   The 24-layer configs may not finish in one 24h job. To resume, simply
#   resubmit the SAME script with the SAME config:
#     cd ~/CoTFormer && bash iridis/base-train/job-template.sh
#   The --use_pretrained auto flag scans the checkpoint dir for the latest
#   ckpt_<step>.pt and resumes from there. The checkpoint path is determined
#   by RESULTS_BASE, dataset, model, and EXP_NAME -- as long as these stay
#   the same, resumption finds the right checkpoint automatically.
#
#   DO NOT change EXP_NAME between resubmissions for the same config, or
#   resumption will start from scratch in a new directory.
#
# MULTI-SEED RUNS:
#   Table 1 reports the mean of 3 runs. To train with a different seed,
#   change SEED in the config section. Each seed gets its own checkpoint dir:
#     table1_12L_5rep_bs8x16_seed0/
#     table1_12L_5rep_bs8x16_seed1/
#     table1_12L_5rep_bs8x16_seed2/
################################################################################

# ========================= CONFIGURATION ======================================
# Change these two variables to select which Table 1 entry to train.
# Then resubmit: cd ~/CoTFormer && bash iridis/base-train/job-template.sh

N_GPUS=2            # Must match --gres=gpu:N above (max 2 on ecsstudents_l4)
N_LAYER=12          # 12 or 24 (paper Table 1 rows)
N_REPEAT=5          # 2, 3, or 5 (paper Table 1 columns)
ITERATIONS=40000    # Paper: 40k steps for all Table 1 entries
BATCH_SIZE=8        # Per-GPU micro-batch (safe for all configs on L4 24GB)
ACC_STEPS=16        # Gradient accumulation steps
                    # Effective batch size = BATCH_SIZE * ACC_STEPS = 128
                    # (DDP halves ACC_STEPS per GPU but multiplies by N_GPUS,
                    #  so effective batch is always BATCH_SIZE * ACC_STEPS.)
CKPT_FREQ=2000      # Save checkpoint every N steps (20 checkpoints at 40k)
SEED=0              # Random seed (paper uses seeds 0, 1, 2 for 3-run mean)

# Table 1 = base CoTFormer, NO reserved layers:
N_LAYER_BEGIN=0     # No prefix layers (Section 3.3 uses 2)
N_LAYER_END=0       # No suffix layers (Section 3.3 uses 1)

# ========================= END CONFIGURATION ==================================

# --- Self-submitting wrapper (runs on login node) ---
# When you run 'bash job.sh', this block detects you're NOT inside a SLURM job,
# creates a log directory, and submits the script to the queue via sbatch.
# The 'exec' replaces this shell process with sbatch, so the script exits here
# on the login node. SLURM then runs the rest on a compute node.
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== CoTFormer Base Training (Table 1) ==="
    echo "  Partition: ecsstudents_l4"
    echo "  GPUs:      $N_GPUS"
    echo "  Config:    ${N_LAYER}L x ${N_REPEAT} repeats (${N_LAYER_BEGIN}+mid*${N_REPEAT}+${N_LAYER_END})"
    echo "  Steps:     $ITERATIONS"
    echo "  Eff. BS:   $((BATCH_SIZE * ACC_STEPS))"
    echo "  Seed:      $SEED"
    echo "  Logs:      $RUN_DIR/"
    echo ""
    exec sbatch \
        --job-name="base_cot_${N_LAYER}L_${N_REPEAT}R" \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR" \
        "$0" "$@"
fi

# =============================================================================
# Everything below runs ON THE COMPUTE NODE inside a SLURM job.
# =============================================================================
set -eo pipefail
export PYTHONUNBUFFERED=1

# REPO_DIR is passed by the self-submitting wrapper via --export.
# Fallback handles the case where someone runs 'sbatch job.sh' directly
# (not recommended -- use 'bash job.sh' instead).
if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
    echo "Tip: use 'bash job.sh' instead of 'sbatch job.sh' for proper setup"
fi

source "$REPO_DIR/iridis/env.sh"

# --- Checkpoint output directory ---
# main.py constructs ckpt_path = RESULTS_BASE / dataset / model / exp_name
# and creates it via os.makedirs. Job scripts must NOT mirror that formula
# in bash — see docs/reprod-notes.md §C4 for why.
#
# EXP_NAME is the LEAF dir name. Must stay IDENTICAL across resubmissions
# for --use_pretrained auto to find the previous checkpoint.
RESULTS_BASE="/scratch/$USER/exps"
EXP_NAME="table1_${N_LAYER}L_${N_REPEAT}rep_bs${BATCH_SIZE}x${ACC_STEPS}_seed${SEED}"
mkdir -p "$RESULTS_BASE" "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"

echo "========================================="
echo " CoTFormer Base Training (Table 1)"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " CPUs:          $SLURM_CPUS_PER_TASK"
echo " GPUs:          $N_GPUS"
echo " Job ID:        $SLURM_JOB_ID"
echo " Model:         cotformer_full_depth"
echo " Config:        ${N_LAYER}L x ${N_REPEAT} repeats"
echo " Eff. layers:   $((N_LAYER * N_REPEAT))"
echo " Iterations:    $ITERATIONS"
echo " Eff. BS:       $((BATCH_SIZE * ACC_STEPS))"
echo " Seed:          $SEED"
echo " Checkpoint:    every $CKPT_FREQ steps -> $RESULTS_BASE/owt2/cotformer_full_depth/$EXP_NAME"
echo " Data dir:      $DATA_DIR"
echo " Started:       $(date)"
echo "========================================="

# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

# Wandb offline-mode and the NCCL Simple protocol are exported by
# iridis/env.sh (single source of truth); compute nodes have no internet
# and NCCL Simple avoids the LL-protocol hang on NCCL 2.15-2.17 / CUDA
# 11.8 during DDP checkpoint coordination (see docs/reprod-notes.md §B9).

cd "$REPO_DIR"

# --- File descriptor limit (prevents "too many open files" on large datasets) ---
ulimit -n 4096
echo "ulimit (open files): $(ulimit -n)"

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
# NOTE: cotformer_full_depth does NOT use --depth_embedding,
# --depth_random_method, or --min_repeat. Those flags exist in the config
# parser but the model ignores them -- it always runs all n_repeat iterations
# unconditionally (see models/cotformer_full_depth.py line 322). Including
# them here would only pollute the auto-generated exp_name with misleading
# values. For the LN-CoTFormer that DOES use these, see iridis/lncot-train.
TRAIN_ARGS=(
    --config_format base
    --model cotformer_full_depth
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
    --seed "$SEED"
    --n_layer_begin "$N_LAYER_BEGIN"
    --n_layer_end "$N_LAYER_END"
    --save_checkpoint_freq "$CKPT_FREQ"
    --results_base_folder "$RESULTS_BASE"
    --exp_name "$EXP_NAME"
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
echo " Checkpoints: $RESULTS_BASE/owt2/cotformer_full_depth/$EXP_NAME"
echo ""
echo " If training did not finish (24h limit), just resubmit:"
echo "   cd ~/CoTFormer && bash iridis/base-train/job-template.sh"
echo "   (auto-resumes from latest checkpoint)"
echo ""
echo " After all training completes, sync WandB:"
echo "   wandb sync $WANDB_DIR/<offline-run-*>"
echo "========================================="

exit $EXIT_CODE
