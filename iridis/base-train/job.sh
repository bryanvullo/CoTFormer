#!/bin/sh

################################################################################
# Job Script - Run main.py
#
# sbatch job.sh
################################################################################

#SBATCH --job-name=base_cotformer_12L_5R
#SBATCH --nodes=1
#SBATCH -p l4               # partition
#SBATCH --gres=gpu:1        # request 1 GPU
#SBATCH --mem=128G          # use 20GB memory
#SBATCH --time=60:00:00     # max wall time is 60 hrs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_%j.out

# --- Activate your env ---
REPO_DIR="$HOME/CoTFormer"
source "$REPO_DIR/iridis/env.sh"
module load conda
eval "$(conda shell.bash hook)"
conda activate /scratch/ab3u21/cotformer-env 

# WandB offline (compute nodes have no internet)
export WANDB_MODE=offline

# --- Sanity checks (first run) ---
# --- Package Configuration ---
echo "========================================="
echo " Slurm Job: $SLURM_JOB_ID"
echo " Node:      $(hostname)"
echo " CPUs:      $SLURM_CPUS_PER_TASK"
echo " Started:   $(date)"
which python
python --version
python -c "import sys; print('PYTHONPATH:', sys.path[:3])"
ulimit -n 4096
echo "ulimit:" $(ulimit -n)
echo "========================================="

# --- GPU diagnostics ---
echo ""
echo "GPU Info:"
nvidia-smi
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
    --model cotformer_full_depth
    --exp_name "BaseCot_12L_5R"
    --n_embd 768
    --n_head 12
    --n_layer 12
    --n_repeat 5
    --min_repeat 5
    --batch_size 32
    --sequence_length 256
    --acc_steps 4
    --dropout 0.0
    --iterations 40000
    --dataset owt2
    --data_dir "$DATA_DIR"
    --data_in_ram
    --lr 1e-3
    --weight_decay 0.1
    --warmup_percent 0.2
    --eval_freq 100
    --seed 0
    --depth_random_method uniform_random_range
    --n_layer_begin 0
    --n_layer_end 0
    --depth_embedding linear_learned
    --save_checkpoint_freq 2000
    --results_base_folder "$RESULTS_DIR"
    --use_pretrained auto
    --wandb
    --wandb_project rcotformer
    "$@"
)

# --- Run ---
python -u ./main.py "${TRAIN_ARGS[@]}" \
# You can add your arguments here, for example:
    # --arg1 arg1 \
    # --arg2 arg2 \

# --- End ---
EXIT_CODE=$?

echo "========================================="
echo " Job finished: $(date)"
echo " Exit code:    $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
