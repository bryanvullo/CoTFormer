#!/bin/bash

################################################################################
# Job Script — Run main.py
#
# Usage:  cd ~/CoTFormer && sbatch iridis/job.sh
################################################################################

#SBATCH --job-name=cotformer
#SBATCH --nodes=1
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=iridis/slurm_%j.out
#SBATCH --error=iridis/slurm_%j.err

# --- Environment ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

module load conda
conda activate "$CONDA_ENV_PREFIX"

# --- Sanity checks ---
echo "========================================="
echo " Slurm Job: $SLURM_JOB_ID"
echo " Node:      $(hostname)"
echo " CPUs:      $SLURM_CPUS_PER_TASK"
echo " DATA_DIR:  $DATA_DIR"
echo " RESULTS:   $RESULTS_DIR"
echo " Started:   $(date)"
which python
python --version
echo "========================================="

nvidia-smi

# --- Run ---
python -u ./main.py \
    --data_dir "$DATA_DIR" \
    --results_base_folder "$RESULTS_DIR" \
    # Add experiment-specific arguments below, e.g.:
    # --model base --n_layer 12 --n_embd 768 --n_head 12 \
    # --batch_size 64 --acc_steps 2 --sequence_length 256 \
    # --iterations 40000 --dataset owt2 --lr 1e-3 \
    # --eval_freq 100 --seed 0

# --- End ---
EXIT_CODE=$?

echo "========================================="
echo " Job finished: $(date)"
echo " Exit code:    $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
