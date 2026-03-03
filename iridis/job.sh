#!/bin/sh

################################################################################
# Job Script — Run main.py
#
# sbatch job.sh
################################################################################

#SBATCH --job-name=base_cotformer
#SBATCH --nodes=1
#SBATCH -p l4               # partition
#SBATCH --gres=gpu:1        # request 1 GPU
#SBATCH --mem=20000         # use 20GB memory
#SBATCH --time=60:00:00     # max wall time is 60 hrs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_%j.out

# --- Activate your env ---
conda init 
conda activate cotformer-env

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
echo "========================================="

nvidia-smi

# --- Run ---
python -u ./main.py \
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
