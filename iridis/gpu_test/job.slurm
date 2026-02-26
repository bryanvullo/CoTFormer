#!/bin/bash
################################################################################
# Slurm Job Script — GPU Smoke Test
#
# Submit from the project root (~/dpdl/):
#   sbatch iridis/gpu_test/job.slurm
################################################################################

#SBATCH --job-name=gpu_test
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=iridis/gpu_test/slurm_%j.out
#SBATCH --error=iridis/gpu_test/slurm_%j.err

# --- Package Configuration ---
PKG_NAME="iridis/gpu_test"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/dpdl}"

echo "========================================="
echo " Slurm Job: $SLURM_JOB_ID"
echo " Package:   $PKG_NAME"
echo " Node:      $(hostname)"
echo " CPUs:      $SLURM_CPUS_PER_TASK"
echo " Started:   $(date)"
echo "========================================="

# --- Load container runtime ---
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || {
    echo "ERROR: Neither apptainer nor singularity module available"
    exit 1
}

cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# --- Run inside container ---
# --nv:       exposes host NVIDIA drivers + GPUs to the container
# --unsquash: required on compute nodes (no fusermount)
# --bind $PWD: makes project files visible inside the container
nvidia-smi

apptainer exec --nv --unsquash --bind "$PWD" \
    iridis/rcotformer.sif \
    python3 "${PKG_NAME}/test_gpu.py"

EXIT_CODE=$?

echo "========================================="
echo " Job finished: $(date)"
echo " Exit code:    $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
