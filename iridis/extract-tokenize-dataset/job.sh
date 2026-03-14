#!/bin/bash
#SBATCH --job-name=owt2_extract_tokenize
#SBATCH --partition=amd_student
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
################################################################################
# Extract, tokenize, and write OpenWebText2 memmap bins
#
# Usage:  cd ~/CoTFormer && bash iridis/extract-tokenize-dataset/job.sh
# Requires: tarball already downloaded (see iridis/download-dataset/job.sh)
#
# Self-submitting: run with bash, it creates a run_N dir and calls sbatch.
##################################################q##############################

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "Submitting to amd_student, logs → $RUN_DIR/"
    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=ALL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR" \
        "$0" "$@"
fi

# --- Actual job (runs on compute node) ---
set -eo pipefail
export PYTHONUNBUFFERED=1  # flush prints immediately to SLURM logs

# REPO_DIR is passed by the self-submitting wrapper via --export.
# Fallback for direct sbatch invocation (prefer 'bash job.sh' instead).
if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set — falling back to $REPO_DIR"
    echo "Tip: use 'bash job.sh' instead of 'sbatch job.sh'"
fi

source "$REPO_DIR/iridis/env.sh"

echo "========================================="
echo " OpenWebText2 Extract & Tokenize"
echo " User:     $USER"
echo " Data dir: $DATA_DIR"
echo " Node:     $(hostname)"
echo " CPUs:     $SLURM_CPUS_PER_TASK"
echo " Job ID:   $SLURM_JOB_ID"
echo " Started:  $(date)"
echo "========================================="

# Conda stuff
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

# Ensure scratch dirs exist
mkdir -p "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR"

# Change to where executables live
cd "$REPO_DIR"

# Extract and tokenize dataset with edited openwebtext2.py (CPU node only)
python -c " 
from data.openwebtext2 import extract_and_tokenize_openwebtext2
extract_and_tokenize_openwebtext2('$DATA_DIR/openwebtext2')
"

echo "========================================="
echo " Done: $(date)"
echo " Verify:"
ls -lh "$DATA_DIR/openwebtext2/"
echo "========================================="
