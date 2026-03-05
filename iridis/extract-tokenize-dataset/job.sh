#!/bin/bash
#SBATCH --job-name=owt2_tokenize
#SBATCH --partition=amd_student
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=iridis/extract-tokenize-dataset/slurm_%j.out
#SBATCH --error=iridis/extract-tokenize-dataset/slurm_%j.err
################################################################################
# Extract, tokenize, and write OpenWebText2 memmap bins
#
# Submit via:  cd ~/CoTFormer && sbatch iridis/extract-tokenize-dataset/job.sh
# Requires:    tarball already downloaded (see iridis/download-dataset/job.sh)
################################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

echo "========================================="
echo " OpenWebText2 Extract & Tokenize"
echo " User:     $USER"
echo " Data dir: $DATA_DIR"
echo " Node:     $(hostname)"
echo " CPUs:     $SLURM_CPUS_PER_TASK"
echo " Started:  $(date)"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

mkdir -p "$DATA_DIR" "$HF_HOME"

cd "$SCRIPT_DIR/../.."
python -c "
from data.openwebtext2 import extract_and_tokenize_openwebtext2
extract_and_tokenize_openwebtext2('$DATA_DIR/openwebtext2')
"

echo "========================================="
echo " Done: $(date)"
echo " Verify:"
ls -lh "$DATA_DIR/openwebtext2/"
echo "========================================="
