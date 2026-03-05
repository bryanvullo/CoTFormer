#!/bin/bash
################################################################################
# Download the OpenWebText2 tarball (~28 GB)
#
# Run on a login node (requires internet access).
# Usage:  cd ~/CoTFormer && bash iridis/download-dataset/job.sh
#
# After this completes, submit the tokenization job:
#   sbatch iridis/extract-tokenize-dataset/job.sh
################################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

echo "========================================="
echo " OpenWebText2 Download"
echo " User:     $USER"
echo " Data dir: $DATA_DIR"
echo " Started:  $(date)"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

mkdir -p "$DATA_DIR"

cd "$SCRIPT_DIR/../.."
python -c "
from data.openwebtext2 import download_openwebtext2
download_openwebtext2('$DATA_DIR/openwebtext2')
"

echo "========================================="
echo " Download complete: $(date)"
echo " Next step:"
echo "   sbatch iridis/extract-tokenize-dataset/job.sh"
echo "========================================="
