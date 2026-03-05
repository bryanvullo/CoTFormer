#!/bin/bash
################################################################################
# Download the OpenWebText2 tarball (~28 GB)
#
# Run on a login node (requires internet access).
# Usage:  cd ~/CoTFormer && bash iridis/download-dataset/job.sh
#
# After this completes, submit the tokenization job:
#   bash iridis/extract-tokenize-dataset/job.sh
################################################################################

set -eo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
source "$REPO_DIR/iridis/env.sh"

RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
exec > >(tee "$RUN_DIR/output.log") 2> >(tee "$RUN_DIR/error.log" >&2)

echo "========================================="
echo " OpenWebText2 Download"
echo " User:     $USER"
echo " Data dir: $DATA_DIR"
echo " Run dir:  $RUN_DIR"
echo " Started:  $(date)"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

mkdir -p "$DATA_DIR"

cd "$REPO_DIR"
python -c "
from data.openwebtext2 import download_openwebtext2
download_openwebtext2('$DATA_DIR/openwebtext2')
"

echo "========================================="
echo " Download complete: $(date)"
echo " Next step:"
echo "   bash iridis/extract-tokenize-dataset/job.sh"
echo "========================================="
