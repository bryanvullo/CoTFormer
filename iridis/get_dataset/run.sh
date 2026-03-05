#!/bin/bash
################################################################################
# Download and tokenize OpenWebText2
#
# Run interactively on a login node (requires internet).
# Usage:  cd ~/CoTFormer && bash iridis/get_dataset/run.sh
################################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

echo "========================================="
echo " OpenWebText2 Dataset Setup"
echo " User:     $USER"
echo " Data dir: $DATA_DIR"
echo " Started:  $(date)"
echo "========================================="

# Activate environment
module load conda
conda activate "$CONDA_ENV_PREFIX"

# Create scratch directories if they don't exist
mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$HF_HOME" "$WANDB_DIR"

# Download, extract, tokenize, cleanup (all handled in Python)
cd "$SCRIPT_DIR/../.."
python -c "
from data.openwebtext2 import get_openwebtext2_data
import argparse
get_openwebtext2_data(argparse.Namespace(data_dir='$DATA_DIR'))
"

echo "========================================="
echo " Done: $(date)"
echo " Verify:"
ls -lh "$DATA_DIR/openwebtext2/"
echo "========================================="
