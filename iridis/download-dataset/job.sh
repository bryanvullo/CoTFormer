#!/bin/bash
################################################################################
# Download OWT2 tarball and cache tiktoken BPE vocab (login node only).
#
# Usage:  cd ~/CoTFormer && bash iridis/download-dataset/job.sh [--reset|--bpe]
#
# Flags:
#   --reset  Wipe ALL dataset and cache state, then re-download from scratch.
#            Cleans: $DATA_DIR/openwebtext2/, $TIKTOKEN_CACHE_DIR/, $HF_HOME/
#   --bpe    Only cache the tiktoken BPE vocabulary (~1 MB).
#            Useful when the tarball is already downloaded but compute nodes
#            fail on tiktoken import.
#   (none)   Interactive prompt to choose between the above modes.
#
# All modes ensure the tiktoken BPE vocab is cached for offline compute nodes.
# Paths are defined in iridis/env.sh.
################################################################################

set -eo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
source "$REPO_DIR/iridis/env.sh"

RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
exec > >(tee "$RUN_DIR/output.log") 2> >(tee "$RUN_DIR/error.log" >&2)

# --- Flag parsing ---
MODE="$1"
if [ -z "$MODE" ]; then
    echo "This script downloads dataset files that require internet access."
    echo ""
    echo "  1) Full download: tarball (~28 GB) + BPE cache [default]"
    echo "  2) BPE cache only (~1 MB, for fixing tiktoken on compute nodes)"
    echo "  3) Full reset: delete everything and re-download"
    echo ""
    read -r -p "Choice [1]: " choice
    case "$choice" in
        2) MODE="--bpe" ;;
        3) MODE="--reset" ;;
        *) MODE="--full" ;;
    esac
fi

echo "========================================="
echo " OpenWebText2 Download"
echo " User:     $USER"
echo " Data dir: $DATA_DIR"
echo " Mode:     $MODE"
echo " Run dir:  $RUN_DIR"
echo " Started:  $(date)"
echo "========================================="

# --- Reset: clean ALL related state before anything else ---
if [ "$MODE" = "--reset" ]; then
    echo "Resetting all dataset and cache state..."
    rm -rf "$DATA_DIR/openwebtext2"
    rm -rf "$TIKTOKEN_CACHE_DIR"
    rm -rf "$HF_HOME/datasets/json"*
    rm -rf "$HF_HOME/datasets/generator"*
    echo "Clean."
fi

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

mkdir -p "$DATA_DIR" "$TIKTOKEN_CACHE_DIR"

cd "$REPO_DIR"

case "$MODE" in
    --bpe)
        python -c "
from data.openwebtext2 import cache_tiktoken_bpe
cache_tiktoken_bpe()
"
        ;;
    --reset|*)
        python -c "
from data.openwebtext2 import download_openwebtext2, cache_tiktoken_bpe
download_openwebtext2('$DATA_DIR/openwebtext2')
cache_tiktoken_bpe()
"
        ;;
esac

echo "========================================="
echo " Done: $(date)"
echo " Next step:"
echo "   bash iridis/extract-tokenize-dataset/job.sh"
echo "========================================="
