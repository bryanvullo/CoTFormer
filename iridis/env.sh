#!/usr/bin/env bash
# iridis/env.sh — Iridis X environment config
# Source this file; do not execute it directly.
#
# Usage:
#   source iridis/env.sh          (from repo root)
#   source "$(dirname "$0")/env.sh"  (from SLURM scripts)

# --- Conda environment ---
CONDA_ENV_PREFIX="/scratch/$USER/cotformer-env"

# --- Per-user scratch paths ---
export DATA_DIR="/scratch/$USER/datasets"
export RESULTS_DIR="/scratch/$USER/cotformer/exps"

# --- Caches (keep off home quota) ---
export HF_HOME="/scratch/$USER/.cache/huggingface"
export WANDB_DIR="/scratch/$USER/.cache/wandb"
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"
export CONDA_PKGS_DIRS="/scratch/$USER/.conda/pkgs"
