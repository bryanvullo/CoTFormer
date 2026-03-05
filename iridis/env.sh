#!/usr/bin/env bash
# iridis/env.sh — Iridis X environment config
# Source this file; do not execute it directly.
#
# Usage (from job.sh scripts):
#   source "$REPO_DIR/iridis/env.sh"

# --- Conda environment ---
CONDA_ENV_PREFIX="/scratch/$USER/cotformer-env"

# --- Per-user scratch paths ---
export DATA_DIR="/scratch/$USER/datasets"
export RESULTS_DIR="/scratch/$USER/cotformer-outputs"

# --- Caches (keep off home quota) ---
export TIKTOKEN_CACHE_DIR="/scratch/$USER/.cache/tiktoken"
export HF_HOME="/scratch/$USER/.cache/huggingface"
export WANDB_DIR="/scratch/$USER/.cache/wandb"
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"
export CONDA_PKGS_DIRS="/scratch/$USER/.conda/pkgs"

# --- Run directory helper ---
# Creates the next run_N directory inside a package dir.
# Usage: RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
next_run_dir() {
    local pkg_dir="$1"
    local last_n
    last_n=$(ls -1d "$pkg_dir"/run_* 2>/dev/null | sed 's/.*\/run_//' | sort -n | tail -1)
    local next_n=$(( ${last_n:--1} + 1 ))
    local run_dir="$pkg_dir/run_$next_n"
    mkdir -p "$run_dir"
    echo "$run_dir"
}
