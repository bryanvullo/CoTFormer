#!/bin/bash
# rq9_launch_pilot1.sh -- Arm A (Pilot 1) launcher.
#
# Iterates the 5 Chang and Bisk 2024 [25]-style seeds and submits the
# iridis/counting-sweep/job.sh once per seed, passing --arm A
# --variant baseline --n-embd 1024. This is the ONLY strict-reproduction
# control in RQ9 (see docs/extend-notes.md §1.2 RQ9 Pilot 1 row and
# DEC-031/DEC-033).
#
# Usage:
#   bash scripts/rq9_launch_pilot1.sh                 # submit all 5 seeds
#   bash scripts/rq9_launch_pilot1.sh --dry-run       # print sbatch invocations
#   bash scripts/rq9_launch_pilot1.sh --seed 1234     # submit one seed only
#
# The launcher does NOT itself call sbatch; it delegates to
# iridis/counting-sweep/job.sh which self-submits via its login-node
# wrapper. The output of --dry-run prints the bash commands that WOULD
# be invoked; each line is exactly what the launcher would shell out.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JOB="$REPO_DIR/iridis/counting-sweep/job.sh"

# Chang and Bisk 2024 [25]-style seed convention: config.py declares
# seed = [1234, 12] and we extend to 5 seeds for symmetry with the
# Arm B panel per DEC-031. The extension values are short integer
# literals in the spirit of the original pair.
SEEDS=(1234 12 42 7 21)
NEMBD=1024
DRY_RUN=0
ONLY_SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --seed) ONLY_SEED="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--seed N]"
            echo "  --dry-run     Print intended bash/sbatch invocations"
            echo "  --seed N      Submit only one cell for seed N"
            exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2 ; exit 2 ;;
    esac
done

if [ ! -f "$JOB" ]; then
    echo "ERROR: $JOB not found" >&2
    exit 2
fi

SUBMITTED=0
SKIPPED=0
for seed in "${SEEDS[@]}"; do
    if [ -n "$ONLY_SEED" ] && [ "$ONLY_SEED" != "$seed" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    CMD=(bash "$JOB" --arm A --variant baseline --seed "$seed" --n-embd "$NEMBD")
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] ${CMD[*]}"
    else
        echo "[submit] ${CMD[*]}"
        "${CMD[@]}"
    fi
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Arm A Pilot 1 launcher complete: submitted=$SUBMITTED (+$SKIPPED skipped)"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run -- no actual sbatch submissions were made."
fi
