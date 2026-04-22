#!/bin/bash
# rq9_launch_arm_b.sh -- Arm B (CoTFormer-native) launcher with secondary-width expansion.
#
# Iterates the 4 CoTFormer variants {V1, V2, V3, V4} across a WIDTH x SEED
# matrix that implements the DEC-033 secondary-width expansion (formerly
# "contingent"; committed unconditionally per DEC-035). The expansion adds
# more granularity on the ln_mid x n_embd interaction and populates
# additional 2x2 cells for the DEC-028 RQ9b factorial ANOVA; the
# initial-wave 12-cell roster at n_embd=1024 had exactly one empty 2x2
# cell under every candidate variant-to-factor mapping (see
# docs/extend-notes.md §1.9 DEC-035 for the full rationale).
#
# Width x seed matrix (reduced seed count at smaller widths per DEC-033):
#
#   n_embd = 1024 -> 3 seeds (primary width, matches initial-wave design)
#   n_embd =  512 -> 2 seeds
#   n_embd =  256 -> 2 seeds
#   n_embd =  128 -> 1 seed
#
# Total cells: 4 variants x (3 + 2 + 2 + 1) = 4 x 8 = 32 cells.
# Per-cell wall time at each width (1x L4, 312.5K steps at BS=32):
#   n_embd = 1024  -> ~12-18 h/cell (dominant cost; 12 cells at 1024 total)
#   n_embd =  512  -> ~6-9 h/cell   (8 cells at 512 total)
#   n_embd =  256  -> ~3-5 h/cell   (8 cells at 256 total)
#   n_embd =  128  -> ~2-3 h/cell   (4 cells at 128 total)
#
# Grand total: ~24-96 L4-hours of additional compute vs the initial-wave
# 12-cell budget. Full width matrix projects to ~9-14 L4-days on 2x L4
# parallel submission.
#
# Per-cell seed assignment from the DEC-031 project convention
# (train=2357, val=8191, ood=19937). The n_embd=1024 cells use all three;
# n_embd=512 and n_embd=256 use the first two (2357, 8191); n_embd=128
# uses only the primary training seed (2357).
#
# Usage:
#   bash scripts/rq9_launch_arm_b.sh                  # submit all 32 cells
#   bash scripts/rq9_launch_arm_b.sh --dry-run        # print sbatch invocations
#   bash scripts/rq9_launch_arm_b.sh --variant V1     # only V1 cells (8 cells)
#   bash scripts/rq9_launch_arm_b.sh --n-embd 1024    # only n_embd=1024 (12 cells)
#   bash scripts/rq9_launch_arm_b.sh --seed 2357      # only seed 2357 (16 cells)
#   bash scripts/rq9_launch_arm_b.sh --variant V3 --seed 8191 --n-embd 512  # single cell
#
# See rq9_launch_pilot1.sh for Arm A (Pilot 1). Arm C is NOT commissioned
# in this DIR and is handled by a follow-up launcher contingent on Arm B
# results.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JOB="$REPO_DIR/iridis/counting-sweep/job.sh"

VARIANTS=(V1 V2 V3 V4)

# Width x seed matrix per DEC-035. Keys are widths; values are space-
# separated seed lists taken from DEC-031's project convention
# (2357=train, 8191=val, 19937=ood). Smaller widths reduce seed count
# proportionally to their reduced scientific weight in the ANOVA (they
# carry fewer parameters and are expected to exhibit lower variance).
declare -A WIDTH_SEEDS=(
    [1024]="2357 8191 19937"
    [512]="2357 8191"
    [256]="2357 8191"
    [128]="2357"
)
WIDTH_ORDER=(1024 512 256 128)

DRY_RUN=0
ONLY_VARIANT=""
ONLY_SEED=""
ONLY_NEMBD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --variant) ONLY_VARIANT="$2"; shift 2 ;;
        --seed) ONLY_SEED="$2"; shift 2 ;;
        --n-embd) ONLY_NEMBD="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--variant V1|V2|V3|V4] [--n-embd 128|256|512|1024] [--seed N]"
            echo "  --dry-run            Print intended bash/sbatch invocations"
            echo "  --variant V          Submit only cells for one variant"
            echo "  --n-embd N           Submit only cells for one width"
            echo "  --seed N             Submit only cells for one seed"
            echo "  Combine filters to submit a single cell."
            exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2 ; exit 2 ;;
    esac
done

if [ ! -f "$JOB" ]; then
    echo "ERROR: $JOB not found" >&2
    exit 2
fi

if [ -n "$ONLY_VARIANT" ]; then
    valid=0
    for v in "${VARIANTS[@]}"; do
        if [ "$v" = "$ONLY_VARIANT" ]; then valid=1; fi
    done
    if [ "$valid" -eq 0 ]; then
        echo "ERROR: --variant='$ONLY_VARIANT' not one of ${VARIANTS[*]}" >&2
        exit 2
    fi
fi

if [ -n "$ONLY_NEMBD" ]; then
    valid=0
    for w in "${WIDTH_ORDER[@]}"; do
        if [ "$w" = "$ONLY_NEMBD" ]; then valid=1; fi
    done
    if [ "$valid" -eq 0 ]; then
        echo "ERROR: --n-embd='$ONLY_NEMBD' not one of ${WIDTH_ORDER[*]}" >&2
        exit 2
    fi
fi

SUBMITTED=0
SKIPPED=0
for variant in "${VARIANTS[@]}"; do
    for nembd in "${WIDTH_ORDER[@]}"; do
        if [ -n "$ONLY_NEMBD" ] && [ "$ONLY_NEMBD" != "$nembd" ]; then
            for seed in ${WIDTH_SEEDS[$nembd]}; do
                SKIPPED=$((SKIPPED + 1))
            done
            continue
        fi
        for seed in ${WIDTH_SEEDS[$nembd]}; do
            if [ -n "$ONLY_VARIANT" ] && [ "$ONLY_VARIANT" != "$variant" ]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            if [ -n "$ONLY_SEED" ] && [ "$ONLY_SEED" != "$seed" ]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            CMD=(bash "$JOB" --arm B --variant "$variant" --seed "$seed" --n-embd "$nembd")
            if [ "$DRY_RUN" -eq 1 ]; then
                echo "[dry-run] ${CMD[*]}"
            else
                echo "[submit] ${CMD[*]}"
                "${CMD[@]}"
            fi
            SUBMITTED=$((SUBMITTED + 1))
        done
    done
done

echo ""
echo "Arm B launcher complete: submitted=$SUBMITTED (+$SKIPPED skipped)"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run -- no actual sbatch submissions were made."
fi
