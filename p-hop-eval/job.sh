#!/bin/bash
#SBATCH --job-name=phop_eval
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ========================= CONFIGURATION ====================================

CHECKPOINT="${CHECKPOINT:-/scratch/ab3u21/exps/p-hop/phop_p32_seq256_a4_final/fixed_cot_attn/phop_phop_p32_seq256_a4_final_fixed_cot_attn_0-1x8-0_8eff_d128_h8_bs32x4_seed0}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-}"
TASK="${TASK:-phop_p32_seq256_a4_final}"
EVAL_SPLITS="${EVAL_SPLITS:-val_constructive test_constructive}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-}"
DIAGNOSTIC_MAX_EXAMPLES="${DIAGNOSTIC_MAX_EXAMPLES:-}"
PHOP_DATA_ROOT="${PHOP_DATA_ROOT:-/scratch/ab3u21/datasets/p-hop}"
OUTPUT_JSON="${OUTPUT_JSON:-/scratch/ab3u21/exps/p-hop/evals/phop_eval_summary.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/scratch/ab3u21/exps/p32-hop/evals/phop_eval_examples.jsonl}"
N_GPUS="${N_GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"

# ========================= END CONFIGURATION ================================

if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== p-hop evaluation ==="
    echo "  Partition:   ecsstudents_l4"
    echo "  GPUs:        $N_GPUS"
    echo "  Task:        $TASK"
    echo "  Checkpoint:  $CHECKPOINT"
    echo "  Ckpt file:   ${CHECKPOINT_FILENAME:-auto}"
    echo "  Eval splits: $EVAL_SPLITS"
    echo "  Data root:   $PHOP_DATA_ROOT"
    echo "  Output JSON: $OUTPUT_JSON"
    echo "  Logs:        $RUN_DIR/"
    echo ""

    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --gres="gpu:${N_GPUS}" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",RUN_DIR="$RUN_DIR" \
        "$0" "$@"
fi

set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
fi

source "$REPO_DIR/iridis/env.sh"
if [ -z "$RUN_DIR" ]; then
    RUN_DIR=$(job_output_dir)
fi

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_JSON")" "$(dirname "$OUTPUT_JSONL")" "$PHOP_DATA_ROOT" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"
exec > >(tee -a "$RUN_DIR/output.log") 2> >(tee -a "$RUN_DIR/error.log" >&2)

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

PHOP_DATA_DIR="$PHOP_DATA_ROOT/$TASK"

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

export WANDB_MODE=offline
export PYTHONPATH="$REPO_DIR:$REPO_DIR/p-hop-induction:$REPO_DIR/tak-shifted-start:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "========================================="
echo " p-hop Evaluation"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " CPUs:          $SLURM_CPUS_PER_TASK"
echo " GPUs:          $N_GPUS"
echo " Job ID:        $SLURM_JOB_ID"
echo " Task:          $TASK"
echo " Eval splits:   $EVAL_SPLITS"
echo " Checkpoint:    $CHECKPOINT"
echo " Ckpt file:     ${CHECKPOINT_FILENAME:-auto}"
echo " Data root:     $PHOP_DATA_ROOT"
echo " Data dir:      $PHOP_DATA_DIR"
echo " Output JSON:   $OUTPUT_JSON"
echo " Output JSONL:  $OUTPUT_JSONL"
echo " Run dir:       $RUN_DIR"
echo " Started:       $(date)"
echo "========================================="

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

if [ ! -f "$REPO_DIR/p-hop-eval/eval.py" ]; then
    die "Missing $REPO_DIR/p-hop-eval/eval.py"
fi

for split in $EVAL_SPLITS; do
    if [ ! -f "$PHOP_DATA_DIR/$split.txt" ]; then
        die "Missing split: $PHOP_DATA_DIR/$split.txt"
    fi
done

EVAL_ARGS=(
    --checkpoint "$CHECKPOINT"
    --phop_task "$TASK"
    --phop_data_root "$PHOP_DATA_ROOT"
    --phop_eval_splits $EVAL_SPLITS
    --diagnostics
    --output_json "$OUTPUT_JSON"
    --output_jsonl "$OUTPUT_JSONL"
)

if [ -n "$CHECKPOINT_FILENAME" ]; then
    EVAL_ARGS+=(--checkpoint_filename "$CHECKPOINT_FILENAME")
fi

if [ -n "$EVAL_BATCH_SIZE" ]; then
    EVAL_ARGS+=(--phop_eval_batch_size "$EVAL_BATCH_SIZE")
fi

if [ -n "$EVAL_MAX_BATCHES" ]; then
    EVAL_ARGS+=(--phop_eval_max_batches "$EVAL_MAX_BATCHES")
fi

if [ -n "$DIAGNOSTIC_MAX_EXAMPLES" ]; then
    EVAL_ARGS+=(--diagnostic_max_examples "$DIAGNOSTIC_MAX_EXAMPLES")
fi

EVAL_ARGS+=("$@")

echo "Evaluation command:"
printf '  %q' python p-hop-eval/eval.py "${EVAL_ARGS[@]}"
echo ""
echo ""

python p-hop-eval/eval.py "${EVAL_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "========================================="
echo " Evaluation finished: $(date)"
echo " Exit code: $EXIT_CODE"
echo " Summary JSON:      $OUTPUT_JSON"
echo " Per-example JSONL: $OUTPUT_JSONL"
echo " Logs:              $RUN_DIR"
echo "========================================="

exit $EXIT_CODE

