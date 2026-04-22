#!/bin/bash
#SBATCH --job-name=analyze_lncot_adm_gpu
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
################################################################################
# analyze-lncot+adm GPU stage -- Protocol A (Logit Lens) + Protocol G Type B
# + Protocol C (Attention Taxonomy).
#
# Wave 1 + 2a + 2c deliverable (MVP observational bundle).
# Runs analysis.logit_lens on every checkpoint in the VERSIONS array; the
# script emits logit_lens_results.json and logit_lens_trajectories.png into
# run_N/<tag>/, and persists the per-(layer, repeat) residual cache into
# /scratch/$USER/analysis_workspace/<tag>/ for Stage-4 consumers
# (analysis.tuned_lens, analysis.residual_diagnostics).
#
# Wave 2a additions (Protocol G Type B, per §1.5 Matrix row "G: KV Compress."
# which applies to C1, C3, C5): analysis.kv_rank runs in --capture-kv mode
# on those three tags to populate the kv_mid_l<L>_r<R>.npy cache and emit
# kv_rank_results.json + kv_rank_plots.png into run_N/<tag>/. Activation-
# level SVD is a ~25 min GPU step per G-enabled checkpoint per §1.3 Protocol
# G row.
#
# Wave 2c additions (Protocol C attention taxonomy + sparsity-across-repeat
# test per §1.3 Protocol C row + §1.4 Prediction 2): analysis.attention_taxonomy
# runs with ATTN_WEIGHTS site + non_flash=True on C1, C2, C3, C4, C5 (all
# five per the §1.3 extended matrix note), emitting attention_taxonomy_results.json
# plus heatmap + sparsity-trajectory PNGs. The non-flash math path + the
# monkey-patched forward's Q/K re-compute incur ~8x the flash-backend per-
# block cost; per-checkpoint wall time is ~9 min L4 (§1.8 compute budget).
#
# Stage-5 synthesis lives on the CPU side in job-cpu.sh; this script
# does not run it.
#
# Output structure:
#   run_N/
#     slurm_<jobid>.out                 SLURM stdout
#     slurm_<jobid>.err                 SLURM stderr
#     <tag>/
#       logit_lens_results.json         per-(layer, repeat) metrics + CV
#       logit_lens_trajectories.png     per-repeat PNG trajectories
#
# Workspace (intermediate, /scratch):
#   /scratch/$USER/analysis_workspace/<tag>/
#     residual_mid_l{L}_r{R}.npy        pre-ln_mid residuals
#     residual_ln_mid_r{R}.npy          post-ln_mid residuals
#     residual_pre_ln_f.npy             input to ln_f (canonical Belrose 2023 target residual)
#     targets.npy                       next-token targets for Tuned Lens
#     meta.json                         collector metadata
#
# Usage:
#   cd ~/CoTFormer && bash iridis/analyze-lncot+adm/job-gpu.sh
################################################################################

# ========================= CONFIGURATION ====================================

# Version matrix: "tag|ckpt_dir_absolute_path|ckpt_file"
# The VERSIONS list is populated per the §1.5 Analysis Matrix (Protocol A
# runs on C1-C5). Replace each entry's ckpt_dir leaf as your run history
# dictates; the leaf names in /scratch/ab3u21/exps/owt2/ follow the
# pattern exp_name=<model>_lr<lr>_bs<bs>x<acc>_seqlen<seq>__<overrides>_seed=<s>.
VERSIONS=(
    "c1_cotres_40k|$EXPS_DIR/owt2/cotformer_full_depth/cotformer_full_depth_res_only_lr0.001_bs8x16_seqlen256|ckpt_40000.pt"
    "c2_lncot_40k|$EXPS_DIR/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256|ckpt_40000.pt"
    "c3_lncot_60k|$EXPS_DIR/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256|ckpt_60000.pt"
    "c4_mod_40k|$EXPS_DIR/owt2/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_lr0.001_bs8x16_seqlen256|ckpt_40000.pt"
    "c5_adm_v2_60k|$EXPS_DIR/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256|ckpt_60000.pt"
)

# Inter-batch robustness batch count and per-batch token budget (§1.2 RQ1).
N_BATCHES=4
MAX_TOKENS=2048
SEQ_LENGTH=256
BATCH_SIZE=8
SEED=2357

# Protocol G tag whitelist per §1.5 Matrix row "G: KV Compress." -- runs on
# C1 (cotres_40k), C3 (lncot_60k), C5 (adm_v2_60k). Other tags fall through
# unchanged.
KV_RANK_TAGS_REGEX='^(c1_cotres_40k|c3_lncot_60k|c5_adm_v2_60k)$'

# Reference MLA kv_lora_rank for the Prediction-1 comparison plot line.
KV_TARGET_RANK=192

# Dataset root on /scratch; used by --capture-kv to bypass a possibly
# absent config.data_dir in the checkpoint's summary.json. Empty means
# "use config.data_dir".
DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/$USER/datasets}"

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    # Pre-flight: verify every checkpoint exists. Fail fast before queueing.
    MISSING=0
    for version_entry in "${VERSIONS[@]}"; do
        IFS='|' read -r v_tag v_dir v_file <<< "$version_entry"
        if [ ! -f "$v_dir/summary.json" ]; then
            echo "ERROR: $v_tag summary.json not found at $v_dir"
            MISSING=$((MISSING+1))
        fi
        if [ ! -f "$v_dir/$v_file" ]; then
            echo "ERROR: $v_tag $v_file not found at $v_dir"
            MISSING=$((MISSING+1))
        fi
    done
    if [ "$MISSING" -gt 0 ]; then
        echo ""
        echo "$MISSING checkpoint(s) missing. Update VERSIONS in $0."
        exit 1
    fi

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")

    # Create run_N/<tag>/ layout on login node
    for version_entry in "${VERSIONS[@]}"; do
        IFS='|' read -r v_tag _ _ <<< "$version_entry"
        mkdir -p "$RUN_DIR/$v_tag"
    done

    echo "=== analyze-lncot+adm GPU ==="
    echo "  Versions:    $(IFS=,; echo "${VERSIONS[*]}" | sed 's/|[^,]*//g')"
    echo "  N batches:   $N_BATCHES"
    echo "  Max tokens:  $MAX_TOKENS per batch"
    echo "  Logs:        $RUN_DIR/"
    echo ""
    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",RUN_DIR="$RUN_DIR" \
        "$0" "$@"
fi

# --- Actual job (runs on compute node) ---
set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
fi

source "$REPO_DIR/iridis/env.sh"

echo "========================================="
echo " analyze-lncot+adm GPU stage"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " GPU:           1x L4"
echo " Job ID:        $SLURM_JOB_ID"
echo " Started:       $(date)"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# ========================= VERSION LOOP =====================================
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r TAG CKPT_DIR CKPT_FILE <<< "$version_entry"

    TAG_RUN_DIR="$RUN_DIR/$TAG"
    mkdir -p "$TAG_RUN_DIR"

    WORKSPACE="/scratch/$USER/analysis_workspace/$TAG"
    mkdir -p "$WORKSPACE"

    echo ""
    echo "###################################################################"
    echo "   $TAG  ($CKPT_DIR/$CKPT_FILE)"
    echo "###################################################################"

    echo ""
    echo "--- Protocol A: Logit Lens (stage 1 collector + stage 3 metrics) ---"
    python -m analysis.logit_lens \
        --checkpoint "$CKPT_DIR" \
        --checkpoint-file "$CKPT_FILE" \
        --workspace "$WORKSPACE" \
        --output-dir "$TAG_RUN_DIR" \
        --seed "$SEED" \
        --max-tokens "$MAX_TOKENS" \
        --n-batches "$N_BATCHES" \
        --seq-length "$SEQ_LENGTH" \
        --batch-size "$BATCH_SIZE" \
        --device cuda

    echo "  Logit Lens outputs:"
    echo "    $TAG_RUN_DIR/logit_lens_results.json"
    echo "    $TAG_RUN_DIR/logit_lens_trajectories.png"
    echo "  Workspace:"
    echo "    $WORKSPACE/"

    # --- Protocol G Type B: activation-level KV rank + KV-CoRE NER ---
    # Runs only on the §1.5 Matrix G row (C1, C3, C5). --capture-kv
    # triggers the KV_C_ATTN collector pass when the workspace is missing
    # the kv_mid_l<L>_r<R>.npy files (it skips the capture when the files
    # are already present, which is the normal re-run case).
    if [[ "$TAG" =~ $KV_RANK_TAGS_REGEX ]]; then
        echo ""
        echo "--- Protocol G Type B: KV activation rank + NER ---"
        if [ -n "$DATA_DIR_OVERRIDE" ] && [ -d "$DATA_DIR_OVERRIDE" ]; then
            KV_DATA_DIR_ARG=(--data-dir "$DATA_DIR_OVERRIDE")
        else
            KV_DATA_DIR_ARG=()
        fi
        python -m analysis.kv_rank \
            --checkpoint "$CKPT_DIR" \
            --checkpoint-file "$CKPT_FILE" \
            --workspace "$WORKSPACE" \
            --output-dir "$TAG_RUN_DIR" \
            --seed "$SEED" \
            --target-rank "$KV_TARGET_RANK" \
            --capture-kv \
            --device cuda \
            --seq-length "$SEQ_LENGTH" \
            --batch-size "$BATCH_SIZE" \
            --max-tokens "$MAX_TOKENS" \
            "${KV_DATA_DIR_ARG[@]}"

        echo "  Protocol G outputs:"
        echo "    $TAG_RUN_DIR/kv_rank_results.json"
        echo "    $TAG_RUN_DIR/kv_rank_plots.png"
    else
        echo ""
        echo "  Skipping Protocol G for $TAG (not in §1.5 Matrix G row)"
    fi

    # --- Protocol C: attention taxonomy + sparsity-across-repeat test ---
    # §1.3 Protocol C row: runs on C1, C2, C3, C4, C5 (all five). The
    # collector flips attn.flash=False and installs a per-attn monkey-
    # patch that re-runs the Q/K math path to recover att post-softmax;
    # see analysis.common.collector._install_attn_monkey_patch and
    # docs/extend-technical.md §8.6 for the rationale and failure modes.
    # Compute: ~4x non-flash slowdown + ~2x Q/K re-compute cost; per
    # checkpoint ~9 min L4 at max_tokens=2048, seq_length=256.
    echo ""
    echo "--- Protocol C: Attention taxonomy (non-flash ATTN_WEIGHTS) ---"
    if [ -n "$DATA_DIR_OVERRIDE" ] && [ -d "$DATA_DIR_OVERRIDE" ]; then
        PROTC_DATA_DIR_ARG=(--data-dir "$DATA_DIR_OVERRIDE")
    else
        PROTC_DATA_DIR_ARG=()
    fi
    python -m analysis.attention_taxonomy \
        --checkpoint "$CKPT_DIR" \
        --checkpoint-file "$CKPT_FILE" \
        --workspace "$WORKSPACE" \
        --output-dir "$TAG_RUN_DIR" \
        --seed "$SEED" \
        --max-tokens "$MAX_TOKENS" \
        --seq-length "$SEQ_LENGTH" \
        --batch-size "$BATCH_SIZE" \
        --device cuda \
        "${PROTC_DATA_DIR_ARG[@]}"

    echo "  Protocol C outputs:"
    echo "    $TAG_RUN_DIR/attention_taxonomy_results.json"
    echo "    $TAG_RUN_DIR/attention_taxonomy_heatmap.png"
    echo "    $TAG_RUN_DIR/attention_sparsity_trajectory.png"

done  # end VERSION LOOP

# ========================= SUMMARY ==========================================
echo ""
echo "========================================="
echo " analyze-lncot+adm GPU stage complete: $(date)"
echo " Results tree: $RUN_DIR/"
echo ""
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag _ _ <<< "$version_entry"
    echo "   $v_tag/"
    echo "     logit_lens_results.json"
    echo "     logit_lens_trajectories.png"
    echo "     attention_taxonomy_results.json"
    echo "     attention_taxonomy_heatmap.png"
    echo "     attention_sparsity_trajectory.png"
done
echo ""
echo " Workspace (intermediate, NOT in run_N):"
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag _ _ <<< "$version_entry"
    echo "   /scratch/$USER/analysis_workspace/$v_tag/"
    if [[ "$v_tag" =~ $KV_RANK_TAGS_REGEX ]]; then
        echo "     (+ kv_mid_l<L>_r<R>.npy from Protocol G capture)"
    fi
    echo "     (+ attn_weights_mid_l<L>_r<R>.npy from Protocol C capture)"
done
echo ""
echo " Next step: submit job-cpu.sh for Tuned Lens + Protocol B/E/F/G-weight + Protocol D"
echo "========================================="
