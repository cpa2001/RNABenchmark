#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ -f /root/chenpengan/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/chenpengan/miniconda3/etc/profile.d/conda.sh
    conda activate eco-rna
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="./outputs/ft/rna-all/NoncodingRNAFamily/logs/controlled-frozen-resume-${TIMESTAMP}"
COMMON_ARGS=("$@")
GPU_IDS=(0 1 2 3 4 5 6)
CHECKPOINT="/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k"
VARIANT="stage-d-100k-control"

mkdir -p "${LOG_ROOT}"

JOB_SPECS=(
  "666|3|cls_tanh|1e-3"
  "666|3|loop_mean_cls|1e-3"
  "42|1|cls|5e-5"
  "42|1|cls_tanh|1e-3"
  "42|1|loop_mean_cls|1e-3"
  "42|2|cls|5e-5"
  "42|2|cls_tanh|1e-3"
  "42|2|loop_mean_cls|1e-3"
  "42|3|cls|5e-5"
  "42|3|cls_tanh|1e-3"
  "42|3|loop_mean_cls|1e-3"
  "3407|1|cls|5e-5"
  "3407|1|cls_tanh|1e-3"
  "3407|1|loop_mean_cls|1e-3"
  "3407|2|cls|5e-5"
  "3407|2|cls_tanh|1e-3"
  "3407|2|loop_mean_cls|1e-3"
  "3407|3|cls|5e-5"
  "3407|3|cls_tanh|1e-3"
  "3407|3|loop_mean_cls|1e-3"
)

PIDS=()
NAMES=()
LOGS=()

cleanup_batch() {
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}

wait_batch() {
    local idx
    for idx in "${!PIDS[@]}"; do
        if wait "${PIDS[$idx]}"; then
            echo "[done] ${NAMES[$idx]}"
        else
            echo "[fail] ${NAMES[$idx]} (log: ${LOGS[$idx]})"
            cleanup_batch
            wait || true
            exit 1
        fi
    done
    PIDS=()
    NAMES=()
    LOGS=()
}

trap 'cleanup_batch' EXIT INT TERM

echo "=========================================="
echo "Resuming EcoRNA controlled frozen validation"
echo "Checkpoint: ${CHECKPOINT}"
echo "Variant: ${VARIANT}"
echo "Candidate jobs: ${#JOB_SPECS[@]}"
echo "Logs: ${LOG_ROOT}"
echo "=========================================="

slot=0
for spec in "${JOB_SPECS[@]}"; do
    IFS='|' read -r seed loops pooling lr <<<"${spec}"
    result_file="./outputs/ft/rna-all/NoncodingRNAFamily/ecorna/${VARIANT}/${pooling}/loops-${loops}/frozen-head/lr-${lr}/${seed}/results/ecorna_ncrna_frozen_head_lr-${lr}/test_results.json"
    if [ -f "${result_file}" ]; then
        echo "[skip] seed=${seed} loops=${loops} pooling=${pooling} lr=${lr}"
        continue
    fi

    gpu="${GPU_IDS[$slot]}"
    slot=$(( (slot + 1) % ${#GPU_IDS[@]} ))
    name="seed-${seed}__loop-${loops}__${pooling}__lr-${lr}"
    log_file="${LOG_ROOT}/${name}.log"

    echo "[launch] gpu=${gpu} ${name}"
    (
        export GPU_DEVICE="${gpu}"
        export NPROC_PER_NODE=1
        export SEED="${seed}"
        export FREEZE_BACKBONE=1
        export LR="${lr}"
        export ECORNA_CHECKPOINT="${CHECKPOINT}"
        export MODEL_VARIANT="${VARIANT}"
        export RUN_TAG="lr-${lr}"
        export ECORNA_POOLING_STRATEGY="${pooling}"
        export ECORNA_USE_CHECKPOINT_DEFAULT_LOOPS=0
        export ECORNA_NUM_LOOPS="${loops}"
        bash scripts/opensource/run_ncrna.sh ecorna "${COMMON_ARGS[@]}"
    ) >"${log_file}" 2>&1 &

    PIDS+=("$!")
    NAMES+=("${name}")
    LOGS+=("${log_file}")

    if [ "${#PIDS[@]}" -eq "${#GPU_IDS[@]}" ]; then
        wait_batch
    fi
done

if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_batch
fi

trap - EXIT INT TERM

echo "=========================================="
echo "Controlled frozen validation resume completed."
echo "Logs: ${LOG_ROOT}"
echo "=========================================="
