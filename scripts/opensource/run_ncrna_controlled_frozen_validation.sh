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
LOG_ROOT="./outputs/ft/rna-all/NoncodingRNAFamily/logs/controlled-frozen-validation-${TIMESTAMP}"
COMMON_ARGS=("$@")
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<<"${GPU_IDS_CSV}"
SEEDS=(666 42 3407)
CHECKPOINT="/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k"
VARIANT="stage-d-100k-control"

mkdir -p "${LOG_ROOT}"

JOB_SPECS=()

add_job() {
    local seed="$1"
    local loops="$2"
    local pooling="$3"
    local lr="$4"
    JOB_SPECS+=("${seed}|${loops}|${pooling}|${lr}")
}

for seed in "${SEEDS[@]}"; do
    for loops in 1 2 3; do
        add_job "${seed}" "${loops}" "cls" "5e-5"
        add_job "${seed}" "${loops}" "cls_tanh" "1e-3"
        add_job "${seed}" "${loops}" "loop_mean_cls" "1e-3"
    done
done

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
echo "Launching EcoRNA controlled frozen validation"
echo "Checkpoint: ${CHECKPOINT}"
echo "Variant: ${VARIANT}"
echo "Total jobs: ${#JOB_SPECS[@]}"
echo "Logs: ${LOG_ROOT}"
echo "=========================================="

for idx in "${!JOB_SPECS[@]}"; do
    IFS='|' read -r seed loops pooling lr <<<"${JOB_SPECS[$idx]}"
    slot=$(( idx % ${#GPU_IDS[@]} ))
    gpu="${GPU_IDS[$slot]}"
    name="seed-${seed}__loop-${loops}__${pooling}__lr-${lr}"
    log_file="${LOG_ROOT}/${name}.log"
    result_file="./outputs/ft/rna-all/NoncodingRNAFamily/ecorna/${VARIANT}/${pooling}/loops-${loops}/frozen-head/lr-${lr}/${seed}/results/ecorna_ncrna_frozen_head_lr-${lr}/test_results.json"

    if [ -f "${result_file}" ]; then
        echo "[skip] ${name} (${result_file})"
        continue
    fi

    echo "[launch] gpu=${gpu} name=${name}"
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
echo "Controlled frozen validation completed."
echo "Logs: ${LOG_ROOT}"
echo "=========================================="
