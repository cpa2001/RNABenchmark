#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ -f /root/chenpengan/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/chenpengan/miniconda3/etc/profile.d/conda.sh
    conda activate eco-rna
fi

SEED="${SEED:-666}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="./outputs/ft/rna-all/NoncodingRNAFamily/logs/frozen-readout-sweep-seed-${SEED}-${TIMESTAMP}"
COMMON_ARGS=("$@")
GPU_IDS=(0 1 2 3 4 5 6 7)
LRS=("5e-5" "2e-4" "1e-3")
LOOP1_CKPT="/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-loop1-stage-d"
RECUR_CKPT="/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k"

mkdir -p "${LOG_ROOT}"

JOB_SPECS=()

add_job() {
    local name="$1"
    local checkpoint="$2"
    local variant="$3"
    local pooling="$4"
    local lr="$5"
    local use_ckpt_loops="$6"
    local loops="${7:-}"
    JOB_SPECS+=("${name}|${checkpoint}|${variant}|${pooling}|${lr}|${use_ckpt_loops}|${loops}")
}

for lr in "${LRS[@]}"; do
    for pooling in mean cls_tanh cls_mean_concat; do
        add_job "loop1-stage-d__${pooling}__lr-${lr}" "${LOOP1_CKPT}" "loop1-stage-d" "${pooling}" "${lr}" "1"
    done
done
add_job "loop1-stage-d__cls__diag__lr-5e-5" "${LOOP1_CKPT}" "loop1-stage-d" "cls" "5e-5" "1"

for loops in 1 2 3; do
    for lr in "${LRS[@]}"; do
        for pooling in mean cls_tanh cls_mean_concat loop_mean_cls; do
            add_job "stage-d-100k__loop-${loops}__${pooling}__lr-${lr}" "${RECUR_CKPT}" "stage-d-100k" "${pooling}" "${lr}" "0" "${loops}"
        done
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
echo "Launching EcoRNA frozen-readout sweep"
echo "Seed: ${SEED}"
echo "Total jobs: ${#JOB_SPECS[@]}"
echo "Logs: ${LOG_ROOT}"
echo "=========================================="

for idx in "${!JOB_SPECS[@]}"; do
    IFS='|' read -r name checkpoint variant pooling lr use_ckpt_loops loops <<<"${JOB_SPECS[$idx]}"
    slot=$(( idx % ${#GPU_IDS[@]} ))
    gpu="${GPU_IDS[$slot]}"
    log_file="${LOG_ROOT}/${name}.log"

    echo "[launch] gpu=${gpu} name=${name}"
    (
        export GPU_DEVICE="${gpu}"
        export NPROC_PER_NODE=1
        export SEED="${SEED}"
        export FREEZE_BACKBONE=1
        export LR="${lr}"
        export ECORNA_CHECKPOINT="${checkpoint}"
        export MODEL_VARIANT="${variant}"
        export RUN_TAG="lr-${lr}"
        export ECORNA_POOLING_STRATEGY="${pooling}"
        if [ "${use_ckpt_loops}" = "1" ]; then
            export ECORNA_USE_CHECKPOINT_DEFAULT_LOOPS=1
            unset ECORNA_NUM_LOOPS
        else
            export ECORNA_USE_CHECKPOINT_DEFAULT_LOOPS=0
            export ECORNA_NUM_LOOPS="${loops}"
        fi
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
echo "EcoRNA frozen-readout sweep completed."
echo "Logs: ${LOG_ROOT}"
echo "=========================================="
