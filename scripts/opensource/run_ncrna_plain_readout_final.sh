#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ -f /root/chenpengan/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/chenpengan/miniconda3/etc/profile.d/conda.sh
    conda activate eco-rna
fi

WINNER_JSON="${WINNER_JSON:-}"
WINNER_POOLING="${WINNER_POOLING:-}"
WINNER_LR="${WINNER_LR:-}"

if [ -n "${WINNER_JSON}" ]; then
    read -r WINNER_POOLING WINNER_LR < <(
        python - "${WINNER_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    payload = json.load(f)
winner = payload.get("winner") or {}
print(winner.get("strategy", ""), winner.get("lr", ""))
PY
    )
fi

if [ -z "${WINNER_POOLING}" ] || [ -z "${WINNER_LR}" ]; then
    echo "Set WINNER_JSON or both WINNER_POOLING and WINNER_LR before launching final runs."
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="./outputs/ft/rna-all/NoncodingRNAFamily/logs/plain-readout-final-${WINNER_POOLING}-lr-${WINNER_LR}-${TIMESTAMP}"
COMMON_ARGS=("$@")
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<<"${GPU_IDS_CSV}"
SEEDS=(666 42 3407)
LOOP1_CKPT="/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-loop1-stage-d"
RECUR_CKPT="/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k"
LOOP1_VARIANT="loop1-stage-d-plain-readout-final"
RECUR_VARIANT="stage-d-100k-plain-readout-final"

mkdir -p "${LOG_ROOT}"

JOB_SPECS=()

add_job() {
    local seed="$1"
    local checkpoint_group="$2"
    local checkpoint="$3"
    local variant="$4"
    local use_ckpt_loops="$5"
    local loops="${6:-}"
    JOB_SPECS+=("${seed}|${checkpoint_group}|${checkpoint}|${variant}|${use_ckpt_loops}|${loops}")
}

for seed in "${SEEDS[@]}"; do
    add_job "${seed}" "loop1-stage-d" "${LOOP1_CKPT}" "${LOOP1_VARIANT}" "1"
    for loops in 1 2 3; do
        add_job "${seed}" "stage-d-100k__loop-${loops}" "${RECUR_CKPT}" "${RECUR_VARIANT}" "0" "${loops}"
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
echo "Launching EcoRNA plain readout final"
echo "Winner pooling: ${WINNER_POOLING}"
echo "Winner lr: ${WINNER_LR}"
echo "Total jobs: ${#JOB_SPECS[@]}"
echo "Logs: ${LOG_ROOT}"
echo "=========================================="

for idx in "${!JOB_SPECS[@]}"; do
    IFS='|' read -r seed checkpoint_group checkpoint variant use_ckpt_loops loops <<<"${JOB_SPECS[$idx]}"
    slot=$(( idx % ${#GPU_IDS[@]} ))
    gpu="${GPU_IDS[$slot]}"
    name="seed-${seed}__${checkpoint_group}__${WINNER_POOLING}__lr-${WINNER_LR}"
    log_file="${LOG_ROOT}/${name}.log"

    if [ "${use_ckpt_loops}" = "1" ]; then
        loop_label="loops-ckpt"
    else
        loop_label="loops-${loops}"
    fi

    result_file="./outputs/ft/rna-all/NoncodingRNAFamily/ecorna/${variant}/${WINNER_POOLING}/${loop_label}/frozen-head/lr-${WINNER_LR}/${seed}/results/ecorna_ncrna_frozen_head_lr-${WINNER_LR}/test_results.json"
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
        export LR="${WINNER_LR}"
        export ECORNA_CHECKPOINT="${checkpoint}"
        export MODEL_VARIANT="${variant}"
        export RUN_TAG="lr-${WINNER_LR}"
        export ECORNA_POOLING_STRATEGY="${WINNER_POOLING}"
        if [ "${use_ckpt_loops}" = "1" ]; then
            export ECORNA_USE_CHECKPOINT_DEFAULT_LOOPS=1
            unset ECORNA_NUM_LOOPS
        else
            export ECORNA_USE_CHECKPOINT_DEFAULT_LOOPS=0
            export ECORNA_NUM_LOOPS="${loops}"
        fi
        bash scripts/opensource/run_ncrna.sh ecorna \
            --metric_for_best_model f1 \
            --greater_is_better True \
            "${COMMON_ARGS[@]}"
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
echo "EcoRNA plain readout final completed."
echo "Logs: ${LOG_ROOT}"
echo "=========================================="
