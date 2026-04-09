#!/bin/bash

set -euo pipefail

MODE="${1:-}"
if [ -z "${MODE}" ]; then
    echo "Usage: $0 <phase0|pilot|final> [extra run_ncrna.sh args...]"
    exit 1
fi
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

activate_conda() {
    if [ -f /root/chenpengan/miniconda3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /root/chenpengan/miniconda3/etc/profile.d/conda.sh
        conda activate eco-rna
    fi
}

read_json_winner_field() {
    local json_path="$1"
    local field="$2"
    python - "${json_path}" "${field}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], "r"))
winner = payload.get("winner") or {}
value = winner.get(sys.argv[2], "")
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

resolve_strategy_list() {
    if [ -n "${STRATEGIES_CSV:-}" ]; then
        IFS=',' read -r -a STRATEGIES <<<"${STRATEGIES_CSV}"
        return
    fi
    if [ -n "${PHASE0_JSON:-}" ]; then
        mapfile -t STRATEGIES < <(python - "${PHASE0_JSON}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], "r"))
winner = payload.get("winner") or {}
winner_strategy = winner.get("strategy")
if winner_strategy:
    print(winner_strategy)
else:
    for strategy in payload.get("passing_strategies", []):
        print(strategy)
PY
)
        return
    fi
    STRATEGIES=("weighted_layer_content" "weighted_cell_content")
}

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

add_job() {
    local seed="$1"
    local checkpoint_group="$2"
    local checkpoint="$3"
    local variant="$4"
    local strategy="$5"
    local lr="$6"
    local use_ckpt_loops="$7"
    local loops="${8:-}"
    JOB_SPECS+=("${seed}|${checkpoint_group}|${checkpoint}|${variant}|${strategy}|${lr}|${use_ckpt_loops}|${loops}")
}

launch_job() {
    local seed="$1"
    local checkpoint_group="$2"
    local checkpoint="$3"
    local variant="$4"
    local strategy="$5"
    local lr="$6"
    local use_ckpt_loops="$7"
    local loops="${8:-}"
    local gpu="$9"
    local name="${10}"
    local log_file="${11}"

    (
        export GPU_DEVICE="${gpu}"
        export NPROC_PER_NODE=1
        export SEED="${seed}"
        export FREEZE_BACKBONE=1
        export LR="${lr}"
        export ECORNA_CHECKPOINT="${checkpoint}"
        export MODEL_VARIANT="${variant}"
        export RUN_TAG="lr-${lr}"
        export ECORNA_POOLING_STRATEGY="${strategy}"
        unset ECORNA_POOLING_CELLS
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
}

activate_conda

COMMON_ARGS=("$@")
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<<"${GPU_IDS_CSV}"
LOOP1_CKPT="/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-loop1-stage-d"
RECUR_CKPT="/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

declare -a JOB_SPECS=()
declare -a PIDS=()
declare -a NAMES=()
declare -a LOGS=()

case "${MODE}" in
    phase0)
        SEEDS=("666")
        LOG_ROOT="./outputs/ft/rna-all/NoncodingRNAFamily/logs/weighted-content-phase0-seed-666-${TIMESTAMP}"
        STRATEGIES=("weighted_layer_content" "weighted_cell_content")
        LRS=("2e-4" "1e-3")
        VARIANT="stage-d-100k-weighted-content-phase0"
        for strategy in "${STRATEGIES[@]}"; do
            for lr in "${LRS[@]}"; do
                add_job "666" "stage-d-100k__loop-3" "${RECUR_CKPT}" "${VARIANT}" "${strategy}" "${lr}" "0" "3"
            done
        done
        ;;
    pilot)
        SEED="${SEED:-666}"
        PHASE0_JSON="${PHASE0_JSON:-}"
        STRATEGIES_CSV="${STRATEGIES_CSV:-}"
        resolve_strategy_list
        if [ "${#STRATEGIES[@]}" -eq 0 ]; then
            echo "No weighted-content strategies available for pilot."
            exit 1
        fi
        LOG_ROOT="./outputs/ft/rna-all/NoncodingRNAFamily/logs/weighted-content-pilot-seed-${SEED}-${TIMESTAMP}"
        LRS=("5e-5" "2e-4" "1e-3")
        LOOP1_VARIANT="loop1-stage-d-weighted-content-pilot"
        RECUR_VARIANT="stage-d-100k-weighted-content-pilot"
        for strategy in "${STRATEGIES[@]}"; do
            for lr in "${LRS[@]}"; do
                add_job "${SEED}" "loop1-stage-d" "${LOOP1_CKPT}" "${LOOP1_VARIANT}" "${strategy}" "${lr}" "1"
            done
        done
        for loops in 1 2 3; do
            for strategy in "${STRATEGIES[@]}"; do
                for lr in "${LRS[@]}"; do
                    add_job "${SEED}" "stage-d-100k__loop-${loops}" "${RECUR_CKPT}" "${RECUR_VARIANT}" "${strategy}" "${lr}" "0" "${loops}"
                done
            done
        done
        ;;
    final)
        WINNER_JSON="${WINNER_JSON:-}"
        if [ -z "${WINNER_JSON}" ]; then
            echo "WINNER_JSON is required for final mode."
            exit 1
        fi
        PASS_PILOT="$(read_json_winner_field "${WINNER_JSON}" passes_gate)"
        if [ "${PASS_PILOT}" != "true" ]; then
            echo "Weighted-content pilot did not pass gate. Refusing to launch final."
            exit 1
        fi
        STRATEGY="$(read_json_winner_field "${WINNER_JSON}" strategy)"
        LR="$(read_json_winner_field "${WINNER_JSON}" lr)"
        SEEDS_CSV="${SEEDS_CSV:-666,42,3407}"
        IFS=',' read -r -a SEEDS <<<"${SEEDS_CSV}"
        LOG_ROOT="./outputs/ft/rna-all/NoncodingRNAFamily/logs/weighted-content-final-${STRATEGY}-lr-${LR}-${TIMESTAMP}"
        LOOP1_VARIANT="loop1-stage-d-weighted-content-final"
        RECUR_VARIANT="stage-d-100k-weighted-content-final"
        for seed in "${SEEDS[@]}"; do
            add_job "${seed}" "loop1-stage-d" "${LOOP1_CKPT}" "${LOOP1_VARIANT}" "${STRATEGY}" "${LR}" "1"
        done
        for seed in "${SEEDS[@]}"; do
            for loops in 1 2 3; do
                add_job "${seed}" "stage-d-100k__loop-${loops}" "${RECUR_CKPT}" "${RECUR_VARIANT}" "${STRATEGY}" "${LR}" "0" "${loops}"
            done
        done
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Supported: phase0, pilot, final"
        exit 1
        ;;
esac

mkdir -p "${LOG_ROOT}"
trap 'cleanup_batch' EXIT INT TERM

echo "=========================================="
echo "Launching EcoRNA weighted-content ${MODE}"
if [ "${MODE}" = "final" ]; then
    echo "Strategy: ${STRATEGY}"
    echo "LR: ${LR}"
    echo "Seeds: ${SEEDS_CSV}"
elif [ "${MODE}" = "pilot" ]; then
    echo "Seed: ${SEED}"
    echo "Phase0 JSON: ${PHASE0_JSON:-<none>}"
    echo "Strategies: ${STRATEGIES[*]}"
else
    echo "Seed: 666"
    echo "Strategies: ${STRATEGIES[*]}"
fi
echo "Total jobs: ${#JOB_SPECS[@]}"
echo "Logs: ${LOG_ROOT}"
echo "=========================================="

for idx in "${!JOB_SPECS[@]}"; do
    IFS='|' read -r seed checkpoint_group checkpoint variant strategy lr use_ckpt_loops loops <<<"${JOB_SPECS[$idx]}"
    slot=$(( idx % ${#GPU_IDS[@]} ))
    gpu="${GPU_IDS[$slot]}"
    name="seed-${seed}__${checkpoint_group}__${strategy}__lr-${lr}"
    [ "${MODE}" = "phase0" ] && name="${checkpoint_group}__${strategy}__lr-${lr}"
    log_file="${LOG_ROOT}/${name}.log"

    if [ "${use_ckpt_loops}" = "1" ]; then
        loop_label="loops-ckpt"
    else
        loop_label="loops-${loops}"
    fi

    result_file="./outputs/ft/rna-all/NoncodingRNAFamily/ecorna/${variant}/${strategy}/${loop_label}/frozen-head/lr-${lr}/${seed}/results/ecorna_ncrna_frozen_head_lr-${lr}/test_results.json"
    if [ -f "${result_file}" ]; then
        echo "[skip] ${name} (${result_file})"
        continue
    fi

    echo "[launch] gpu=${gpu} name=${name}"
    launch_job "${seed}" "${checkpoint_group}" "${checkpoint}" "${variant}" "${strategy}" "${lr}" "${use_ckpt_loops}" "${loops}" "${gpu}" "${name}" "${log_file}"

    if [ "${#PIDS[@]}" -eq "${#GPU_IDS[@]}" ]; then
        wait_batch
    fi
done

if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_batch
fi

trap - EXIT INT TERM
echo "=========================================="
echo "EcoRNA weighted-content ${MODE} completed."
echo "Logs: ${LOG_ROOT}"
echo "=========================================="
