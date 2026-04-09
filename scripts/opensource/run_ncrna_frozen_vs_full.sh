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
RNAFM_CHECKPOINT="${RNAFM_CHECKPOINT:-/root/chenpengan/CUHK/eco/eco-rna-2/benchmarks/RNABenchmark/checkpoint/opensource/rna-fm}"
ECORNA_CHECKPOINT="${ECORNA_CHECKPOINT:-/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k}"
ECORNA_POOLING_STRATEGY="${ECORNA_POOLING_STRATEGY:-cls}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="./outputs/ft/rna-all/NoncodingRNAFamily/logs/frozen-vs-full-seed-${SEED}-${TIMESTAMP}"
COMMON_ARGS=("$@")

mkdir -p "${LOG_ROOT}"

PIDS=()
NAMES=()
LOGS=()

cleanup() {
    local status=$?
    if [ "${status}" -ne 0 ]; then
        echo "[cleanup] Aborting remaining ncRNA jobs..."
        for pid in "${PIDS[@]:-}"; do
            kill "${pid}" 2>/dev/null || true
        done
    fi
}

trap cleanup EXIT INT TERM

launch_job() {
    local gpu="$1"
    local name="$2"
    local model_type="$3"
    local freeze_backbone="$4"
    local loops="${5:-}"
    local log_file="${LOG_ROOT}/${name}.log"

    echo "[launch] gpu=${gpu} name=${name} log=${log_file}"

    (
        export GPU_DEVICE="${gpu}"
        export NPROC_PER_NODE=1
        export SEED="${SEED}"
        export FREEZE_BACKBONE="${freeze_backbone}"
        export RNAFM_CHECKPOINT="${RNAFM_CHECKPOINT}"
        export ECORNA_CHECKPOINT="${ECORNA_CHECKPOINT}"
        if [ "${model_type}" = "ecorna" ]; then
            export ECORNA_POOLING_STRATEGY="${ECORNA_POOLING_STRATEGY}"
            export ECORNA_NUM_LOOPS="${loops}"
        fi
        bash scripts/opensource/run_ncrna.sh "${model_type}" "${COMMON_ARGS[@]}"
    ) >"${log_file}" 2>&1 &

    PIDS+=("$!")
    NAMES+=("${name}")
    LOGS+=("${log_file}")
}

echo "=========================================="
echo "Launching NoncodingRNAFamily frozen-vs-full matrix"
echo "Seed: ${SEED}"
echo "RNA-FM checkpoint: ${RNAFM_CHECKPOINT}"
echo "EcoRNA checkpoint: ${ECORNA_CHECKPOINT}"
echo "EcoRNA pooling: ${ECORNA_POOLING_STRATEGY}"
echo "Logs: ${LOG_ROOT}"
echo "=========================================="

launch_job 0 "rnafm_frozen" "rna-fm" "1"
launch_job 1 "ecorna_loop1_frozen" "ecorna" "1" "1"
launch_job 2 "ecorna_loop2_frozen" "ecorna" "1" "2"
launch_job 3 "ecorna_loop3_frozen" "ecorna" "1" "3"
launch_job 4 "rnafm_full" "rna-fm" "0"
launch_job 5 "ecorna_loop1_full" "ecorna" "0" "1"
launch_job 6 "ecorna_loop2_full" "ecorna" "0" "2"
launch_job 7 "ecorna_loop3_full" "ecorna" "0" "3"

for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[done] ${NAMES[$i]}"
    else
        echo "[fail] ${NAMES[$i]} (log: ${LOGS[$i]})"
        for j in "${!PIDS[@]}"; do
            if [ "${j}" -ne "${i}" ]; then
                kill "${PIDS[$j]}" 2>/dev/null || true
            fi
        done
        wait || true
        exit 1
    fi
done

trap - EXIT INT TERM

echo "=========================================="
echo "All ncRNA jobs completed."
echo "Logs: ${LOG_ROOT}"
echo "=========================================="
