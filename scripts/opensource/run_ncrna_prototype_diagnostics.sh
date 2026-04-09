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
DATA_DIR="./data/NoncodingRNAFamily"
LOG_ROOT="./outputs/diagnostics/NoncodingRNAFamily/prototype/logs-${TIMESTAMP}"
OUT_ROOT="./outputs/diagnostics/NoncodingRNAFamily/prototype"
GPU_IDS=(0 1 2 3 4 5 6 7)

mkdir -p "${LOG_ROOT}"

JOBS=(
  "loop1-stage-d|/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-loop1-stage-d|cls|-1"
  "loop1-stage-d|/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-loop1-stage-d|mean|-1"
  "stage-d-100k|/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k|cls|1"
  "stage-d-100k|/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k|mean|1"
  "stage-d-100k|/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k|cls|2"
  "stage-d-100k|/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k|mean|2"
  "stage-d-100k|/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k|cls|3"
  "stage-d-100k|/root/chenpengan/CUHK/eco/eco-rna-2/output/ecorna-RNA-stage-d-100k|mean|3"
)

PIDS=()
NAMES=()
LOGS=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}

trap cleanup EXIT INT TERM

for idx in "${!JOBS[@]}"; do
    IFS='|' read -r variant checkpoint pooling loops <<<"${JOBS[$idx]}"
    gpu="${GPU_IDS[$idx]}"
    loop_label="loops-${loops}"
    if [ "${loops}" = "-1" ]; then
        loop_label="loops-ckpt"
    fi
    name="${variant}__${pooling}__${loop_label}"
    log_file="${LOG_ROOT}/${name}.log"
    output_json="${OUT_ROOT}/${variant}/${pooling}/${loop_label}/results.json"

    echo "[launch] gpu=${gpu} name=${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/opensource/eval_ncrna_prototype.py \
        --checkpoint_path "${checkpoint}" \
        --data_dir "${DATA_DIR}" \
        --pooling "${pooling}" \
        --num_loops "${loops}" \
        --output_json "${output_json}" >"${log_file}" 2>&1 &

    PIDS+=("$!")
    NAMES+=("${name}")
    LOGS+=("${log_file}")
done

for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[$idx]}"; then
        echo "[done] ${NAMES[$idx]}"
    else
        echo "[fail] ${NAMES[$idx]} (log: ${LOGS[$idx]})"
        cleanup
        wait || true
        exit 1
    fi
done

trap - EXIT INT TERM

echo "=========================================="
echo "Prototype diagnostics completed."
echo "Logs: ${LOG_ROOT}"
echo "=========================================="
