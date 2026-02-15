#!/bin/bash
# Script to run NoncodingRNAFamily task with RNA-FM or EcoRNA
# Usage:
#   bash scripts/opensource/run_ncrna.sh rna-fm
#   bash scripts/opensource/run_ncrna.sh ecorna

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

gpu_device="${GPU_DEVICE:-0}"
nproc_per_node="${NPROC_PER_NODE:-1}"
master_port=$(shuf -i 10000-45000 -n 1)
echo "Using port $master_port for communication."

data_root=./data
model_root=./checkpoint
MODEL_TYPE="${1:-rna-fm}"
if [ "$#" -gt 0 ]; then
    shift
fi
EXTRA_ARGS=("$@")
model_extra_args=()

case "$MODEL_TYPE" in
    "rna-fm")
        token='single'
        model_max_length=1024
        MODEL_PATH="${RNAFM_CHECKPOINT:-${model_root}/opensource/rna-fm}"
        ;;
    "ecorna")
        token='single'
        model_max_length=1024
        MODEL_PATH="${ECORNA_CHECKPOINT:-../../output/ecorna-RNA-stage-d-100k}"
        precision_args=(--bf16)
        ECORNA_POOLING_STRATEGY="${ECORNA_POOLING_STRATEGY:-cls_tanh}"
        ECORNA_NUM_LOOPS="${ECORNA_NUM_LOOPS:--1}"
        model_extra_args=(
            --ecorna_pooling_strategy "${ECORNA_POOLING_STRATEGY}"
            --ecorna_num_loops "${ECORNA_NUM_LOOPS}"
        )
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        echo "Supported: rna-fm, ecorna"
        exit 1
        ;;
esac

seed="${SEED:-666}"
task='NoncodingRNAFamily'
batch_size="${BATCH_SIZE:-16}"
lr="${LR:-5e-5}"
num_train_epochs="${NUM_TRAIN_EPOCHS:-30}"

if [ -d "${data_root}/${task}" ]; then
    DATA_PATH="${data_root}/${task}"
elif [ -d "${data_root}/downstream/${task}" ]; then
    DATA_PATH="${data_root}/downstream/${task}"
else
    echo "Data directory for ${task} not found."
    echo "Checked: ${data_root}/${task} and ${data_root}/downstream/${task}"
    exit 1
fi

if [ -f "${DATA_PATH}/train_new.csv" ]; then
    data_file_train="train_new.csv"
elif [ -f "${DATA_PATH}/train.csv" ]; then
    data_file_train="train.csv"
else
    echo "Neither train_new.csv nor train.csv found in ${DATA_PATH}"
    exit 1
fi

if [ -f "${DATA_PATH}/test.csv" ]; then
    data_file_test="test.csv"
elif [ -f "${DATA_PATH}/test" ]; then
    data_file_test="test"
else
    echo "Neither test.csv nor test found in ${DATA_PATH}"
    exit 1
fi

data_file_val="val.csv"
if [ ! -f "${DATA_PATH}/${data_file_val}" ]; then
    echo "val.csv not found in ${DATA_PATH}"
    exit 1
fi

OUTPUT_PATH="./outputs/ft/rna-all/${task}/${MODEL_TYPE}"
if [ "${MODEL_TYPE}" = "ecorna" ]; then
    OUTPUT_PATH="${OUTPUT_PATH}/${ECORNA_POOLING_STRATEGY}/loops-${ECORNA_NUM_LOOPS}"
fi
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=${gpu_device} torchrun --nproc_per_node=${nproc_per_node} --master_port=${master_port}"
precision_args=("${precision_args[@]:---fp16}")

echo "=========================================="
echo "Task: ${task}"
echo "Model: ${MODEL_TYPE}"
echo "Model Path: ${MODEL_PATH}"
if [ "${MODEL_TYPE}" = "ecorna" ]; then
    echo "EcoRNA pooling: ${ECORNA_POOLING_STRATEGY} | infer loops: ${ECORNA_NUM_LOOPS}"
fi
echo "Data Path: ${DATA_PATH}"
echo "GPUs: ${gpu_device} | nproc_per_node=${nproc_per_node}"
echo "=========================================="

${EXEC_PREFIX} \
downstream/train_ncrna.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --data_train_path "${data_file_train}" \
    --data_val_path "${data_file_val}" \
    --data_test_path "${data_file_test}" \
    --run_name "${MODEL_TYPE}_ncrna" \
    --model_max_length "${model_max_length}" \
    --per_device_train_batch_size "${batch_size}" \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate "${lr}" \
    --num_train_epochs "${num_train_epochs}" \
    "${precision_args[@]}" \
    --save_steps 400 \
    --output_dir "${OUTPUT_PATH}/${seed}" \
    --eval_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --report_to none \
    --seed "${seed}" \
    --token_type "${token}" \
    --model_type "${MODEL_TYPE}" \
    "${model_extra_args[@]:-}" \
    "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Completed: ${task} with ${MODEL_TYPE}"
echo "=========================================="
