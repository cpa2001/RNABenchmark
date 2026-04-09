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
master_port=$(
    python - <<'PY'
import socket

sock = socket.socket()
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
PY
)
echo "Using port $master_port for communication."

freeze_backbone="${FREEZE_BACKBONE:-0}"
model_variant="${MODEL_VARIANT:-}"
run_tag="${RUN_TAG:-}"
run_tag_safe=""
run_tag_path=""
run_tag_name=""
if [ -n "${run_tag}" ]; then
    run_tag_safe="$(printf '%s' "${run_tag}" | tr '/ ' '__')"
    run_tag_path="/${run_tag_safe}"
    run_tag_name="_${run_tag_safe}"
fi
case "${freeze_backbone,,}" in
    1|true|yes)
        freeze_backbone_cli="True"
        mode_label="frozen-head"
        run_suffix="frozen_head"
        ;;
    0|false|no)
        freeze_backbone_cli="False"
        mode_label="full-ft"
        run_suffix="full_ft"
        ;;
    *)
        echo "Invalid FREEZE_BACKBONE=${freeze_backbone}. Use 0/1 or false/true."
        exit 1
        ;;
esac

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
        ECORNA_POOLING_STRATEGY="${ECORNA_POOLING_STRATEGY:-weighted_layer_content}"
        ECORNA_POOLING_CELLS="${ECORNA_POOLING_CELLS:-}"
        ECORNA_NUM_LOOPS="${ECORNA_NUM_LOOPS:-}"
        ECORNA_USE_CHECKPOINT_DEFAULT_LOOPS="${ECORNA_USE_CHECKPOINT_DEFAULT_LOOPS:-0}"
        model_extra_args=(--ecorna_pooling_strategy "${ECORNA_POOLING_STRATEGY}")
        if [ -n "${ECORNA_POOLING_CELLS}" ]; then
            model_extra_args+=(--ecorna_pooling_cells "${ECORNA_POOLING_CELLS}")
        fi
        use_checkpoint_default_loops=0
        case "${ECORNA_USE_CHECKPOINT_DEFAULT_LOOPS,,}" in
            1|true|yes)
                use_checkpoint_default_loops=1
                ;;
        esac
        if [ "${use_checkpoint_default_loops}" -eq 0 ] && [ -n "${ECORNA_NUM_LOOPS}" ] && [ "${ECORNA_NUM_LOOPS}" != "-1" ]; then
            model_extra_args+=(--ecorna_num_loops "${ECORNA_NUM_LOOPS}")
            loop_output_label="loops-${ECORNA_NUM_LOOPS}"
        else
            loop_output_label="loops-ckpt"
        fi
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
    if [ -n "${model_variant}" ]; then
        OUTPUT_PATH="${OUTPUT_PATH}/${model_variant}/${ECORNA_POOLING_STRATEGY}/${loop_output_label}"
    else
        if [ "${loop_output_label}" = "loops-ckpt" ]; then
            OUTPUT_PATH="${OUTPUT_PATH}/${ECORNA_POOLING_STRATEGY}/loops--1"
        else
            OUTPUT_PATH="${OUTPUT_PATH}/${ECORNA_POOLING_STRATEGY}/${loop_output_label}"
        fi
    fi
fi
OUTPUT_PATH="${OUTPUT_PATH}/${mode_label}${run_tag_path}"
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=${gpu_device} torchrun --nproc_per_node=${nproc_per_node} --master_port=${master_port}"
precision_args=("${precision_args[@]:---fp16}")
run_name="${MODEL_TYPE}_ncrna_${run_suffix}${run_tag_name}"

echo "=========================================="
echo "Task: ${task}"
echo "Model: ${MODEL_TYPE}"
echo "Model Path: ${MODEL_PATH}"
if [ -n "${model_variant}" ]; then
    echo "Variant: ${model_variant}"
fi
echo "Mode: ${mode_label} (FREEZE_BACKBONE=${freeze_backbone_cli})"
if [ "${MODEL_TYPE}" = "ecorna" ]; then
    echo "EcoRNA pooling: ${ECORNA_POOLING_STRATEGY} | infer loops: ${loop_output_label}"
    if [ -n "${ECORNA_POOLING_CELLS:-}" ]; then
        echo "EcoRNA pooling cells: ${ECORNA_POOLING_CELLS}"
    fi
fi
if [ -n "${run_tag_safe}" ]; then
    echo "Run tag: ${run_tag_safe}"
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
    --run_name "${run_name}" \
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
    --freeze_backbone "${freeze_backbone_cli}" \
    "${model_extra_args[@]}" \
    "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Completed: ${task} with ${MODEL_TYPE}"
echo "=========================================="
