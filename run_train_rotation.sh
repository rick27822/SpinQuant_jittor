#!/usr/bin/env bash
# Like official scripts/10_optimize_rotation.sh:
#   bash run_train_rotation.sh /path/to/model 16 4 4
# Optional named arguments at the end override the settings below.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

# Paths: edit these defaults, set environment variables, or pass CLI arguments.
INPUT_MODEL="${MODEL_DIR:-$HERE/data/models/TinyLlama-1.1B}"
OUTPUT_ROTATION="${OUTPUT_ROTATION_DIR:-$HERE/outputs/rotation}"
W_BITS=16
A_BITS=4
KV_BITS=4
if [[ $# -gt 0 && "$1" != -* ]]; then
    if [[ $# -lt 4 || "$2" == -* || "$3" == -* || "$4" == -* ]]; then
        printf 'Usage: bash run_train_rotation.sh MODEL W_BITS A_BITS KV_BITS [--options ...]\n' >&2
        exit 2
    fi
    INPUT_MODEL="$1"
    W_BITS="$2"
    A_BITS="$3"
    KV_BITS="$4"
    shift 4
fi

# Single-card Jittor uses python rather than the official multi-GPU torchrun.
if [ -n "${JITTOR_CUDA_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="$JITTOR_CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
PY="${JITTOR_PYTHON:-python3}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Script usage: bash run_train_rotation.sh MODEL W_BITS A_BITS KV_BITS [--options ...]\n\n'
fi

# Official-style parameter list. K/V groups are resolved from model head_dim
# (TinyLlama: 64, Mistral: 128); use --fp16 for FP16 base weights.
exec "$PY" -u -m jittor_spinquant.train.train_rotation \
    --input_model "$INPUT_MODEL" \
    --output_rotation_path "$OUTPUT_ROTATION" \
    --model_max_length 2048 \
    --bsz 1 \
    --logging_steps 1 \
    --learning_rate 1.5 \
    --lr_scheduler_type cosine \
    --rotate_iters 100 \
    --seed 0 \
    --w_bits "$W_BITS" \
    --a_bits "$A_BITS" \
    --k_bits "$KV_BITS" \
    --v_bits "$KV_BITS" \
    --w_clip \
    --a_asym \
    --k_asym \
    --v_asym \
    "$@"
