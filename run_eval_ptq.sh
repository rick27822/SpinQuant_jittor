#!/usr/bin/env bash
# Like official scripts/2_eval_ptq.sh:
#   bash run_eval_ptq.sh /path/to/model 4 4 4 \
#       --rotate --optimized_rotation_path /path/to/R.npy --w_quant_method gptq
# Omit --rotate and its checkpoint option for direct quantization.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

INPUT_MODEL="${MODEL_DIR:-$HERE/data/models/TinyLlama-1.1B}"
# Preserve floating-baseline behavior when no positional bit widths are given.
W_BITS=16
A_BITS=16
KV_BITS=16
if [[ $# -gt 0 && "$1" != -* ]]; then
    if [[ $# -lt 4 || "$2" == -* || "$3" == -* || "$4" == -* ]]; then
        printf 'Usage: bash run_eval_ptq.sh MODEL W_BITS A_BITS KV_BITS [--options ...]\n' >&2
        exit 2
    fi
    INPUT_MODEL="$1"
    W_BITS="$2"
    A_BITS="$3"
    KV_BITS="$4"
    shift 4
fi

if [ -n "${JITTOR_CUDA_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="$JITTOR_CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
PY="${JITTOR_PYTHON:-python3}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Script usage: bash run_eval_ptq.sh MODEL W_BITS A_BITS KV_BITS [--options ...]\n\n'
fi

# RTN is the existing Jittor default; select --w_quant_method gptq for GPTQ.
# The official R.bin checkpoint is replaced by the Jittor R.npy checkpoint.
exec "$PY" -u -m jittor_spinquant.pipeline.eval_ptq \
    --input_model "$INPUT_MODEL" \
    --model_max_length 2048 \
    --bsz 1 \
    --seed 0 \
    --w_bits "$W_BITS" \
    --a_bits "$A_BITS" \
    --k_bits "$KV_BITS" \
    --v_bits "$KV_BITS" \
    --w_quant_method rtn \
    --w_clip \
    --a_asym \
    --k_asym \
    --v_asym \
    --nsamples 128 \
    --percdamp 0.01 \
    --gptq_blocksize 128 \
    "$@"
