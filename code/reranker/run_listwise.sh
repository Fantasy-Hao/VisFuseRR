#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" != serve ]]; then
  exec python -m reranker.listwise "$@"
fi
shift

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' 'Start the local FP8 vLLM service.' \
    'QWEN38_MODEL_DIR overrides the weights directory. QWEN38_PORT defaults to 8000.'
  exit 0
fi
if (( $# > 0 )); then
  printf 'Usage: bash run.sh serve\n' >&2
  exit 2
fi

model_dir="${QWEN38_MODEL_DIR:-model/Qwen3.8-27B-FP8}"
if [[ -z "${QWEN38_MODEL_DIR:-}" && ! -d "$model_dir" && -d Qwen3.8-27B-FP8 ]]; then
  model_dir=Qwen3.8-27B-FP8
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec vllm serve "$model_dir" \
  --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 \
  --port "${QWEN38_PORT:-8000}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":10,"video":0}' \
  --generation-config vllm \
  --reasoning-parser qwen3
