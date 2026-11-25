#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'HELP'
Usage: bash run.sh <command> [options]

  encode     Encode queries and pages with Qwen3-VL-Embedding-8B
  retrieve   Retrieve candidate pages from saved embeddings
  pointwise  Rerank with Qwen3-VL-Reranker-8B
  serve      Start the local Qwen3.8-27B-FP8 service
  listwise   Rerank candidates with the local FP8 service
  rerank     Dispatch to pointwise or listwise

Run a command with --help for its options. The default dataset is mmdocir.
HELP
}

command="${1:-help}"
if (( $# > 0 )); then
  shift
fi
case "$command" in
  encode)    exec bash code/retriever/run_encode.sh "$@" ;;
  retrieve)  exec bash code/retriever/run_retrieve.sh "$@" ;;
  pointwise) exec bash code/reranker/run_pointwise.sh "$@" ;;
  serve)     exec bash code/reranker/run_listwise.sh serve "$@" ;;
  listwise)  exec bash code/reranker/run_listwise.sh "$@" ;;
  rerank)    exec bash code/reranker/run_rerank.sh "$@" ;;
  help|-h|--help) usage ;;
  *)
    printf 'Unknown command: %s\n' "$command" >&2
    usage >&2
    exit 2
    ;;
esac
