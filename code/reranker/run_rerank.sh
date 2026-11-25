#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$0")"
case "${1:-}" in
  pointwise|listwise)
    method="$1"
    shift
    exec bash "$script_dir/run_$method.sh" "$@"
    ;;
  -h|--help)
    printf '%s\n' 'Usage: bash run.sh rerank <pointwise|listwise> [options]'
    ;;
  "")
    exec bash "$script_dir/run_pointwise.sh" "$@"
    ;;
  *)
    printf 'Unknown rerank method: %s\n' "$1" >&2
    exit 2
    ;;
esac
