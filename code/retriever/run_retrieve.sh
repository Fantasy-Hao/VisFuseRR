#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
exec python -m retriever.search "$@"
