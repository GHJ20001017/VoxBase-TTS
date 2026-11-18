#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# First positional argument is an optional CONFIG path; otherwise use the default.
# Example: PYTHON_BIN=python3 bash scripts/train.sh configs/train.yaml --epochs 2
CONFIG="configs/train.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$ROOT/train.py" --config "$CONFIG" "$@"
