#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Example: bash scripts/infer.sh --checkpoint outputs/tts/best.pt --text '你好' --num-frames 100 --output generated_codes.pt
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$ROOT/infer.py" "$@"
