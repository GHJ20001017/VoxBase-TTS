#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Cosine inference can optionally decode the generated .pt tokens to WAV with Mimi.
PYTHON_BIN="${PYTHON_BIN:-python3}"
MIMI_MODEL="${MIMI_MODEL:-}"
WAV_OUTPUT=""
INFER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mimi-model)
      [[ $# -ge 2 ]] || { echo "--mimi-model requires a path" >&2; exit 2; }
      MIMI_MODEL="$2"
      shift 2
      ;;
    --mimi-model=*)
      MIMI_MODEL="${1#*=}"
      shift
      ;;
    --wav-output)
      [[ $# -ge 2 ]] || { echo "--wav-output requires a path" >&2; exit 2; }
      WAV_OUTPUT="$2"
      shift 2
      ;;
    --wav-output=*)
      WAV_OUTPUT="${1#*=}"
      shift
      ;;
    *)
      INFER_ARGS+=("$1")
      shift
      ;;
  esac
done

"$PYTHON_BIN" "$ROOT/infer.py" "${INFER_ARGS[@]}"

if [[ -n "$MIMI_MODEL" ]]; then
  CODE_FILE=""
  for ((index = 0; index < ${#INFER_ARGS[@]}; index++)); do
    if [[ "${INFER_ARGS[index]}" == "--output" && $((index + 1)) -lt ${#INFER_ARGS[@]} ]]; then
      CODE_FILE="${INFER_ARGS[index + 1]}"
      break
    elif [[ "${INFER_ARGS[index]}" == --output=* ]]; then
      CODE_FILE="${INFER_ARGS[index]#*=}"
      break
    fi
  done
  [[ -n "$CODE_FILE" ]] || { echo "--output is required with --mimi-model" >&2; exit 2; }
  if [[ -z "$WAV_OUTPUT" ]]; then
    WAV_OUTPUT="${CODE_FILE%.pt}.wav"
  fi
  "$PYTHON_BIN" "$ROOT/scripts/decode_codes.py" \
    --codes-file "$CODE_FILE" \
    --output-wav "$WAV_OUTPUT" \
    --mimi-model "$MIMI_MODEL"
fi
