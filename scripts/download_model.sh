#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT_DIR/benchmarks/config.env}"

if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model}"
LOCAL_MODEL_DIR="$MODEL_DIR/$MODEL_ID"

echo "[download] model: $MODEL_ID"
echo "[download] target dir: $LOCAL_MODEL_DIR"
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[download] huggingface-cli not found. Install with: pip install huggingface_hub"
  exit 1
fi

mkdir -p "$LOCAL_MODEL_DIR"
huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_MODEL_DIR"
echo "[download] done"
