#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/benchmarks/config.env"
CLI_MODEL_ID=""
CLI_MODEL_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_model.sh  #default is deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  bash scripts/download_model.sh --model-id <model> [--model-dir <dir>]
  bash scripts/download_model.sh --config benchmarks/config.env --model-id <model> [--model-dir <dir>]

Options:
  --config <path>      Optional config file path (default: benchmarks/config.env)
  --model-id <id>      Override MODEL_ID from config
  --model-dir <path>   Override MODEL_DIR from config
  -h, --help           Show this help
EOF
}

POSITIONAL_CONFIG_SET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "[error] --config requires a value"
        usage
        exit 1
      fi
      CONFIG_FILE="$2"
      shift 2
      ;;
    --model-id)
      if [[ $# -lt 2 ]]; then
        echo "[error] --model-id requires a value"
        usage
        exit 1
      fi
      CLI_MODEL_ID="$2"
      shift 2
      ;;
    --model-dir)
      if [[ $# -lt 2 ]]; then
        echo "[error] --model-dir requires a value"
        usage
        exit 1
      fi
      CLI_MODEL_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$POSITIONAL_CONFIG_SET" -eq 0 ]] && [[ "$1" != -* ]]; then
        CONFIG_FILE="$1"
        POSITIONAL_CONFIG_SET=1
        shift
      else
        echo "[error] unknown argument: $1"
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model}"

if [[ -n "$CLI_MODEL_ID" ]]; then
  MODEL_ID="$CLI_MODEL_ID"
fi

if [[ -n "$CLI_MODEL_DIR" ]]; then
  MODEL_DIR="$CLI_MODEL_DIR"
fi

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
