#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT_DIR/benchmarks/config.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[error] config not found: $CONFIG_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Print runtime Python info to avoid conda/venv confusion.
echo "[env] python=$(command -v python)"
python -V || true

MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
BACKEND="${BACKEND:-auto}"
ENABLE_FALLBACK="${ENABLE_FALLBACK:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TIMEOUT_S="${TIMEOUT_S:-120}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.90}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
PROMPTS_FILE="${PROMPTS_FILE:-benchmarks/prompts.txt}"
PRELOAD_MODEL="${PRELOAD_MODEL:-0}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT_DIR/$OUTPUT_DIR/$RUN_TS"
mkdir -p "$RUN_DIR"

SERVER_PID=""
ACTUAL_BACKEND=""

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[cleanup] stopping server pid=$SERVER_PID"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

has_cuda() {
  command -v nvidia-smi >/dev/null 2>&1
}

detect_backend() {
  if [[ "$BACKEND" != "auto" ]]; then
    echo "$BACKEND"
    return
  fi

  case "$(uname -s)" in
    Linux)
      if has_cuda; then
        echo "vllm"
      else
        echo "transformers"
      fi
      ;;
    Darwin)
      # Apple Silicon path defaults to transformers for compatibility.
      echo "transformers"
      ;;
    *)
      echo "transformers"
      ;;
  esac
}

wait_for_health() {
  local health_url="$1"
  local max_wait="${2:-120}"
  local pid="${3:-}"
  local waited=0

  while (( waited < max_wait )); do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "[server] process exited before healthy (pid=$pid)"
      return 2
    fi
    if curl -sSf "$health_url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

start_server() {
  local backend="$1"
  local log_file="$RUN_DIR/server_${backend}.log"

  echo "[server] starting backend=$backend"
  if [[ "$backend" == "transformers" ]]; then
    if ! python -c 'import torch; print(torch.__version__)' >/dev/null 2>&1; then
      echo "[error] local torch runtime is broken (python cannot import torch)."
      echo "[error] try a clean env and reinstall pytorch for your platform."
      return 1
    fi
  fi

  if [[ "$backend" == "vllm" ]]; then
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_ID" \
      --host "$HOST" \
      --port "$PORT" \
      >"$log_file" 2>&1 &
  elif [[ "$backend" == "transformers" ]]; then
    python -m benchmarks.transformers_server \
      --model-id "$MODEL_ID" \
      --host "$HOST" \
      --port "$PORT" \
      >"$log_file" 2>&1 &
  else
    echo "[error] unknown backend: $backend"
    return 1
  fi

  SERVER_PID=$!
  echo "[server] pid=$SERVER_PID log=$log_file"

  if wait_for_health "http://$HOST:$PORT/health" "$TIMEOUT_S" "$SERVER_PID"; then
    echo "[server] healthy"
    return 0
  fi

  local rc=$?
  if [[ $rc -eq 2 ]]; then
    echo "[server] health check failed because server process exited early"
  else
    echo "[server] health check timed out after ${TIMEOUT_S}s for backend=$backend"
  fi
  echo "[server] last log lines:"
  tail -n 60 "$log_file" || true
  return 1
}

if [[ "$PRELOAD_MODEL" == "1" ]]; then
  "$ROOT_DIR/scripts/download_model.sh" "$CONFIG_FILE"
fi

ACTUAL_BACKEND="$(detect_backend)"
if ! start_server "$ACTUAL_BACKEND"; then
  if [[ "$ACTUAL_BACKEND" == "vllm" && "$ENABLE_FALLBACK" == "1" ]]; then
    echo "[fallback] vllm failed, switching to transformers"
    cleanup
    ACTUAL_BACKEND="transformers"
    start_server "$ACTUAL_BACKEND"
  else
    echo "[error] failed to start backend=$ACTUAL_BACKEND"
    exit 1
  fi
fi

python "$ROOT_DIR/benchmarks/evaluate.py" \
  --model-id "$MODEL_ID" \
  --backend "$ACTUAL_BACKEND" \
  --host "$HOST" \
  --port "$PORT" \
  --prompts-file "$ROOT_DIR/$PROMPTS_FILE" \
  --output-dir "$RUN_DIR" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --timeout-s "$TIMEOUT_S" \
  --warmup-requests "$WARMUP_REQUESTS" \
  --min-success-rate "$MIN_SUCCESS_RATE"

echo "[done] report: $RUN_DIR/report.txt"
