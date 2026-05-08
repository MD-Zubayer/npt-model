#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-./outputs/best_model_weights_only.pt}"
TOKENIZER_NAME="${TOKENIZER_NAME:-deepseek-ai/DeepSeek-V4-Pro}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[run_infer] Model file not found: $MODEL_PATH"
  echo "[run_infer] Put your checkpoint in outputs/ or set MODEL_PATH=..."
  exit 1
fi

echo "[run_infer] ROOT_DIR=$ROOT_DIR"
echo "[run_infer] MODEL_PATH=$MODEL_PATH"
echo "[run_infer] TOKENIZER_NAME=$TOKENIZER_NAME"
echo "[run_infer] HOST=$HOST PORT=$PORT"

MODEL_PATH="$MODEL_PATH" TOKENIZER_NAME="$TOKENIZER_NAME" uvicorn inference.server:app --host "$HOST" --port "$PORT" --log-level info
