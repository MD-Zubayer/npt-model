#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

STAGE="${1:-all}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B}"
EPOCHS="${EPOCHS:-1}"
DATASET_SIZE="${DATASET_SIZE:-2048}"
SEQ_LENGTH="${SEQ_LENGTH:-64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PORT="${PORT:-8000}"

run_pipeline_stage() {
  local s="$1"
  echo "[auto] pipeline stage: $s"
  python3 scripts/run_pipeline.py --stage "$s" --config pipeline_config.json --output-dir ./outputs
}

run_train() {
  echo "[auto] training with tokenizer=$TOKENIZER_NAME epochs=$EPOCHS dataset_size=$DATASET_SIZE"
  python3 training/pretraining/train.py \
    --num-epochs "$EPOCHS" \
    --dataset-size "$DATASET_SIZE" \
    --seq-length "$SEQ_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --tokenizer-name "$TOKENIZER_NAME" \
    --output-dir ./outputs
}

run_server() {
  echo "[auto] starting server on port $PORT with tokenizer=$TOKENIZER_NAME"
  TOKENIZER_NAME="$TOKENIZER_NAME" python3 inference/server.py
}

show_help() {
  cat <<HELP
Usage: bash scripts/auto_run.sh [stage]

Stages:
  collect      Run collection only
  clean        Run cleaning only
  filter       Run filtering only
  dedup        Run deduplication only
  tokenize     Run tokenization only (if enabled in config)
  train        Run training only
  serve        Start inference server only
  prep         Run collect->clean->filter->dedup
  all          Run prep->train (default)
  fullserve    Run prep->train->serve

Env overrides:
  TOKENIZER_NAME (default: Qwen/Qwen2.5-7B)
  EPOCHS (default: 1)
  DATASET_SIZE (default: 2048)
  SEQ_LENGTH (default: 64)
  BATCH_SIZE (default: 2)
  PORT (default: 8000)

Examples:
  bash scripts/auto_run.sh prep
  TOKENIZER_NAME=deepseek-ai/DeepSeek-V4-Pro EPOCHS=2 DATASET_SIZE=5000 bash scripts/auto_run.sh train
  bash scripts/auto_run.sh fullserve
HELP
}

case "$STAGE" in
  collect)
    run_pipeline_stage collection
    ;;
  clean)
    run_pipeline_stage cleaning
    ;;
  filter)
    run_pipeline_stage filtering
    ;;
  dedup)
    run_pipeline_stage deduplication
    ;;
  tokenize)
    run_pipeline_stage tokenization
    ;;
  train)
    run_train
    ;;
  serve)
    run_server
    ;;
  prep)
    run_pipeline_stage collection
    run_pipeline_stage cleaning
    run_pipeline_stage filtering
    run_pipeline_stage deduplication
    ;;
  all)
    run_pipeline_stage collection
    run_pipeline_stage cleaning
    run_pipeline_stage filtering
    run_pipeline_stage deduplication
    run_train
    ;;
  fullserve)
    run_pipeline_stage collection
    run_pipeline_stage cleaning
    run_pipeline_stage filtering
    run_pipeline_stage deduplication
    run_train
    run_server
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    echo "Unknown stage: $STAGE"
    show_help
    exit 1
    ;;
esac
