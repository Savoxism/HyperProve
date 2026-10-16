#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing .venv. Create it first with: uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt" >&2
  exit 1
fi
# hundreds of worker threads each hold sockets and log handles; the default 1024 is too low
ulimit -n 65536
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_hyperprove}"
LIMIT_QUESTIONS="${LIMIT_QUESTIONS:-1000}"

COMMON_ARGS=(
  --output-root "${OUTPUT_ROOT}"
  --limit-questions "${LIMIT_QUESTIONS}"
  --max-workers "${MAX_WORKERS:-128}"
  --llm-max-workers "${LLM_MAX_WORKERS:-256}"
  --synonym-threshold "${SYNONYM_THRESHOLD:-0.8}"
)

if [[ "${REBUILD:-0}" == "1" ]]; then
  COMMON_ARGS+=(--rebuild)
fi

"${PYTHON_BIN}" main.py \
  --dataset-name musique \
  --dataset-path datasets/musique_questions.json \
  --corpus-path datasets/musique_corpus.json \
  "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" main.py \
  --dataset-name hotpotqa \
  --dataset-path datasets/hotpotqa_questions.json \
  --corpus-path datasets/hotpotqa_corpus.json \
  "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" main.py \
  --dataset-name 2wikimultihopqa \
  --dataset-path datasets/2wikimultihopqa_questions.json \
  --corpus-path datasets/2wikimultihopqa_corpus.json \
  "${COMMON_ARGS[@]}"
