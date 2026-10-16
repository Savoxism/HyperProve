#!/usr/bin/env bash
# Serve the HyperProve backends with vLLM (OpenAI-compatible):
#   LLM        LLM_MODEL (gpt-oss-20b)     on LLM_GPUS (one data-parallel replica per GPU), port LLM_PORT
#   embedding  EMB_MODEL (BAAI/bge-m3)    on EMB_GPU (shares it with an LLM replica),    port EMB_PORT
# Usage: bash scripts/serve_models.sh {start|start_emb|start_llm|stop [llm|emb]|status}
set -euo pipefail
cd "$(dirname "$0")/.."

LLM_GPUS="${LLM_GPUS:-0,1,2,3,4,5,6,7}"
EMB_GPU="${EMB_GPU:-7}"
LLM_MEM="${LLM_MEM:-0.75}"
EMB_MEM="${EMB_MEM:-0.15}"
EMB_MODEL="${EMB_MODEL:-BAAI/bge-m3}"
LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-20b}"
LLM_PORT="${LLM_PORT:-8000}"
EMB_PORT="${EMB_PORT:-8001}"
VLLM=".venv/bin/vllm"
# vLLM JIT-compiles kernels at startup and needs the venv's ninja on PATH
export PATH="$PWD/.venv/bin:$PATH"
mkdir -p logs/serve

start() {
  start_emb
  start_llm
}

start_emb() {
  CUDA_VISIBLE_DEVICES="$EMB_GPU" nohup "$VLLM" serve "$EMB_MODEL" \
    --served-model-name "$EMB_MODEL" --runner pooling --convert embed \
    --gpu-memory-utilization "$EMB_MEM" \
    --host 127.0.0.1 --port "$EMB_PORT" > logs/serve/emb.log 2>&1 &
  echo $! > logs/serve/emb.pid
  wait_ready emb "$EMB_PORT"
}

start_llm() {
  local dp; dp=$(awk -F, '{print NF}' <<<"$LLM_GPUS")
  CUDA_VISIBLE_DEVICES="$LLM_GPUS" nohup "$VLLM" serve "$LLM_MODEL" \
    --served-model-name "$LLM_MODEL" --data-parallel-size "$dp" \
    --gpu-memory-utilization "$LLM_MEM" --max-num-seqs 256 \
    --host 127.0.0.1 --port "$LLM_PORT" > logs/serve/llm.log 2>&1 &
  echo $! > logs/serve/llm.pid
  wait_ready llm "$LLM_PORT"
}

wait_ready() {
  local name="$1" port="$2"
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
    kill -0 "$(cat "logs/serve/$name.pid")" 2>/dev/null || break
    sleep 10
  done
  if ! curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
    echo "server on :$port failed; see logs/serve/$name.log" >&2
    stop "$name"
    exit 1
  fi
  echo "server on :$port ready"
}

stop() {
  local names="${1:-llm emb}" n
  for n in $names; do f="logs/serve/$n.pid"
    [[ -f "$f" ]] && kill "$(cat "$f")" 2>/dev/null || true
    rm -f "$f"
  done
}

status() {
  for port in "$LLM_PORT" "$EMB_PORT"; do
    curl -sf "http://127.0.0.1:$port/v1/models" || echo "server on :$port down"
    echo
  done
}

"${1:-start}" "${@:2}"
