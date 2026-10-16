#!/usr/bin/env bash
# One dataset end to end with the current defaults: builds (or reuses a compatible) index, then
# retrieval + QA + judge. Brings the vLLM servers up if needed; stops them at the end unless
# KEEP_SERVERS=1.
# Usage: bash scripts/run_dataset.sh <musique|musique_subset|hotpotqa|2wikimultihopqa> <output_root> [extra main.py args]
set -uo pipefail
cd "$(dirname "$0")/.."
ulimit -n 65536
dataset="$1"; out="$2"; shift 2
case "$dataset" in
  musique_subset) files=(datasets/subset_musique_questions.json datasets/subset_musique_corpus.json) ;;
  musique|hotpotqa|2wikimultihopqa) files=("datasets/${dataset}_questions.json" "datasets/${dataset}_corpus.json") ;;
  *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
esac
log() { echo "[$(date "+%F %T %Z")] $*"; }
if ! curl -sf http://127.0.0.1:8000/health >/dev/null || ! curl -sf http://127.0.0.1:8001/health >/dev/null; then
  log "serve start"
  bash scripts/serve_models.sh start || { log "serve failed"; exit 1; }
fi
log "$dataset -> $out $*"
.venv/bin/python main.py --dataset-name "$dataset" --dataset-path "${files[0]}" --corpus-path "${files[1]}" \
  --output-root "$out" --limit-questions 1000 --max-workers 128 --llm-max-workers 256 "$@"
rc=$?
log "$dataset -> $out exit=$rc"
[[ "${KEEP_SERVERS:-0}" == 1 ]] || bash scripts/serve_models.sh stop
exit $rc
