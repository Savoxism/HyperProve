#!/usr/bin/env bash
# Run several datasets one after another on the same vLLM servers; a dataset starts only after the
# previous one exited 0. Writes logs/<run_id>/{run.pid,<dataset>.log,<dataset>.exit,exit_code}.
# Usage: nohup bash scripts/run_sequence.sh <run_id> <output_root> <dataset> [<dataset> ...] &
set -uo pipefail
cd "$(dirname "$0")/.."
run_id="$1"; out="$2"; shift 2
dir="logs/$run_id"
mkdir -p "$dir"
echo $$ > "$dir/run.pid"
log() { echo "[$(date "+%F %T %Z")] $*" | tee -a "$dir/sequence.log"; }
log "start run_id=$run_id out=$out datasets=$* commit=$(git rev-parse HEAD)"
rc=0
for dataset in "$@"; do
  log "dataset $dataset start"
  KEEP_SERVERS=1 bash scripts/run_dataset.sh "$dataset" "$out" > "$dir/$dataset.log" 2>&1
  rc=$?
  echo "$rc" > "$dir/$dataset.exit"
  log "dataset $dataset exit=$rc"
  [[ $rc -eq 0 ]] || break
done
bash scripts/serve_models.sh stop
echo "$rc" > "$dir/exit_code"
log "done exit=$rc"
exit $rc
