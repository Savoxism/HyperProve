#!/usr/bin/env bash
# Read-only progress view for run hp-hotpot-2wiki-20261015 (Ctrl-C stops only this watcher).
# Usage: INTERVAL=10 bash scripts/watch_hp-hotpot-2wiki-20261015.sh
cd "$(dirname "$0")/.."
RUN_ID=hp-hotpot-2wiki-20261015
OUT=outputs_hyperprove
DIR="logs/$RUN_ID"
INTERVAL="${INTERVAL:-10}"
trap 'echo; exit 0' INT

while true; do
  clear
  echo "HyperProve run $RUN_ID   $(date "+%F %T %Z")   (refresh ${INTERVAL}s, Ctrl-C to quit)"
  pid=$(cat "$DIR/run.pid" 2>/dev/null)
  if [[ -f "$DIR/exit_code" ]]; then state="finished exit=$(cat "$DIR/exit_code")"
  elif [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then state="running (pid $pid)"
  else state="not running"; fi
  echo "state: $state"
  start=$(head -1 "$DIR/sequence.log" 2>/dev/null | grep -o "^\[[^]]*\]" | tr -d "[]")
  [[ -n "$start" ]] && echo "elapsed: $(( ($(date +%s) - $(date -d "$start" +%s)) / 60 )) min since $start"
  echo
  for ds in hotpotqa 2wikimultihopqa; do
    log="$DIR/$ds.log"
    if [[ -f "$DIR/$ds.exit" ]]; then status="exit=$(cat "$DIR/$ds.exit")"
    elif [[ -f "$log" ]]; then status="running"
    else status="pending"; fi
    # last tqdm line: "<phase>:  42%|####      | 420/1000 [05:00<06:54, 1.40it/s]"
    line=$(tr "\r" "\n" < "$log" 2>/dev/null | grep -E "it/s\]|s/it\]|\]$" | tail -1)
    phase=$(sed -E "s/:.*//" <<<"$line")
    pct=$(grep -oE "[0-9]+%" <<<"$line" | head -1); pct=${pct%\%}
    steps=$(grep -oE "[0-9]+/[0-9]+" <<<"$line" | head -1)
    eta=$(grep -oE "<[0-9:]+" <<<"$line" | tr -d "<")
    bar=""; if [[ -n "$pct" ]]; then n=$((pct / 4)); bar="[$(printf "%${n}s" | tr " " "#")$(printf "%$((25 - n))s")]"; fi
    printf "%-16s %-9s %-22s %s %4s%%  %-11s ETA %s\n" "$ds" "$status" "${phase:-–}" "${bar:-[                         ]}" "${pct:-0}" "${steps:-}" "${eta:-–}"
    summary="$OUT/$ds/openai_gpt-oss-120b_BAAI_bge-m3/run_summary.json"
    [[ -f "$summary" ]] && .venv/bin/python -c "
import json,sys; m=json.load(open(sys.argv[1]))['metrics']
print('                 Acc %.2f  EM %.2f  F1 %.2f  R@10 %.2f' % (m['judge']['JudgeAccuracy']*100, m['qa']['ExactMatch']*100, m['qa']['F1']*100, m['retrieval']['Recall@10']*100))" "$summary"
    retries=$(grep -c "Retrying LLM" "$log" 2>/dev/null); errs=$(tr "\r" "\n" < "$log" 2>/dev/null | grep -cE "Traceback|Too many open files|OutOfMemory")
    [[ -f "$log" ]] && echo "                 retries=${retries:-0} errors=${errs:-0}"
  done
  echo
  echo "GPU  util  mem(MiB)      temp  power"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits \
    | awk -F", " '{printf "%-4s %3s%%  %6s/%-6s %3sC  %5.0fW\n", $1, $2, $3, $4, $5, $6}'
  echo "disk /nvme: $(df -h /nvme | awk 'NR==2{print $4 " free (" $5 " used)"}')"
  sleep "$INTERVAL"
done
