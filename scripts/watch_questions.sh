#!/usr/bin/env bash
# Follow a run question by question: decomposition, each hop (rewritten sub-query -> answer),
# QA prediction and judge verdict, read from the structured logs of the latest run.
# Usage: bash scripts/watch_questions.sh [dataset] [lines]   (dataset: musique; lines: past events to show, default 20)
cd "$(dirname "$0")/.."
dataset="${1:-musique}"; lines="${2:-20}"
inference=$(ls -t logs/"${dataset}"_inference_*.log 2>/dev/null | head -1)
evaluate=$(ls -t logs/"${dataset}"_evaluate_*.log 2>/dev/null | head -1)
[[ -n "$inference" ]] || { echo "no logs/${dataset}_inference_*.log yet" >&2; exit 1; }
echo "following $inference ${evaluate:+and $evaluate} (Ctrl-C to stop)"

tail -n "$lines" -q -F "$inference" ${evaluate:+"$evaluate"} 2>/dev/null \
  | grep --line-buffered -E "\| (retrieval_completed|prediction_completed|judge_completed) \|" \
  | stdbuf -oL sed -E "s/^[^|]*\| [A-Z]+ \| ([a-z_]+) \| /\1 /" \
  | while read -r event payload; do
      case "$event" in
        retrieval_completed)
          jq -r '"\n#\(.query_index)  \(.question)",
                 (.iterations[] | "   hop \(.sub_index): \(.retrieval_query)  ->  \(.sub_answer // "UNANSWERED")"),
                 "   => \(if .answer_source == "integrated_sub_answers" then "integrated: \(.final_answer)" else "reader fallback (answer comes at the QA step)" end)"' <<<"$payload" ;;
        prediction_completed)
          jq -r 'select(.answer_source == "reader_fallback") | "#\(.query_index)  reader answer: \(.prediction)"' <<<"$payload" ;;
        judge_completed)
          jq -r '"#\(.query_index)  judge: \(if .verdict == true then "CORRECT" elif .verdict == false then "wrong" else "no verdict" end)"' <<<"$payload" ;;
      esac
    done
