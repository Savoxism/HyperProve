from __future__ import annotations

import re
from typing import Any

from src.prompts import ANSWER_JUDGE_PROMPT
from utils.checkpoint import merge_records, rows_from_payload, run_checkpointed_parallel
from utils.config import HyperProveConfig
from utils.io import read_json_if_exists, write_json
from utils.llm_client import LLMClient
from utils.metrics import exact_match, recall_at_k, token_f1
from utils.run_logging import log_event


_VERDICT_RE = re.compile(r"\[\[\s*(yes|no)\s*\]\]", flags=re.IGNORECASE)
# A judge reply with no verdict (empty content when reasoning hits max_tokens, or no [[yes]]/[[no]])
# is asked again (sampled anew) up to this many times in total.
JUDGE_ATTEMPTS = 3


def _parse_verdict(text: str) -> bool | None:
    match = _VERDICT_RE.search(text or "")
    if match:
        return match.group(1).lower() == "yes"
    lowered = str(text or "").strip().lower()
    if lowered.startswith("yes"):
        return True
    if lowered.startswith("no"):
        return False
    return None


def _judge_one(record: dict[str, Any]) -> dict[str, Any]:
    raw = ""
    verdict = None
    error = None
    attempts = 0
    while verdict is None and attempts < JUDGE_ATTEMPTS:
        attempts += 1
        try:
            raw = str(LLMClient().call(
                prompt=ANSWER_JUDGE_PROMPT.format(
                    question=record["question"],
                    ground_truth=" | ".join(record.get("gold_answers", [])),
                    prediction=record.get("prediction", ""),
                ),
                artifact_metadata={"step": "judge", "item_id": record["idx"], "attempt": attempts},
            ))
            verdict = _parse_verdict(raw)
            error = None
        except Exception as exc:
            error = str(exc)
    result = {
        "idx": record["idx"],
        "question_id": record.get("question_id"),
        "question_fingerprint": record.get("question_fingerprint"),
        "run_signature": record.get("run_signature"),
        "prediction": record.get("prediction", ""),
        "judge_response": raw,
        "verdict": verdict,
        "attempts": attempts,
        "error": error,
    }
    log_event(
        "evaluate",
        "judge_completed",
        query_index=record["idx"],
        question_id=record.get("question_id"),
        verdict=verdict,
        attempts=attempts,
        error=error,
        raw_response=raw,
    )
    return result


def evaluate_predictions(
    config: HyperProveConfig,
    qa_predictions: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
    gold_doc_ids: list[list[int]],
) -> dict[str, Any]:
    judge_output_path = f"{config.run_dir}/judge_results.json"
    existing_rows = [] if config.rebuild else rows_from_payload(read_json_if_exists(judge_output_path), "per_question")
    existing_by_id = {str(row.get("question_id")): row for row in existing_rows if row.get("question_id")}
    judge_rows: list[dict[str, Any] | None] = []
    for prediction in qa_predictions:
        row = existing_by_id.get(str(prediction.get("question_id")))
        compatible = (
            isinstance(row, dict)
            and row.get("run_signature") == prediction.get("run_signature")
            and row.get("question_fingerprint") == prediction.get("question_fingerprint")
            and row.get("verdict") is not None
            and row.get("prediction") == prediction.get("prediction", "")
        )
        judge_rows.append(row if compatible else None)

    def write_judge_checkpoint(rows: list[dict[str, Any] | None]) -> None:
        merged = merge_records(existing_rows, rows, key=lambda row: str(row["question_id"]))
        write_json(judge_output_path, {"dataset_name": config.dataset_name, "per_question": merged})

    log_event(
        "evaluate",
        "judge_resume",
        total=len(judge_rows),
        skipped=sum(row is not None for row in judge_rows),
        skip_judge=config.skip_judge,
    )

    if not config.skip_judge:
        run_checkpointed_parallel(
            results=judge_rows,
            desc="Judge",
            max_workers=config.judge_workers or config.llm_max_workers,
            checkpoint_interval=config.checkpoint_interval,
            submit_one=lambda idx: _judge_one(qa_predictions[idx]),
            write_checkpoint=write_judge_checkpoint,
        )

    recall_totals = {f"Recall@{k}": 0.0 for k in (1, 3, 5, 10)}
    em_total = 0.0
    f1_total = 0.0
    judge_yes = 0
    judge_count = 0
    per_query: list[dict[str, Any]] = []
    finalized_judge = [
        row
        or {
            "idx": idx,
            "question_id": qa_predictions[idx].get("question_id"),
            "question_fingerprint": qa_predictions[idx].get("question_fingerprint"),
            "run_signature": qa_predictions[idx].get("run_signature"),
            "judge_response": None,
            "verdict": None,
            "error": None,
        }
        for idx, row in enumerate(judge_rows)
    ]

    for idx, prediction in enumerate(qa_predictions):
        retrieved = [int(item) for item in retrieval_results[idx].get("retrieved_doc_ids", [])]
        row = {
            "idx": idx,
            "question": prediction["question"],
            "gold_answers": prediction.get("gold_answers", []),
            "prediction": prediction.get("prediction", ""),
            "retrieved_doc_ids": retrieved,
        }
        for k in (1, 3, 5, 10):
            value = recall_at_k(gold_doc_ids[idx], retrieved, k)
            row[f"Recall@{k}"] = value
            recall_totals[f"Recall@{k}"] += value
        em = exact_match(prediction.get("gold_answers", []), prediction.get("prediction", ""))
        f1 = token_f1(prediction.get("gold_answers", []), prediction.get("prediction", ""))
        row["ExactMatch"] = em
        row["F1"] = f1
        em_total += em
        f1_total += f1
        verdict = finalized_judge[idx].get("verdict")
        row["JudgeCorrect"] = verdict
        if verdict is not None:
            judge_count += 1
            judge_yes += int(bool(verdict))
        per_query.append(row)
        log_event("evaluate", "sample_metrics", **row)

    total = max(1, len(qa_predictions))
    metrics = {
        "dataset_name": config.dataset_name,
        "retrieval": {key: round(value / total, 4) for key, value in recall_totals.items()},
        "qa": {
            "ExactMatch": round(em_total / total, 4),
            "F1": round(f1_total / total, 4),
        },
        "judge": {
            # Over every prediction: one still without a verdict after JUDGE_ATTEMPTS counts as wrong.
            "JudgeAccuracy": round(judge_yes / total, 4),
            "counted_predictions": judge_count,
            "unjudged_predictions": len(qa_predictions) - judge_count,
        },
    }
    merged_judges = merge_records(existing_rows, finalized_judge, key=lambda row: str(row.get("question_id")))
    write_json(
        judge_output_path,
        {"dataset_name": config.dataset_name, "per_question": merged_judges, "summary": metrics["judge"]},
    )
    write_json(f"{config.run_dir}/metrics.json", metrics)
    write_json(f"{config.run_dir}/evaluation_per_query.json", per_query)
    log_event("evaluate", "evaluation_completed", metrics=metrics)
    return metrics
