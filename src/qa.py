from __future__ import annotations

import re
from typing import Any

from src.prompts import format_qa_reader_prompt
from utils.checkpoint import merge_records, rows_from_payload, run_checkpointed_parallel
from utils.config import HyperProveConfig
from utils.io import read_json_if_exists, write_json
from utils.llm_client import LLMClient
from utils.run_logging import log_event


def parse_final_answer(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    match = re.search(r"FINAL_ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().splitlines()[0].rstrip(".")
    match = re.search(r"(?:final[_\s-]*answer|answer)\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().splitlines()[0].rstrip(".")
    if "\n" in text:
        return text.splitlines()[-1].strip().rstrip(".")
    return text.rstrip(".")


def _answer_one(config: HyperProveConfig, retrieval_result: dict[str, Any], gold_answers: list[str]) -> dict[str, Any]:
    query_index = int(retrieval_result["query_index"])
    final_answer = str(retrieval_result.get("final_answer") or "").strip()
    source = str(retrieval_result.get("answer_source") or "")
    raw_response = final_answer
    reasoning = None
    error = None
    answer = final_answer
    qa_source = "integrated_sub_answers"

    if source != "integrated_sub_answers" or not final_answer:
        passages = retrieval_result.get("retrieved_contexts", [])[: config.qa_top_k]
        hyperedges = [item["text"] for item in retrieval_result.get("accumulated_hyperedges", [])[: config.top_hyperedges]]
        system_prompt, user_prompt = format_qa_reader_prompt(
            retrieval_result["question"],
            passages,
            hyperedges,
            retriever_hypothesis=final_answer,
        )
        qa_source = "reader_fallback"
        try:
            response = LLMClient().call(
                prompt=user_prompt,
                system_prompt=system_prompt,
                return_metadata=True,
                artifact_metadata={"step": "qa_reader", "item_id": query_index},
            )
            raw_response = str(response["content"])
            reasoning = response.get("reasoning_content")
            answer = parse_final_answer(raw_response)
        except Exception as exc:
            error = str(exc)
            answer = final_answer
            raw_response = final_answer

    result = {
        "idx": query_index,
        "question_id": retrieval_result.get("question_id"),
        "question": retrieval_result["question"],
        "gold_answers": gold_answers,
        "gold_contexts": retrieval_result.get("gold_contexts", []),
        "retrieved_doc_ids": [int(item["idx"]) for item in retrieval_result.get("retrieved_contexts", [])[: config.qa_top_k]],
        "retrieved_contexts": retrieval_result.get("retrieved_contexts", [])[: config.qa_top_k],
        "prediction": answer,
        "raw_response": raw_response,
        "reasoning_content": reasoning,
        "answer_source": qa_source,
        "retrieval_answer_source": source,
        "error": error,
        "question_fingerprint": retrieval_result.get("question_fingerprint"),
        "run_signature": retrieval_result.get("run_signature"),
    }
    log_event(
        "inference",
        "prediction_completed",
        query_index=query_index,
        question_id=result["question_id"],
        question=result["question"],
        prediction=answer,
        answer_source=qa_source,
        error=error,
    )
    return result


def run_qa(
    config: HyperProveConfig,
    retrieval_results: list[dict[str, Any]],
    gold_answers: list[list[str]],
) -> list[dict[str, Any]]:
    output_path = f"{config.run_dir}/qa_predictions.json"
    existing_rows = [] if config.rebuild else rows_from_payload(read_json_if_exists(output_path), "predictions")
    existing_by_id = {str(row.get("question_id")): row for row in existing_rows if row.get("question_id")}
    results: list[dict[str, Any] | None] = []
    for retrieval_result in retrieval_results:
        row = existing_by_id.get(str(retrieval_result.get("question_id")))
        compatible = (
            isinstance(row, dict)
            and row.get("run_signature") == retrieval_result.get("run_signature")
            and row.get("question_fingerprint") == retrieval_result.get("question_fingerprint")
        )
        results.append(row if compatible else None)
    log_event(
        "inference",
        "prediction_resume",
        total=len(results),
        skipped=sum(row is not None for row in results),
    )

    def write_checkpoint(rows: list[dict[str, Any] | None]) -> None:
        merged = merge_records(existing_rows, rows, key=lambda row: str(row["question_id"]))
        write_json(output_path, {"dataset_name": config.dataset_name, "predictions": merged})

    return run_checkpointed_parallel(
        results=results,
        desc="QA",
        max_workers=config.llm_max_workers,
        checkpoint_interval=config.checkpoint_interval,
        submit_one=lambda idx: _answer_one(config, retrieval_results[idx], gold_answers[idx]),
        write_checkpoint=write_checkpoint,
    )
