from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.prompts import (
    format_decomposition_prompt,
    format_final_answer_integration_prompt,
    format_subquery_evaluation_prompt,
    format_subquery_rewrite_prompt,
)
from src.extraction import parse_json_object
from src.retrieval import HypergraphRetriever, top_k_indices
from utils.checkpoint import merge_records, rows_from_payload, run_checkpointed_parallel
from utils.config import HyperProveConfig
from utils.embeddings import EmbeddingGenerator
from utils.io import read_json_if_exists, write_json
from utils.llm_client import LLMClient
from utils.manifest import text_sha256
from utils.metrics import recall_at_k
from utils.run_logging import log_event


def _parse_sub_questions(raw: str, question: str) -> list[str]:
    try:
        parsed = parse_json_object(raw)
        items = parsed.get("sub_questions", [])
        if isinstance(items, list):
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            return cleaned[:4] or [question]
    except Exception:
        pass
    return [question]


def _parse_eval(raw: str) -> dict[str, Any]:
    try:
        parsed = parse_json_object(raw)
    except Exception as exc:
        return {"status": "UNANSWERED", "answer": None, "supporting_hyperedge_ids": [], "parse_error": str(exc)}
    status = str(parsed.get("status", "")).strip().upper()
    answer = parsed.get("answer")
    ids = parsed.get("supporting_hyperedge_ids", [])
    if not isinstance(ids, list):
        ids = []
    return {
        "status": "ANSWERED" if status == "ANSWERED" and answer else "UNANSWERED",
        "answer": str(answer).strip() if answer else None,
        "supporting_hyperedge_ids": ids,
    }


def _parse_rewrite(raw: str) -> str:
    try:
        return str(parse_json_object(raw).get("rewritten_query") or "").strip()
    except Exception:
        return ""


def _parse_final_answer(raw: str) -> str:
    try:
        parsed = parse_json_object(raw)
        return str(parsed.get("final_answer", "")).strip()
    except Exception:
        return ""


def _rank_frontier(frontier: list[int], embeddings: np.ndarray, query_embedding: np.ndarray, k: int) -> list[int]:
    valid = [int(idx) for idx in dict.fromkeys(frontier) if 0 <= int(idx) < embeddings.shape[0]]
    if not valid or k <= 0:
        return []
    scores = embeddings[np.asarray(valid, dtype=np.int64)] @ query_embedding
    return [valid[int(local)] for local in top_k_indices(scores, min(k, len(valid))).tolist()]


def _extract_supporting_global_indices(
    local_ids: list[Any],
    top_hyperedges: list[dict[str, Any]],
) -> list[int]:
    output: list[int] = []
    for raw in local_ids:
        try:
            local = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= local < len(top_hyperedges):
            output.append(int(top_hyperedges[local]["index"]))
    return list(dict.fromkeys(output))


class HyperProveReasoner:
    def __init__(self, config: HyperProveConfig, index_state: dict[str, Any]) -> None:
        self.config = config
        self.retriever = HypergraphRetriever(config, index_state)
        self.embedding_generator = EmbeddingGenerator()

    def decompose(self, question: str, query_index: int) -> tuple[list[str], str]:
        raw = ""
        try:
            raw = str(LLMClient().call(
                prompt=format_decomposition_prompt(question),
                artifact_metadata={"step": "decompose", "item_id": query_index},
            ))
        except Exception:
            return [question], raw
        return _parse_sub_questions(raw, question), raw

    def rewrite(
        self,
        sub_query: str,
        answered_pairs: list[tuple[str, str]],
        query_index: int,
        sub_index: int,
    ) -> tuple[str, str]:
        """Rewrite(q_t, R): fill references with validated answers; q'_t = q_t when R is empty."""
        if not answered_pairs:
            return sub_query, ""
        raw = ""
        try:
            raw = str(LLMClient().call(
                prompt=format_subquery_rewrite_prompt(sub_query, answered_pairs),
                artifact_metadata={"step": "subquery_rewrite", "item_id": query_index, "sub_index": sub_index},
            ))
        except Exception:
            pass
        # An unparsable or failed rewrite keeps the sub-query as decomposed.
        return _parse_rewrite(raw) or sub_query, raw

    def answer_question(
        self,
        question: str,
        query_embedding: np.ndarray,
        query_index: int,
        gold_doc_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        sub_queries, decomposition_raw = self.decompose(question, query_index)
        answered_pairs: list[tuple[str, str]] = []
        sub_answer_pairs: list[tuple[str, str | None]] = []
        frontier: list[int] = []
        memory_h: dict[int, dict[str, Any]] = {}
        memory_d: dict[int, dict[str, Any]] = {}
        iterations: list[dict[str, Any]] = []
        all_answered = True

        for sub_index, sub_query in enumerate(sub_queries):
            retrieval_query, rewrite_raw = self.rewrite(sub_query, answered_pairs, query_index, sub_index)
            if retrieval_query == question:
                q_embedding = query_embedding
            else:
                q_embedding = self.embedding_generator.batch_encode([retrieval_query], batch_size=1, show_progress=False)[0]

            if sub_index == 0:
                seeds = self.retriever.dense_topk(q_embedding, self.config.kd)
                carry_seeds: list[int] = []
            else:
                carry_seeds = _rank_frontier(frontier, self.retriever.hyperedge_embeddings, q_embedding, self.config.kc)
                dense_seeds = self.retriever.dense_topk(q_embedding, self.config.kd, exclude=set(carry_seeds))
                seeds = list(dict.fromkeys(carry_seeds + dense_seeds))

            hop = self.retriever.retrieve_hop(q_embedding, seeds)
            top_hyperedges = hop["top_hyperedges"]
            top_passages = hop["top_passages"]
            for item in top_hyperedges:
                idx = int(item["index"])
                previous = memory_h.get(idx)
                if previous is None or float(previous["score"]) < float(item["score"]):
                    memory_h[idx] = item
            for passage in top_passages:
                idx = int(passage["idx"])
                previous = memory_d.get(idx)
                if previous is None or float(previous["score"]) < float(passage["score"]):
                    memory_d[idx] = passage

            eval_raw = ""
            eval_result = {"status": "UNANSWERED", "answer": None, "supporting_hyperedge_ids": []}
            try:
                eval_raw = str(LLMClient().call(
                    prompt=format_subquery_evaluation_prompt(
                        retrieval_query,
                        [item["text"] for item in top_hyperedges],
                        [{"title": item["title"], "text": item["text"]} for item in top_passages],
                    ),
                    artifact_metadata={"step": "subquery_evaluate", "item_id": query_index, "sub_index": sub_index},
                ))
                eval_result = _parse_eval(eval_raw)
            except Exception as exc:
                eval_result = {"status": "UNANSWERED", "answer": None, "supporting_hyperedge_ids": [], "error": str(exc)}

            support_indices = _extract_supporting_global_indices(
                eval_result.get("supporting_hyperedge_ids", []),
                top_hyperedges,
            )
            answered = eval_result.get("status") == "ANSWERED" and bool(eval_result.get("answer"))
            sub_answer = str(eval_result["answer"]).strip() if answered else None
            if answered and sub_answer:
                answered_pairs.append((retrieval_query, sub_answer))
                supporting_entity_ids: list[str] = []
                for idx in support_indices:
                    supporting_entity_ids.extend(self.retriever.hyperedges[idx].get("entity_ids", []))
                frontier = self.retriever.find_hyperedges_by_entities([sub_answer], supporting_entity_ids)
            else:
                all_answered = False
                frontier = [int(item["index"]) for item in top_hyperedges]
            sub_answer_pairs.append((retrieval_query, sub_answer))

            iteration = {
                "sub_index": sub_index,
                "sub_query": sub_query,
                "retrieval_query": retrieval_query,
                "rewrite_raw": rewrite_raw,
                "carry_seed_indices": carry_seeds,
                "seed_indices": hop["seed_indices"],
                "top_hyperedges": top_hyperedges,
                "top_passages": top_passages,
                "eval_raw": eval_raw,
                "eval_result": eval_result,
                "supporting_hyperedge_indices": support_indices,
                "answered": answered,
                "sub_answer": sub_answer,
                "next_frontier_size": len(frontier),
            }
            iterations.append(iteration)

        final_answer = ""
        answer_source = "reader_fallback"
        if all_answered and sub_answer_pairs:
            try:
                raw = str(LLMClient().call(
                    prompt=format_final_answer_integration_prompt(question, sub_answer_pairs),
                    artifact_metadata={"step": "final_answer_integration", "item_id": query_index},
                ))
                final_answer = _parse_final_answer(raw)
            except Exception:
                final_answer = ""
            if not final_answer:
                for _, answer in reversed(sub_answer_pairs):
                    if answer:
                        final_answer = answer
                        break
            answer_source = "integrated_sub_answers" if final_answer else "reader_fallback"

        ranked_h = sorted(memory_h.values(), key=lambda item: -float(item["score"]))
        ranked_d = sorted(memory_d.values(), key=lambda item: (-float(item["score"]), int(item["idx"])))
        retrieved_doc_ids = [int(item["idx"]) for item in ranked_d[: self.config.qa_top_k]]
        result: dict[str, Any] = {
            "query_index": query_index,
            "question": question,
            "sub_queries": sub_queries,
            "sub_queries_raw": decomposition_raw,
            "sub_answer_pairs": [
                {"sub_query": sub_query, "answer": answer} for sub_query, answer in sub_answer_pairs
            ],
            "iterations": iterations,
            "accumulated_hyperedges": ranked_h,
            "retrieved_contexts": [
                {"idx": int(item["idx"]), "title": item["title"], "text": item["text"], "score": item["score"]}
                for item in ranked_d[: self.config.qa_top_k]
            ],
            "retrieved_doc_ids": retrieved_doc_ids,
            "final_answer": final_answer,
            "answer_source": answer_source,
            "gold_doc_ids": gold_doc_ids or [],
        }
        if gold_doc_ids is not None:
            result["recall"] = {f"Recall@{k}": recall_at_k(gold_doc_ids, retrieved_doc_ids, k) for k in (1, 3, 5, 10)}
        log_event(
            "inference",
            "retrieval_completed",
            query_index=query_index,
            question=question,
            final_answer=final_answer,
            answer_source=answer_source,
            retrieved_doc_ids=retrieved_doc_ids,
            sub_answers=result["sub_answer_pairs"],
            iterations=iterations,
        )
        return result


def retrieve_all(
    config: HyperProveConfig,
    index_state: dict[str, Any],
    questions: list[str],
    question_ids: list[str | None],
    gold_doc_ids: list[list[int]],
    gold_answers: list[list[str]],
    gold_contexts: list[list[dict[str, Any]]],
    run_signature: str,
) -> list[dict[str, Any]]:
    reasoner = HyperProveReasoner(config, index_state)
    output_path = config.retrieval_results_path
    existing_rows = [] if config.rebuild else rows_from_payload(read_json_if_exists(output_path), "results")
    existing_by_id = {str(row.get("question_id")): row for row in existing_rows if row.get("question_id")}
    results: list[dict[str, Any] | None] = []
    stable_question_ids = [str(qid or f"query-{idx}") for idx, qid in enumerate(question_ids)]
    for idx, question in enumerate(questions):
        row = existing_by_id.get(stable_question_ids[idx])
        compatible = (
            isinstance(row, dict)
            and row.get("run_signature") == run_signature
            and row.get("question_fingerprint") == text_sha256(question)
        )
        results.append(row if compatible else None)

    pending = [idx for idx, row in enumerate(results) if row is None]
    pending_embeddings: dict[int, np.ndarray] = {}
    if pending:
        encoded = EmbeddingGenerator().batch_encode(
            [questions[idx] for idx in pending],
            batch_size=min(32, len(pending)),
            show_progress=False,
            stage="inference",
        )
        pending_embeddings = {
            idx: np.asarray(encoded[pos], dtype=np.float32)
            for pos, idx in enumerate(pending)
        }
    log_event("inference", "retrieval_resume", total=len(questions), skipped=len(questions) - len(pending))

    def write_checkpoint(rows: list[dict[str, Any] | None]) -> None:
        merged = merge_records(existing_rows, rows, key=lambda row: str(row["question_id"]))
        write_json(output_path, {"dataset_name": config.dataset_name, "results": merged})

    return run_checkpointed_parallel(
        results=results,
        desc="HyperProve retrieval",
        max_workers=config.max_workers,
        checkpoint_interval=config.checkpoint_interval,
        submit_one=lambda idx: _answer_and_attach(
            reasoner,
            questions[idx],
            pending_embeddings[idx],
            idx,
            stable_question_ids[idx],
            gold_doc_ids[idx],
            gold_answers[idx],
            gold_contexts[idx],
            run_signature,
        ),
        write_checkpoint=write_checkpoint,
    )


def _answer_and_attach(
    reasoner: HyperProveReasoner,
    question: str,
    query_embedding: np.ndarray,
    query_index: int,
    question_id: str,
    gold_doc_ids: list[int],
    gold_answers: list[str],
    gold_contexts: list[dict[str, Any]],
    run_signature: str,
) -> dict[str, Any]:
    result = reasoner.answer_question(question, query_embedding, query_index, gold_doc_ids)
    result["question_id"] = question_id
    result["gold_answers"] = gold_answers
    result["gold_contexts"] = gold_contexts
    result["question_fingerprint"] = text_sha256(question)
    result["run_signature"] = run_signature
    return result
