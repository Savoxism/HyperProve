from __future__ import annotations

from typing import Any

from src import evaluation, extraction, indexing, qa, reasoning
from utils.data import (
    get_gold_answers,
    get_gold_doc_ids,
    load_corpus,
    load_dataset,
    prepare_passages,
)
from utils.config import HyperProveConfig
from utils.io import ensure_dir, read_json, write_json
from utils.manifest import index_manifest, inference_signature
from utils.run_logging import configure_run_logs, log_event


def _ensure_output_dirs(config: HyperProveConfig) -> None:
    ensure_dir(config.artifacts_dir)
    ensure_dir(config.run_dir)


def run_pipeline(config: HyperProveConfig) -> dict[str, Any]:
    _ensure_output_dirs(config)
    log_paths = configure_run_logs(config.dataset_name, config.logs_dir)

    dataset = load_dataset(config.dataset_path)
    corpus = load_corpus(config.corpus_path)
    if config.limit_questions is not None:
        dataset = dataset[: config.limit_questions]

    gold_answers = get_gold_answers(dataset)
    gold_doc_ids = get_gold_doc_ids(dataset, corpus)
    gold_contexts = [item.get("contexts", []) for item in dataset]
    question_ids = [item.get("question_id") for item in dataset]
    questions = [str(item["question"]) for item in dataset]
    expected_index_manifest = index_manifest(config, len(corpus))
    run_signature = inference_signature(config, expected_index_manifest)
    stable_question_ids = [str(item or f"query-{idx}") for idx, item in enumerate(question_ids)]

    log_event(
        "indexing",
        "pipeline_started",
        dataset_name=config.dataset_name,
        corpus_count=len(corpus),
        rebuild=config.rebuild,
        skip_index=config.skip_index,
    )
    log_event("inference", "pipeline_started", dataset_name=config.dataset_name, question_count=len(questions))
    log_event("evaluate", "pipeline_started", dataset_name=config.dataset_name, question_count=len(questions))

    if config.qa_only:
        retrieval_payload = read_json(config.retrieval_results_path)
        by_id = {str(row.get("question_id")): row for row in retrieval_payload.get("results", [])}
        missing = [qid for qid in stable_question_ids if qid not in by_id]
        if missing:
            raise FileNotFoundError(f"Cannot use --qa-only; retrieval results are missing {len(missing)} questions.")
        retrieval_results = [by_id[qid] for qid in stable_question_ids]
    else:
        if config.skip_index:
            if not indexing.required_artifacts_exist(config, expected_index_manifest):
                raise FileNotFoundError(
                    "Cannot use --skip-index because HyperProve index artifacts are missing or incompatible."
                )
            index_state = indexing.load_index(config)
            log_event("indexing", "index_reused", reason="skip_index")
        elif indexing.required_artifacts_exist(config, expected_index_manifest) and not config.rebuild:
            index_state = indexing.load_index(config)
            log_event("indexing", "index_reused", reason="compatible_manifest")
        else:
            passages = prepare_passages(corpus)
            entity_results = extraction.run_entity_extraction(config, passages)
            hyperedge_results = extraction.run_hyperedge_extraction(config, passages, entity_results)
            index_state = indexing.build_index(config, passages, entity_results, hyperedge_results)

        if config.index_only:
            summary = {
                "dataset_name": config.dataset_name,
                "dataset_path": config.dataset_path,
                "corpus_path": config.corpus_path,
                "run_dir": config.run_dir,
                "num_questions": len(questions),
                "num_corpus_passages": len(corpus),
                "index_artifacts_ready": True,
                "log_paths": log_paths,
            }
            write_json(f"{config.run_dir}/run_summary.json", summary)
            return summary

        retrieval_results = reasoning.retrieve_all(
            config=config,
            index_state=index_state,
            questions=questions,
            question_ids=question_ids,
            gold_doc_ids=gold_doc_ids,
            gold_answers=gold_answers,
            gold_contexts=gold_contexts,
            run_signature=run_signature,
        )

    if config.retrieval_only:
        summary = {
            "dataset_name": config.dataset_name,
            "run_dir": config.run_dir,
            "num_questions": len(questions),
            "retrieval_results_path": config.retrieval_results_path,
            "log_paths": log_paths,
        }
        write_json(f"{config.run_dir}/run_summary.json", summary)
        return summary

    qa_predictions = qa.run_qa(config, retrieval_results, gold_answers)
    metrics = evaluation.evaluate_predictions(config, qa_predictions, retrieval_results, gold_doc_ids)
    summary = {
        "dataset_name": config.dataset_name,
        "dataset_path": config.dataset_path,
        "corpus_path": config.corpus_path,
        "run_dir": config.run_dir,
        "num_questions": len(questions),
        "num_corpus_passages": len(corpus),
        "metrics": metrics,
        "log_paths": log_paths,
    }
    write_json(f"{config.run_dir}/run_summary.json", summary)
    return summary
