from __future__ import annotations

import json
from typing import Any

from src.prompts import (
    format_entity_extraction_prompt,
    format_hyperedge_extraction_prompt,
)
from utils.checkpoint import merge_records, rows_from_payload, run_checkpointed_parallel
from utils.config import HyperProveConfig, settings
from utils.io import read_json_if_exists, write_json
from utils.llm_client import LLMClient
from utils.run_logging import log_event


def extract_first_json_object(text: str) -> str:
    start = None
    stack = 0
    in_string = False
    escape = False
    for index, char in enumerate(text or ""):
        if start is None:
            if char == "{":
                start = index
                stack = 1
            continue
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            stack += 1
        elif char == "}":
            stack -= 1
            if stack == 0:
                return text[start : index + 1]
    raise ValueError("No JSON object found in model output.")


def parse_json_object(text: str) -> dict[str, Any]:
    payload = json.loads(extract_first_json_object(text))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item and item.strip()))


def _extract_entities_one(config: HyperProveConfig, passage: dict[str, Any]) -> dict[str, Any]:
    raw = ""
    error = None
    entities: list[str] = []
    try:
        raw = str(LLMClient().call(
            prompt=format_entity_extraction_prompt(passage["content"]),
            artifact_metadata={"step": "entity_extraction", "item_id": passage["idx"]},
            reasoning_effort=settings.index_reasoning_effort,
        ))
        parsed = parse_json_object(raw)
        entities = dedupe([str(item) for item in parsed.get("entities", [])])
    except Exception as exc:
        error = str(exc)
    result = {
        "passage_idx": int(passage["idx"]),
        "passage_id": passage["passage_id"],
        "title": passage["title"],
        "entities": entities,
        "raw_response": raw,
        "error": error,
    }
    log_event(
        "indexing",
        "entity_extraction_completed",
        passage_idx=passage["idx"],
        passage_id=passage["passage_id"],
        title=passage["title"],
        entity_count=len(entities),
        error=error,
        raw_response=raw,
    )
    return result


def run_entity_extraction(config: HyperProveConfig, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_path = f"{config.artifacts_dir}/entities.json"
    existing_rows = [] if config.rebuild else rows_from_payload(read_json_if_exists(output_path))
    existing_by_id = {
        str(row.get("passage_id")): row
        for row in existing_rows
        if row.get("passage_id") and not row.get("error")
    }
    results: list[dict[str, Any] | None] = [existing_by_id.get(str(item["passage_id"])) for item in passages]
    log_event(
        "indexing",
        "entity_extraction_resume",
        total=len(passages),
        skipped=sum(item is not None for item in results),
    )

    def write_checkpoint(rows: list[dict[str, Any] | None]) -> None:
        merged = merge_records(existing_rows, rows, key=lambda row: str(row["passage_id"]))
        write_json(output_path, merged)

    return run_checkpointed_parallel(
        results=results,
        desc="Entity extraction",
        max_workers=config.llm_max_workers,
        checkpoint_interval=config.checkpoint_interval,
        submit_one=lambda idx: _extract_entities_one(config, passages[idx]),
        write_checkpoint=write_checkpoint,
    )


def _extract_hyperedges_one(
    config: HyperProveConfig,
    passage: dict[str, Any],
    entity_result: dict[str, Any],
) -> dict[str, Any]:
    raw = ""
    error = None
    hyperedges: list[dict[str, Any]] = []
    entities = dedupe([str(item) for item in entity_result.get("entities", [])])
    allowed = set(entities)
    try:
        raw = str(LLMClient().call(
            prompt=format_hyperedge_extraction_prompt(passage["content"], entities),
            artifact_metadata={"step": "hyperedge_extraction", "item_id": passage["idx"]},
            reasoning_effort=settings.index_reasoning_effort,
        ))
        parsed = parse_json_object(raw)
        for item in parsed.get("hyperedges", []):
            text = str(item.get("text", "")).strip()
            # Keep the fact and drop only entities outside the passage's entity list.
            item_entities = [ent for ent in dedupe([str(ent) for ent in item.get("entities", [])]) if ent in allowed]
            if text and item_entities:
                hyperedges.append({"text": text, "entities": item_entities})
    except Exception as exc:
        error = str(exc)
    result = {
        "passage_idx": int(passage["idx"]),
        "passage_id": passage["passage_id"],
        "title": passage["title"],
        "entities": entities,
        "hyperedges": hyperedges,
        "raw_response": raw,
        "error": error,
    }
    log_event(
        "indexing",
        "hyperedge_extraction_completed",
        passage_idx=passage["idx"],
        passage_id=passage["passage_id"],
        title=passage["title"],
        hyperedge_count=len(hyperedges),
        error=error,
        raw_response=raw,
    )
    return result


def run_hyperedge_extraction(
    config: HyperProveConfig,
    passages: list[dict[str, Any]],
    entity_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_path = f"{config.artifacts_dir}/hyperedges.json"
    existing_rows = [] if config.rebuild else rows_from_payload(read_json_if_exists(output_path))
    existing_by_id = {
        str(row.get("passage_id")): row
        for row in existing_rows
        if row.get("passage_id") and not row.get("error")
    }
    results: list[dict[str, Any] | None] = [existing_by_id.get(str(item["passage_id"])) for item in passages]
    log_event(
        "indexing",
        "hyperedge_extraction_resume",
        total=len(passages),
        skipped=sum(item is not None for item in results),
    )

    def write_checkpoint(rows: list[dict[str, Any] | None]) -> None:
        merged = merge_records(existing_rows, rows, key=lambda row: str(row["passage_id"]))
        write_json(output_path, merged)

    return run_checkpointed_parallel(
        results=results,
        desc="Hyperedge extraction",
        max_workers=config.llm_max_workers,
        checkpoint_interval=config.checkpoint_interval,
        submit_one=lambda idx: _extract_hyperedges_one(config, passages[idx], entity_results[idx]),
        write_checkpoint=write_checkpoint,
    )
