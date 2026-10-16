"""Dataset normalization and stable identifier helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from utils.io import read_json


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def passage_content(title: str, text: str) -> str:
    return f"{normalize_whitespace(title)}\n{normalize_whitespace(text)}"


def load_dataset(path: str) -> list[dict[str, Any]]:
    return read_json(path)


def load_corpus(path: str) -> list[dict[str, Any]]:
    return read_json(path)


def prepare_passages(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    for item in corpus:
        content = passage_content(item.get("title", ""), item.get("text", ""))
        passages.append(
            {
                "idx": int(item["idx"]),
                "title": str(item.get("title", "")),
                "text": str(item.get("text", "")),
                "content": content,
                "passage_id": stable_id("passage", content),
            }
        )
    return passages


def get_gold_answers(samples: list[dict[str, Any]]) -> list[list[str]]:
    answers: list[list[str]] = []
    for sample in samples:
        answer = sample.get("answer", "")
        if isinstance(answer, list):
            answers.append([str(item) for item in answer])
        else:
            answers.append([str(answer)])
    return answers


def get_gold_doc_ids(samples: list[dict[str, Any]], corpus: list[dict[str, Any]]) -> list[list[int]]:
    lookup = {
        passage_content(item.get("title", ""), item.get("text", "")): int(item["idx"])
        for item in corpus
    }
    by_idx = {int(item["idx"]): item for item in corpus}
    output: list[list[int]] = []
    for sample in samples:
        doc_ids: list[int] = []
        for context in sample.get("contexts", []) or []:
            raw_idx = context.get("idx")
            if raw_idx is not None and int(raw_idx) in by_idx:
                doc_ids.append(int(raw_idx))
                continue
            key = passage_content(context.get("title", ""), context.get("text", ""))
            if key in lookup:
                doc_ids.append(int(lookup[key]))
        output.append(list(dict.fromkeys(doc_ids)))
    return output
