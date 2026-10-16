"""Artifact fingerprints and compatibility manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.config import HyperProveConfig, settings
from utils.io import read_json_if_exists, write_json


ALGORITHM_VERSION = "hyperprove-paper-v2"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def index_manifest(config: HyperProveConfig, corpus_count: int) -> dict[str, Any]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "corpus_path": str(Path(config.corpus_path).resolve()),
        "corpus_sha256": file_sha256(config.corpus_path),
        "corpus_count": int(corpus_count),
        "llm_model": settings.llm_model,
        "index_reasoning_effort": settings.index_reasoning_effort,
        "llm_temperature": settings.llm_temperature,
        "embedding_model": settings.embedding_model,
        "synonym_threshold": float(config.synonym_threshold),
    }


def inference_signature(config: HyperProveConfig, expected_index_manifest: dict[str, Any]) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "index": expected_index_manifest,
        "kc": config.kc,
        "kd": config.kd,
        "local_subgraph_size": config.local_subgraph_size,
        "top_hyperedges": config.top_hyperedges,
        "qa_top_k": config.qa_top_k,
        "tau": config.tau,
        "theta": config.theta,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "ppr_max_iter": config.ppr_max_iter,
        "reasoning_effort": settings.llm_reasoning_effort,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return text_sha256(canonical)


def manifest_matches(path: str, expected: dict[str, Any]) -> bool:
    actual = read_json_if_exists(path)
    return isinstance(actual, dict) and actual == expected


def write_manifest(path: str, payload: dict[str, Any]) -> None:
    write_json(path, payload)
