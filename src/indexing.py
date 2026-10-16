from __future__ import annotations

from collections import defaultdict
import os
from typing import Any

import numpy as np
import scipy.sparse as sp

from utils.config import HyperProveConfig
from utils.data import stable_id
from utils.embeddings import EmbeddingGenerator
from utils.io import atomic_output_path, read_json, write_json
from utils.manifest import index_manifest, manifest_matches, write_manifest
from utils.run_logging import log_event


def _save_numpy(path: str, array: np.ndarray) -> None:
    with atomic_output_path(path, ".npy") as tmp_path:
        np.save(tmp_path, array)


def _save_sparse(path: str, matrix: sp.csr_matrix) -> None:
    with atomic_output_path(path, ".npz") as tmp_path:
        sp.save_npz(tmp_path, matrix)


def _load_or_embed(path: str, texts: list[str]) -> np.ndarray:
    if os.path.exists(path):
        try:
            arr = np.load(path)
            if arr.shape[0] == len(texts):
                return np.asarray(arr, dtype=np.float32)
        except Exception:
            pass
    log_event("indexing", "embedding_started", path=path, count=len(texts))
    arr = EmbeddingGenerator().batch_encode(texts, batch_size=128, show_progress=True, stage="indexing")
    _save_numpy(path, arr)
    log_event("indexing", "embedding_completed", path=path, shape=list(arr.shape))
    return arr


def _build_entity_catalog(
    entity_results: list[dict[str, Any]],
    hyperedge_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    text_to_item: dict[str, dict[str, Any]] = {}
    for entity_result, hyperedge_result in zip(entity_results, hyperedge_results):
        passage_idx = int(entity_result["passage_idx"])
        combined = list(entity_result.get("entities", []))
        for hyperedge in hyperedge_result.get("hyperedges", []):
            combined.extend(hyperedge.get("entities", []))
        for text in combined:
            entity_text = str(text).strip()
            if not entity_text:
                continue
            if entity_text not in text_to_item:
                text_to_item[entity_text] = {
                    "entity_id": stable_id("entity", entity_text),
                    "text": entity_text,
                    "passage_indices": [],
                }
            text_to_item[entity_text]["passage_indices"].append(passage_idx)
    entities = sorted(text_to_item.values(), key=lambda item: item["entity_id"])
    for item in entities:
        item["passage_indices"] = sorted(set(int(idx) for idx in item["passage_indices"]))
    return entities, text_to_item


def _build_hyperedge_catalog(
    hyperedge_results: list[dict[str, Any]],
    entity_text_to_item: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    hyperedges: list[dict[str, Any]] = []
    for result in hyperedge_results:
        passage_idx = int(result["passage_idx"])
        for ordinal, hyperedge in enumerate(result.get("hyperedges", [])):
            text = str(hyperedge.get("text", "")).strip()
            entity_texts = [str(ent).strip() for ent in hyperedge.get("entities", []) if str(ent).strip()]
            entity_ids = [entity_text_to_item[text]["entity_id"] for text in entity_texts if text in entity_text_to_item]
            if not text or not entity_ids:
                continue
            identity = f"{passage_idx}:{ordinal}:{text}"
            hyperedges.append(
                {
                    "hyperedge_id": stable_id("hyperedge", identity),
                    "text": text,
                    "entities": list(dict.fromkeys(entity_texts)),
                    "entity_ids": list(dict.fromkeys(entity_ids)),
                    "passage_idx": passage_idx,
                }
            )
    hyperedges.sort(key=lambda item: item["hyperedge_id"])
    return hyperedges


def _compute_synonym_map(
    entities: list[dict[str, Any]],
    entity_embeddings: np.ndarray,
    threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    synonym_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if len(entities) < 2 or entity_embeddings.size == 0:
        return {}
    batch_size = 512
    for start in range(0, len(entities), batch_size):
        end = min(start + batch_size, len(entities))
        sims = entity_embeddings[start:end] @ entity_embeddings.T
        rows, cols = np.where(sims >= float(threshold))
        for local_row, col in zip(rows.tolist(), cols.tolist()):
            row = start + local_row
            if col <= row:
                continue
            left = entities[row]["entity_id"]
            right = entities[col]["entity_id"]
            score = float(sims[local_row, col])
            synonym_map[left].append({"entity_id": right, "similarity": score})
            synonym_map[right].append({"entity_id": left, "similarity": score})
    return dict(synonym_map)


def _expanded_entities(entity_ids: list[str], synonym_map: dict[str, list[dict[str, Any]]]) -> set[str]:
    expanded = set(entity_ids)
    for entity_id in entity_ids:
        for item in synonym_map.get(entity_id, []):
            expanded.add(str(item["entity_id"]))
    return expanded


def build_structural_transition(
    hyperedges: list[dict[str, Any]],
    synonym_map: dict[str, list[dict[str, Any]]],
) -> sp.csr_matrix:
    num_h = len(hyperedges)
    if num_h == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    expanded_sets = [
        _expanded_entities(list(item.get("entity_ids", [])), synonym_map)
        for item in hyperedges
    ]
    entity_ids = sorted(set().union(*expanded_sets)) if expanded_sets else []
    entity_to_col = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
    passage_ids = sorted(set(int(item["passage_idx"]) for item in hyperedges))
    passage_to_col = {passage_idx: idx for idx, passage_idx in enumerate(passage_ids)}

    rows_he: list[int] = []
    cols_he: list[int] = []
    for row, entity_set in enumerate(expanded_sets):
        for entity_id in entity_set:
            rows_he.append(row)
            cols_he.append(entity_to_col[entity_id])
    b_he = sp.csr_matrix(
        (np.ones(len(rows_he), dtype=np.float32), (rows_he, cols_he)),
        shape=(num_h, len(entity_to_col)),
    )

    rows_hd = np.arange(num_h, dtype=np.int64)
    cols_hd = np.asarray([passage_to_col[int(item["passage_idx"])] for item in hyperedges], dtype=np.int64)
    b_hd = sp.csr_matrix(
        (np.ones(num_h, dtype=np.float32), (rows_hd, cols_hd)),
        shape=(num_h, len(passage_to_col)),
    )

    deg_h = (np.asarray(b_he.sum(axis=1)).ravel() + np.asarray(b_hd.sum(axis=1)).ravel()).astype(np.float32)
    deg_e = np.asarray(b_he.sum(axis=0)).ravel().astype(np.float32)
    deg_d = np.asarray(b_hd.sum(axis=0)).ravel().astype(np.float32)

    inv_h = np.divide(1.0, deg_h, out=np.zeros_like(deg_h), where=deg_h > 0)
    inv_e = np.divide(1.0, deg_e, out=np.zeros_like(deg_e), where=deg_e > 0)
    inv_d = np.divide(1.0, deg_d, out=np.zeros_like(deg_d), where=deg_d > 0)
    d_h_inv = sp.diags(inv_h.astype(np.float32))
    d_e_inv = sp.diags(inv_e.astype(np.float32))
    d_d_inv = sp.diags(inv_d.astype(np.float32))

    adjacency = d_h_inv @ (b_he @ d_e_inv @ b_he.T + b_hd @ d_d_inv @ b_hd.T)
    adjacency = adjacency.tocsr().astype(np.float32)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    row_inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0).astype(np.float32)
    return (sp.diags(row_inv) @ adjacency).tocsr().astype(np.float32)


def required_artifacts_exist(config: HyperProveConfig, expected_manifest: dict[str, Any]) -> bool:
    required = [
        "passages.json",
        "entities.json",
        "hyperedges.json",
        "entity_embeddings.npy",
        "hyperedge_embeddings.npy",
        "structural_transition.npz",
        "index_state.json",
    ]
    files_exist = all(os.path.exists(os.path.join(config.artifacts_dir, name)) for name in required)
    return files_exist and manifest_matches(config.index_manifest_path, expected_manifest)


def build_index(
    config: HyperProveConfig,
    passages: list[dict[str, Any]],
    entity_results: list[dict[str, Any]],
    hyperedge_results: list[dict[str, Any]],
) -> dict[str, Any]:
    write_json(f"{config.artifacts_dir}/passages.json", passages)
    entities, entity_text_to_item = _build_entity_catalog(entity_results, hyperedge_results)
    hyperedges = _build_hyperedge_catalog(hyperedge_results, entity_text_to_item)

    entity_embeddings = _load_or_embed(
        f"{config.artifacts_dir}/entity_embeddings.npy",
        [item["text"] for item in entities],
    ) if entities else np.zeros((0, 0), dtype=np.float32)
    hyperedge_embeddings = _load_or_embed(
        f"{config.artifacts_dir}/hyperedge_embeddings.npy",
        [item["text"] for item in hyperedges],
    ) if hyperedges else np.zeros((0, 0), dtype=np.float32)

    synonym_map = _compute_synonym_map(entities, entity_embeddings, config.synonym_threshold)
    log_event(
        "indexing",
        "synonym_map_completed",
        entity_count=len(entities),
        synonym_links=sum(len(items) for items in synonym_map.values()),
        threshold=config.synonym_threshold,
    )
    transition = build_structural_transition(hyperedges, synonym_map)
    _save_sparse(f"{config.artifacts_dir}/structural_transition.npz", transition)

    state = {
        "passages": passages,
        "entity_catalog": entities,
        "hyperedges": hyperedges,
        "synonym_map": synonym_map,
    }
    write_json(f"{config.artifacts_dir}/index_state.json", state)
    expected_manifest = index_manifest(config, len(passages))
    write_manifest(config.index_manifest_path, expected_manifest)
    log_event(
        "indexing",
        "index_build_completed",
        passages=len(passages),
        entities=len(entities),
        hyperedges=len(hyperedges),
        transition_shape=list(transition.shape),
        transition_nnz=int(transition.nnz),
    )
    return load_index(config)


def load_index(config: HyperProveConfig) -> dict[str, Any]:
    state = read_json(f"{config.artifacts_dir}/index_state.json")
    entity_embeddings = np.load(f"{config.artifacts_dir}/entity_embeddings.npy")
    hyperedge_embeddings = np.load(f"{config.artifacts_dir}/hyperedge_embeddings.npy")
    transition = sp.load_npz(f"{config.artifacts_dir}/structural_transition.npz").tocsr().astype(np.float32)
    hyperedges = state["hyperedges"]
    entities = state["entity_catalog"]
    passages = state["passages"]

    entity_to_hyperedges: dict[str, list[int]] = defaultdict(list)
    passage_to_hyperedges: dict[int, list[int]] = defaultdict(list)
    for index, hyperedge in enumerate(hyperedges):
        passage_to_hyperedges[int(hyperedge["passage_idx"])].append(index)
        expanded_ids = _expanded_entities(list(hyperedge.get("entity_ids", [])), state.get("synonym_map", {}))
        for entity_id in expanded_ids:
            entity_to_hyperedges[str(entity_id)].append(index)

    return {
        "passages": passages,
        "entities": entities,
        "hyperedges": hyperedges,
        "synonym_map": state.get("synonym_map", {}),
        "entity_embeddings": entity_embeddings,
        "hyperedge_embeddings": hyperedge_embeddings,
        "structural_transition": transition,
        "entity_id_to_item": {item["entity_id"]: item for item in entities},
        "entity_text_to_id": {item["text"]: item["entity_id"] for item in entities},
        "entity_to_hyperedges": dict(entity_to_hyperedges),
        "passage_idx_to_item": {int(item["idx"]): item for item in passages},
        "passage_to_hyperedges": dict(passage_to_hyperedges),
    }
