from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from utils.config import HyperProveConfig
from utils.metrics import normalize_answer


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0 or k <= 0:
        return np.asarray([], dtype=np.int64)
    k = min(int(k), scores.size)
    if k == scores.size:
        return np.argsort(scores)[::-1]
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


class HypergraphRetriever:
    def __init__(self, config: HyperProveConfig, index_state: dict[str, Any]) -> None:
        self.config = config
        self.index_state = index_state
        self.hyperedges: list[dict[str, Any]] = index_state["hyperedges"]
        self.passages: dict[int, dict[str, Any]] = index_state["passage_idx_to_item"]
        self.hyperedge_embeddings = np.asarray(index_state["hyperedge_embeddings"], dtype=np.float32)
        self.t_s: sp.csr_matrix = index_state["structural_transition"].tocsr().astype(np.float32)
        self.t_s_t: sp.csr_matrix = self.t_s.T.tocsr()
        self.entity_id_to_item: dict[str, dict[str, Any]] = index_state["entity_id_to_item"]
        self.entity_text_to_id: dict[str, str] = index_state["entity_text_to_id"]
        self.entity_to_hyperedges: dict[str, list[int]] = index_state["entity_to_hyperedges"]

    @property
    def size(self) -> int:
        return len(self.hyperedges)

    def dense_topk(self, query_embedding: np.ndarray, k: int, exclude: set[int] | None = None) -> list[int]:
        if self.size == 0 or k <= 0:
            return []
        exclude = exclude or set()
        scores = self.hyperedge_embeddings @ query_embedding
        fetched = top_k_indices(scores, min(scores.size, k + len(exclude)))
        out: list[int] = []
        for idx in fetched.tolist():
            idx = int(idx)
            if idx in exclude:
                continue
            out.append(idx)
            if len(out) >= k:
                break
        return out

    def find_hyperedges_by_entities(self, entity_texts: list[str], entity_ids: list[str] | None = None) -> list[int]:
        ids = set(entity_ids or [])
        normalized_texts = [normalize_answer(text) for text in entity_texts if normalize_answer(text)]
        for text, entity_id in self.entity_text_to_id.items():
            norm = normalize_answer(text)
            if any(norm == item or norm in item or item in norm for item in normalized_texts):
                ids.add(entity_id)
        candidates: list[int] = []
        for entity_id in ids:
            candidates.extend(int(idx) for idx in self.entity_to_hyperedges.get(entity_id, []))
        return list(dict.fromkeys(candidates))

    def retrieve_hop(
        self,
        query_embedding: np.ndarray,
        seed_indices: list[int],
    ) -> dict[str, Any]:
        seeds = [idx for idx in dict.fromkeys(int(i) for i in seed_indices) if 0 <= idx < self.size]
        if not seeds:
            seeds = self.dense_topk(query_embedding, self.config.kd)
        local_nodes = self._local_subgraph(seeds)
        pi = self._rank_local(query_embedding, seeds, local_nodes)
        top_h = top_k_indices(pi, min(self.config.top_hyperedges, pi.size))
        top_hyperedges = [self._hyperedge_payload(int(idx), float(pi[int(idx)])) for idx in top_h if pi[int(idx)] > 0]
        top_passages = self._top_passages(pi, top_h, self.config.qa_top_k)
        return {
            "seed_indices": seeds,
            "local_nodes": local_nodes.tolist(),
            "scores": pi,
            "top_hyperedges": top_hyperedges,
            "top_passages": top_passages,
        }

    def _hyperedge_payload(self, index: int, score: float) -> dict[str, Any]:
        item = self.hyperedges[index]
        return {
            "index": index,
            "hyperedge_id": item["hyperedge_id"],
            "text": item["text"],
            "entities": item.get("entities", []),
            "entity_ids": item.get("entity_ids", []),
            "passage_idx": int(item["passage_idx"]),
            "score": score,
        }

    def _local_subgraph(self, seeds: list[int]) -> np.ndarray:
        if self.size == 0:
            return np.asarray([], dtype=np.int64)
        target_size = int(self.config.local_subgraph_size)
        if target_size <= 0 or target_size >= self.size:
            return np.arange(self.size, dtype=np.int64)
        u = np.zeros(self.size, dtype=np.float32)
        for seed in seeds:
            u[seed] = 1.0 / len(seeds)
        pi = u.copy()
        for _ in range(int(self.config.ppr_max_iter)):
            new_pi = (1.0 - self.config.gamma) * u + self.config.gamma * np.asarray(self.t_s_t @ pi).ravel()
            if np.linalg.norm(new_pi - pi, ord=1) < 1e-6:
                pi = new_pi
                break
            pi = new_pi
        selected: list[int] = []
        seen: set[int] = set()
        for seed in seeds:
            if seed not in seen:
                selected.append(seed)
                seen.add(seed)
                if len(selected) >= target_size:
                    return np.asarray(sorted(selected), dtype=np.int64)
        for idx in top_k_indices(pi, min(target_size, self.size)).tolist():
            idx = int(idx)
            if idx in seen:
                continue
            selected.append(idx)
            seen.add(idx)
            if len(selected) >= target_size:
                break
        return np.asarray(sorted(selected), dtype=np.int64)

    def _rank_local(
        self,
        query_embedding: np.ndarray,
        seeds: list[int],
        local_nodes: np.ndarray,
    ) -> np.ndarray:
        full = np.zeros(self.size, dtype=np.float32)
        if local_nodes.size == 0:
            return full
        node_to_local = {int(node): pos for pos, node in enumerate(local_nodes.tolist())}
        local_seeds = [node_to_local[seed] for seed in seeds if seed in node_to_local]
        if not local_seeds:
            return full

        t_s_raw = self.t_s[local_nodes, :][:, local_nodes].tocsr()
        t_s_star = self._row_normalize(t_s_raw)
        c = (self.hyperedge_embeddings @ query_embedding).astype(np.float32)
        c_local = c[local_nodes]
        t_q_star = self._query_biased_transition(t_s_raw, c_local)
        alpha = max(0.0, min(1.0, float(self.config.alpha)))
        # A row with no neighbour above theta has an all-zero T_q row; it relies on the
        # structural transition alone (weight 1), so M stays row-stochastic and PPR keeps its mass.
        gated = np.asarray(t_q_star.sum(axis=1)).ravel() > 0
        row_alpha = np.where(gated, alpha, 1.0).astype(np.float32)
        m = (sp.diags(row_alpha) @ t_s_star + sp.diags(1.0 - row_alpha) @ t_q_star).tocsr()

        u = np.zeros(local_nodes.size, dtype=np.float32)
        for seed in local_seeds:
            u[seed] = 1.0 / len(local_seeds)
        pi = u.copy()
        m_t = m.T.tocsr()
        for _ in range(int(self.config.ppr_max_iter)):
            new_pi = (1.0 - self.config.gamma) * u + self.config.gamma * np.asarray(m_t @ pi).ravel()
            if np.linalg.norm(new_pi - pi, ord=1) < 1e-6:
                pi = new_pi
                break
            pi = new_pi
        full[local_nodes] = pi
        return full

    def _query_biased_transition(self, structural: sp.csr_matrix, c_local: np.ndarray) -> sp.csr_matrix:
        if structural.nnz == 0:
            return sp.csr_matrix(structural.shape, dtype=np.float32)
        tau = max(float(self.config.tau), 1e-6)
        theta = float(self.config.theta)
        data = np.zeros(structural.nnz, dtype=np.float32)
        indices = structural.indices.copy()
        indptr = structural.indptr.copy()
        for row in range(structural.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            if start == end:
                continue
            cols = indices[start:end]
            mask = c_local[cols] > theta
            if not mask.any():
                continue
            weights = np.exp(c_local[cols[mask]] / tau).astype(np.float32)
            weights = weights / max(float(weights.sum()), 1e-12)
            row_data = np.zeros(end - start, dtype=np.float32)
            row_data[mask] = weights
            data[start:end] = row_data
        return sp.csr_matrix((data, indices, indptr), shape=structural.shape, dtype=np.float32)

    @staticmethod
    def _row_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        row_inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0).astype(np.float32)
        return (sp.diags(row_inv) @ matrix).tocsr().astype(np.float32)

    def _top_passages(self, pi: np.ndarray, nodes: np.ndarray, k: int) -> list[dict[str, Any]]:
        best: dict[int, float] = {}
        for idx in nodes.tolist():
            if idx < 0 or idx >= self.size:
                continue
            score = float(pi[idx])
            if score <= 0:
                continue
            passage_idx = int(self.hyperedges[idx]["passage_idx"])
            if passage_idx not in best or best[passage_idx] < score:
                best[passage_idx] = score
        ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            {
                "idx": int(passage_idx),
                "title": self.passages[passage_idx]["title"],
                "text": self.passages[passage_idx]["text"],
                "score": float(score),
            }
            for passage_idx, score in ranked
        ]
