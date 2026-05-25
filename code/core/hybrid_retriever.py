from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config import INDEX_DIR
from core.document import DocumentChunk
from core.embedder import Embedder


@dataclass
class _HybridArtifacts:
    """Container for index artifacts loaded from disk once and reused."""

    faiss_index: faiss.IndexFlatIP
    bm25_index: BM25Okapi
    chunk_metadata: List[dict]
    embeddings: np.ndarray


class HybridRetriever:
    """Hybrid retriever combining dense FAISS and sparse BM25 ranking.

    Design notes:
    - Dense and sparse scores are normalized per query over the candidate pool to
      keep weights stable across different score scales.
    - Retrieval runs over a union of top dense and top sparse candidates for
      higher recall, then applies weighted fusion.
    - Filepath-level dedup is applied at the very end so we do not return many
      chunks from the same source document.
    """

    DENSE_WEIGHT = 0.65
    BM25_WEIGHT = 0.35

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.index_dir = self._resolve_index_dir(index_dir)
        self._artifacts = self._load_cached_artifacts(str(self.index_dir)) if self.index_dir else None
        self._embedder: Optional[Embedder] = None
        self._query_cache: Dict[Tuple[str, int, str], List[DocumentChunk]] = {}

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        company_filter: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Return top chunks by hybrid score with filepath-level deduplication."""

        normalized_query = (query or "").strip()
        if not normalized_query or top_k <= 0 or self._artifacts is None:
            return []

        cache_key = (normalized_query, top_k, (company_filter or "").strip().lower())
        if cache_key in self._query_cache:
            return [chunk.model_copy(deep=True) for chunk in self._query_cache[cache_key]]

        candidate_limit = min(max(top_k * 6, 24), len(self._artifacts.chunk_metadata))
        query_vector = self._embed_query(normalized_query)

        dense_scores, dense_indices = self._dense_candidates(query_vector, candidate_limit)
        sparse_scores_all = self._sparse_scores(normalized_query)
        sparse_indices = self._top_indices(sparse_scores_all, candidate_limit)

        candidate_indices = set(dense_indices.tolist()) | set(sparse_indices.tolist())
        if not candidate_indices:
            return []

        dense_map = self._build_dense_map(query_vector, dense_indices, dense_scores, candidate_indices)
        sparse_map = {idx: float(sparse_scores_all[idx]) for idx in candidate_indices}

        dense_norm = self._normalize_score_map(dense_map)
        sparse_norm = self._normalize_score_map(sparse_map)

        ranked = sorted(
            candidate_indices,
            key=lambda idx: self.DENSE_WEIGHT * dense_norm[idx] + self.BM25_WEIGHT * sparse_norm[idx],
            reverse=True,
        )

        company_filter_norm = (company_filter or "").strip().lower()
        seen_paths = set()
        results: List[DocumentChunk] = []

        for idx in ranked:
            metadata = self._artifacts.chunk_metadata[idx]
            filepath = str(metadata.get("filepath", ""))
            company = str(metadata.get("company", "")).lower()

            if company_filter_norm and company_filter_norm not in {company, company.title().lower()}:
                continue
            if filepath in seen_paths:
                continue

            fused_score = self.DENSE_WEIGHT * dense_norm[idx] + self.BM25_WEIGHT * sparse_norm[idx]
            chunk = DocumentChunk.model_validate(
                {
                    **metadata,
                    "embedding": None,
                    "metadata": {
                        **dict(metadata.get("metadata") or {}),
                        "hybrid_score": round(float(fused_score), 6),
                        "dense_score": round(float(dense_norm[idx]), 6),
                        "bm25_score": round(float(sparse_norm[idx]), 6),
                    },
                }
            )

            seen_paths.add(filepath)
            results.append(chunk)
            if len(results) >= top_k:
                break

        self._query_cache[cache_key] = [chunk.model_copy(deep=True) for chunk in results]
        return results

    @classmethod
    @lru_cache(maxsize=4)
    def _load_cached_artifacts(cls, index_dir: str) -> Optional[_HybridArtifacts]:
        """Disk-loading cache to avoid repeated FAISS/BM25 deserialization."""

        resolved = Path(index_dir)
        required_files = {
            "faiss": resolved / "faiss.index",
            "bm25": resolved / "bm25_index.pkl",
            "metadata": resolved / "chunks_metadata.pkl",
            "embeddings": resolved / "embeddings.npy",
        }
        if not all(path.exists() for path in required_files.values()):
            return None

        faiss_index = faiss.read_index(str(required_files["faiss"]))
        with required_files["bm25"].open("rb") as handle:
            bm25_index = pickle.load(handle)
        with required_files["metadata"].open("rb") as handle:
            chunk_metadata = pickle.load(handle)
        embeddings = np.load(required_files["embeddings"])

        return _HybridArtifacts(
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            chunk_metadata=chunk_metadata,
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )

    def _resolve_index_dir(self, explicit_index_dir: Optional[Path]) -> Optional[Path]:
        candidates: Sequence[Path] = [
            explicit_index_dir if explicit_index_dir is not None else Path(INDEX_DIR),
            self.repo_root / "code" / "index",
            self.repo_root / "index",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate_path = Path(candidate)
            if (candidate_path / "faiss.index").exists():
                return candidate_path
        return None

    def _embed_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            self._embedder = Embedder()

        vector = self._embedder.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32)

    def _dense_candidates(self, query_vector: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
        assert self._artifacts is not None
        scores, indices = self._artifacts.faiss_index.search(query_vector.reshape(1, -1), limit)
        return scores[0], indices[0]

    def _sparse_scores(self, query: str) -> np.ndarray:
        assert self._artifacts is not None
        tokens = [token for token in query.lower().split() if token]
        scores = self._artifacts.bm25_index.get_scores(tokens)
        return np.asarray(scores, dtype=np.float32)

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
        if scores.size == 0:
            return np.asarray([], dtype=np.int64)
        bounded = min(limit, scores.size)
        partition = np.argpartition(scores, -bounded)[-bounded:]
        return partition[np.argsort(scores[partition])[::-1]]

    def _build_dense_map(
        self,
        query_vector: np.ndarray,
        dense_indices: np.ndarray,
        dense_scores: np.ndarray,
        candidate_indices: set[int],
    ) -> Dict[int, float]:
        assert self._artifacts is not None

        dense_map: Dict[int, float] = {int(idx): float(score) for idx, score in zip(dense_indices, dense_scores) if int(idx) >= 0}
        missing = [idx for idx in candidate_indices if idx not in dense_map]
        if missing:
            missing_matrix = self._artifacts.embeddings[np.asarray(missing, dtype=np.int64)]
            recovered = missing_matrix @ query_vector
            dense_map.update({idx: float(score) for idx, score in zip(missing, recovered)})
        return dense_map

    @staticmethod
    def _normalize_score_map(score_map: Dict[int, float]) -> Dict[int, float]:
        if not score_map:
            return {}

        values = np.asarray(list(score_map.values()), dtype=np.float32)
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value - min_value < 1e-9:
            return {idx: 1.0 for idx in score_map}

        return {idx: (float(score) - min_value) / (max_value - min_value) for idx, score in score_map.items()}
