from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi

from core.chunker import DocumentChunker
from core.corpus_loader import CorpusLoader
from core.embedder import Embedder


@dataclass
class HybridRetriever:
    """Stub container for the hybrid index artifacts."""

    faiss_index: faiss.Index
    bm25_index: BM25Okapi
    chunk_metadata: List[dict]
    embeddings: np.ndarray


def build_full_index() -> HybridRetriever:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    repo_root = Path(__file__).resolve().parents[1]
    index_dir = repo_root / "code" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    loader = CorpusLoader(repo_root=repo_root)
    documents = loader.load_documents()

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)

    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)
    embedder.save_artifacts(chunks, embeddings, index_dir)

    faiss_index = _build_faiss_index(embeddings)
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))

    tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    with (index_dir / "bm25_index.pkl").open("wb") as handle:
        pickle.dump(bm25_index, handle)

    chunk_metadata = [chunk.dict(exclude={"embedding"}) for chunk in chunks]
    return HybridRetriever(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunk_metadata=chunk_metadata,
        embeddings=embeddings,
    )


def load_hybrid_index(index_dir: Optional[Path] = None) -> HybridRetriever:
    repo_root = Path(__file__).resolve().parents[1]
    resolved_index_dir = index_dir or (repo_root / "code" / "index")

    faiss_index = faiss.read_index(str(resolved_index_dir / "faiss.index"))
    with (resolved_index_dir / "bm25_index.pkl").open("rb") as handle:
        bm25_index = pickle.load(handle)
    with (resolved_index_dir / "chunks_metadata.pkl").open("rb") as handle:
        chunk_metadata = pickle.load(handle)
    embeddings = np.load(resolved_index_dir / "embeddings.npy")

    return HybridRetriever(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunk_metadata=chunk_metadata,
        embeddings=embeddings,
    )


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.size == 0:
        raise ValueError("Cannot build a FAISS index without embeddings")

    vector_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(vector_dim)
    index.add(np.asarray(embeddings, dtype=np.float32))
    return index


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


if __name__ == "__main__":
    build_full_index()