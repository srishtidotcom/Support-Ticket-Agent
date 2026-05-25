from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from core.document import DocumentChunk


class Embedder:
    """Generate and persist chunk embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        if not texts:
            raise ValueError("Cannot embed an empty chunk list")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def save_artifacts(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
        index_dir: Path,
    ) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = index_dir / "chunks_metadata.pkl"
        embeddings_path = index_dir / "embeddings.npy"

        metadata = [chunk.dict(exclude={"embedding"}) for chunk in chunks]
        with metadata_path.open("wb") as handle:
            pickle.dump(metadata, handle)

        np.save(embeddings_path, embeddings)