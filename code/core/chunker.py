from __future__ import annotations

import hashlib
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.document import Document, DocumentChunk


class DocumentChunker:
    """Split documents into overlapping chunks for retrieval."""

    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=650,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []

        for document in documents:
            texts = self.splitter.split_text(document.content)
            total_chunks = len(texts)

            for chunk_index, text in enumerate(texts):
                chunk_id = self._stable_chunk_id(document.id, chunk_index)
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_count": total_chunks,
                    }
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.id,
                        filepath=document.filepath,
                        company=document.company,
                        text=text,
                        metadata=metadata,
                    )
                )

        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    @staticmethod
    def _stable_chunk_id(document_id: str, chunk_index: int) -> str:
        source = f"{document_id}:{chunk_index}".encode("utf-8")
        return hashlib.md5(source).hexdigest()