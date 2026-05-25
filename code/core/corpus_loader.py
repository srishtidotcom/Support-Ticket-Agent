from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from core.document import Document


class CorpusLoader:
    """Load the challenge corpus from the repo's data directories."""

    ALLOWED_EXTENSIONS = {".md", ".txt", ".html", ".htm"}

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.data_dirs = [
            self.repo_root / "data" / "devplatform",
            self.repo_root / "data" / "claude",
            self.repo_root / "data" / "visa",
        ]

    def load_documents(self) -> List[Document]:
        documents: List[Document] = []

        for file_path in self._iter_files():
            relative_path = file_path.relative_to(self.repo_root).as_posix()
            company = self._company_from_relative_path(relative_path)
            content = file_path.read_text(encoding="utf-8", errors="replace")
            last_modified = datetime.fromtimestamp(
                file_path.stat().st_mtime, tz=timezone.utc
            )

            documents.append(
                Document(
                    id=self._stable_document_id(relative_path),
                    filepath=relative_path,
                    company=company,
                    title=self._clean_title(file_path),
                    content=content,
                    metadata={
                        "source_company": company,
                        "extension": file_path.suffix.lower(),
                    },
                    last_modified=last_modified,
                    word_count=len(content.split()),
                )
            )

        print(f"Loaded {len(documents)} documents")
        return documents

    def _iter_files(self) -> Iterable[Path]:
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                continue
            for file_path in sorted(data_dir.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in self.ALLOWED_EXTENSIONS:
                    yield file_path

    @staticmethod
    def _stable_document_id(relative_path: str) -> str:
        return hashlib.md5(relative_path.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_title(file_path: Path) -> str:
        stem = file_path.stem.replace("_", " ").replace("-", " ")
        return " ".join(stem.split()).strip().title()

    @staticmethod
    def _company_from_relative_path(relative_path: str) -> str:
        parts = Path(relative_path).parts
        if len(parts) >= 2 and parts[0] == "data":
            return parts[1]
        raise ValueError(f"Unable to infer company from path: {relative_path}")