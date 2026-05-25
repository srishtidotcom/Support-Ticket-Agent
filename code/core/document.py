from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
	id: str
	filepath: str
	company: str
	title: str
	content: str
	metadata: Dict[str, Any] = Field(default_factory=dict)
	last_modified: Optional[datetime] = None
	word_count: int


class DocumentChunk(BaseModel):
	chunk_id: str
	document_id: str
	filepath: str
	company: str
	text: str
	embedding: Optional[List[float]] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)
