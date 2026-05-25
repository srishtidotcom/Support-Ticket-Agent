from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SafetyResult(BaseModel):
    """Structured safety verdict for a single ticket."""

    is_adversarial: bool
    attack_type: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class TicketClassification(BaseModel):
    """Structured routing decision for a support ticket."""

    company: str
    product_area: str
    request_type: str
    risk_level: str
    pii_detected: bool
    language: str
    confidence: float = Field(ge=0.0, le=1.0)