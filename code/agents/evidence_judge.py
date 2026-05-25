from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Sequence

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import OLLAMA_MODEL_ROUTING, TEMPERATURE, SEED


class EvidenceResult(BaseModel):
	has_sufficient_evidence: bool
	confidence: float = Field(ge=0.0, le=1.0)
	reasoning: str
	recommended_action: Literal["reply", "escalate", "ask_clarification"]
	top_sources: List[str]
	evidence_quality_score: float = Field(ge=0.0, le=1.0)


class EvidenceJudge:
	"""Assess if retrieved evidence is enough to safely answer the ticket.

	This component explicitly checks sufficiency, consistency, action support,
	and hallucination risk before allowing a direct reply path.
	"""

	def __init__(self, model_name: str = OLLAMA_MODEL_ROUTING) -> None:
		self.model_name = model_name

	def evaluate(
		self,
		ticket: Dict[str, Any],
		retrieved_chunks: List[Dict[str, Any]],
		classification: Dict[str, Any],
	) -> EvidenceResult:
		if not retrieved_chunks:
			return EvidenceResult(
				has_sufficient_evidence=False,
				confidence=0.1,
				reasoning="No supporting documents were retrieved, so answering now would carry a high hallucination risk.",
				recommended_action="ask_clarification",
				top_sources=[],
				evidence_quality_score=0.0,
			)

		llm = self._build_llm()
		if llm is not None:
			llm_result = self._evaluate_with_llm(ticket=ticket, retrieved_chunks=retrieved_chunks, classification=classification, llm=llm)
			if llm_result is not None:
				return llm_result

		return self._evaluate_with_rules(ticket=ticket, retrieved_chunks=retrieved_chunks, classification=classification)

	def _evaluate_with_llm(
		self,
		ticket: Dict[str, Any],
		retrieved_chunks: List[Dict[str, Any]],
		classification: Dict[str, Any],
		llm: Any,
	) -> EvidenceResult | None:
		prompt = ChatPromptTemplate.from_messages(
			[
				(
					"system",
					"You are an evidence quality judge for support operations. Evaluate relevance, consistency, action support, and hallucination risk. Return only the structured result.",
				),
				(
					"human",
					"Ticket Subject: {subject}\n"
					"Company: {company}\n"
					"Product Area: {product_area}\n"
					"Latest Request: {latest_request}\n\n"
					"Retrieved Evidence:\n{evidence}\n\n"
					"You must answer these explicitly in your reasoning:\n"
					"1) Is the evidence strong and relevant?\n"
					"2) Are there conflicting statements?\n"
					"3) Does evidence directly support requested action?\n"
					"4) Hallucination risk if we reply now?\n\n"
					"Decision philosophy:\n"
					"- Strong, consistent, directly relevant -> reply\n"
					"- Weak, conflicting, or missing -> escalate\n"
					"- Partial evidence -> ask_clarification\n",
				),
			]
		)

		evidence_blob = self._format_evidence(retrieved_chunks)
		try:
			chain = prompt | llm.with_structured_output(EvidenceResult)
			result = chain.invoke(
				{
					"subject": str(ticket.get("subject", "") or ""),
					"company": str(classification.get("company", "") or ""),
					"product_area": str(classification.get("product_area", "") or ""),
					"latest_request": self._latest_user_request(ticket),
					"evidence": evidence_blob,
				}
			)
			verdict = self._coerce_result(result)
			verdict.top_sources = self._top_sources(retrieved_chunks)
			verdict.confidence = self._clamp(verdict.confidence)
			verdict.evidence_quality_score = self._clamp(verdict.evidence_quality_score)
			return verdict
		except Exception:
			return None

	def _evaluate_with_rules(
		self,
		ticket: Dict[str, Any],
		retrieved_chunks: List[Dict[str, Any]],
		classification: Dict[str, Any],
	) -> EvidenceResult:
		top_sources = self._top_sources(retrieved_chunks)
		texts = [str(chunk.get("text", "")) for chunk in retrieved_chunks]
		avg_score = sum(float(chunk.get("score", 0.0)) for chunk in retrieved_chunks) / max(1, len(retrieved_chunks))

		conflict_found = self._has_conflicts(texts)
		direct_support = self._direct_support_score(ticket=ticket, texts=texts)
		relevance = self._relevance_score(classification=classification, texts=texts)

		quality = self._clamp((0.45 * avg_score) + (0.30 * direct_support) + (0.25 * relevance))
		if conflict_found:
			quality = self._clamp(quality - 0.25)

		if quality >= 0.72 and direct_support >= 0.6 and not conflict_found:
			recommended_action: Literal["reply", "escalate", "ask_clarification"] = "reply"
			sufficient = True
			confidence = self._clamp(0.78 + 0.18 * quality)
			hallucination_risk = "low"
		elif quality >= 0.45 and not conflict_found:
			recommended_action = "ask_clarification"
			sufficient = False
			confidence = self._clamp(0.45 + 0.30 * quality)
			hallucination_risk = "medium"
		else:
			recommended_action = "escalate"
			sufficient = False
			confidence = self._clamp(0.55 + 0.25 * max(0.2, quality))
			hallucination_risk = "high"

		reasoning = (
			f"Evidence relevance={'strong' if relevance >= 0.6 else 'limited'}, "
			f"conflicts={'detected' if conflict_found else 'not detected'}, "
			f"direct_action_support={direct_support:.2f}, "
			f"hallucination_risk={hallucination_risk}."
		)

		return EvidenceResult(
			has_sufficient_evidence=sufficient,
			confidence=confidence,
			reasoning=reasoning,
			recommended_action=recommended_action,
			top_sources=top_sources,
			evidence_quality_score=quality,
		)

	@staticmethod
	def _top_sources(retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
		seen = set()
		sources: List[str] = []
		for chunk in retrieved_chunks:
			filepath = str(chunk.get("filepath", "")).strip()
			if filepath and filepath not in seen:
				seen.add(filepath)
				sources.append(filepath)
		return sources[:5]

	@staticmethod
	def _format_evidence(chunks: List[Dict[str, Any]]) -> str:
		sections = []
		for idx, chunk in enumerate(chunks[:5], start=1):
			text = str(chunk.get("text", "")).strip().replace("\n", " ")
			truncated = (text[:700] + "...") if len(text) > 700 else text
			sections.append(
				f"[{idx}] source={chunk.get('filepath', '')} score={chunk.get('score', 0.0):.4f}\n{truncated}"
			)
		return "\n\n".join(sections)

	@staticmethod
	def _latest_user_request(ticket: Dict[str, Any]) -> str:
		issue = ticket.get("issue", "")
		parsed = issue
		if isinstance(issue, str):
			try:
				parsed = json.loads(issue)
			except Exception:
				parsed = issue

		if isinstance(parsed, list):
			for message in reversed(parsed):
				if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
					return str(message.get("content", "")).strip()
			return str(parsed[-1]) if parsed else ""
		if isinstance(parsed, dict):
			return str(parsed.get("content", "")).strip()
		return str(parsed or "")

	@staticmethod
	def _has_conflicts(texts: Sequence[str]) -> bool:
		if len(texts) < 2:
			return False

		positive_markers = ("can", "supported", "available", "allowed", "enabled")
		negative_markers = ("cannot", "not supported", "unavailable", "not allowed", "disabled")

		has_positive = any(any(marker in text.lower() for marker in positive_markers) for text in texts)
		has_negative = any(any(marker in text.lower() for marker in negative_markers) for text in texts)
		return has_positive and has_negative

	@staticmethod
	def _direct_support_score(ticket: Dict[str, Any], texts: Sequence[str]) -> float:
		request = EvidenceJudge._latest_user_request(ticket).lower()
		if not request:
			return 0.0

		action_terms = []
		for keyword in [
			"refund",
			"dispute",
			"cancel",
			"delete",
			"unlock",
			"restore",
			"access",
			"pricing",
			"subscription",
			"api",
			"billing",
			"policy",
		]:
			if keyword in request:
				action_terms.append(keyword)

		if not action_terms:
			action_terms = [token for token in request.split() if len(token) > 4][:5]

		text_blob = "\n".join(texts).lower()
		matches = sum(1 for term in set(action_terms) if term in text_blob)
		return matches / max(1, len(set(action_terms)))

	@staticmethod
	def _relevance_score(classification: Dict[str, Any], texts: Sequence[str]) -> float:
		tokens = []
		tokens.extend(str(classification.get("company", "")).lower().split())
		tokens.extend(str(classification.get("product_area", "")).lower().split("_"))
		tokens = [token for token in tokens if token]
		if not tokens:
			return 0.5

		text_blob = "\n".join(texts).lower()
		matches = sum(1 for token in set(tokens) if token in text_blob)
		return matches / max(1, len(set(tokens)))

	@staticmethod
	def _coerce_result(value: Any) -> EvidenceResult:
		if isinstance(value, EvidenceResult):
			return value
		if hasattr(EvidenceResult, "model_validate"):
			return EvidenceResult.model_validate(value)
		return EvidenceResult.parse_obj(value)

	@staticmethod
	def _clamp(value: float) -> float:
		return round(max(0.0, min(1.0, float(value))), 4)

	def _build_llm(self) -> Any:
		try:
			from langchain_ollama import ChatOllama

			return ChatOllama(model=self.model_name, temperature=TEMPERATURE, seed=SEED)
		except Exception:
			return None

