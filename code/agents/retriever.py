from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import OLLAMA_MODEL_ROUTING, TEMPERATURE, SEED
from core.hybrid_retriever import HybridRetriever


class RetrievalQueryPlan(BaseModel):
	"""Structured retrieval-plan output for deterministic query generation."""

	conversation_summary: str
	intent: str
	key_entities: List[str] = Field(default_factory=list)
	queries: List[str] = Field(default_factory=list)


class RetrievalAgent:
	"""Generate optimized retrieval queries and return top evidence chunks.

	The agent keeps reranking lightweight and deterministic with score fusion to
	avoid introducing unnecessary LLM variance in the hot path.
	"""

	def __init__(self, model_name: str = OLLAMA_MODEL_ROUTING) -> None:
		self.model_name = model_name
		self.hybrid_retriever = HybridRetriever()

	def retrieve(
		self,
		ticket: Dict[str, Any],
		classification: Dict[str, Any],
		top_k: int = 5,
	) -> List[Dict[str, Any]]:
		"""Retrieve and rerank evidence chunks for the given ticket."""

		query_plan = self._build_query_plan(ticket=ticket, classification=classification)
		company_filter = self._normalize_company(classification.get("company", ""))

		candidate_chunks: List[Dict[str, Any]] = []
		for query in query_plan.queries[:2]:
			chunks = self.hybrid_retriever.retrieve(query=query, top_k=8, company_filter=company_filter)
			for chunk in chunks:
				hybrid_score = float(chunk.metadata.get("hybrid_score", 0.0))
				candidate_chunks.append(
					{
						"text": chunk.text,
						"filepath": chunk.filepath,
						"company": chunk.company,
						"hybrid_score": hybrid_score,
						"query": query,
					}
				)

		reranked = self._rerank_candidates(
			candidates=candidate_chunks,
			query_plan=query_plan,
			classification=classification,
			top_k=top_k,
		)
		return reranked

	def _build_query_plan(self, ticket: Dict[str, Any], classification: Dict[str, Any]) -> RetrievalQueryPlan:
		history = self._normalize_issue_messages(ticket.get("issue", ""))
		latest_user_message = self._latest_user_message(history)

		llm = self._build_llm()
		if llm is not None:
			prompt = ChatPromptTemplate.from_messages(
				[
					(
						"system",
						"You optimize support retrieval queries. Summarize conversation context, extract intent and entities, and output 1-2 precise retrieval queries for documentation lookup.",
					),
					(
						"human",
						"Subject: {subject}\n"
						"Company: {company}\n"
						"Product Area: {product_area}\n"
						"Conversation History:\n{history}\n\n"
						"Latest User Message:\n{latest_user_message}\n\n"
						"Requirements:\n"
						"- Produce exactly 1 or 2 retrieval queries.\n"
						"- Include company and product area clues.\n"
						"- Keep each query concise and grounded in user intent.\n",
					),
				]
			)

			try:
				chain = prompt | llm.with_structured_output(RetrievalQueryPlan)
				result = chain.invoke(
					{
						"subject": str(ticket.get("subject", "") or ""),
						"company": str(classification.get("company", "") or ""),
						"product_area": str(classification.get("product_area", "") or ""),
						"history": history,
						"latest_user_message": latest_user_message,
					}
				)
				plan = self._coerce_plan(result)
				if plan.queries:
					plan.queries = plan.queries[:2]
					return plan
			except Exception:
				pass

		return self._build_rule_plan(ticket=ticket, classification=classification, history=history, latest_user_message=latest_user_message)

	def _build_rule_plan(
		self,
		ticket: Dict[str, Any],
		classification: Dict[str, Any],
		history: str,
		latest_user_message: str,
	) -> RetrievalQueryPlan:
		subject = str(ticket.get("subject", "") or "").strip()
		company = str(classification.get("company", "") or "").strip()
		product_area = str(classification.get("product_area", "") or "").strip()
		intent = self._infer_intent(subject=subject, latest_user_message=latest_user_message)
		entities = self._extract_entities(latest_user_message)

		query_1 = " ".join(part for part in [company, product_area, intent, subject] if part).strip()
		query_2 = " ".join(part for part in [company, product_area, " ".join(entities[:4])] if part).strip()
		queries = [q for q in [query_1, query_2] if q]

		summary = self._summarize_history(history, latest_user_message)
		return RetrievalQueryPlan(
			conversation_summary=summary,
			intent=intent,
			key_entities=entities[:8],
			queries=queries[:2] or [latest_user_message or subject or "general support"],
		)

	def _rerank_candidates(
		self,
		candidates: List[Dict[str, Any]],
		query_plan: RetrievalQueryPlan,
		classification: Dict[str, Any],
		top_k: int,
	) -> List[Dict[str, Any]]:
		if not candidates:
			return []

		topic_terms = self._build_topic_terms(query_plan, classification)
		per_path_best: Dict[str, Dict[str, Any]] = {}

		for candidate in candidates:
			text = str(candidate.get("text", ""))
			filepath = str(candidate.get("filepath", ""))
			if not filepath:
				continue

			lexical_coverage = self._coverage_score(text=text, terms=topic_terms)
			hybrid_score = float(candidate.get("hybrid_score", 0.0))

			# Blend lexical alignment with hybrid retrieval score.
			final_score = (0.7 * hybrid_score) + (0.3 * lexical_coverage)
			merged = {
				"text": text,
				"filepath": filepath,
				"company": candidate.get("company", classification.get("company", "unknown")),
				"score": round(float(final_score), 6),
			}

			if filepath not in per_path_best or float(per_path_best[filepath]["score"]) < final_score:
				per_path_best[filepath] = merged

		ranked = sorted(per_path_best.values(), key=lambda item: item["score"], reverse=True)
		return ranked[:top_k]

	@staticmethod
	def _build_topic_terms(query_plan: RetrievalQueryPlan, classification: Dict[str, Any]) -> List[str]:
		terms = []
		terms.extend(query_plan.intent.lower().split())
		terms.extend(entity.lower() for entity in query_plan.key_entities)
		terms.extend(str(classification.get("company", "")).lower().split())
		terms.extend(str(classification.get("product_area", "")).lower().split("_"))
		return [term for term in terms if len(term) > 2]

	@staticmethod
	def _coverage_score(text: str, terms: Sequence[str]) -> float:
		if not text or not terms:
			return 0.0
		lowered = text.lower()
		unique_terms = sorted(set(terms))
		matches = sum(1 for term in unique_terms if term in lowered)
		return matches / max(1, len(unique_terms))

	@staticmethod
	def _normalize_issue_messages(issue: Any) -> str:
		if isinstance(issue, str):
			parsed = RetrievalAgent._maybe_parse_json(issue)
			if parsed is not issue:
				issue = parsed

		if isinstance(issue, list):
			return "\n".join(RetrievalAgent._message_to_text(item) for item in issue if item)
		if isinstance(issue, dict):
			return RetrievalAgent._message_to_text(issue)
		return str(issue or "")

	@staticmethod
	def _latest_user_message(history_text: str) -> str:
		lines = [line.strip() for line in history_text.splitlines() if line.strip()]
		for line in reversed(lines):
			if line.lower().startswith("user:"):
				return line.split(":", 1)[1].strip()
		return lines[-1] if lines else ""

	@staticmethod
	def _summarize_history(history: str, latest_user_message: str) -> str:
		lines = [line.strip() for line in history.splitlines() if line.strip()]
		condensed = " ".join(lines[-6:])
		summary = condensed[:420].strip()
		if latest_user_message and latest_user_message not in summary:
			summary = f"{summary} | latest: {latest_user_message}".strip(" |")
		return summary or latest_user_message[:420]

	@staticmethod
	def _extract_entities(text: str) -> List[str]:
		tokens = re.findall(r"[A-Za-z0-9_./-]+", text or "")
		frequencies: Dict[str, int] = defaultdict(int)
		for token in tokens:
			if len(token) < 4:
				continue
			normalized = token.strip().lower()
			if normalized in {"http", "https", "with", "that", "this", "from", "your", "have", "please"}:
				continue
			frequencies[normalized] += 1
		ranked = sorted(frequencies.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
		return [token for token, _ in ranked[:12]]

	@staticmethod
	def _infer_intent(subject: str, latest_user_message: str) -> str:
		text = f"{subject} {latest_user_message}".lower()
		if any(marker in text for marker in ["refund", "chargeback", "dispute"]):
			return "resolve refund or dispute request"
		if any(marker in text for marker in ["cannot", "can't", "unable", "not working", "error", "failed"]):
			return "troubleshoot product issue"
		if any(marker in text for marker in ["how", "where", "what", "guide", "steps"]):
			return "provide policy or procedural guidance"
		return "resolve customer support request"

	@staticmethod
	def _message_to_text(message: Any) -> str:
		if not isinstance(message, dict):
			return str(message).strip()

		role = str(message.get("role", "user") or "user").strip()
		content = message.get("content", "")
		if isinstance(content, list):
			content_text = " ".join(str(item) for item in content)
		else:
			content_text = str(content)
		return f"{role}: {content_text.strip()}"

	@staticmethod
	def _maybe_parse_json(raw_text: str) -> Any:
		try:
			return json.loads(raw_text)
		except Exception:
			return raw_text

	@staticmethod
	def _coerce_plan(value: Any) -> RetrievalQueryPlan:
		if isinstance(value, RetrievalQueryPlan):
			return value
		if hasattr(RetrievalQueryPlan, "model_validate"):
			return RetrievalQueryPlan.model_validate(value)
		return RetrievalQueryPlan.parse_obj(value)

	@staticmethod
	def _normalize_company(company: str) -> str:
		lowered = (company or "").strip().lower()
		mapping = {
			"visa": "visa",
			"claude": "claude",
			"devplatform": "devplatform",
		}
		return mapping.get(lowered, lowered)

	def _build_llm(self) -> Any:
		try:
			from langchain_ollama import ChatOllama

			return ChatOllama(model=self.model_name, temperature=TEMPERATURE, seed=SEED)
		except Exception:
			return None

