from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import OLLAMA_MODEL_ROUTING, TEMPERATURE, SEED
from core.hybrid_retriever import HybridRetriever
from agents.safety import SafetyAgent


class RetrievalQueryPlan(BaseModel):
	"""Structured retrieval-plan output for deterministic query generation."""

	conversation_summary: str
	intent: str
	key_entities: List[str] = Field(default_factory=list)
	queries: List[str] = Field(default_factory=list)


class CorpusSanitizer:
	"""Filter retrieved corpus chunks that look like indirect prompt injection."""

	_BLOCKED_ATTACK_TYPES = {
		"prompt_injection",
		"jailbreak",
		"policy_forgery",
		"policy_bypass",
		"role_hijack",
		"system_prompt_extraction",
		"output_field_manipulation",
	}

	def __init__(self, safety_agent: SafetyAgent | None = None) -> None:
		self.safety_agent = safety_agent or SafetyAgent()
		self.last_filtered_count = 0
		self.last_filtered_reasons: List[str] = []

	def sanitize(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		clean_chunks: List[Dict[str, Any]] = []
		filtered_reasons: List[str] = []

		for chunk in chunks:
			text = str(chunk.get("text", "") or "")
			verdict = self.safety_agent.assess_text(text, use_llm=False)
			attack_type = str(verdict.attack_type or "")
			if verdict.is_adversarial and attack_type in self._BLOCKED_ATTACK_TYPES:
				filtered_reasons.append(f"{chunk.get('filepath', 'unknown')}:{attack_type}")
				continue
			clean_chunks.append(chunk)

		self.last_filtered_count = len(chunks) - len(clean_chunks)
		self.last_filtered_reasons = filtered_reasons[:5]
		return clean_chunks


class RetrievalAgent:
	"""Generate optimized retrieval queries and return top evidence chunks.

	The agent keeps reranking lightweight and deterministic with score fusion to
	avoid introducing unnecessary LLM variance in the hot path.
	"""

	def __init__(self, model_name: str = OLLAMA_MODEL_ROUTING) -> None:
		self.model_name = model_name
		self.hybrid_retriever = HybridRetriever()
		self.corpus_sanitizer = CorpusSanitizer()
		self.last_sanitizer_filtered_count = 0
		self.last_second_pass_used = False

	def retrieve(
		self,
		ticket: Dict[str, Any],
		classification: Dict[str, Any],
		top_k: int = 5,
	) -> List[Dict[str, Any]]:
		"""Retrieve and rerank evidence chunks for the given ticket."""

		query_plan = self._build_query_plan(ticket=ticket, classification=classification)
		company_filter = self._normalize_company(classification.get("company", ""))
		subject = str(ticket.get("subject", "") or "").strip() or "unknown subject"
		print(
			f"[RetrievalAgent] Ticket={subject!r} queries={len(query_plan.queries[:2])} company_filter={company_filter or 'none'}"
		)

		candidate_chunks: List[Dict[str, Any]] = []
		queries = query_plan.queries[:2] or [subject]
		for query_index, query in enumerate(queries, start=1):
			print(f"[RetrievalAgent] Ticket={subject!r} query={query_index} text={query!r}")
			chunks = self.hybrid_retriever.retrieve(query=query, top_k=8, company_filter=company_filter)
			print(f"[RetrievalAgent] Ticket={subject!r} query={query_index} retrieved={len(chunks)} chunks")

			if query_index == 1 and not chunks:
				fallback_query = query_plan.queries[1] if len(query_plan.queries) > 1 else query
				print(
					f"[RetrievalAgent] Ticket={subject!r} query={query_index} fallback_text={fallback_query!r} company_filter=None"
				)
				chunks = self.hybrid_retriever.retrieve(query=fallback_query, top_k=8, company_filter=None)
				print(
					f"[RetrievalAgent] Ticket={subject!r} query={query_index} fallback_retrieved={len(chunks)} chunks"
				)

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
		reranked = self._sanitize_reranked(reranked)
		print(
			f"[RetrievalAgent] Ticket={subject!r} candidate_chunks={len(candidate_chunks)} "
			f"sanitized_filtered={self.last_sanitizer_filtered_count} reranked_chunks={len(reranked)}"
		)
		return reranked

	def retrieve_second_pass(
		self,
		ticket: Dict[str, Any],
		classification: Dict[str, Any],
		existing_chunks: List[Dict[str, Any]],
		top_k: int = 5,
	) -> List[Dict[str, Any]]:
		"""Run one broader deterministic retry and merge with first-pass chunks."""

		self.last_second_pass_used = True
		query_plan = self._build_second_pass_plan(ticket=ticket, classification=classification)
		subject = str(ticket.get("subject", "") or "").strip() or "unknown subject"
		candidate_chunks = list(existing_chunks)

		for query_index, query in enumerate(query_plan.queries[:2], start=1):
			print(f"[RetrievalAgent] Ticket={subject!r} second_pass_query={query_index} text={query!r}")
			chunks = self.hybrid_retriever.retrieve(query=query, top_k=10, company_filter=None)
			for chunk in chunks:
				candidate_chunks.append(
					{
						"text": chunk.text,
						"filepath": chunk.filepath,
						"company": chunk.company,
						"hybrid_score": float(chunk.metadata.get("hybrid_score", 0.0)),
						"score": float(chunk.metadata.get("hybrid_score", 0.0)),
						"query": query,
					}
				)

		reranked = self._rerank_candidates(
			candidates=candidate_chunks,
			query_plan=query_plan,
			classification={**classification, "company": ""},
			top_k=top_k,
		)
		reranked = self._sanitize_reranked(reranked)
		print(
			f"[RetrievalAgent] Ticket={subject!r} second_pass_candidates={len(candidate_chunks)} "
			f"sanitized_filtered={self.last_sanitizer_filtered_count} merged_chunks={len(reranked)}"
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

	def _build_second_pass_plan(self, ticket: Dict[str, Any], classification: Dict[str, Any]) -> RetrievalQueryPlan:
		history = self._normalize_issue_messages(ticket.get("issue", ""))
		latest_user_message = self._latest_user_message(history)
		subject = str(ticket.get("subject", "") or "").strip()
		intent = self._infer_intent(subject=subject, latest_user_message=latest_user_message)
		entities = self._extract_entities(latest_user_message)
		synonyms = self._intent_synonyms(intent, str(classification.get("request_type", "")))
		base = latest_user_message or subject or intent or "support policy"
		entity_text = " ".join(entities[:5])
		queries = [
			" ".join(part for part in [base, synonyms] if part).strip(),
			" ".join(part for part in [intent, entity_text, "policy troubleshooting guidance"] if part).strip(),
		]
		return RetrievalQueryPlan(
			conversation_summary=self._summarize_history(history, latest_user_message),
			intent=intent,
			key_entities=entities[:8],
			queries=[query for query in queries if query][:2] or [base],
		)

	def _sanitize_reranked(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		clean_chunks = self.corpus_sanitizer.sanitize(chunks)
		self.last_sanitizer_filtered_count = self.corpus_sanitizer.last_filtered_count
		for chunk in clean_chunks:
			chunk["sanitizer_filtered_count"] = self.last_sanitizer_filtered_count
		return clean_chunks

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
	def _intent_synonyms(intent: str, request_type: str) -> str:
		lowered = f"{intent} {request_type}".lower()
		if any(term in lowered for term in ("refund", "chargeback", "dispute", "billing")):
			return "refund chargeback billing dispute payment reversal"
		if any(term in lowered for term in ("login", "password", "account", "security")):
			return "account access login password security verification"
		if any(term in lowered for term in ("privacy", "delete", "data", "gdpr")):
			return "privacy data deletion export gdpr compliance"
		if any(term in lowered for term in ("api", "integration", "technical", "error")):
			return "api integration error troubleshooting configuration"
		if any(term in lowered for term in ("subscription", "plan", "cancel", "upgrade", "downgrade")):
			return "subscription plan cancel upgrade downgrade billing"
		return "policy documentation support guidance troubleshooting"

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
