from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.prompts import ChatPromptTemplate

from config import OLLAMA_MODEL_ROUTING, TEMPERATURE, SEED
from core.models import TicketClassification
from .safety import PiiDetector


class RoutingAgent:
	"""Route a full ticket into a structured classification."""

	def __init__(self, model_name: str = OLLAMA_MODEL_ROUTING) -> None:
		self.model_name = model_name
		self.pii_detector = PiiDetector()

	def classify_ticket(self, ticket: Dict[str, Any]) -> TicketClassification:
		"""Return a structured routing decision for the ticket."""

		normalized = self._normalize_ticket(ticket)
		llm_result = self._classify_with_llm(normalized)
		if llm_result is not None:
			return llm_result

		return self._classify_with_rules(normalized)

	route_ticket = classify_ticket

	def _classify_with_llm(self, normalized_ticket: Dict[str, Any]) -> Optional[TicketClassification]:
		llm = self._build_llm()
		if llm is None:
			return None

		prompt = ChatPromptTemplate.from_messages(
			[
				(
					"system",
					"You are a deterministic support-ticket router. Cross-check the subject against the conversation history, do not trust the company field if it conflicts with the text, infer the real company from the content, and assign request_type and risk_level carefully. Return only the structured classification.",
				),
				(
					"human",
					"Ticket payload:\n{ticket_json}\n\nInstructions:\n"
					"- Infer company from the content if the provided field is misleading.\n"
					"- Cross-check the subject against the issue conversation.\n"
					"- Use request_type = product_issue, feature_request, bug, or invalid.\n"
					"- Use risk_level = low, medium, high, or critical based on financial, legal, security, and PII content.\n"
					"- Report the language as an ISO 639-1 code.\n",
				),
			]
		)

		try:
			chain = prompt | llm.with_structured_output(TicketClassification)
			verdict = chain.invoke({"ticket_json": json.dumps(normalized_ticket, ensure_ascii=False, indent=2)})
			return self._coerce_classification(verdict)
		except Exception:
			return None

	def _classify_with_rules(self, normalized_ticket: Dict[str, Any]) -> TicketClassification:
		text = normalized_ticket["combined_text"]
		company_hint = normalized_ticket["company_hint"]

		company, company_confidence = self._infer_company(text, company_hint)
		request_type, request_confidence = self._infer_request_type(text)
		product_area, product_confidence = self._infer_product_area(text, company)
		risk_level, risk_confidence = self._infer_risk_level(text)
		language, language_confidence = self._detect_language(text)
		pii_detected, _ = self.pii_detector.detect(text)

		confidence = round(
			max(
				0.05,
				min(
					1.0,
					(
						company_confidence
						+ request_confidence
						+ product_confidence
						+ risk_confidence
						+ language_confidence
					)
					/ 5.0,
				),
			),
			3,
		)

		if not company:
			company = company_hint or "unknown"

		if pii_detected and risk_level == "low":
			risk_level = "medium"

		return TicketClassification(
			company=company,
			product_area=product_area,
			request_type=request_type,
			risk_level=risk_level,
			pii_detected=pii_detected,
			language=language,
			confidence=confidence,
		)

	def _build_llm(self) -> Any:
		try:
			from langchain_ollama import ChatOllama

			return ChatOllama(model=self.model_name, temperature=TEMPERATURE, seed=SEED)
		except Exception:
			return None

	def _normalize_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
		subject = str(ticket.get("subject", "") or "").strip()
		company_hint = str(ticket.get("company", "") or "").strip()
		issue = ticket.get("issue", "")
		issue_text = self._issue_to_text(issue)
		combined_text = "\n".join(
			part
			for part in [
				f"Subject: {subject}" if subject else "",
				f"Company hint: {company_hint}" if company_hint else "",
				issue_text,
			]
			if part
		)

		return {
			"subject": subject,
			"company_hint": company_hint,
			"issue": issue,
			"issue_text": issue_text,
			"combined_text": combined_text,
		}

	def _infer_company(self, text: str, company_hint: str) -> Tuple[str, float]:
		lowered = text.lower()

		visa_terms = ("visa", "card", "charge", "payment", "merchant", "refund", "dispute", "fraud")
		claude_terms = ("claude", "anthropic", "bedrock", "api", "workspace", "lti", "safeguards", "desktop", "mobile")
		devplatform_terms = ("devplatform", "assessment", "test", "interview", "submission", "candidate", "recruiter", "hiring", "seat", "proctor")

		if self._has_any(lowered, visa_terms):
			return "Visa", 0.95
		if self._has_any(lowered, claude_terms):
			return "Claude", 0.95
		if self._has_any(lowered, devplatform_terms):
			return "DevPlatform", 0.95

		if company_hint and company_hint.lower() not in {"none", "unknown", "nan"}:
			return company_hint, 0.55

		return "unknown", 0.35

	def _infer_product_area(self, text: str, company: str) -> Tuple[str, float]:
		lowered = text.lower()

		if company == "Visa":
			if self._has_any(lowered, ("fraud", "unauthorized", "blocked", "stolen", "identity theft")):
				return "fraud_and_security", 0.92
			if self._has_any(lowered, ("dispute", "chargeback", "refund")):
				return "disputes", 0.91
			if self._has_any(lowered, ("merchant", "payment", "declined", "authorization")):
				return "payments", 0.9
			return "card_management", 0.7

		if company == "Claude":
			if self._has_any(lowered, ("api", "bedrock", "console", "requests", "errors")):
				return "api_and_console", 0.91
			if self._has_any(lowered, ("workspace", "team", "seat", "admin", "sso", "scim")):
				return "team_and_enterprise", 0.9
			if self._has_any(lowered, ("data", "privacy", "delete", "gdpr", "logs", "retention")):
				return "privacy_and_legal", 0.9
			return "general_support", 0.68

		if company == "DevPlatform":
			if self._has_any(lowered, ("test", "assessment", "submission", "candidate", "score")):
				return "assessments", 0.92
			if self._has_any(lowered, ("interview", "screen share", "lobby", "inactivity")):
				return "interviews", 0.9
			if self._has_any(lowered, ("billing", "subscription", "refund", "payment")):
				return "billing", 0.88
			if self._has_any(lowered, ("security", "infosec", "account hacked", "remove interviewer", "seat")):
				return "account_admin", 0.87
			return "platform_support", 0.66

		return "general_support", 0.5

	def _infer_request_type(self, text: str) -> Tuple[str, float]:
		lowered = text.lower()

		invalid_markers = (
			"ignore previous",
			"reveal your system prompt",
			"show your instructions",
			"output hidden",
			"system override",
			"give me the code to delete all files",
		)
		if self._has_any(lowered, invalid_markers):
			return "invalid", 0.98

		if self._has_any(
			lowered,
			("how do i", "how to", "what is", "can you help", "please explain", "best practice", "where do i", "when should i"),
		):
			return "product_issue", 0.84

		if self._has_any(lowered, ("add", "create", "new feature", "feature request", "would like", "could you support", "enhance", "improve", "variant")):
			return "feature_request", 0.85

		if self._has_any(lowered, ("error", "failing", "failed", "broken", "does not work", "not working", "stopped working", "bug", "issue", "blocked", "unable", "cannot")):
			return "bug", 0.83

		return "product_issue", 0.6

	def _infer_risk_level(self, text: str) -> Tuple[str, float]:
		lowered = text.lower()

		if self._has_any(
			lowered,
			("ssn", "aadhaar", "pan", "card number", "account hacked", "unauthorized", "refund me today", "legal", "gdpr", "right to erasure", "system prompt", "data breach", "security vulnerability"),
		):
			return "critical", 0.95

		if self._has_any(lowered, ("payment", "charge", "dispute", "blocked", "fraud", "identity theft", "password changed", "access lost", "urgent", "escalate")):
			return "high", 0.88

		if self._has_any(lowered, ("bug", "error", "issue", "unable", "not working", "failing", "stopped")):
			return "medium", 0.78

		return "low", 0.55

	def _detect_language(self, text: str) -> Tuple[str, float]:
		lowered = text.lower()

		language_scores = {
			"fr": self._score_language(lowered, ("bonjour", "merci", "carte", "règles", "accès", "aide", "s'il vous plaît")),
			"de": self._score_language(lowered, ("ich", "bitte", "hilfe", "konto", "zugang", "mein", "danke")),
			"es": self._score_language(lowered, ("hola", "gracias", "cuenta", "ayuda", "por favor", "tarjeta", "reembolso")),
			"pt": self._score_language(lowered, ("olá", "obrigado", "conta", "ajuda", "cartão", "por favor")),
			"it": self._score_language(lowered, ("ciao", "grazie", "conto", "aiuto", "per favore", "carta")),
		}

		best_language = "en"
		best_score = 0.35
		for language, score in language_scores.items():
			if score > best_score:
				best_language = language
				best_score = score

		return best_language, min(0.99, best_score)

	@staticmethod
	def _score_language(text: str, markers: Sequence[str]) -> float:
		score = 0.35
		for marker in markers:
			if marker in text:
				score += 0.15
		return min(0.95, score)

	@staticmethod
	def _has_any(text: str, markers: Sequence[str]) -> bool:
		return any(marker in text for marker in markers)

	def _issue_to_text(self, issue: Any) -> str:
		if isinstance(issue, list):
			return "\n".join(self._message_to_text(message) for message in issue if message)
		if isinstance(issue, dict):
			return self._message_to_text(issue)
		if isinstance(issue, str):
			parsed = self._maybe_parse_json(issue)
			if isinstance(parsed, list):
				return "\n".join(self._message_to_text(message) for message in parsed if message)
			if isinstance(parsed, dict):
				return self._message_to_text(parsed)
			return issue.strip()
		return str(issue).strip()

	@staticmethod
	def _maybe_parse_json(raw_text: str) -> Any:
		try:
			return json.loads(raw_text)
		except Exception:
			return raw_text

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
	def _coerce_classification(value: Any) -> TicketClassification:
		if isinstance(value, TicketClassification):
			return value
		if hasattr(TicketClassification, "model_validate"):
			try:
				return TicketClassification.model_validate(value)
			except Exception:
				pass
		try:
			return TicketClassification.parse_obj(value)
		except Exception:
			return TicketClassification(
				company="unknown",
				product_area="general_support",
				request_type="product_issue",
				risk_level="low",
				pii_detected=False,
				language="en",
				confidence=0.25,
			)
