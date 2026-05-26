from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import BASE_DIR, OLLAMA_MODEL_ROUTING, SEED, TEMPERATURE


class ToolIntent(BaseModel):
	"""High-level tool intent only; never raw executable JSON from the model."""

	intent: Literal[
		"none",
		"verify_identity",
		"issue_refund",
		"reset_password",
		"lock_account",
		"modify_subscription",
		"escalate_to_human",
	]
	confidence: float = Field(ge=0.0, le=1.0)
	reasoning: str


class ToolEngine:
	"""Build strict validated tool calls from ticket context and policy rules.

	The local LLM may propose an intent label, but the rules engine owns every
	parameter and prerequisite. This prevents prompt injection from smuggling raw
	tool JSON and ensures destructive operations are preceded by identity checks.
	"""

	_DESTRUCTIVE_TOOLS = {"issue_refund", "modify_subscription"}

	def __init__(
		self,
		spec_path: Path | None = None,
		model_name: str = OLLAMA_MODEL_ROUTING,
	) -> None:
		self.spec_path = spec_path or (BASE_DIR / "data" / "api_specs" / "internal_tools.json")
		self.model_name = model_name
		self.tool_specs = self._load_specs(self.spec_path)
		self.spec_by_name = {str(spec.get("name")): spec for spec in self.tool_specs}

	def build_actions(
		self,
		ticket: Dict[str, Any],
		classification: Dict[str, Any],
		evidence_result: Dict[str, Any],
		reflection_result: Dict[str, Any],
		status: str,
	) -> List[Dict[str, Any]]:
		if status == "replied" and str(reflection_result.get("final_action", "")) == "accept":
			intent = self._propose_intent(ticket=ticket, classification=classification, evidence_result=evidence_result)
		else:
			intent = ToolIntent(intent="escalate_to_human", confidence=0.9, reasoning="Pipeline status requires human review.")

		text = self._ticket_text(ticket)
		rule_intent = self._rule_intent(text=text, classification=classification, evidence_result=evidence_result, status=status)
		if rule_intent is not None:
			intent = rule_intent

		if intent.intent == "none":
			return []

		actions = self._build_validated_calls(intent=intent, text=text, classification=classification, status=status)
		return actions

	def _propose_intent(
		self,
		ticket: Dict[str, Any],
		classification: Dict[str, Any],
		evidence_result: Dict[str, Any],
	) -> ToolIntent:
		llm = self._build_llm()
		if llm is not None:
			prompt = ChatPromptTemplate.from_messages(
				[
					(
						"system",
						"You classify support tool intent only. Do not output JSON tool calls or parameters. "
						"Return one intent label from the schema. Simple informational replies use intent='none'.",
					),
					(
						"human",
						"Ticket text:\n{text}\n\nClassification: {classification_json}\n"
						"Evidence verdict: {evidence_json}\n\n"
						"Available intent labels: none, verify_identity, issue_refund, reset_password, "
						"lock_account, modify_subscription, escalate_to_human.",
					),
				]
			)
			try:
				chain = prompt | llm.with_structured_output(ToolIntent)
				result = chain.invoke(
					{
						"text": self._ticket_text(ticket),
						"classification_json": json.dumps(classification, ensure_ascii=False),
						"evidence_json": json.dumps(evidence_result, ensure_ascii=False),
					}
				)
				return self._coerce_intent(result)
			except Exception:
				pass

		return self._rule_intent(
			text=self._ticket_text(ticket),
			classification=classification,
			evidence_result=evidence_result,
			status="replied",
		) or ToolIntent(intent="none", confidence=0.7, reasoning="No actionable tool intent detected.")

	def _rule_intent(
		self,
		text: str,
		classification: Dict[str, Any],
		evidence_result: Dict[str, Any],
		status: str,
	) -> Optional[ToolIntent]:
		lowered = text.lower()
		risk_level = str(classification.get("risk_level", "low")).lower()
		if status == "escalated" or risk_level in {"high", "critical"} or str(evidence_result.get("recommended_action")) == "escalate":
			return ToolIntent(intent="escalate_to_human", confidence=0.9, reasoning="Risk or evidence status requires human escalation.")
		if any(term in lowered for term in ("hacked", "account takeover", "identity theft", "unauthorized access")):
			return ToolIntent(intent="lock_account", confidence=0.88, reasoning="Account compromise language detected.")
		if any(term in lowered for term in ("refund", "chargeback", "dispute")):
			return ToolIntent(intent="issue_refund", confidence=0.78, reasoning="Refund or dispute request detected.")
		if any(term in lowered for term in ("cancel subscription", "downgrade", "upgrade", "pause subscription")):
			return ToolIntent(intent="modify_subscription", confidence=0.76, reasoning="Subscription modification request detected.")
		if any(term in lowered for term in ("reset password", "forgot password", "password reset")):
			return ToolIntent(intent="reset_password", confidence=0.78, reasoning="Password reset request detected.")
		return None

	def _build_validated_calls(
		self,
		intent: ToolIntent,
		text: str,
		classification: Dict[str, Any],
		status: str,
	) -> List[Dict[str, Any]]:
		if intent.intent == "escalate_to_human":
			return [self._tool_call("escalate_to_human", self._escalation_params(classification, intent.reasoning))]

		if intent.intent in self._DESTRUCTIVE_TOOLS and not self._identity_verified(text):
			target = self._extract_email(text) or self._extract_phone(text) or "customer_contact_on_file"
			return [self._tool_call("verify_identity", {"method": "email_otp", "target": target})]

		if intent.intent == "verify_identity":
			target = self._extract_email(text) or self._extract_phone(text) or "customer_contact_on_file"
			return [self._tool_call("verify_identity", {"method": "email_otp", "target": target})]

		if intent.intent == "issue_refund":
			transaction_id = self._extract_transaction_id(text)
			amount = self._extract_amount(text)
			if not transaction_id or amount is None or amount > 500:
				return [self._tool_call("escalate_to_human", self._escalation_params(classification, "Refund lacks required ID/amount or exceeds authorization."))]
			return [self._tool_call("issue_refund", {"transaction_id": transaction_id, "amount": amount, "reason": self._refund_reason(text)})]

		if intent.intent == "reset_password":
			email = self._extract_email(text)
			if not email:
				return [self._tool_call("verify_identity", {"method": "email_otp", "target": "customer_contact_on_file"})]
			return [self._tool_call("reset_password", {"user_email": email})]

		if intent.intent == "lock_account":
			identifier = self._extract_email(text) or self._extract_account_id(text) or "customer_account_on_file"
			return [self._tool_call("lock_account", {"user_identifier": identifier, "lock_reason": "suspected_fraud"})]

		if intent.intent == "modify_subscription":
			user_id = self._extract_account_id(text)
			action = self._subscription_action(text)
			if not user_id:
				return [self._tool_call("verify_identity", {"method": "email_otp", "target": self._extract_email(text) or "customer_contact_on_file"})]
			params: Dict[str, Any] = {"user_id": user_id, "action": action}
			target_plan = self._target_plan(text)
			if target_plan:
				params["target_plan"] = target_plan
			return [self._tool_call("modify_subscription", params)]

		return []

	def _tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
		self._validate_tool_call(name, arguments)
		return {"tool": name, "arguments": arguments}

	def _validate_tool_call(self, name: str, arguments: Dict[str, Any]) -> None:
		spec = self.spec_by_name.get(name)
		if not spec:
			raise ValueError(f"Unknown tool: {name}")
		required = spec.get("parameters", {}).get("required", [])
		missing = [field for field in required if field not in arguments or arguments[field] in (None, "")]
		if missing:
			raise ValueError(f"Tool {name} missing required arguments: {missing}")

	@staticmethod
	def _escalation_params(classification: Dict[str, Any], summary: str) -> Dict[str, Any]:
		risk = str(classification.get("risk_level", "low")).lower()
		product_area = str(classification.get("product_area", "general_support")).lower()
		if risk == "critical":
			priority = "urgent"
		elif risk == "high":
			priority = "high"
		else:
			priority = "normal"

		if "billing" in product_area or "dispute" in product_area or "payment" in product_area:
			department = "billing"
		elif "security" in product_area or "fraud" in product_area or risk in {"high", "critical"}:
			department = "security"
		elif "legal" in product_area or risk == "critical":
			department = "legal"
		elif "api" in product_area or "technical" in product_area:
			department = "technical"
		else:
			department = "general"
		return {"priority": priority, "department": department, "summary": summary[:240]}

	@staticmethod
	def _identity_verified(text: str) -> bool:
		return bool(re.search(r"\b(identity verified|verified user|otp verified|verification complete)\b", text, re.IGNORECASE))

	@staticmethod
	def _extract_email(text: str) -> Optional[str]:
		match = re.search(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b", text)
		return match.group(0) if match else None

	@staticmethod
	def _extract_phone(text: str) -> Optional[str]:
		match = re.search(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}\b", text)
		return match.group(0) if match else None

	@staticmethod
	def _extract_transaction_id(text: str) -> Optional[str]:
		match = re.search(r"\b(?:txn|transaction)[_-]?[A-Za-z0-9]{4,}\b", text, re.IGNORECASE)
		return match.group(0) if match else None

	@staticmethod
	def _extract_account_id(text: str) -> Optional[str]:
		match = re.search(r"\b(?:user|acct|account)[_-]?[A-Za-z0-9]{5,}\b", text, re.IGNORECASE)
		return match.group(0) if match else None

	@staticmethod
	def _extract_amount(text: str) -> Optional[float]:
		match = re.search(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)", text)
		return float(match.group(1)) if match else None

	@staticmethod
	def _refund_reason(text: str) -> str:
		lowered = text.lower()
		if "duplicate" in lowered:
			return "duplicate"
		if any(term in lowered for term in ("fraud", "unauthorized")):
			return "fraud"
		if any(term in lowered for term in ("failed", "outage", "service")):
			return "service_failure"
		return "customer_request"

	@staticmethod
	def _subscription_action(text: str) -> str:
		lowered = text.lower()
		for action in ("upgrade", "downgrade", "cancel", "pause"):
			if action in lowered:
				return action
		return "cancel"

	@staticmethod
	def _target_plan(text: str) -> Optional[str]:
		lowered = text.lower()
		for plan in ("free", "pro", "team", "enterprise"):
			if plan in lowered:
				return plan
		return None

	@staticmethod
	def _ticket_text(ticket: Dict[str, Any]) -> str:
		issue = ticket.get("issue", "")
		if isinstance(issue, str):
			try:
				issue = json.loads(issue)
			except Exception:
				return "\n".join([str(ticket.get("subject", "")), str(ticket.get("company", "")), issue])
		if isinstance(issue, list):
			body = "\n".join(
				str(message.get("content", message)) if isinstance(message, dict) else str(message)
				for message in issue
			)
		elif isinstance(issue, dict):
			body = str(issue.get("content", issue))
		else:
			body = str(issue or "")
		return "\n".join([str(ticket.get("subject", "")), str(ticket.get("company", "")), body])

	@staticmethod
	def _load_specs(path: Path) -> List[Dict[str, Any]]:
		with path.open("r", encoding="utf-8") as handle:
			data = json.load(handle)
		if not isinstance(data, list):
			raise ValueError(f"Tool spec must be a list: {path}")
		return data

	@staticmethod
	def _coerce_intent(value: Any) -> ToolIntent:
		if isinstance(value, ToolIntent):
			return value
		if hasattr(ToolIntent, "model_validate"):
			return ToolIntent.model_validate(value)
		return ToolIntent.parse_obj(value)

	def _build_llm(self) -> Any:
		try:
			from langchain_ollama import ChatOllama

			return ChatOllama(model=self.model_name, temperature=TEMPERATURE, seed=SEED)
		except Exception:
			return None
