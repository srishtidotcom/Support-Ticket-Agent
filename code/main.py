from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from agents import RoutingAgent, SafetyAgent
from agents.evidence_judge import EvidenceJudge
from agents.reflector import ReflectionAgent
from agents.resolver import ResponseGenerator
from agents.retriever import RetrievalAgent
from tools.tool_engine import ToolEngine


@dataclass
class TicketAnalysisPipeline:
	safety_agent: SafetyAgent
	routing_agent: RoutingAgent
	retrieval_agent: RetrievalAgent
	evidence_judge: EvidenceJudge
	response_generator: ResponseGenerator
	reflection_agent: ReflectionAgent
	tool_engine: ToolEngine

	def analyze(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
		safety_result = self.safety_agent.assess_safety(ticket)
		pii_detected, redacted_text = self.safety_agent.detect_pii(ticket)
		classification = self.routing_agent.classify_ticket(ticket)
		classification_payload = _dump_model(classification)

		retrieved_chunks = self.retrieval_agent.retrieve(ticket=ticket, classification=classification_payload, top_k=5)
		evidence_result = self.evidence_judge.evaluate(
			ticket=ticket,
			retrieved_chunks=retrieved_chunks,
			classification=classification_payload,
		)
		evidence_payload = _dump_model(evidence_result)
		source_documents = _collect_source_documents(evidence_payload, retrieved_chunks)

		generated = self.response_generator.generate(
			ticket=ticket,
			classification=classification_payload,
			retrieved_chunks=retrieved_chunks,
			evidence_result=evidence_payload,
		)
		generated_payload = _dump_model(generated)
		reflection_result = self.reflection_agent.validate(
			ticket=ticket,
			classification=classification_payload,
			retrieved_chunks=retrieved_chunks,
			evidence_result=evidence_payload,
			generated_response=str(generated_payload.get("response", "")),
			source_documents=source_documents,
		)
		reflection_payload = _dump_model(reflection_result)
		status = _decide_status(_dump_model(safety_result), classification_payload, evidence_payload, reflection_payload)
		actions = self.tool_engine.build_actions(
			ticket=ticket,
			classification=classification_payload,
			evidence_result=evidence_payload,
			reflection_result=reflection_payload,
			status=status,
		)

		return {
			"ticket": ticket,
			"safety": _dump_model(safety_result),
			"pii_detected": pii_detected,
			"redacted_text": redacted_text,
			"classification": classification_payload,
			"retrieval": retrieved_chunks,
			"evidence": evidence_payload,
			"generated": generated_payload,
			"reflection": reflection_payload,
			"status": status,
			"actions": actions,
			"source_documents": source_documents,
		}


def build_pipeline() -> TicketAnalysisPipeline:
	retrieval_agent = RetrievalAgent()
	print(f"Index status: {'LOADED' if retrieval_agent.hybrid_retriever._artifacts is not None else 'NOT FOUND'}")
	return TicketAnalysisPipeline(
		safety_agent=SafetyAgent(),
		routing_agent=RoutingAgent(),
		retrieval_agent=retrieval_agent,
		evidence_judge=EvidenceJudge(),
		response_generator=ResponseGenerator(),
		reflection_agent=ReflectionAgent(),
		tool_engine=ToolEngine(),
	)


OUTPUT_COLUMNS = [
	"issue",
	"subject",
	"company",
	"response",
	"product_area",
	"status",
	"request_type",
	"justification",
	"confidence_score",
	"source_documents",
	"risk_level",
	"pii_detected",
	"language",
	"actions_taken",
]


def _dump_model(model: Any) -> Dict[str, Any]:
	if hasattr(model, "model_dump"):
		return model.model_dump()
	if hasattr(model, "dict"):
		return model.dict()
	return dict(model)


def _build_justification(
	safety: Dict[str, Any],
	classification: Dict[str, Any],
	evidence: Dict[str, Any],
	reflection: Dict[str, Any],
	generated: Dict[str, Any],
	status: str,
) -> str:
	safety_reason = str(safety.get("reasoning") or "safety check passed")
	risk_level = str(classification.get("risk_level") or "low")
	routing_reason = (
		f"routed as {classification.get('company', 'unknown')} / "
		f"{classification.get('product_area', 'general_support')} / "
		f"{classification.get('request_type', 'product_issue')}"
	)
	evidence_reason = str(evidence.get("reasoning") or "evidence check unavailable")
	evidence_action = str(evidence.get("recommended_action") or "ask_clarification")
	reflection_reason = str(reflection.get("reasoning") or "reflection unavailable")
	reflection_action = str(reflection.get("final_action") or "unknown")
	generation_reason = str(generated.get("reasoning") or "response generated from retrieved chunks")
	status_reason = f"status={status} because risk_level={risk_level}, evidence_action={evidence_action}, reflection_action={reflection_action}"
	return " | ".join(
		[
			safety_reason,
			evidence_reason,
			generation_reason,
			reflection_reason,
			status_reason,
			routing_reason,
		]
	)


def _decide_status(
	safety: Dict[str, Any],
	classification: Dict[str, Any],
	evidence: Dict[str, Any],
	reflection: Dict[str, Any],
) -> str:
	if bool(safety.get("is_adversarial", False)):
		return "escalated"
	if str(classification.get("risk_level", "low")).lower() in {"high", "critical"}:
		return "escalated"
	if str(evidence.get("recommended_action", "ask_clarification")) == "escalate":
		return "escalated"
	if str(reflection.get("final_action", "escalate")) != "accept":
		return "escalated"
	return "replied"


def _collect_source_documents(evidence: Dict[str, Any], retrieval: Any) -> str:
	sources = []
	for source in evidence.get("top_sources", []) or []:
		source_text = str(source).strip()
		if source_text and source_text not in sources:
			sources.append(source_text)

	if not sources and retrieval:
		for chunk in retrieval:
			filepath = str(chunk.get("filepath", "")).strip()
			if filepath and filepath not in sources:
				sources.append(filepath)

	return "|".join(sources)


def main() -> None:
	repo_root = Path(__file__).resolve().parents[1]
	tickets_path = repo_root / "support_tickets" / "support_tickets.csv"
	output_path = repo_root / "support_tickets" / "output.csv"

	print(f"Loading tickets from {tickets_path}")
	df = pd.read_csv(tickets_path).fillna("")
	print(f"Found {len(df)} tickets")

	pipeline = build_pipeline()

	rows = []
	for idx, row in df.iterrows():
		try:
			issue_raw = row.get("Issue", "")
			try:
				issue = json.loads(issue_raw) if isinstance(issue_raw, str) and issue_raw.strip().startswith("[") else issue_raw
			except Exception:
				issue = issue_raw

			ticket = {"subject": row.get("Subject", ""), "company": row.get("Company", ""), "issue": issue}
			print(f"Processing ticket {idx+1}/{len(df)}: {ticket.get('subject')}")
			result = pipeline.analyze(ticket)

			classification = result.get("classification", {})
			safety = result.get("safety", {})
			evidence = result.get("evidence", {})
			generated = result.get("generated", {})
			reflection = result.get("reflection", {})
			retrieval = result.get("retrieval", [])
			status = result.get("status") or _decide_status(safety, classification, evidence, reflection)
			source_documents = result.get("source_documents") or _collect_source_documents(evidence, retrieval)
			actions_taken = json.dumps(result.get("actions", []), ensure_ascii=False)

			row_out = {
				"issue": row.get("Issue", ""),
				"subject": row.get("Subject", ""),
				"company": row.get("Company", ""),
				"response": generated.get("response", ""),
				"product_area": classification.get("product_area", "general_support"),
				"status": status,
				"request_type": classification.get("request_type", "product_issue"),
				"justification": _build_justification(safety, classification, evidence, reflection, generated, status),
				"confidence_score": min(
					float(evidence.get("confidence", classification.get("confidence", 0.0)) or 0.0),
					float(generated.get("confidence", 1.0) or 1.0),
					float(reflection.get("confidence", 1.0) or 1.0),
				),
				"source_documents": source_documents,
				"risk_level": classification.get("risk_level", "low"),
				"pii_detected": result.get("pii_detected", False),
				"language": classification.get("language", "en"),
				"actions_taken": actions_taken,
			}
			rows.append(row_out)
		except Exception as exc:  # keep processing other tickets
			print(f"Error processing ticket {idx + 1}: {exc}")
			print(traceback.format_exc(limit=1).strip())
			rows.append(
				{
					"issue": row.get("Issue", ""),
					"subject": row.get("Subject", ""),
					"company": row.get("Company", ""),
					"response": "",
					"product_area": "general_support",
					"status": "escalated",
					"request_type": "product_issue",
					"justification": f"Pipeline error: {exc} | status=escalated because risk could not be computed | routed as general_support / general_support / product_issue",
					"confidence_score": 0.0,
					"source_documents": "",
					"risk_level": "low",
					"pii_detected": False,
					"language": "en",
					"actions_taken": json.dumps(
						[
							{
								"tool": "escalate_to_human",
								"arguments": {
									"priority": "high",
									"department": "general",
									"summary": "Pipeline error prevented safe automated resolution.",
								},
							}
						]
					),
				}
			)

	out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
	out_df.to_csv(output_path, index=False)
	print(f"Wrote output to {output_path}")


if __name__ == "__main__":
	main()
