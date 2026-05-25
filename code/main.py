from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from agents import RoutingAgent, SafetyAgent


@dataclass
class TicketAnalysisPipeline:
	safety_agent: SafetyAgent
	routing_agent: RoutingAgent

	def analyze(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
		safety_result = self.safety_agent.assess_safety(ticket)
		pii_detected, redacted_text = self.safety_agent.detect_pii(ticket)
		classification = self.routing_agent.classify_ticket(ticket)

		return {
			"ticket": ticket,
			"safety": _dump_model(safety_result),
			"pii_detected": pii_detected,
			"redacted_text": redacted_text,
			"classification": _dump_model(classification),
		}


def build_pipeline() -> TicketAnalysisPipeline:
	return TicketAnalysisPipeline(safety_agent=SafetyAgent(), routing_agent=RoutingAgent())


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


def _build_justification(safety: Dict[str, Any], classification: Dict[str, Any], status: str) -> str:
	safety_reason = str(safety.get("reasoning") or "safety check passed")
	risk_level = str(classification.get("risk_level") or "low")
	routing_reason = (
		f"routed as {classification.get('company', 'unknown')} / "
		f"{classification.get('product_area', 'general_support')} / "
		f"{classification.get('request_type', 'product_issue')}"
	)
	status_reason = f"status={status} because risk_level={risk_level}"
	return " | ".join([safety_reason, status_reason, routing_reason])


def _decide_status(safety: Dict[str, Any], classification: Dict[str, Any]) -> str:
	if bool(safety.get("is_adversarial")):
		return "escalated"
	if str(classification.get("risk_level", "low")) in {"high", "critical"}:
		return "escalated"
	return "replied"


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
			status = _decide_status(safety, classification)

			row_out = {
				"issue": row.get("Issue", ""),
				"subject": row.get("Subject", ""),
				"company": row.get("Company", ""),
				"response": "",
				"product_area": classification.get("product_area", "general_support"),
				"status": status,
				"request_type": classification.get("request_type", "product_issue"),
				"justification": _build_justification(safety, classification, status),
				"confidence_score": classification.get("confidence", 0.0),
				"source_documents": "",
				"risk_level": classification.get("risk_level", "low"),
				"pii_detected": result.get("pii_detected", False),
				"language": classification.get("language", "en"),
				"actions_taken": "safety_check -> routing_check -> stub_response",
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
					"actions_taken": "error_logged",
				}
			)

	out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
	out_df.to_csv(output_path, index=False)
	print(f"Wrote output to {output_path}")


if __name__ == "__main__":
	main()
