from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
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
from config import SEED
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
		sanitizer_filtered_count = self.retrieval_agent.last_sanitizer_filtered_count
		evidence_result = self.evidence_judge.evaluate(
			ticket=ticket,
			retrieved_chunks=retrieved_chunks,
			classification=classification_payload,
		)
		evidence_payload = _dump_model(evidence_result)
		second_pass_used = False
		if _needs_second_pass(evidence_payload):
			second_pass_used = True
			retrieved_chunks = self.retrieval_agent.retrieve_second_pass(
				ticket=ticket,
				classification=classification_payload,
				existing_chunks=retrieved_chunks,
				top_k=5,
			)
			sanitizer_filtered_count += self.retrieval_agent.last_sanitizer_filtered_count
			evidence_result = self.evidence_judge.evaluate(
				ticket=ticket,
				retrieved_chunks=retrieved_chunks,
				classification=classification_payload,
			)
			evidence_payload = _dump_model(evidence_result)
			evidence_payload["reasoning"] = (
				f"Second-pass retrieval triggered after weak evidence and merged broader corpus results. "
				f"{evidence_payload.get('reasoning', '')}"
			).strip()
		if sanitizer_filtered_count:
			evidence_payload["reasoning"] = (
				f"Filtered {sanitizer_filtered_count} suspicious chunks from corpus "
				f"(potential indirect injection). {evidence_payload.get('reasoning', '')}"
			).strip()
		evidence_payload["second_pass_retrieval"] = second_pass_used
		evidence_payload["corpus_sanitizer_filtered_count"] = sanitizer_filtered_count
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
		status = _decide_status(_dump_model(safety_result), classification_payload, evidence_payload, reflection_payload, generated_payload)
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


def _seed_everything(seed: int = SEED) -> None:
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	try:
		import numpy as np

		np.random.seed(seed)
	except Exception:
		pass
	try:
		import torch

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
	except Exception:
		pass


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
	generated: Dict[str, Any] | None = None,
) -> str:
	if bool(safety.get("is_adversarial", False)):
		return "escalated"
	if str(classification.get("risk_level", "low")).lower() in {"high", "critical"}:
		return "escalated"
	if str(evidence.get("recommended_action", "ask_clarification")) == "escalate":
		return "escalated"
	if str(reflection.get("final_action", "escalate")) != "accept":
		return "escalated"
	if generated and "escalat" in str(generated.get("response", "")).lower():
		return "escalated"
	return "replied"


def _collect_source_documents(evidence: Dict[str, Any], retrieval: Any) -> str:
	sources = []
	for source in evidence.get("top_sources", []) or []:
		source_text = str(source).strip()
		if _valid_source(source_text) and source_text not in sources:
			sources.append(source_text)

	if not sources and retrieval:
		for chunk in retrieval:
			filepath = str(chunk.get("filepath", "")).strip()
			if _valid_source(filepath) and filepath not in sources:
				sources.append(filepath)

	return "|".join(sources)


def _valid_source(source: str) -> bool:
	return bool(source and source.startswith("data/") and len(source) > 8)


def _weighted_confidence(
	evidence: Dict[str, Any],
	reflection: Dict[str, Any],
	classification: Dict[str, Any],
) -> float:
	evidence_score = _as_score(evidence.get("confidence"), 0.0)
	reflection_score = _as_score(reflection.get("confidence"), 0.0)
	routing_score = _as_score(classification.get("confidence"), 0.0)
	return round((0.40 * evidence_score) + (0.40 * reflection_score) + (0.20 * routing_score), 4)


def _needs_second_pass(evidence: Dict[str, Any]) -> bool:
	action = str(evidence.get("recommended_action", "ask_clarification"))
	confidence = _as_score(evidence.get("confidence"), 0.0)
	return action == "ask_clarification" or confidence < 0.60


def _as_score(value: Any, default: float) -> float:
	try:
		return max(0.0, min(1.0, float(value)))
	except (TypeError, ValueError):
		return default


def _actions_json(actions: Any) -> str:
	if not isinstance(actions, list):
		return "[]"
	try:
		return json.dumps(actions, ensure_ascii=False)
	except (TypeError, ValueError):
		return "[]"


def _clean_response_text(response: Any, source_documents: str) -> str:
	text = str(response or "")
	text = re.sub(r"\s+", " ", text).strip()
	text = re.sub(r"\s*Sources:\s*.*$", "", text, flags=re.IGNORECASE).strip()
	if source_documents:
		return f"{text}\nSources: {source_documents}".strip()
	return text


def _print_final_stats(rows: list[Dict[str, Any]], runtime_seconds: float, validation_code: int) -> None:
	total = len(rows)
	replied = sum(1 for row in rows if str(row.get("status", "")).lower() == "replied")
	escalated = sum(1 for row in rows if str(row.get("status", "")).lower() == "escalated")
	with_sources = sum(1 for row in rows if str(row.get("source_documents", "")).strip())
	avg_confidence = (
		sum(_as_score(row.get("confidence_score"), 0.0) for row in rows) / total
		if total
		else 0.0
	)
	source_pct = (with_sources / total * 100.0) if total else 0.0
	print("\nFinal stats")
	print(f"- Tickets processed: {total}")
	print(f"- Replied/escalated: {replied}/{escalated}")
	print(f"- Rows with sources: {source_pct:.1f}%")
	print(f"- Average confidence: {avg_confidence:.4f}")
	print(f"- Runtime: {runtime_seconds:.1f}s ({runtime_seconds / 60.0:.1f} min)")
	print(f"- validate_output.py exit code: {validation_code}")


def main() -> None:
	start_time = time.perf_counter()
	_seed_everything()
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
			status = _decide_status(safety, classification, evidence, reflection, generated)
			source_documents = result.get("source_documents") or _collect_source_documents(evidence, retrieval)
			response_text = _clean_response_text(generated.get("response", ""), source_documents)
			actions_taken = _actions_json(result.get("actions", []))

			row_out = {
				"issue": row.get("Issue", ""),
				"subject": row.get("Subject", ""),
				"company": row.get("Company", ""),
				"response": response_text,
				"product_area": classification.get("product_area", "general_support"),
				"status": status,
				"request_type": classification.get("request_type", "product_issue"),
				"justification": _build_justification(safety, classification, evidence, reflection, generated, status),
				"confidence_score": _weighted_confidence(evidence, reflection, classification),
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
	print("Running validate_output.py")
	validation = subprocess.run([sys.executable, str(repo_root / "code" / "validate_output.py")], check=False)
	_print_final_stats(rows, time.perf_counter() - start_time, validation.returncode)


if __name__ == "__main__":
	main()
