# Architecture

## High-Level Flow

```text
support_tickets/support_tickets.csv
        |
        v
code/main.py
        |
        v
TicketAnalysisPipeline
        |
        +--> SafetyAgent
        |       - prompt-injection and adversarial pattern checks
        |       - PII detection and redaction
        |
        +--> RoutingAgent
        |       - company, product area, request type, risk level, language
        |       - local LLM first, deterministic rules fallback
        |
        +--> RetrievalAgent
        |       - query planning
        |       - hybrid retrieval from FAISS embeddings + BM25 lexical index
        |       - deterministic reranking and source deduplication
        |
        +--> EvidenceJudge
        |       - evidence sufficiency
        |       - conflict checks
        |       - direct action support and hallucination risk
        |
        +--> ResponseGenerator
        |       - concise customer-facing response from retrieved evidence only
        |       - citation formatting and PII cleanup
        |
        +--> ReflectionAgent
        |       - final grounding, citation, PII, injection, and high-risk checks
        |       - can force escalation
        |
        +--> ToolEngine
                - safe validated action JSON
                - destructive actions require identity checks or escalation

support_tickets/output.csv
        |
        v
code/validate_output.py
```

## Component Responsibilities

`main.py` is the evaluator entry point. It loads tickets, runs the pipeline, builds all required output columns, computes weighted confidence, writes `support_tickets/output.csv`, runs `validate_output.py`, and prints final stats and runtime.

`build_index.py` builds the retrieval artifacts. It loads the corpus from `data/`, chunks documents, embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`, saves normalized embeddings, builds a FAISS inner-product index, and builds a BM25 index over chunk tokens.

`SafetyAgent` provides the first gate. It detects prompt injection, jailbreaks, prompt extraction attempts, data exfiltration requests, label manipulation, and common PII patterns. It redacts PII before downstream prose generation where possible.

`RoutingAgent` classifies the ticket into company, product area, request type, risk level, language, and confidence. It uses Ollama structured output when available and rule-based fallback logic when the model is unavailable.

`RetrievalAgent` creates one or two focused search queries from the subject, company, product area, latest user message, and extracted entities. It retrieves candidates with hybrid lexical/vector retrieval, applies company filtering when useful, and reranks by hybrid score plus lexical topic coverage.

`EvidenceJudge` decides whether retrieved chunks are sufficient to answer safely. It checks relevance, conflicts, direct support for requested actions, and hallucination risk. It returns top source paths and a recommended action: reply, ask clarification, or escalate.

`ResponseGenerator` writes the customer-facing answer using retrieved evidence only. It keeps wording concise, strips unsupported source lines, redacts PII, and appends complete `Sources: data/...` citations.

`ReflectionAgent` is the final quality gate. It catches missing or incomplete citations, PII leakage, prompt-injection compliance, unsupported policy claims, hallucinated action steps, missing evidence, and high/critical-risk tickets. Major issues force escalation.

`ToolEngine` creates valid `actions_taken` JSON arrays. It never accepts raw tool JSON from ticket text or the LLM. It maps a safe intent to validated internal tool calls and requires verification before destructive operations such as refunds or subscription changes.

## Key Design Decisions And Trade-Offs

Hybrid retrieval was chosen because the corpus mixes highly specific support articles, broad policy pages, and multilingual or terse tickets. BM25 helps exact terms such as "chargeback", "GDPR", "CodePair", or "LTI"; embeddings help paraphrased questions and messy customer language.

The system uses local Ollama models with deterministic rule fallbacks. This keeps the agent usable when model calls fail and makes hidden-test behavior more predictable. The trade-off is that fallback responses can sound more template-like and routing can be conservative.

Risk handling is deliberately safety-first. High and critical tickets escalate even if relevant documents exist, because fraud, legal threats, account compromise, privacy incidents, and data leakage should not be fully resolved by an automated reply. This improves safety but can reduce direct-resolution rate.

The response builder separates evidence sufficiency from final writing. The generator can produce a polished answer, but the reflector can still reject it if citations are incomplete, PII leaks, or action steps are unsupported. This adds latency but catches a class of errors that simple RAG pipelines often miss.

Tool use is intent-based rather than free-form. The LLM may suggest an intent label, but code constructs every parameter and validates required fields against the tool spec. This prevents prompt injection from smuggling executable action JSON.

## Escalation Logic

A ticket is escalated when any of these conditions apply:

- Safety detects adversarial or prompt-manipulation behavior.
- Routing marks risk as `high` or `critical`.
- Evidence judge recommends escalation.
- Reflection result is anything other than `accept`.
- The generated response itself indicates escalation.
- Pipeline errors prevent safe automated resolution.

Otherwise the ticket is marked `replied`.

## Confidence Scoring

`confidence_score` is a weighted average:

```text
40% evidence confidence
40% reflection confidence
20% routing confidence
```

This weights final answer grounding and safety as heavily as initial retrieval quality while still accounting for routing uncertainty.

## Safety And Adversarial Handling

The ticket text is treated as untrusted input at every stage. Safety rules catch direct injection patterns, and prompts instruct local models not to follow hidden-instruction requests. PII detection covers emails, phone numbers, SSNs, Aadhaar/PAN-like IDs, card numbers with Luhn validation, addresses, account identifiers, and secret-like tokens. Response generation and reflection both run cleanup checks so sensitive data is less likely to appear in final prose.

### Adversarial Robustness Enhancements

The latest red-team pass added targeted deterministic coverage for social-engineering and output-manipulation attacks that looked benign to generic prompt-injection checks. `SafetyAgent` now detects authority impersonation involving executives, support leads, internal evaluators, authorized testers, fake colleagues, and fake previous assistants. It also flags false consensus claims, praise or trust-building followed by unsafe requests, gradual setup-then-attack language, forged internal memos or support articles, forged corpus/source citations, direct evaluator-column manipulation, and partial-compliance traps.

`ReflectionAgent` mirrors these checks at the final gate. It explicitly scans the original ticket for attempts to manipulate `status`, `confidence_score`, `actions_taken`, `source_documents`, or `risk_level`; requests to answer only selected parts; authority plus emotional pressure; fake previous-assistant statements; and fake policy/corpus language. If any high-risk adversarial pattern is present and the router has assigned `medium`, `high`, or `critical` risk, reflection forces escalation even if the generated response is otherwise well cited.

`ResponseGenerator` now refuses requests for hidden prompts, developer instructions, internal tools, tool schemas, action JSON, confidence formulas, routing labels, source-document internals, or retrieval metadata. It also applies stronger final redaction for SSNs, Aadhaar/PAN-like IDs, card-like numbers, account identifiers, bearer tokens, passwords, OTPs, API keys, and JWT-shaped secrets. The refusal tone is intentionally short, polite, and firm: it does not debate the malicious framing and routes the case to a human review path.

## Known Limitations And Failure Modes

The biggest known failure mode is retrieval confusion across companies when a ticket is short, ambiguous, or intentionally cross-domain. The router and company filter reduce this, but some output rows can still cite plausible-looking documents from the wrong company if the subject is vague or the issue mentions several products.

The fallback rules are intentionally conservative but imperfect. They may escalate benign tickets with scary language and may reply to nuanced policy questions when the best answer should be a specialist review.

Multilingual handling is partial. The router can identify non-English content in simple cases, and retrieval may still find relevant English documents, but the agent does not translate or localize responses deeply.

The validator checks structure, not semantic correctness. Passing `validate_output.py` means the CSV can be evaluated, not that every answer is optimal.

## Self-Assessment

### Evaluation Dimension Ratings

| Dimension | Rating | Rationale |
| --- | ---: | --- |
| Correctness and helpfulness | 7/10 | The pipeline usually returns grounded, concise answers with citations, but short ambiguous tickets can still route or retrieve weak evidence. |
| Retrieval quality | 8/10 | Hybrid BM25 + embeddings works well for exact policies and paraphrases; company filtering and reranking help, but cross-domain tickets remain difficult. |
| Safety and escalation | 9/10 | High-risk, PII, injection, social engineering, forged policy/corpus claims, evaluator-field manipulation, and unsupported claims are checked in multiple stages; this favors safe escalation over risky automation. |
| Output format compliance | 9/10 | `main.py` writes the required columns, validates JSON actions, appends citations, and runs the provided validator automatically. |
| Architecture and code quality | 8/10 | The pipeline is modular, deterministic where practical, and documented. Some heuristics are still embedded in agents rather than external policy config. |
| Hidden-test robustness | 8/10 | Rule fallbacks and conservative reflection now cover more red-team patterns, but unseen products, new languages, or adversarially vague tickets may still expose retrieval gaps. |

### Three Hardest Tickets

1. `Multiple Issues Across Platforms`

   This is hard because it blends companies and likely contains more than one user intent. The router must infer the primary company and product area from content rather than trusting a missing or generic company field. The pipeline handles it by classifying risk conservatively, retrieving multiple evidence chunks, and escalating when the combined context is not safe for a single automated answer.

2. `Claude billing refund + Visa chargeback`

   This mixes two domains with financial action language. A naive agent might answer from only Claude billing docs or only Visa dispute docs. The system handles it by allowing cross-domain evidence retrieval, then escalating high-risk financial ambiguity instead of pretending one policy fully resolves both requests.

3. `LEGAL THREAT - DISCRIMINATION LAWSUIT`

   This is difficult because there may be relevant DevPlatform documentation, but legal-threat language makes automated resolution unsafe. The router marks it critical, and the reflector enforces the policy that high/critical-risk tickets must escalate even when evidence exists.

### Predicted Hidden Test Challenges

Hidden tests will likely include prompt injection embedded in normal support conversations, misleading company fields, multilingual fraud or privacy tickets, requests with real-looking PII, and tickets requiring exact policy distinctions between refund, chargeback, account recovery, data deletion, and proctoring disputes.

I also expect hidden tests to include sparse subjects like "Help", follow-up-only conversations, cross-company requests, fake internal authority claims, forged support-article citations, and output-column injection where the correct behavior is to avoid over-answering and escalate or ask for clarification.

### One Known Failure Mode Not Fixed

The agent does not perform a second retrieval pass after reflection detects weak support. It can escalate safely, but it does not automatically reformulate the query and try again. A second-pass retriever would improve answer rate on tickets where the first retrieval set is close but not enough.
