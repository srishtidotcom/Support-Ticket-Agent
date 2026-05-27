# MLE Hiring Challenge - Support Ticket Triage Agent

A robust, safety-first, hybrid retrieval agent that triages and resolves customer support tickets across **DevPlatform**, **Claude**, and **Visa** using only the provided documentation corpus.

**Primary Entry Point:** `code/main.py`

## Setup

```bash
cd ~/MLE-hiring

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r code/requirements.txt
```

## Local LLM Setup (Ollama)

```bash
ollama pull llama3.2:3b
ollama pull llama3.2:8b
```

## Build Retrieval Index

```bash
python code/build_index.py
```

This builds the hybrid FAISS + BM25 index used by the agent.

## Run the Agent

```bash
python code/main.py
```

This processes all tickets from `support_tickets/support_tickets.csv` and writes results to `support_tickets/output.csv`.

## Run Validation

```bash
python code/validate_output.py
```

## Expected Runtime

Approximately 1.5 to 2.5 minutes on CPU with warm Ollama models.

## Key Features

- Hybrid retrieval using FAISS embeddings plus BM25
- Multi-stage safety and adversarial detection
- Corpus sanitization for indirect prompt injection defense
- Adaptive second-pass retrieval
- Structured reflection and grounded response generation
- Deterministic tool calling with verify-before-destruct
- Full output schema compliance

## Project Structure

```text
code/
├── main.py              # Main pipeline entry point
├── build_index.py       # Builds hybrid retrieval index
├── config.py            # Model and path configuration
├── agents/              # Safety, routing, retrieval, etc.
├── core/                # Document models, chunking, retriever
├── tools/               # Tool engine and validation
└── requirements.txt
```
