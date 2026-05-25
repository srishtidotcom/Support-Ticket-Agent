import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "index"

OLLAMA_MODEL_SAFETY = "llama3.2:3b"
OLLAMA_MODEL_ROUTING = "llama3.2:3b"
OLLAMA_MODEL_RESPONSE = "llama3.2:8b"

TEMPERATURE = 0.0
SEED = 42
