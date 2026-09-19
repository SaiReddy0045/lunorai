"""
LunorAI - Configuration and Settings
Centralized management of environment variables, models, and retrieval parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Ollama LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "128"))
LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "1536"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Embedding Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")  # 'ollama' or 'huggingface'
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Document Ingestion & Chunking Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval Configuration
TOP_K_RETRIEVAL = max(3, min(5, int(os.getenv("TOP_K_RETRIEVAL", "4"))))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))
# Kept for backwards-compatible configuration. Retrieval is top-k first and does
# not discard candidates using a fixed score cutoff because score ranges vary by
# embedding model and FAISS distance strategy.
RELEVANCE_SCORE_THRESHOLD = float(os.getenv("RELEVANCE_SCORE_THRESHOLD", "0.0"))

# Prompt Configuration
PROMPTS_DIR = BASE_DIR / "prompts"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_prompt.txt"

# Vectorstore Storage Directory
VECTORSTORE_DIR = BASE_DIR / "storage" / "faiss_index"
VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)
