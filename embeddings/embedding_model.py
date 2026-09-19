"""
LunorAI - Embedding Model Factory
Provides local, API-key-free embeddings using Ollama (nomic-embed-text)
with configurable fallback support.
"""

import logging
from functools import lru_cache
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from config import settings

logger = logging.getLogger(__name__)


def _embedding_keep_alive_seconds() -> int:
    value = settings.OLLAMA_KEEP_ALIVE
    if isinstance(value, int):
        return value
    value = str(value).strip().lower()
    if value.endswith("m"):
        return int(float(value[:-1]) * 60)
    if value.endswith("h"):
        return int(float(value[:-1]) * 3600)
    return int(float(value))


@lru_cache(maxsize=8)
def get_embedding_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Embeddings:
    """
    Returns an instantiated LangChain Embeddings model.
    Default uses Ollama with 'nomic-embed-text' for fast, offline, free embedding generation.
    """
    prov = (provider or settings.EMBEDDING_PROVIDER).lower()
    model = model_name or settings.EMBEDDING_MODEL

    if prov == "ollama":
        logger.info(f"Initializing OllamaEmbeddings with model: {model} at {settings.OLLAMA_BASE_URL}")
        return OllamaEmbeddings(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
            keep_alive=_embedding_keep_alive_seconds(),
        )

    elif prov == "huggingface":
        # If user explicitly chooses huggingface in settings
        logger.info(f"Initializing HuggingFaceEmbeddings with model: {model}")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    else:
        # Default fallback to Ollama
        logger.warning(f"Unknown provider '{prov}'. Falling back to OllamaEmbeddings.")
        return OllamaEmbeddings(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
            keep_alive=_embedding_keep_alive_seconds(),
        )
