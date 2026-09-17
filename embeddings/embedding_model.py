"""
LunorAI - Embedding Model Factory
Provides local, API-key-free embeddings using Ollama (nomic-embed-text)
with configurable fallback support.
"""

import logging
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from config import settings

logger = logging.getLogger(__name__)


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
        )
