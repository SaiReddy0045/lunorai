"""
LunorAI - Ollama LLM Module
Initializes ChatOllama with configured parameters and provides connectivity diagnostics.
"""

import logging
from typing import Optional, Tuple
import requests
from langchain_ollama import ChatOllama
from config import settings

logger = logging.getLogger(__name__)


def check_ollama_status() -> Tuple[bool, str]:
    """
    Checks if the local Ollama server is running and if the configured model is available.
    Returns (is_ready, status_message).
    """
    try:
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Check if model name matches (e.g. 'llama3.2' or 'llama3.2:latest')
            target = settings.OLLAMA_MODEL
            model_found = any(target in m for m in models)
            if model_found:
                return True, f"Ollama is online. Model '{target}' is available."
            else:
                return False, (
                    f"Ollama is running, but model '{target}' was not found. "
                    f"Run `ollama pull {target}` in your terminal."
                )
        return False, f"Ollama returned unexpected HTTP {response.status_code}."
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}. Ensure the Ollama app/service is running."
    except Exception as e:
        return False, f"Ollama check failed: {str(e)}"


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = True,
) -> ChatOllama:
    """
    Returns an initialized LangChain ChatOllama instance.
    """
    model_name = model or settings.OLLAMA_MODEL
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE

    logger.info(f"Initializing ChatOllama with model='{model_name}', temp={temp}")
    return ChatOllama(
        model=model_name,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temp,
        streaming=streaming,
        num_predict=settings.LLM_MAX_TOKENS,
        num_ctx=settings.LLM_CONTEXT_SIZE,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
    )
