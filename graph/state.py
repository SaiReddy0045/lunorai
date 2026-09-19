"""
LunorAI - LangGraph State Definition
Defines the state structure that flows through the RAG workflow graph.
"""

from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    Represents the state of the LunorAI RAG workflow.
    """
    # The incoming user question
    question: str

    # Retrieved candidate documents from FAISS
    documents: List[Document]

    # Similarity/Relevance scores for retrieved documents
    relevance_scores: List[float]

    # Flag set by check_relevance node
    is_relevant: bool

    # The generated answer from the LLM or fallback message
    answer: str

    # Deduplicated list of source dictionaries {source, page, snippet}
    sources: List[Dict[str, Any]]

    # Development diagnostics for one question-answer request
    timings: Dict[str, float]
