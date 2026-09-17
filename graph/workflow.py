"""
LunorAI - LangGraph RAG Workflow Orchestrator
Builds and compiles the stateful RAG workflow graph:
START -> retrieve_documents -> check_relevance -> generate_answer -> format_sources -> END
"""

import logging
from typing import Callable, Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import settings
from graph.state import RAGState
from vectorstore.faiss_store import FAISSStore
from llm.ollama_model import get_llm
from utils.helpers import deduplicate_sources

logger = logging.getLogger(__name__)


def load_prompt_template() -> str:
    """Reads the strict RAG prompt from the filesystem."""
    try:
        with open(settings.RAG_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Could not load custom prompt file ({e}), using default fallback prompt.")
        return (
            "You are LunorAI. Answer the question strictly using the provided context.\n"
            "If the context does not contain the answer, state that the information was not found.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )


def create_rag_graph(
    faiss_store: FAISSStore,
    llm=None,
    token_callback: Optional[Callable[[str], None]] = None,
):
    """
    Constructs and compiles the LangGraph state machine for LunorAI.
    """
    llm = llm or get_llm(streaming=True)
    prompt_raw = load_prompt_template()
    prompt_template = PromptTemplate(
        template=prompt_raw,
        input_variables=["context", "question"],
    )

    # -------------------------------------------------------------
    # 1. Node: retrieve_documents
    # -------------------------------------------------------------
    def retrieve_documents(state: RAGState) -> Dict[str, Any]:
        question = state.get("question", "").strip()
        logger.info(f"[Graph] retrieve_documents for: '{question}'")

        if not question:
            return {"documents": [], "relevance_scores": []}

        scored_docs = faiss_store.similarity_search_with_scores(
            query=question,
            k=settings.TOP_K_RETRIEVAL,
            score_threshold=settings.RELEVANCE_SCORE_THRESHOLD,
        )

        docs: List[Document] = [doc for doc, _ in scored_docs]
        scores: List[float] = [float(score) for _, score in scored_docs]

        logger.info(f"[Graph] Retrieved {len(docs)} documents.")
        return {
            "documents": docs,
            "relevance_scores": scores,
        }

    # -------------------------------------------------------------
    # 2. Node: check_relevance
    # -------------------------------------------------------------
    def check_relevance(state: RAGState) -> Dict[str, Any]:
        docs = state.get("documents", [])
        scores = state.get("relevance_scores", [])

        # Check if we have documents and non-empty content
        if not docs:
            logger.info("[Graph] check_relevance: No documents retrieved -> is_relevant=False")
            return {"is_relevant": False}

        has_meaningful_content = any(len(d.page_content.strip()) > 20 for d in docs)
        if not has_meaningful_content:
            logger.info("[Graph] check_relevance: Content empty or negligible -> is_relevant=False")
            return {"is_relevant": False}

        logger.info(f"[Graph] check_relevance: Context validated ({len(docs)} chunks) -> is_relevant=True")
        return {"is_relevant": True}

    # -------------------------------------------------------------
    # 3. Node: generate_answer (for relevant context)
    # -------------------------------------------------------------
    def generate_answer(state: RAGState) -> Dict[str, Any]:
        question = state.get("question", "")
        docs = state.get("documents", [])

        formatted_context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'Doc')}, Page: {d.metadata.get('page', 1)}]\n{d.page_content}"
            for d in docs
        )

        formatted_prompt = prompt_template.format(
            context=formatted_context,
            question=question,
        )

        try:
            if token_callback is None:
                response = llm.invoke(formatted_prompt)
                answer_text = response.content if hasattr(response, "content") else str(response)
            else:
                answer_parts = []
                for chunk in llm.stream(formatted_prompt):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        answer_parts.append(content)
                        token_callback(content)
                answer_text = "".join(answer_parts)
        except Exception as e:
            logger.error(f"[Graph] LLM invocation failed: {e}")
            answer_text = (
                f"An error occurred while communicating with the local LLM: {str(e)}. "
                f"Please ensure Ollama is running with model '{settings.OLLAMA_MODEL}'."
            )

        return {"answer": answer_text}

    # -------------------------------------------------------------
    # 4. Node: fallback_unsupported (when context is insufficient)
    # -------------------------------------------------------------
    def fallback_unsupported(state: RAGState) -> Dict[str, Any]:
        logger.info("[Graph] fallback_unsupported: Returning honest out-of-context response.")
        fallback_msg = (
            "I could not find the information to answer this question in the uploaded document(s). "
            "Please verify if the topic is covered in your uploaded PDFs or try rephrasing your question."
        )
        return {"answer": fallback_msg}

    # -------------------------------------------------------------
    # 5. Node: format_sources
    # -------------------------------------------------------------
    def format_sources(state: RAGState) -> Dict[str, Any]:
        is_relevant = state.get("is_relevant", False)
        docs = state.get("documents", [])
        answer = state.get("answer", "").lower()

        # If context was not relevant or answer indicated missing info, omit false citations
        if (
            not is_relevant
            or not docs
            or "could not find the information" in answer
            or "information was not found" in answer
            or "not found in the uploaded document" in answer
        ):
            return {"sources": []}

        citations = deduplicate_sources(docs)
        logger.info(f"[Graph] format_sources: Formatted {len(citations)} unique citations.")
        return {"sources": citations}

    # -------------------------------------------------------------
    # Routing logic
    # -------------------------------------------------------------
    def decide_generation_path(state: RAGState) -> str:
        if state.get("is_relevant", False):
            return "generate_answer"
        return "fallback_unsupported"

    # Build Workflow Graph
    workflow = StateGraph(RAGState)

    # Register nodes
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("fallback_unsupported", fallback_unsupported)
    workflow.add_node("format_sources", format_sources)

    # Wire edges
    workflow.add_edge(START, "retrieve_documents")
    workflow.add_edge("retrieve_documents", "check_relevance")
    workflow.add_conditional_edges(
        "check_relevance",
        decide_generation_path,
        {
            "generate_answer": "generate_answer",
            "fallback_unsupported": "fallback_unsupported",
        },
    )
    workflow.add_edge("generate_answer", "format_sources")
    workflow.add_edge("fallback_unsupported", "format_sources")
    workflow.add_edge("format_sources", END)

    return workflow.compile()
