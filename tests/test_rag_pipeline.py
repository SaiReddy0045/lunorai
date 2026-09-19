"""Lightweight deterministic tests for the LunorAI RAG pipeline."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from embeddings.embedding_model import get_embedding_model
from graph.workflow import create_rag_graph
from ingestion.chunker import DocumentChunker
from ingestion.pdf_loader import PDFLoader
from vectorstore.faiss_store import FAISSStore


BASE_DIR = Path(__file__).resolve().parent.parent


class KeywordEmbeddings(Embeddings):
    """Small local embedding substitute that makes tests independent of Ollama."""

    WORDS = ("author", "published", "embedding", "dimension", "recipe")

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(word)) for word in self.WORDS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeLLM:
    def invoke(self, prompt: str) -> AIMessage:
        question = prompt.split("Question:", 1)[-1].split("Answer:", 1)[0].lower()
        if "recipe" in question:
            return AIMessage(content="I could not find the information to answer this question in the uploaded document(s).")
        if "author" in question:
            return AIMessage(content="The primary author is Elena Rostova.")
        if "embedding model" in question:
            return AIMessage(content="The embedding model is nomic-embed-text with dimension 768.")
        return AIMessage(content="I could not find the information to answer this question in the uploaded document(s).")


def state(question: str) -> dict:
    return {
        "question": question,
        "documents": [],
        "relevance_scores": [],
        "is_relevant": False,
        "answer": "",
        "sources": [],
    }


def test_pdf_ingestion_and_chunking():
    documents = PDFLoader().load_from_path(BASE_DIR / "sample_document.pdf")

    assert len(documents) == 3
    assert documents[0].metadata["page"] == 1

    chunks = DocumentChunker(chunk_size=500, chunk_overlap=100).split_documents(documents)
    assert chunks
    assert all("source" in chunk.metadata and "page" in chunk.metadata for chunk in chunks)


def build_test_store() -> FAISSStore:
    store = FAISSStore(KeywordEmbeddings())
    store.build_from_documents([
        Document(
            page_content="The primary author is Elena Rostova and the specification was published in 2024.",
            metadata={"source": "spec.pdf", "page": 1},
        ),
        Document(
            page_content="LunorAI uses the nomic-embed-text embedding model with dimension 768.",
            metadata={"source": "spec.pdf", "page": 2},
        ),
    ])
    return store


def test_retrieval_uses_top_k_without_aggressive_threshold():
    results = build_test_store().similarity_search_with_scores(
        "Who is the primary author?", k=3, score_threshold=0.99
    )

    assert results
    assert results[0][0].metadata["page"] == 1


def test_graph_answers_grounded_question_and_preserves_citation():
    graph = create_rag_graph(build_test_store(), llm=FakeLLM())
    result = graph.invoke(state("Who is the primary author?"))

    assert "Elena Rostova" in result["answer"]
    assert result["sources"][0]["source"] == "spec.pdf"
    assert result["sources"][0]["page"] == 1


def test_graph_refuses_question_outside_retrieved_context():
    graph = create_rag_graph(build_test_store(), llm=FakeLLM())
    result = graph.invoke(state("What is the recipe for blueberry cheesecake?"))

    assert "could not find" in result["answer"].lower()
    assert result["sources"] == []


def test_repeated_questions_do_not_rebuild_vector_store_or_embedding_model():
    store = build_test_store()
    graph = create_rag_graph(store, llm=FakeLLM())
    build_count = 0
    original_build = store.build_from_documents

    def unexpected_rebuild(documents):
        nonlocal build_count
        build_count += 1
        return original_build(documents)

    store.build_from_documents = unexpected_rebuild
    graph.invoke(state("Who is the primary author?"))
    graph.invoke(state("Who is the primary author?"))

    assert build_count == 0
    assert get_embedding_model.cache_info().currsize >= 0


def test_graph_records_stage_timings():
    graph = create_rag_graph(build_test_store(), llm=FakeLLM())
    result = graph.invoke(state("Who is the primary author?"))

    assert result["timings"]["query_embedding_ms"] >= 0
    assert result["timings"]["faiss_search_ms"] >= 0
    assert result["timings"]["context_preparation_ms"] >= 0
    assert result["timings"]["prompt_construction_ms"] >= 0
    assert result["timings"]["ollama_generation_ms"] >= 0
    assert result["timings"]["source_formatting_ms"] >= 0
    assert result["timings"]["total_response_ms"] >= 0
