"""
LunorAI - Comprehensive End-to-End Pipeline Verification
Tests PDF loading, chunking, FAISS vector indexing, relevance filtering,
grounded question answering with llama3.2, and out-of-context safety behavior.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import DocumentChunker
from embeddings.embedding_model import get_embedding_model
from vectorstore.faiss_store import FAISSStore
from graph.workflow import create_rag_graph


def run_pipeline_tests():
    print("=" * 60)
    print("  LunorAI Automated Verification Test Suite")
    print("=" * 60)

    sample_pdf = BASE_DIR / "sample_document.pdf"
    if not sample_pdf.exists():
        raise FileNotFoundError(f"Missing test PDF at {sample_pdf}")

    # 1. Test Ingestion
    print("\n[1/5] Testing PDFLoader...")
    loader = PDFLoader()
    documents = loader.load_from_path(sample_pdf)
    print(f"-> Successfully extracted {len(documents)} pages.")
    assert len(documents) == 3, f"Expected 3 pages, got {len(documents)}"
    for doc in documents:
        print(f"   Page {doc.metadata.get('page')}: {len(doc.page_content)} characters")
        assert "page" in doc.metadata and doc.metadata["page"] >= 1

    # 2. Test Chunking
    print("\n[2/5] Testing DocumentChunker...")
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.split_documents(documents)
    print(f"-> Generated {len(chunks)} text chunks.")
    assert len(chunks) >= 3, "Expected at least 3 chunks"
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata

    # 3. Test Embeddings & FAISS Indexing
    print("\n[3/5] Testing Embedding Model & FAISS Vector Index...")
    embeddings = get_embedding_model()
    store = FAISSStore(embeddings=embeddings)
    store.build_from_documents(chunks)
    print("-> FAISS index created successfully.")

    # 4. Test LangGraph Workflow Compilation
    print("\n[4/5] Compiling LangGraph State Machine...")
    graph = create_rag_graph(faiss_store=store)
    print("-> LangGraph workflow compiled successfully.")

    # 5. Test RAG Retrieval & In-Context Answering
    print("\n[5/5] Testing Question Answering & Safety Behavior...")

    # Case A: Factual in-context question (Page 1)
    q1 = "Who is the primary author of the LunorAI specification and when was it published?"
    print(f"\n--- Test Case A (Grounded Question): '{q1}' ---")
    res1 = graph.invoke({
        "question": q1,
        "documents": [],
        "relevance_scores": [],
        "is_relevant": False,
        "answer": "",
        "sources": [],
    })
    print("Answer:\n", res1["answer"])
    print("Sources:\n", res1["sources"])
    assert "Elena Rostova" in res1["answer"], "Failed to identify author Elena Rostova"
    assert len(res1["sources"]) > 0, "Expected at least one citation"
    assert res1["sources"][0]["page"] == 1, f"Expected Page 1 citation, got {res1['sources'][0]['page']}"

    # Case B: Factual in-context question (Page 2)
    q2 = "What embedding model and vector dimension does LunorAI use?"
    print(f"\n--- Test Case B (Technical Spec): '{q2}' ---")
    res2 = graph.invoke({
        "question": q2,
        "documents": [],
        "relevance_scores": [],
        "is_relevant": False,
        "answer": "",
        "sources": [],
    })
    print("Answer:\n", res2["answer"])
    print("Sources:\n", res2["sources"])
    assert "nomic-embed-text" in res2["answer"] or "768" in res2["answer"], "Failed to identify embedding specifications"
    assert any(s["page"] == 2 for s in res2["sources"]), "Expected citation from Page 2"

    # Case C: Out-of-Context / Hallucination-check question
    q3 = "What is the recipe for baking a blueberry cheesecake with dark chocolate ganache?"
    print(f"\n--- Test Case C (Out-of-Context Safety Check): '{q3}' ---")
    res3 = graph.invoke({
        "question": q3,
        "documents": [],
        "relevance_scores": [],
        "is_relevant": False,
        "answer": "",
        "sources": [],
    })
    print("Answer:\n", res3["answer"])
    print("Sources:\n", res3["sources"])
    # Assert assistant refuses or explicitly states information is missing
    answer_lower = res3["answer"].lower()
    assert (
        "could not find" in answer_lower
        or "not mentioned" in answer_lower
        or "does not contain" in answer_lower
        or "not found" in answer_lower
    ), "Safety check failed: LLM hallucinated out-of-context recipe!"

    print("\n" + "=" * 60)
    print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY! [PASS]")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_tests()
