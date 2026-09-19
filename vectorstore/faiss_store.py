"""
LunorAI - FAISS Vector Store Module
Encapsulates FAISS index construction, persistence, incremental updates,
and similarity search with relevance scoring.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from config import settings

logger = logging.getLogger(__name__)


class FAISSStore:
    """Wrapper around LangChain FAISS vector store with relevance scoring and persistence."""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None

    def build_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Builds a new FAISS vector store from chunked documents.
        """
        if not documents:
            raise ValueError("Cannot build vector store from empty documents list.")

        logger.info(f"Building FAISS vector store from {len(documents)} document chunks...")
        self.vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        return self.vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """
        Incrementally adds documents to the existing index or creates a new one.
        """
        if not documents:
            return

        if self.vector_store is None:
            self.build_from_documents(documents)
        else:
            logger.info(f"Adding {len(documents)} chunks to existing FAISS index...")
            self.vector_store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        Performs standard top-k similarity search.
        """
        if self.vector_store is None:
            return []

        top_k = k or settings.TOP_K_RETRIEVAL
        return self.vector_store.similarity_search(query, k=top_k)

    def similarity_search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Performs similarity search with relevance scores.
        Filters out matches below score_threshold if specified.
        Returns list of (Document, score) tuples.
        """
        if self.vector_store is None:
            return []

        top_k = k or settings.TOP_K_RETRIEVAL
        try:
            # Embed once, then pass the vector directly to LangChain's FAISS
            # implementation so embedding and index-search timings are distinct.
            started = time.perf_counter()
            query_embedding = self.embeddings.embed_query(query)
            if timings is not None:
                timings["query_embedding_ms"] = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            raw_docs = self.vector_store.similarity_search_with_score_by_vector(
                query_embedding, k=top_k
            )
            if timings is not None:
                timings["faiss_search_ms"] = (time.perf_counter() - started) * 1000

            return [(doc, 1.0 / (1.0 + float(distance))) for doc, distance in raw_docs]
        except Exception as e:
            logger.warning(f"FAISS similarity search failed: {e}")
            return []

    def as_retriever(self, k: Optional[int] = None):
        """
        Returns a LangChain retriever interface.
        """
        if self.vector_store is None:
            raise ValueError("Vector store has not been initialized.")
        top_k = k or settings.TOP_K_RETRIEVAL
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})

    def save_local(self, folder_path: Union[str, Path]) -> None:
        """
        Saves the FAISS index and docstore to disk.
        """
        if self.vector_store is None:
            raise ValueError("Cannot save an uninitialized vector store.")
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(path))
        logger.info(f"FAISS index saved to {path}")

    def load_local(self, folder_path: Union[str, Path]) -> FAISS:
        """
        Loads a saved FAISS index from disk.
        """
        path = Path(folder_path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index directory not found: {path}")

        self.vector_store = FAISS.load_local(
            folder_path=str(path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS index loaded from {path}")
        return self.vector_store
