"""
LunorAI - Document Chunker Module
Splits loaded Documents into semantically coherent chunks using RecursiveCharacterTextSplitter.
Preserves page-level metadata across all chunk fragments.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings


class DocumentChunker:
    """Splits Documents into overlapping character windows while preserving source and page metadata."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Documents into chunks.
        Appends chunk_index metadata to each chunk for transparent tracking.
        """
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        # Enhance chunks with chunk index per document
        chunk_counters = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 1)
            key = f"{source}_p{page}"
            chunk_counters[key] = chunk_counters.get(key, 0) + 1
            chunk.metadata["chunk_num"] = chunk_counters[key]

        return chunks
