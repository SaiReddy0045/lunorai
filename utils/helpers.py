"""
LunorAI - Helpers & Formatting Utilities
Validation, source citation formatting, and deduplication logic.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document


def validate_pdf_file(filename: str, file_bytes: bytes, max_size_mb: int = 50) -> Tuple[bool, str]:
    """
    Validates uploaded file extension, content header, and size limit.
    """
    if not filename.lower().endswith(".pdf"):
        return False, f"Invalid file '{filename}'. LunorAI only supports PDF files."

    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return False, f"File '{filename}' exceeds maximum allowed size of {max_size_mb} MB."

    # Check PDF magic bytes (%PDF-)
    if not file_bytes.startswith(b"%PDF-"):
        return False, f"File '{filename}' is not a valid or readable PDF document."

    return True, "Valid PDF file."


def deduplicate_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Deduplicates retrieved chunks by document name and page number.
    Returns a sorted list of unique citation metadata dictionaries with snippets.
    """
    seen = set()
    citations = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        key = (source, page)

        if key not in seen:
            seen.add(key)
            citations.append({
                "source": source,
                "page": page,
                "total_pages": doc.metadata.get("total_pages", None),
                "snippet": doc.page_content[:250].replace("\n", " ").strip() + "...",
            })

    # Keep citations ordered by retrieval relevance (most relevant first)
    return citations


def format_citation(source: str, page: int) -> str:
    """
    Returns standard citation line:
    Source: filename.pdf
    Page: 5
    """
    return f"Source: {source}\nPage: {page}"
