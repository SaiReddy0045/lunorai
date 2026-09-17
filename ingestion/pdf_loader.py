"""
LunorAI - PDF Loader Module
Extracts text and page-level metadata from single or multiple PDF documents.
Supports both file paths and Streamlit file buffers (BytesIO).
"""

import io
from pathlib import Path
from typing import List, Union, BinaryIO
from langchain_core.documents import Document

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    import pypdf


class PDFLoader:
    """Loads PDF files and returns a list of LangChain Document objects with accurate page metadata."""

    def __init__(self):
        pass

    def load_from_bytes(self, file_bytes: Union[bytes, BinaryIO], filename: str) -> List[Document]:
        """
        Extract pages from an in-memory PDF byte buffer (e.g. from Streamlit file uploader).
        Returns a list of Documents with 1-based page numbers.
        """
        if isinstance(file_bytes, bytes):
            stream = io.BytesIO(file_bytes)
        else:
            stream = file_bytes

        documents: List[Document] = []

        if PYMUPDF_AVAILABLE:
            # Using PyMuPDF for fast and robust extraction
            if isinstance(stream, io.BytesIO):
                doc = pymupdf.open(stream=stream.getvalue(), filetype="pdf")
            else:
                doc = pymupdf.open(stream=stream.read(), filetype="pdf")

            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1,  # 1-based page indexing
                                "total_pages": total_pages,
                            }
                        )
                    )
            doc.close()
        else:
            # Fallback using pypdf
            reader = pypdf.PdfReader(stream)
            total_pages = len(reader.pages)
            for page_idx, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_idx + 1,
                                "total_pages": total_pages,
                            }
                        )
                    )

        return documents

    def load_from_path(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Extract pages from a PDF file path on the filesystem.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        with open(path, "rb") as f:
            return self.load_from_bytes(f.read(), filename=path.name)

    def load_multiple(self, file_items: List[dict]) -> List[Document]:
        """
        Extract pages from multiple files.
        Each item in file_items can be:
        - {'bytes': bytes_or_stream, 'name': 'file.pdf'}
        - or a Path/str
        """
        all_documents: List[Document] = []
        for item in file_items:
            if isinstance(item, (str, Path)):
                all_documents.extend(self.load_from_path(item))
            elif isinstance(item, dict) and "bytes" in item and "name" in item:
                all_documents.extend(self.load_from_bytes(item["bytes"], filename=item["name"]))
        return all_documents
