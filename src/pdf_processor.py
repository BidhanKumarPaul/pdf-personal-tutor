"""
pdf_processor.py
Turns a raw PDF (as bytes or a file path) into a list of Chunk objects
ready for embedding. Keeps page and source-filename metadata so answers
can be traced back to exactly where they came from.
"""

from dataclasses import dataclass
from io import BytesIO
from typing import Union

import pdfplumber


@dataclass
class Chunk:
    text: str
    source: str  # original filename
    page: int    # 1-indexed page number where this chunk starts


def extract_pages(pdf_source: Union[str, bytes, BytesIO]) -> list[tuple[int, str]]:
    """
    Extract text from every page of a PDF.
    Returns a list of (page_number, page_text) tuples.
    Accepts a file path, raw bytes, or a BytesIO (e.g. from a Streamlit uploader).
    """
    if isinstance(pdf_source, (bytes, bytearray)):
        pdf_source = BytesIO(pdf_source)

    pages = []
    with pdfplumber.open(pdf_source) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))
    return pages


def chunk_document(
    pdf_source: Union[str, bytes, BytesIO],
    source_name: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """
    Full pipeline: extract text page-by-page, then split into overlapping
    word-count chunks. Each chunk remembers which page it started on so
    the tutor can cite it later.
    """
    pages = extract_pages(pdf_source)
    chunks: list[Chunk] = []

    for page_num, page_text in pages:
        words = page_text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append(Chunk(text=chunk_text, source=source_name, page=page_num))
            if end >= len(words):
                break
            start += chunk_size - overlap

    return chunks
