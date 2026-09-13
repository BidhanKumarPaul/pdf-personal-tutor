"""
Basic sanity tests for the chunking logic — the part most likely to
silently break (off-by-one errors, empty pages, etc.) without a test.

Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import Chunk, chunk_document


class FakePage:
    """Mimics a pdfplumber page for testing without needing a real PDF file."""

    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


def test_chunk_document_splits_long_page(monkeypatch):
    words = ["word"] * 700  # longer than default chunk size
    fake_text = " ".join(words)

    import src.pdf_processor as pp

    def fake_extract_pages(_source):
        return [(1, fake_text)]

    monkeypatch.setattr(pp, "extract_pages", fake_extract_pages)

    chunks = chunk_document(
        pdf_source=b"fake",
        source_name="test.pdf",
        chunk_size=300,
        overlap=60,
    )

    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "test.pdf" for c in chunks)
    assert all(c.page == 1 for c in chunks)


def test_chunk_document_handles_empty_page(monkeypatch):
    import src.pdf_processor as pp

    def fake_extract_pages(_source):
        return []  # no extractable text, e.g. a scanned page

    monkeypatch.setattr(pp, "extract_pages", fake_extract_pages)

    chunks = chunk_document(
        pdf_source=b"fake",
        source_name="scanned.pdf",
        chunk_size=300,
        overlap=60,
    )

    assert chunks == []


def test_chunk_short_page_produces_one_chunk(monkeypatch):
    import src.pdf_processor as pp

    def fake_extract_pages(_source):
        return [(1, "short page with only a few words")]

    monkeypatch.setattr(pp, "extract_pages", fake_extract_pages)

    chunks = chunk_document(
        pdf_source=b"fake",
        source_name="short.pdf",
        chunk_size=300,
        overlap=60,
    )

    assert len(chunks) == 1
