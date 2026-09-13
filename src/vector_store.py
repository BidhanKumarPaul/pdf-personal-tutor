"""
vector_store.py
Wraps a local sentence-transformers embedding model and a FAISS index
so the rest of the app can just call `.build()` and `.search()`.
Everything here runs locally — no API calls, no cost.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL_NAME
from src.pdf_processor import Chunk


class VectorStore:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and build a fresh FAISS index in memory."""
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # cosine similarity via normalized inner product
        index.add(embeddings)

        self.index = index
        self.chunks = chunks

    def add(self, chunks: list[Chunk]) -> None:
        """Add more chunks to an existing index (e.g. a second uploaded PDF)."""
        if self.index is None:
            self.build(chunks)
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int) -> list[Chunk]:
        """Return the k chunks most relevant to the query."""
        if self.index is None or not self.chunks:
            return []

        q_vec = self.embedder.encode([query], normalize_embeddings=True)
        q_vec = np.asarray(q_vec, dtype="float32")

        k = min(k, len(self.chunks))
        _, ids = self.index.search(q_vec, k)
        return [self.chunks[i] for i in ids[0] if i != -1]

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0


def load_embedder() -> SentenceTransformer:
    """Loads the local embedding model. Call once and cache in the app layer."""
    return SentenceTransformer(EMBED_MODEL_NAME)
