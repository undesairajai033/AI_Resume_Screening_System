"""Sentence embeddings for semantic matching (all-MiniLM-L6-v2).

The model is loaded once, on first use. Long texts are split into chunks
(the model reads ~256 tokens at a time) and the chunk vectors are averaged,
so the whole resume counts, not just its first page.
"""

import numpy as np

from src.preprocessing.text_cleaning import chunk_words

MODEL_NAME = "all-MiniLM-L6-v2"
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(MODEL_NAME)
    return _encoder


def set_encoder(encoder) -> None:
    """Swap in another encoder (anything with .encode(list[str]) -> array). Used by tests."""
    global _encoder
    _encoder = encoder


def get_embedding(text: str) -> np.ndarray:
    """One L2-normalised vector for a whole document."""
    return get_embeddings([text])[0]


def get_embeddings(texts: list[str]) -> np.ndarray:
    """L2-normalised document vectors, one row per text."""
    encoder = _get_encoder()
    chunks, owner = [], []
    for i, text in enumerate(texts):
        parts = chunk_words(text) or [""]
        chunks.extend(parts)
        owner.extend([i] * len(parts))

    vectors = np.asarray(encoder.encode(chunks), dtype=float)
    vectors = _normalise(vectors)

    docs = np.zeros((len(texts), vectors.shape[1]))
    np.add.at(docs, np.asarray(owner), vectors)
    return _normalise(docs)


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)
