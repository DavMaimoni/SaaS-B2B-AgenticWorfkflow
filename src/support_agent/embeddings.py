"""Text embeddings for the semantic cache.

- `e5`: intfloat/multilingual-e5-small through sentence-transformers (the model used in our RAG project).
- `hashing`: character n-gram hashing, no download, deterministic. Good enough for tests and for near-duplicate
  detection; calibrate the cache threshold again whenever you switch embedder.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from support_agent.config import settings

DIM = 768


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def normalize_text(text: str) -> str:
    """Strip greetings, signatures and noise so two phrasings of the same request look alike."""
    text = text.lower()
    text = re.sub(r"\b(hello|hi|dear [a-z ]+|dear|thanks|thank you|best regards|regards|kind regards)\b", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)            # dataset placeholders such as <tel_num>
    text = re.sub(r"[^a-z0-9à-ÿ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class HashingEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            t = normalize_text(t)
            words = t.split()
            grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
            grams += [t[j:j + 4] for j in range(max(0, len(t) - 3))]
            for g in grams:
                h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
                out[i, h % DIM] += 1.0 if h & 1 else -1.0
        return _normalize(out)


class E5Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode([f"query: {normalize_text(t)}" for t in texts], normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


@lru_cache(maxsize=2)
def get_embedder(kind: str | None = None):
    kind = kind or settings().embedder
    if kind == "e5":
        return E5Embedder(settings().e5_model)
    return HashingEmbedder()
