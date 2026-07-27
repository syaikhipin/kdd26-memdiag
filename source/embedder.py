"""Embeddings for the memory providers. Offline default: hashed TF (numpy only)."""
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
from memory_store import tokenize

_DIM = 2048


def _stable_hash(token: str, dim: int = _DIM) -> int:
    h = 0
    for ch in token:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h % dim


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)


class TfidfHashEmbedder(Embedder):
    dim = _DIM

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts = Counter(_stable_hash(tok, self.dim) for tok in tokenize(text))
        for bucket, c in counts.items():
            vec[bucket] = 1.0 + math.log(c)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


class OpenAIEmbedder(Embedder):
    def __init__(self, client, model: str = "text-embedding-3-small", batch_size: int = 64) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.dim = 0

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            for item in resp.data:
                vec = np.asarray(item.embedding, dtype=np.float32)
                if self.dim == 0:
                    self.dim = vec.shape[0]
                out.append(vec)
        return out


def make_embedder(backend: str, base_url: str | None = None, api_key: str | None = None,
                  model: str = "text-embedding-3-small") -> Embedder:
    return TfidfHashEmbedder()
