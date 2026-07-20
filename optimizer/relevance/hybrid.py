"""Hybrid relevance scorer (BM25 + Embeddings) for Syntaxis-AI."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

from .bm25 import BM25Scorer

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


def embedding_available() -> bool:
    return _ST_AVAILABLE


@dataclass
class RelevanceScore:
    """Container for a relevance score."""
    score: float
    method: str  # "bm25", "embedding", "hybrid"


class RelevanceScorer:
    """Base class for relevance scorers."""

    def score(self, query: str, document: str) -> float:
        raise NotImplementedError


class EmbeddingScorer(RelevanceScorer):
    """Pure embedding-based scorer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._embeddings = None

    def fit(self, documents: List[str]):
        if not _ST_AVAILABLE:
            raise ImportError("sentence-transformers required for EmbeddingScorer")
        self._model = SentenceTransformer(self.model_name)
        self._embeddings = self._model.encode(documents)

    def score(self, query: str, document: str) -> float:
        if not self._model:
            return 0.0
        q_emb = self._model.encode([query])[0]
        d_emb = self._model.encode([document])[0]
        return float(
            sum(a * b for a, b in zip(q_emb, d_emb))
            / (sum(a * a for a in q_emb) ** 0.5 * sum(b * b for b in d_emb) ** 0.5)
        )


class HybridScorer(RelevanceScorer):
    """BM25 + embedding hybrid. Falls back to pure BM25 if transformers missing."""

    def __init__(
        self,
        weights: tuple[float, float] = (0.6, 0.4),
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.bm25_weight, self.emb_weight = weights
        self._bm25 = BM25Scorer()
        self._embedder = None
        self._embedding_model = embedding_model
        self._use_embeddings = _ST_AVAILABLE
        self._doc_embeddings = None

    def fit(self, documents: list[str]) -> "HybridScorer":
        self._bm25.fit(documents)
        if self._use_embeddings:
            try:
                self._embedder = SentenceTransformer(self._embedding_model)
                self._doc_embeddings = self._embedder.encode(
                    documents, convert_to_numpy=True
                ).tolist()
            except Exception as e:
                logger.warning(f"Embedding failed, falling back to BM25: {e}")
                self._use_embeddings = False
        return self

    def score(self, query: str, document: str, doc_index: int | None = None) -> float:
        bm25_score = self._bm25.score(query, document)
        if not self._use_embeddings or not self._doc_embeddings:
            return min(1.0, bm25_score / 15.0)

        try:
            q_emb = self._embedder.encode([query])[0].tolist()
            d_emb = (
                self._doc_embeddings[doc_index]
                if doc_index is not None
                else self._embedder.encode([document])[0].tolist()
            )
            emb_score = max(
                0.0,
                sum(a * b for a, b in zip(q_emb, d_emb))
                / (
                    sum(a * a for a in q_emb) ** 0.5
                    * sum(b * b for b in d_emb) ** 0.5
                ),
            )
        except Exception:
            emb_score = 0.0

        return (
            self.bm25_weight * min(1.0, bm25_score / 15.0)
            + self.emb_weight * emb_score
        )


def create_scorer(use_embeddings: bool = True) -> HybridScorer:
    scorer = HybridScorer()
    if not use_embeddings:
        scorer._use_embeddings = False
    return scorer
