"""BM25 relevance scorer for Syntaxis-AI."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}|[\w]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


class BM25Scorer:
    """Okapi BM25 scorer. Scores content blocks against a user query."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_freqs: Counter[str] = Counter()
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    def fit(self, documents: Iterable[str]) -> "BM25Scorer":
        docs = list(documents)
        self._n_docs = len(docs)
        self._doc_lengths = []
        self._doc_freqs = Counter()
        for doc in docs:
            tokens = _tokenize(doc)
            self._doc_lengths.append(len(tokens))
            for t in set(tokens):
                self._doc_freqs[t] += 1
        self._avg_dl = sum(self._doc_lengths) / self._n_docs if self._n_docs else 0.0
        return self

    def score(self, query: str, document: str) -> float:
        if not query or not document or self._n_docs == 0:
            return 0.0
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(document)
        d_len = len(d_tokens)
        d_counts = Counter(d_tokens)
        avg_dl = self._avg_dl or 1.0
        n = self._n_docs
        total = 0.0
        for t in set(q_tokens):
            df = self._doc_freqs.get(t, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = d_counts.get(t, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * d_len / avg_dl)
            total += idf * (tf * (self.k1 + 1)) / denom if denom else 0.0
        return total

    def score_many(self, query: str, documents: list[str]) -> list[float]:
        return [self.score(query, d) for d in documents]
