"""Sparse text index over the posts (TF-IDF, word 1-2-grams, sublinear tf).  Retrieval is *lineage-blind* by design:
the lineage layer re-weights what was retrieved, it does not change the retriever (any retriever can be plugged in)."""
from __future__ import annotations
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextIndex:
    def __init__(self, texts: list[str]):
        t0 = time.perf_counter()
        self.texts = texts
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english", dtype=np.float32)
        self.M = self.vec.fit_transform(texts).tocsr()         # rows are L2-normalised
        self.build_seconds = time.perf_counter() - t0

    def search(self, query: str, k: int):
        q = self.vec.transform([query])
        s = (self.M @ q.T).toarray().ravel()
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return top.astype(np.int64), s[top].astype(np.float64)

    def pairwise(self, ids: np.ndarray) -> np.ndarray:
        sub = self.M[ids]
        return (sub @ sub.T).toarray()
