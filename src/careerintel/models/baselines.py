"""Lexical baseline: TF-IDF + cosine similarity."""
from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfRanker:
    def __init__(self, ngram_range=(1, 2), min_df=1):
        self.vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, stop_words="english")
        self.matrix = None
        self.ids: list = []

    def fit(self, ids: Sequence, texts: Sequence[str]) -> "TfidfRanker":
        self.ids = list(ids)
        self.matrix = self.vec.fit_transform(texts)
        return self

    def rank(self, query: str, top_k: int = 10) -> list[tuple]:
        q = self.vec.transform([query])
        scores = linear_kernel(q, self.matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in order]
