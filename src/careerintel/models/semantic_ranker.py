"""CareerIntel: Semantic job ranking using transformer embeddings."""

from typing import List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRanker:
    """Rank job descriptions by semantic similarity to a user profile."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.ids: List = []
        self.embeddings = None

    def fit(
        self,
        ids: Sequence,
        texts: Sequence[str],
    ) -> "SemanticRanker":
        """Encode job descriptions into normalized embeddings."""

        if len(ids) != len(texts):
            raise ValueError(
                "The number of job IDs must match the number of texts."
            )

        if not texts:
            raise ValueError("At least one job description is required.")

        if len(set(ids)) != len(ids):
            raise ValueError("Job IDs must be unique.")

        cleaned_texts = [
            str(text).strip() if text is not None else ""
            for text in texts
        ]

        if any(not text for text in cleaned_texts):
            raise ValueError("Job descriptions cannot be empty.")

        self.ids = list(ids)

        self.embeddings = self.model.encode(
            cleaned_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return self

    def rank(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple]:
        """Return job IDs and cosine similarity scores, highest first."""

        if self.embeddings is None:
            raise RuntimeError("Call fit() before rank().")

        if not query or not query.strip():
            raise ValueError("The user profile query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Normalized vectors make the dot product cosine similarity.
        scores = (query_embedding @ self.embeddings.T).ravel()

        # Stable ordering makes ties reproducible.
        order = np.argsort(-scores, kind="stable")[:top_k]

        return [
            (self.ids[i], float(scores[i]))
            for i in order
        ]