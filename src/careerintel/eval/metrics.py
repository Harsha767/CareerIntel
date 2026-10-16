"""Ranking metrics. `ranked` is an ordered list of item ids (best first)."""
import math
from typing import Dict, Sequence, Set


def precision_at_k(ranked: Sequence, relevant: Set, k: int) -> float:
    top = ranked[:k]
    return sum(1 for r in top if r in relevant) / k if k > 0 else 0.0


def recall_at_k(ranked: Sequence, relevant: Set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in ranked[:k] if r in relevant) / len(relevant)


def hit_rate_at_k(ranked: Sequence, relevant: Set, k: int) -> float:
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def mrr(ranked: Sequence, relevant: Set) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence, gains: Dict, k: int) -> float:
    """Graded NDCG. `gains` maps item id -> relevance grade (0 if absent)."""
    dcg = sum(gains.get(r, 0) / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
