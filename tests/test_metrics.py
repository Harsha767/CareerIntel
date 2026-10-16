import math
from careerintel.eval.metrics import (
    precision_at_k, recall_at_k, hit_rate_at_k, mrr, ndcg_at_k,
)


def test_precision_recall():
    ranked = ["a", "b", "c", "d"]
    rel = {"a", "c", "z"}
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert math.isclose(recall_at_k(ranked, rel, 4), 2 / 3)


def test_hit_and_mrr():
    assert hit_rate_at_k(["x", "y"], {"y"}, 1) == 0.0
    assert hit_rate_at_k(["x", "y"], {"y"}, 2) == 1.0
    assert mrr(["x", "y"], {"y"}) == 0.5


def test_ndcg_perfect_and_worse():
    gains = {"a": 3, "b": 1}
    assert math.isclose(ndcg_at_k(["a", "b"], gains, 2), 1.0)
    assert ndcg_at_k(["b", "a"], gains, 2) < 1.0
