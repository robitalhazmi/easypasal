"""Deterministic ranked retrieval metrics: Hit@k, Recall@k, Precision@k, MRR, and nDCG@k."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass


def hit_at_k(
    retrieved: Sequence[str],
    relevant: Collection[str],
    k: int,
) -> float:
    """Compute binary Hit@k (1.0 if at least one relevant document is in top-k, else 0.0).

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Collection of gold relevant document/chunk IDs.
        k: Ranking cut-off depth (must be > 0).

    Returns:
        1.0 if a relevant chunk is retrieved within top-k, 0.0 otherwise.
    """
    if k <= 0 or not retrieved:
        return 0.0
    for doc_id in retrieved[:k]:
        if doc_id in relevant:
            return 1.0
    return 0.0


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Collection[str],
    k: int,
) -> float:
    """Compute multi-relevant Recall@k.

    Recall@k = |Retrieved[:k] ∩ Relevant| / |Relevant|

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Collection of gold relevant document/chunk IDs.
        k: Ranking cut-off depth.

    Returns:
        Proportion of gold relevant documents retrieved in top-k.
        Returns 0.0 if relevant is empty or k <= 0.
    """
    if k <= 0 or not relevant:
        return 0.0
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    hits = sum(1 for doc_id in set(retrieved[:k]) if doc_id in rel_set)
    return hits / len(rel_set)


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Collection[str],
    k: int,
) -> float:
    """Compute Precision@k.

    Precision@k = |Retrieved[:k] ∩ Relevant| / k

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Collection of gold relevant document/chunk IDs.
        k: Ranking cut-off depth.

    Returns:
        Fraction of the top-k retrieved documents that are relevant.
        Returns 0.0 if k <= 0.
    """
    if k <= 0:
        return 0.0
    window = retrieved[:k]
    if not window:
        return 0.0
    hits = sum(1 for doc_id in window if doc_id in relevant)
    return hits / k


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: Collection[str],
    k: int | None = None,
) -> float:
    """Compute Reciprocal Rank (RR) at cut-off k.

    RR = 1 / rank of the first relevant document in top-k (1-indexed).

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Collection of gold relevant document/chunk IDs.
        k: Optional cut-off depth. If None, considers entire retrieved list.

    Returns:
        Reciprocal rank value in [0.0, 1.0].
        Returns 0.0 if no relevant document is found within top-k.
    """
    if k is not None and k <= 0:
        return 0.0
    items = retrieved if k is None else retrieved[:k]
    for rank_0, doc_id in enumerate(items):
        if doc_id in relevant:
            return 1.0 / (rank_0 + 1)
    return 0.0


def dcg_at_k(
    retrieved: Sequence[str],
    relevant: Collection[str] | Mapping[str, float],
    k: int,
) -> float:
    """Compute Discounted Cumulative Gain at rank k (DCG@k).

    DCG@k = sum_{i=1}^k (gain_i / log2(i + 1))

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Collection of binary relevant IDs or Mapping of ID -> graded gain.
        k: Cut-off depth.

    Returns:
        DCG score.
    """
    if k <= 0 or not retrieved:
        return 0.0

    dcg = 0.0
    if isinstance(relevant, Mapping):
        for i, doc_id in enumerate(retrieved[:k]):
            gain = relevant.get(doc_id, 0.0)
            if gain > 0.0:
                dcg += gain / math.log2(i + 2)
    else:
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(i + 2)

    return dcg


def idcg_at_k(
    relevant: Collection[str] | Mapping[str, float],
    k: int,
) -> float:
    """Compute Ideal Discounted Cumulative Gain at rank k (IDCG@k).

    Args:
        relevant: Collection of gold relevant IDs or Mapping of ID -> graded gain.
        k: Cut-off depth.

    Returns:
        IDCG score for ideal monotonic descending ranking.
    """
    if k <= 0:
        return 0.0

    if isinstance(relevant, Mapping):
        sorted_gains = sorted((g for g in relevant.values() if g > 0.0), reverse=True)
    else:
        # Binary relevance
        rel_count = len(set(relevant))
        sorted_gains = [1.0] * rel_count

    if not sorted_gains:
        return 0.0

    idcg = 0.0
    for i, gain in enumerate(sorted_gains[:k]):
        idcg += gain / math.log2(i + 2)

    return idcg


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Collection[str] | Mapping[str, float],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k (nDCG@k).

    nDCG@k = DCG@k / IDCG@k

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Collection of gold relevant IDs or Mapping of ID -> graded gain.
        k: Cut-off depth.

    Returns:
        nDCG score in [0.0, 1.0]. Returns 0.0 if IDCG is 0.
    """
    if k <= 0:
        return 0.0

    idcg = idcg_at_k(relevant, k)
    if idcg <= 0.0:
        return 0.0

    dcg = dcg_at_k(retrieved, relevant, k)
    return dcg / idcg


@dataclass(frozen=True)
class RetrievalMetrics:
    """Dataclass capturing all deterministic retrieval metrics for a single query."""

    k_values: list[int]
    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    has_relevant_items: bool = True

    def to_dict(self) -> dict[str, object]:
        """Convert metrics to a flat dictionary suitable for logging and tables."""
        res: dict[str, object] = {
            "has_relevant_items": self.has_relevant_items,
        }
        for k in self.k_values:
            res[f"hit@{k}"] = self.hit_at_k.get(k, 0.0)
            res[f"recall@{k}"] = self.recall_at_k.get(k, 0.0)
            res[f"precision@{k}"] = self.precision_at_k.get(k, 0.0)
            res[f"mrr@{k}"] = self.mrr_at_k.get(k, 0.0)
            res[f"ndcg@{k}"] = self.ndcg_at_k.get(k, 0.0)
        return res


def compute_retrieval_metrics(
    retrieved: Sequence[str],
    relevant: Collection[str] | Mapping[str, float],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    """Compute all standard retrieval metrics across multiple k thresholds for a single query.

    Args:
        retrieved: Ranked sequence of retrieved chunk IDs.
        relevant: Gold relevant chunk IDs or graded gain mapping.
        k_values: Cut-off depths to compute metrics for (default: 1, 3, 5, 10).

    Returns:
        Populated RetrievalMetrics dataclass.
    """
    sorted_k = sorted(k_values)
    if isinstance(relevant, Mapping):
        has_rel = any(v > 0.0 for v in relevant.values())
        rel_collection: Collection[str] = set(relevant.keys())
    else:
        has_rel = len(relevant) > 0
        rel_collection = relevant

    hits: dict[int, float] = {}
    recalls: dict[int, float] = {}
    precisions: dict[int, float] = {}
    mrrs: dict[int, float] = {}
    ndcgs: dict[int, float] = {}

    for k in sorted_k:
        hits[k] = hit_at_k(retrieved, rel_collection, k)
        recalls[k] = recall_at_k(retrieved, rel_collection, k)
        precisions[k] = precision_at_k(retrieved, rel_collection, k)
        mrrs[k] = reciprocal_rank(retrieved, rel_collection, k)
        ndcgs[k] = ndcg_at_k(retrieved, relevant, k)

    return RetrievalMetrics(
        k_values=list(sorted_k),
        hit_at_k=hits,
        recall_at_k=recalls,
        precision_at_k=precisions,
        mrr_at_k=mrrs,
        ndcg_at_k=ndcgs,
        has_relevant_items=has_rel,
    )
