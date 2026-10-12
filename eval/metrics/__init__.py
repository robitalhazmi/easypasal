"""Metrics package: deterministic retrieval, citation, and abstention metrics."""

from __future__ import annotations

from eval.metrics.abstention import (
    AbstentionMetrics,
    abstention_accuracy,
    abstention_f1,
    abstention_precision,
    abstention_recall,
    abstention_specificity,
    compute_abstention_metrics,
    compute_confusion_matrix,
)
from eval.metrics.citations import (
    CitationMetrics,
    citation_f1,
    citation_precision,
    citation_recall,
    compute_citation_metrics,
    extract_article_key,
    normalize_citation,
)
from eval.metrics.retrieval import (
    RetrievalMetrics,
    compute_retrieval_metrics,
    dcg_at_k,
    hit_at_k,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "AbstentionMetrics",
    "CitationMetrics",
    "RetrievalMetrics",
    "abstention_accuracy",
    "abstention_f1",
    "abstention_precision",
    "abstention_recall",
    "abstention_specificity",
    "citation_f1",
    "citation_precision",
    "citation_recall",
    "compute_abstention_metrics",
    "compute_citation_metrics",
    "compute_confusion_matrix",
    "compute_retrieval_metrics",
    "dcg_at_k",
    "extract_article_key",
    "hit_at_k",
    "idcg_at_k",
    "ndcg_at_k",
    "normalize_citation",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
