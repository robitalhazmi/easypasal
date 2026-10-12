"""Statutory citation fidelity metrics: precision, recall, and F1 (exact and article-level)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

_ARTICLE_PATTERN = re.compile(
    r"^(?P<doc>.+?)(?:,\s*|\s+)(?P<pasal>pasal\s+\d+[a-z]?)",
    re.IGNORECASE,
)


def normalize_citation(cite: str) -> str:
    """Normalize citation string by stripping brackets, extra whitespace, and lowercase.

    Example:
        '[POJK 22 Tahun 2023, Pasal 75 ayat (5)]' -> 'pojk 22 tahun 2023, pasal 75 ayat (5)'
    """
    c = cite.strip().strip("[]").strip()
    return re.sub(r"\s+", " ", c).lower()


def extract_article_key(cite: str) -> str:
    """Extract document and article prefix for relaxed article-level matching.

    Examples:
        'pojk 22 tahun 2023, pasal 75 ayat (5)' -> 'pojk 22 tahun 2023, pasal 75'
        'uu nomor 4 tahun 2023, pasal 243' -> 'uu nomor 4 tahun 2023, pasal 243'
    """
    norm = normalize_citation(cite)
    match = _ARTICLE_PATTERN.search(norm)
    if match:
        doc = match.group("doc").strip()
        pasal = match.group("pasal").strip()
        return f"{doc}, {pasal}"
    # Fallback if structure does not have Pasal
    return norm


def citation_precision(
    predicted: Sequence[str],
    gold: Sequence[str],
    mode: Literal["exact", "article"] = "exact",
) -> float:
    """Compute precision of predicted citations against gold citations.

    Citation Precision = |Predicted ∩ Gold| / |Predicted|

    Special cases:
        - If both predicted and gold are empty: returns 1.0 (correctly cited nothing, e.g. on abstention).
        - If predicted is empty and gold is non-empty: returns 0.0 (missed citations).
        - If predicted is non-empty and gold is empty: returns 0.0 (hallucinated citations).

    Args:
        predicted: Sequence of predicted citation strings.
        gold: Sequence of gold citation strings.
        mode: 'exact' (canonical match) or 'article' (matches doc + pasal regardless of ayat).

    Returns:
        Precision score in [0.0, 1.0].
    """
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0

    if mode == "article":
        pred_set = {extract_article_key(c) for c in predicted}
        gold_set = {extract_article_key(c) for c in gold}
    else:
        pred_set = {normalize_citation(c) for c in predicted}
        gold_set = {normalize_citation(c) for c in gold}

    if not pred_set:
        return 0.0

    hits = sum(1 for c in pred_set if c in gold_set)
    return hits / len(pred_set)


def citation_recall(
    predicted: Sequence[str],
    gold: Sequence[str],
    mode: Literal["exact", "article"] = "exact",
) -> float:
    """Compute recall of predicted citations against gold citations.

    Citation Recall = |Predicted ∩ Gold| / |Gold|

    Special cases:
        - If both predicted and gold are empty: returns 1.0.
        - If gold is empty and predicted is non-empty: returns 0.0.
        - If predicted is empty and gold is non-empty: returns 0.0.

    Args:
        predicted: Sequence of predicted citation strings.
        gold: Sequence of gold citation strings.
        mode: 'exact' or 'article'.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0

    if mode == "article":
        pred_set = {extract_article_key(c) for c in predicted}
        gold_set = {extract_article_key(c) for c in gold}
    else:
        pred_set = {normalize_citation(c) for c in predicted}
        gold_set = {normalize_citation(c) for c in gold}

    if not gold_set:
        return 0.0

    hits = sum(1 for c in pred_set if c in gold_set)
    return hits / len(gold_set)


def citation_f1(
    predicted: Sequence[str],
    gold: Sequence[str],
    mode: Literal["exact", "article"] = "exact",
) -> float:
    """Compute F1 score for citations.

    Citation F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        predicted: Sequence of predicted citation strings.
        gold: Sequence of gold citation strings.
        mode: 'exact' or 'article'.

    Returns:
        F1 score in [0.0, 1.0].
    """
    p = citation_precision(predicted, gold, mode=mode)
    r = citation_recall(predicted, gold, mode=mode)
    if p + r == 0.0:
        return 0.0
    return 2.0 * (p * r) / (p + r)


@dataclass(frozen=True)
class CitationMetrics:
    """Dataclass capturing citation evaluation metrics for a single query."""

    exact_precision: float
    exact_recall: float
    exact_f1: float
    exact_match: bool
    article_precision: float
    article_recall: float
    article_f1: float
    article_match: bool
    predicted_count: int
    gold_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert metrics to a flat dictionary."""
        return {
            "exact_precision": self.exact_precision,
            "exact_recall": self.exact_recall,
            "exact_f1": self.exact_f1,
            "exact_match": self.exact_match,
            "article_precision": self.article_precision,
            "article_recall": self.article_recall,
            "article_f1": self.article_f1,
            "article_match": self.article_match,
            "predicted_count": self.predicted_count,
            "gold_count": self.gold_count,
        }


def compute_citation_metrics(
    predicted: Sequence[str],
    gold: Sequence[str],
) -> CitationMetrics:
    """Compute full citation metrics (exact and article-level) for a single query.

    Args:
        predicted: Sequence of predicted citation strings.
        gold: Sequence of gold canonical citation strings.

    Returns:
        Populated CitationMetrics dataclass.
    """
    p_exact = citation_precision(predicted, gold, mode="exact")
    r_exact = citation_recall(predicted, gold, mode="exact")
    f1_exact = citation_f1(predicted, gold, mode="exact")

    p_art = citation_precision(predicted, gold, mode="article")
    r_art = citation_recall(predicted, gold, mode="article")
    f1_art = citation_f1(predicted, gold, mode="article")

    norm_pred = {normalize_citation(c) for c in predicted}
    norm_gold = {normalize_citation(c) for c in gold}
    exact_match = norm_pred == norm_gold

    art_pred = {extract_article_key(c) for c in predicted}
    art_gold = {extract_article_key(c) for c in gold}
    art_match = art_pred == art_gold

    return CitationMetrics(
        exact_precision=p_exact,
        exact_recall=r_exact,
        exact_f1=f1_exact,
        exact_match=exact_match,
        article_precision=p_art,
        article_recall=r_art,
        article_f1=f1_art,
        article_match=art_match,
        predicted_count=len(norm_pred),
        gold_count=len(norm_gold),
    )
