"""Unit tests for deterministic evaluation metrics (T3.1).

Verifies mathematically exact toy examples, boundary conditions, edge cases,
and dataclass serialization for retrieval, citation, and abstention metrics.
"""

from __future__ import annotations

import math

import pytest
from eval.metrics import (
    AbstentionMetrics,
    CitationMetrics,
    RetrievalMetrics,
    abstention_accuracy,
    abstention_f1,
    abstention_precision,
    abstention_recall,
    abstention_specificity,
    citation_f1,
    citation_precision,
    citation_recall,
    compute_abstention_metrics,
    compute_citation_metrics,
    compute_confusion_matrix,
    compute_retrieval_metrics,
    dcg_at_k,
    extract_article_key,
    hit_at_k,
    idcg_at_k,
    ndcg_at_k,
    normalize_citation,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ==============================================================================
# 1. Retrieval Metrics Tests
# ==============================================================================


class TestRetrievalMetrics:
    """Test suite for ranked retrieval metrics."""

    def test_empty_and_zero_cutoffs(self) -> None:
        """Verify boundary cases with empty inputs or k <= 0."""
        assert hit_at_k([], {"c1"}, k=5) == 0.0
        assert hit_at_k(["c1"], set(), k=5) == 0.0
        assert hit_at_k(["c1"], {"c1"}, k=0) == 0.0
        assert hit_at_k(["c1"], {"c1"}, k=-2) == 0.0

        assert recall_at_k([], {"c1"}, k=5) == 0.0
        assert recall_at_k(["c1"], set(), k=5) == 0.0
        assert recall_at_k(["c1"], {"c1"}, k=0) == 0.0

        assert precision_at_k([], {"c1"}, k=5) == 0.0
        assert precision_at_k(["c1"], set(), k=5) == 0.0
        assert precision_at_k(["c1"], {"c1"}, k=0) == 0.0

        assert reciprocal_rank([], {"c1"}) == 0.0
        assert reciprocal_rank(["c1"], set()) == 0.0
        assert reciprocal_rank(["c1"], {"c1"}, k=0) == 0.0

        assert dcg_at_k([], {"c1"}, k=5) == 0.0
        assert idcg_at_k(set(), k=5) == 0.0
        assert ndcg_at_k([], {"c1"}, k=5) == 0.0
        assert ndcg_at_k(["c1"], set(), k=5) == 0.0

    def test_single_target_at_rank_1(self) -> None:
        """Verify perfect single-document retrieval at top rank."""
        retrieved = ["c1", "c2", "c3"]
        relevant = {"c1"}

        assert hit_at_k(retrieved, relevant, k=1) == 1.0
        assert hit_at_k(retrieved, relevant, k=3) == 1.0

        assert recall_at_k(retrieved, relevant, k=1) == 1.0
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

        assert precision_at_k(retrieved, relevant, k=1) == 1.0
        assert pytest.approx(precision_at_k(retrieved, relevant, k=3), rel=1e-5) == 1.0 / 3

        assert reciprocal_rank(retrieved, relevant, k=1) == 1.0
        assert reciprocal_rank(retrieved, relevant, k=3) == 1.0

        # DCG@1 = 1 / log2(2) = 1.0; IDCG@1 = 1.0 -> nDCG@1 = 1.0
        assert ndcg_at_k(retrieved, relevant, k=1) == 1.0
        assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_single_target_at_rank_2(self) -> None:
        """Verify metrics when the relevant document appears at rank 2."""
        retrieved = ["c2", "c1", "c3"]
        relevant = {"c1"}

        assert hit_at_k(retrieved, relevant, k=1) == 0.0
        assert hit_at_k(retrieved, relevant, k=2) == 1.0

        assert recall_at_k(retrieved, relevant, k=1) == 0.0
        assert recall_at_k(retrieved, relevant, k=2) == 1.0

        assert precision_at_k(retrieved, relevant, k=1) == 0.0
        assert precision_at_k(retrieved, relevant, k=2) == 0.5
        assert pytest.approx(precision_at_k(retrieved, relevant, k=3), rel=1e-5) == 1.0 / 3

        assert reciprocal_rank(retrieved, relevant, k=1) == 0.0
        assert reciprocal_rank(retrieved, relevant, k=2) == 0.5
        assert reciprocal_rank(retrieved, relevant, k=3) == 0.5

        assert ndcg_at_k(retrieved, relevant, k=1) == 0.0
        # DCG@2 = 1 / log2(3) ≈ 0.63092975; IDCG@2 = 1.0
        expected_ndcg = 1.0 / math.log2(3)
        assert pytest.approx(ndcg_at_k(retrieved, relevant, k=2), rel=1e-5) == expected_ndcg

    def test_multi_target_exact_hand_calculated(self) -> None:
        """Verify multi-relevant calculation matching exact theoretical formulas."""
        retrieved = ["c1", "c3", "c2", "c4"]
        relevant = {"c1", "c2"}  # 2 gold targets

        assert recall_at_k(retrieved, relevant, k=1) == 0.5
        assert recall_at_k(retrieved, relevant, k=2) == 0.5
        assert recall_at_k(retrieved, relevant, k=3) == 1.0
        assert recall_at_k(retrieved, relevant, k=4) == 1.0

        assert precision_at_k(retrieved, relevant, k=1) == 1.0
        assert precision_at_k(retrieved, relevant, k=2) == 0.5
        assert pytest.approx(precision_at_k(retrieved, relevant, k=3), rel=1e-5) == 2.0 / 3
        assert precision_at_k(retrieved, relevant, k=4) == 0.5

        # RR: first hit is at rank 1 -> 1.0
        assert reciprocal_rank(retrieved, relevant, k=3) == 1.0

        # DCG@3:
        # rank 1 ('c1'): 1 / log2(2) = 1.0
        # rank 2 ('c3'): 0
        # rank 3 ('c2'): 1 / log2(4) = 0.5
        # Total DCG@3 = 1.5
        # IDCG@3:
        # ideal has both at rank 1 & 2 -> 1 / log2(2) + 1 / log2(3) = 1.0 + 1/log2(3)
        expected_idcg_3 = 1.0 + (1.0 / math.log2(3))
        expected_ndcg_3 = 1.5 / expected_idcg_3
        assert pytest.approx(dcg_at_k(retrieved, relevant, k=3), rel=1e-5) == 1.5
        assert pytest.approx(idcg_at_k(relevant, k=3), rel=1e-5) == expected_idcg_3
        assert pytest.approx(ndcg_at_k(retrieved, relevant, k=3), rel=1e-5) == expected_ndcg_3

    def test_graded_relevance_ndcg(self) -> None:
        """Verify graded relevance mapping in DCG / nDCG calculation."""
        retrieved_perfect = ["c1", "c2"]
        retrieved_inverted = ["c2", "c1"]
        graded = {"c1": 2.0, "c2": 1.0}

        # Perfect: DCG = 2/log2(2) + 1/log2(3) = 2.0 + 1/log2(3)
        idcg_2 = 2.0 + (1.0 / math.log2(3))
        assert pytest.approx(idcg_at_k(graded, k=2), rel=1e-5) == idcg_2
        assert ndcg_at_k(retrieved_perfect, graded, k=2) == 1.0

        # Inverted: DCG = 1/log2(2) + 2/log2(3) = 1.0 + 2/log2(3)
        dcg_inv = 1.0 + (2.0 / math.log2(3))
        expected_inv_ndcg = dcg_inv / idcg_2
        assert (
            pytest.approx(ndcg_at_k(retrieved_inverted, graded, k=2), rel=1e-5) == expected_inv_ndcg
        )

    def test_compute_retrieval_metrics_dataclass(self) -> None:
        """Verify compute_retrieval_metrics aggregates across multiple k values and exports to dict."""
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = {"c1", "c3"}

        res: RetrievalMetrics = compute_retrieval_metrics(retrieved, relevant, k_values=(1, 3, 5))

        assert res.k_values == [1, 3, 5]
        assert res.has_relevant_items is True
        assert res.hit_at_k[1] == 1.0
        assert res.recall_at_k[1] == 0.5
        assert res.recall_at_k[3] == 1.0
        assert res.recall_at_k[5] == 1.0

        d = res.to_dict()
        assert d["has_relevant_items"] is True
        assert "hit@1" in d
        assert "recall@3" in d
        assert "ndcg@5" in d


# ==============================================================================
# 2. Citation Metrics Tests
# ==============================================================================


class TestCitationMetrics:
    """Test suite for statutory citation fidelity metrics."""

    def test_normalization_and_article_extraction(self) -> None:
        """Verify citation normalization and article key regex extraction."""
        cite1 = "  [POJK 22 Tahun 2023, Pasal 75 ayat (5)]  "
        assert normalize_citation(cite1) == "pojk 22 tahun 2023, pasal 75 ayat (5)"
        assert extract_article_key(cite1) == "pojk 22 tahun 2023, pasal 75"

        cite2 = "[UU Nomor 4 Tahun 2023, Pasal 243]"
        assert normalize_citation(cite2) == "uu nomor 4 tahun 2023, pasal 243"
        assert extract_article_key(cite2) == "uu nomor 4 tahun 2023, pasal 243"

        cite3 = "[Penjelasan POJK 11/POJK.03/2022, Pasal 5 ayat (2)]"
        assert extract_article_key(cite3) == "penjelasan pojk 11/pojk.03/2022, pasal 5"

    def test_empty_citation_sets(self) -> None:
        """Verify precision, recall, and F1 when one or both citation sets are empty."""
        # Both empty: 100% agreement (e.g. valid abstention case)
        assert citation_precision([], []) == 1.0
        assert citation_recall([], []) == 1.0
        assert citation_f1([], []) == 1.0

        # Predicted citations on an unanswerable query (hallucination)
        pred = ["POJK 22 Tahun 2023, Pasal 75"]
        assert citation_precision(pred, []) == 0.0
        assert citation_recall(pred, []) == 0.0
        assert citation_f1(pred, []) == 0.0

        # Missed required citations
        gold = ["POJK 22 Tahun 2023, Pasal 75"]
        assert citation_precision([], gold) == 0.0
        assert citation_recall([], gold) == 0.0
        assert citation_f1([], gold) == 0.0

    def test_exact_vs_article_level_matching(self) -> None:
        """Verify exact canonical matching vs relaxed article-level matching."""
        pred = ["POJK 22 Tahun 2023, Pasal 75 ayat (1)"]
        gold = ["POJK 22 Tahun 2023, Pasal 75 ayat (5)"]

        # Exact: mismatch due to ayat (1) vs (5)
        assert citation_precision(pred, gold, mode="exact") == 0.0
        assert citation_recall(pred, gold, mode="exact") == 0.0
        assert citation_f1(pred, gold, mode="exact") == 0.0

        # Article-level: match since both target POJK 22 Tahun 2023, Pasal 75
        assert citation_precision(pred, gold, mode="article") == 1.0
        assert citation_recall(pred, gold, mode="article") == 1.0
        assert citation_f1(pred, gold, mode="article") == 1.0

    def test_over_and_under_citation(self) -> None:
        """Verify precision and recall balance on partial overlap."""
        c1 = "POJK 22 Tahun 2023, Pasal 75 ayat (5)"
        c2 = "POJK 61/POJK.07/2020, Pasal 4"
        c3 = "UU Nomor 27 Tahun 2022, Pasal 4"

        # Over-citation: predicted 2, gold 1 -> P = 0.5, R = 1.0
        assert citation_precision([c1, c2], [c1]) == 0.5
        assert citation_recall([c1, c2], [c1]) == 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3 ≈ 0.6667
        assert pytest.approx(citation_f1([c1, c2], [c1]), rel=1e-5) == 2.0 / 3

        # Under-citation: predicted 1, gold 2 -> P = 1.0, R = 0.5
        assert citation_precision([c1], [c1, c2]) == 1.0
        assert citation_recall([c1], [c1, c2]) == 0.5
        assert pytest.approx(citation_f1([c1], [c1, c2]), rel=1e-5) == 2.0 / 3

        # Disjoint: predicted 1, gold 1 -> 0.0
        assert citation_precision([c1], [c3]) == 0.0
        assert citation_recall([c1], [c3]) == 0.0
        assert citation_f1([c1], [c3]) == 0.0

    def test_compute_citation_metrics_dataclass(self) -> None:
        """Verify compute_citation_metrics computes all fields accurately."""
        pred = ["[POJK 22 Tahun 2023, Pasal 75 ayat (1)]"]
        gold = ["POJK 22 Tahun 2023, Pasal 75 ayat (5)"]

        metrics: CitationMetrics = compute_citation_metrics(pred, gold)

        assert metrics.exact_match is False
        assert metrics.article_match is True
        assert metrics.exact_precision == 0.0
        assert metrics.article_precision == 1.0
        assert metrics.article_recall == 1.0
        assert metrics.article_f1 == 1.0
        assert metrics.predicted_count == 1
        assert metrics.gold_count == 1

        d = metrics.to_dict()
        assert d["article_f1"] == 1.0
        assert d["exact_match"] is False


# ==============================================================================
# 3. Abstention Metrics Tests
# ==============================================================================


class TestAbstentionMetrics:
    """Test suite for abstention and safety metrics."""

    def test_unequal_length_raises_error(self) -> None:
        """Verify ValueError when predictions and gold sequences have different lengths."""
        with pytest.raises(ValueError, match="Predictions length"):
            compute_confusion_matrix([True, False], [True])

    def test_empty_predictions(self) -> None:
        """Verify behavior on empty lists."""
        assert abstention_accuracy([], []) == 0.0
        assert abstention_precision([], []) == 0.0
        assert abstention_recall([], []) == 0.0
        assert abstention_f1([], []) == 0.0
        assert abstention_specificity([], []) == 0.0

    def test_perfect_classifications(self) -> None:
        """Verify 100% accuracy, precision, recall, F1, and specificity."""
        pred = [True, False, True, False]
        gold = [True, False, True, False]

        tp, fp, fn, tn = compute_confusion_matrix(pred, gold)
        assert (tp, fp, fn, tn) == (2, 0, 0, 2)

        assert abstention_accuracy(pred, gold) == 1.0
        assert abstention_precision(pred, gold) == 1.0
        assert abstention_recall(pred, gold) == 1.0
        assert abstention_f1(pred, gold) == 1.0
        assert abstention_specificity(pred, gold) == 1.0

        metrics = compute_abstention_metrics(pred, gold)
        assert metrics.satisfies_benchmark is True

    def test_all_false_negatives_hallucinations(self) -> None:
        """Verify zero accuracy when model fails to refuse unanswerable queries."""
        pred = [False, False]  # Attempted to answer both
        gold = [True, True]  # Both should have abstained

        tp, fp, fn, tn = compute_confusion_matrix(pred, gold)
        assert (tp, fp, fn, tn) == (0, 0, 2, 0)

        assert abstention_accuracy(pred, gold) == 0.0
        assert abstention_precision(pred, gold) == 0.0
        assert abstention_recall(pred, gold) == 0.0
        assert abstention_f1(pred, gold) == 0.0
        assert abstention_specificity(pred, gold) == 1.0

        metrics = compute_abstention_metrics(pred, gold)
        assert metrics.satisfies_benchmark is False

    def test_mixed_confusion_matrix_hand_calculated(self) -> None:
        """Verify hand-computed mixed matrix."""
        # Query 1: Pred=T, Gold=T -> TP
        # Query 2: Pred=T, Gold=F -> FP
        # Query 3: Pred=F, Gold=T -> FN
        # Query 4: Pred=F, Gold=F -> TN
        # Query 5: Pred=T, Gold=F -> FP
        pred = [True, True, False, False, True]
        gold = [True, False, True, False, False]

        tp, fp, fn, tn = compute_confusion_matrix(pred, gold)
        assert tp == 1
        assert fp == 2
        assert fn == 1
        assert tn == 1

        # Accuracy = (1 + 1) / 5 = 0.40
        assert abstention_accuracy(pred, gold) == 0.40

        # Precision = 1 / (1 + 2) = 1/3
        assert pytest.approx(abstention_precision(pred, gold), rel=1e-5) == 1.0 / 3

        # Recall = 1 / (1 + 1) = 0.50
        assert abstention_recall(pred, gold) == 0.50

        # F1 = 2 * (1/3 * 0.5) / (1/3 + 0.5) = (1/3) / (5/6) = 2/5 = 0.40
        assert pytest.approx(abstention_f1(pred, gold), rel=1e-5) == 0.40

        # Specificity = TN / (TN + FP) = 1 / (1 + 2) = 1/3
        assert pytest.approx(abstention_specificity(pred, gold), rel=1e-5) == 1.0 / 3

    def test_compute_abstention_metrics_dataclass_and_format(self) -> None:
        """Verify AbstentionMetrics formatting and dictionary export."""
        pred = [True] * 19 + [False]  # 19 abstains, 1 answered
        gold = [True] * 20  # 20 should abstain
        # 19 TP, 0 FP, 1 FN, 0 TN -> Accuracy = 19/20 = 95.0%

        metrics: AbstentionMetrics = compute_abstention_metrics(pred, gold, target_threshold=0.95)
        assert metrics.accuracy == 0.95
        assert metrics.satisfies_benchmark is True
        assert metrics.tp == 19
        assert metrics.fn == 1

        summary = metrics.format_summary()
        assert "95.0%" in summary
        assert "PASS" in summary

        d = metrics.to_dict()
        assert d["accuracy"] == 0.95
        assert d["satisfies_benchmark"] is True


def test_top_level_eval_package_imports() -> None:
    """Verify that all core metrics can be imported directly from the top-level eval package."""
    import eval as ev

    assert callable(ev.compute_retrieval_metrics)
    assert callable(ev.compute_citation_metrics)
    assert callable(ev.compute_abstention_metrics)
    assert callable(ev.hit_at_k)
    assert callable(ev.recall_at_k)
    assert callable(ev.reciprocal_rank)
    assert callable(ev.ndcg_at_k)
    assert callable(ev.citation_precision)
    assert callable(ev.citation_recall)
    assert callable(ev.citation_f1)
    assert callable(ev.abstention_accuracy)

    assert ev.RetrievalMetrics is RetrievalMetrics
    assert ev.CitationMetrics is CitationMetrics
    assert ev.AbstentionMetrics is AbstentionMetrics
