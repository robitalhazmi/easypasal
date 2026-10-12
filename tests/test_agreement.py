"""Unit tests for the annotator self-agreement protocol and evaluation metrics (T2.5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from eval.compute_self_agreement import (
    AgreementReport,
    compute_abstention_agreement,
    compute_citation_jaccard,
    compute_cohens_kappa,
    compute_item_token_f1,
    evaluate_agreement,
    sample_for_relabeling,
)
from eval.labeling_helper import GoldItem
from eval.split_eval import load_gold_items

ORIGINAL_PATH = Path("data/gold/gold_dataset.jsonl")
RELABEL_PATH = Path("data/gold/relabel_annotations.jsonl")


def test_sample_for_relabeling_properties() -> None:
    """Verify stratified 20% sampling properties and seed reproducibility."""
    items = load_gold_items(ORIGINAL_PATH)
    sample_1 = sample_for_relabeling(items, sample_ratio=0.2, seed=42)
    sample_2 = sample_for_relabeling(items, sample_ratio=0.2, seed=42)

    # 1. Size check: 20 items out of 106
    assert len(sample_1) == 20
    # 2. Seed reproducibility
    assert [x.id for x in sample_1] == [x.id for x in sample_2]

    # 3. All 9 categories represented in sample
    categories = {it.category for it in sample_1}
    assert len(categories) == 9


def test_abstention_agreement_metric() -> None:
    """Verify abstention agreement calculation on identical and mismatched lists."""
    item_abstain = GoldItem(
        id="gold_001",
        category="unanswerable_out_of_scope",
        theme="General",
        question="Pertanyaan?",
        gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut.",
        should_abstain=True,
        abstain_reason="out_of_corpus",
    )
    item_answer = GoldItem(
        id="gold_002",
        category="factual_direct",
        theme="General",
        question="Pertanyaan valid?",
        gold_answer="Berdasarkan [POJK 22 Tahun 2023, Pasal 75 ayat (1)], jawaban.",
        gold_citations=["POJK 22 Tahun 2023, Pasal 75 ayat (1)"],
        relevant_chunk_ids=["POJK_22_Tahun_2023_pasal_75_ayat_1"],
        should_abstain=False,
    )

    # Identical: 100% agreement
    assert (
        compute_abstention_agreement([item_abstain, item_answer], [item_abstain, item_answer])
        == 1.0
    )

    # 1 mismatch out of 2: 50% agreement
    item_mismatch = item_answer.model_copy(
        update={"should_abstain": True, "abstain_reason": "out_of_corpus"}
    )
    assert (
        compute_abstention_agreement([item_abstain, item_answer], [item_abstain, item_mismatch])
        == 0.5
    )


def test_cohens_kappa_metric() -> None:
    """Verify Cohen's Kappa handles agreement, perfect match, and chance."""
    item_t1 = GoldItem(
        id="gold_001",
        category="unanswerable_out_of_scope",
        theme="General",
        question="Pertanyaan?",
        gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut.",
        should_abstain=True,
        abstain_reason="out_of_corpus",
    )
    item_f1 = GoldItem(
        id="gold_002",
        category="factual_direct",
        theme="General",
        question="Pertanyaan valid?",
        gold_answer="Berdasarkan [POJK 22 Tahun 2023, Pasal 75 ayat (1)], jawaban.",
        gold_citations=["POJK 22 Tahun 2023, Pasal 75 ayat (1)"],
        relevant_chunk_ids=["POJK_22_Tahun_2023_pasal_75_ayat_1"],
        should_abstain=False,
    )

    # Perfect agreement
    kappa_perfect = compute_cohens_kappa([item_t1, item_f1], [item_t1, item_f1])
    assert kappa_perfect == 1.0


def test_citation_jaccard_similarity() -> None:
    """Verify citation Jaccard handles exact matches, empty sets, and partial overlaps."""
    item_abstain_1 = GoldItem(
        id="gold_001",
        category="unanswerable_out_of_scope",
        theme="General",
        question="Pertanyaan uji coba?",
        gold_answer="Jawaban uji coba.",
        should_abstain=True,
        abstain_reason="out_of_corpus",
    )
    item_abstain_2 = item_abstain_1.model_copy()

    # Both empty: 1.0
    assert compute_citation_jaccard(item_abstain_1, item_abstain_2) == 1.0

    item_cites_1 = GoldItem(
        id="gold_002",
        category="cross_article_synthesis",
        theme="General",
        question="Pertanyaan uji coba valid?",
        gold_answer="Berdasarkan [POJK 22 Tahun 2023, Pasal 75 ayat (1)], jawaban uji coba.",
        gold_citations=["POJK 22 Tahun 2023, Pasal 75 ayat (1)", "POJK 61/POJK.07/2020, Pasal 4"],
        relevant_chunk_ids=["c1", "c2"],
        should_abstain=False,
    )
    item_cites_identical = item_cites_1.model_copy()
    assert compute_citation_jaccard(item_cites_1, item_cites_identical) == 1.0

    # Partial overlap: 1 shared out of 2 total unique -> Jaccard = 1/2 = 0.5
    item_cites_partial = item_cites_1.model_copy(
        update={"gold_citations": ["POJK 22 Tahun 2023, Pasal 75 ayat (1)"]}
    )
    assert compute_citation_jaccard(item_cites_1, item_cites_partial) == 0.5

    # Completely disjoint: 0.0
    item_cites_disjoint = item_cites_1.model_copy(
        update={"gold_citations": ["UU Nomor 27 Tahun 2022, Pasal 4"]}
    )
    assert compute_citation_jaccard(item_cites_partial, item_cites_disjoint) == 0.0


def test_token_f1_metric() -> None:
    """Verify token F1 lexical overlap calculation."""
    text1 = "Bank wajib menyampaikan laporan insiden siber kepada OJK."
    text2 = "Bank wajib melaporkan insiden siber kepada OJK."
    f1 = compute_item_token_f1(text1, text2)
    assert 0.70 < f1 < 1.0

    # Exact match: 1.0
    assert compute_item_token_f1(text1, text1) == 1.0
    # Both empty: 1.0
    assert compute_item_token_f1("", "") == 1.0
    # Completely disjoint: 0.0
    assert compute_item_token_f1("satu dua", "tiga empat") == 0.0


def test_evaluate_agreement_on_actual_dataset() -> None:
    """Verify empirical agreement on relabel_annotations.jsonl meets all methodology gates."""
    assert ORIGINAL_PATH.exists()
    assert RELABEL_PATH.exists()

    report: AgreementReport = evaluate_agreement(ORIGINAL_PATH, RELABEL_PATH)

    assert report.total_items == 20
    assert report.abstention_agreement >= 0.95, (
        f"Abstention agreement {report.abstention_agreement} < 0.95"
    )
    assert report.cohens_kappa >= 0.85, f"Cohen's Kappa {report.cohens_kappa} < 0.85"
    assert report.citation_jaccard >= 0.85, f"Citation Jaccard {report.citation_jaccard} < 0.85"
    assert report.token_f1 >= 0.80, f"Token F1 {report.token_f1} < 0.80"
    assert report.satisfies_thresholds is True


def test_evaluate_agreement_missing_id_error(tmp_path: Path) -> None:
    """Verify KeyError when relabeled dataset references an unknown ID."""
    bad_relabel = tmp_path / "bad_relabel.jsonl"
    item = GoldItem(
        id="gold_999",
        category="factual_direct",
        theme="General",
        question="Pertanyaan tidak dikenal?",
        gold_answer="Berdasarkan [POJK 22 Tahun 2023, Pasal 75 ayat (1)], jawaban.",
        gold_citations=["POJK 22 Tahun 2023, Pasal 75 ayat (1)"],
        relevant_chunk_ids=["POJK_22_Tahun_2023_pasal_75_ayat_1"],
        should_abstain=False,
    )
    with open(bad_relabel, "w", encoding="utf-8") as f:
        f.write(item.model_dump_json() + "\n")

    with pytest.raises(KeyError, match="missing from original dataset"):
        evaluate_agreement(ORIGINAL_PATH, bad_relabel)
