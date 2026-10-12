"""Unit tests for the EasyPasal gold evaluation dataset and validation suite."""

from __future__ import annotations

import json
from pathlib import Path

from eval.labeling_helper import VALID_CATEGORIES, GoldItem
from eval.validate_gold import (
    DEFAULT_META_PATH,
    STANDARD_ABSTAIN_EN,
    STANDARD_ABSTAIN_ID,
    validate_gold_dataset,
)

GOLD_DATASET_PATH = Path("data/gold/gold_dataset.jsonl")


def test_gold_dataset_exists() -> None:
    """Ensure gold evaluation dataset file exists."""
    assert GOLD_DATASET_PATH.exists(), f"Dataset file not found at {GOLD_DATASET_PATH}"


def test_gold_dataset_full_validation() -> None:
    """Run full schema and corpus grounding validation on gold dataset."""
    report = validate_gold_dataset(
        dataset_path=GOLD_DATASET_PATH,
        meta_path=DEFAULT_META_PATH,
        min_items=100,
        require_verified_status=True,
    )
    assert report.is_valid, f"Validation failed with errors: {report.errors}"
    assert report.total_items >= 100, f"Expected >= 100 items, got {report.total_items}"
    assert report.valid_items == report.total_items
    assert len(report.errors) == 0


def test_gold_dataset_taxonomy_and_category_coverage() -> None:
    """Verify that all 9 standardized categories are represented."""
    report = validate_gold_dataset(GOLD_DATASET_PATH)
    assert report.is_valid

    for category in VALID_CATEGORIES:
        count = report.category_counts.get(category, 0)
        assert count > 0, f"Category '{category}' has 0 items in dataset"


def test_gold_dataset_multilingual_coverage() -> None:
    """Verify language distribution covers both Indonesian and English slices."""
    report = validate_gold_dataset(GOLD_DATASET_PATH)
    assert report.is_valid
    assert report.language_counts.get("id", 0) >= 80, "Expected >= 80 Indonesian items"
    assert report.language_counts.get("en", 0) >= 10, "Expected >= 10 English items"


def test_gold_dataset_thematic_coverage() -> None:
    """Verify representation across all 5 regulatory domains."""
    report = validate_gold_dataset(GOLD_DATASET_PATH)
    assert report.is_valid

    required_themes = [
        "Consumer Protection",
        "IT & Cyber Risk",
        "AML / CFT",
        "Data Privacy & Secrecy",
        "Capital Markets",
    ]
    for theme in required_themes:
        assert report.theme_counts.get(theme, 0) >= 10, f"Theme '{theme}' underrepresented"


def test_gold_dataset_abstention_strictness() -> None:
    """Verify abstaining items have exact standard refusals and empty citations."""
    with open(GOLD_DATASET_PATH, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        item = GoldItem.model_validate(json.loads(line))
        if item.should_abstain:
            assert len(item.gold_citations) == 0
            assert len(item.relevant_chunk_ids) == 0
            assert item.abstain_reason in (
                "out_of_corpus",
                "adversarial",
                "superseded",
                "unanswerable",
            )
            if item.language == "id":
                assert item.gold_answer == STANDARD_ABSTAIN_ID
            elif item.language == "en":
                assert item.gold_answer == STANDARD_ABSTAIN_EN
        else:
            assert len(item.gold_citations) >= 1
            assert len(item.relevant_chunk_ids) >= 1
            assert item.abstain_reason is None


def test_validator_detects_malformed_items(tmp_path: Path) -> None:
    """Verify validator flags schema errors, duplicate IDs, and invalid abstentions."""
    bad_dataset = tmp_path / "bad_dataset.jsonl"

    bad_items = [
        # 1. Invalid ID pattern
        {
            "id": "invalid_id",
            "category": "factual_direct",
            "theme": "Consumer Protection",
            "difficulty": "easy",
            "language": "id",
            "question": "Pertanyaan uji coba?",
            "gold_answer": "Berdasarkan [POJK 22 Tahun 2023, Pasal 75 ayat (1)], jawaban valid.",
            "gold_citations": ["POJK 22 Tahun 2023, Pasal 75 ayat (1)"],
            "relevant_chunk_ids": ["POJK_22_Tahun_2023_pasal_75_ayat_1"],
            "grounded_facts": ["Fakta uji coba"],
            "should_abstain": False,
            "abstain_reason": None,
            "target_regulations": ["POJK 22 Tahun 2023"],
            "review_status": "verified",
        },
        # 2. should_abstain=True but non-empty citations
        {
            "id": "gold_001",
            "category": "unanswerable_out_of_scope",
            "theme": "Consumer Protection",
            "difficulty": "easy",
            "language": "id",
            "question": "Pertanyaan out of scope?",
            "gold_answer": STANDARD_ABSTAIN_ID,
            "gold_citations": ["POJK 22 Tahun 2023, Pasal 75 ayat (1)"],
            "relevant_chunk_ids": [],
            "grounded_facts": ["Fakta"],
            "should_abstain": True,
            "abstain_reason": "out_of_corpus",
            "target_regulations": [],
            "review_status": "verified",
        },
        # 3. Duplicate ID (gold_001)
        {
            "id": "gold_001",
            "category": "factual_direct",
            "theme": "Consumer Protection",
            "difficulty": "easy",
            "language": "id",
            "question": "Pertanyaan duplikat?",
            "gold_answer": "Berdasarkan [POJK 22 Tahun 2023, Pasal 75 ayat (1)], jawaban.",
            "gold_citations": ["POJK 22 Tahun 2023, Pasal 75 ayat (1)"],
            "relevant_chunk_ids": ["POJK_22_Tahun_2023_pasal_75_ayat_1"],
            "grounded_facts": ["Fakta"],
            "should_abstain": False,
            "abstain_reason": None,
            "target_regulations": ["POJK 22 Tahun 2023"],
            "review_status": "verified",
        },
    ]

    with open(bad_dataset, "w", encoding="utf-8") as f:
        for it in bad_items:
            f.write(json.dumps(it) + "\n")

    report = validate_gold_dataset(
        dataset_path=bad_dataset,
        meta_path=DEFAULT_META_PATH,
        min_items=1,
        require_verified_status=True,
    )
    assert not report.is_valid
    assert len(report.errors) >= 3
