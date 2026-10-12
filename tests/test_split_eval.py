"""Unit tests for the stratified train/dev/test dataset splitting module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.labeling_helper import VALID_CATEGORIES, GoldItem
from eval.split_eval import load_gold_items, split_dataset, stratified_split
from eval.validate_gold import validate_gold_dataset

DEFAULT_DATASET_PATH = Path("data/gold/gold_dataset.jsonl")
DEV_PATH = Path("data/gold/dev.jsonl")
TEST_PATH = Path("data/gold/test.jsonl")


def test_load_gold_items() -> None:
    """Verify loading and parsing of gold items."""
    items = load_gold_items(DEFAULT_DATASET_PATH)
    assert len(items) >= 100
    assert all(isinstance(it, GoldItem) for it in items)


def test_load_gold_items_nonexistent() -> None:
    """Verify FileNotFoundError on missing dataset path."""
    with pytest.raises(FileNotFoundError):
        load_gold_items(Path("nonexistent/path/data.jsonl"))


def test_stratified_split_partition_integrity() -> None:
    """Verify disjoint partition, complete union, and sizing of splits."""
    items = load_gold_items(DEFAULT_DATASET_PATH)
    dev_items, test_items = stratified_split(items, dev_ratio=0.4, seed=42)

    total_len = len(items)
    assert len(dev_items) + len(test_items) == total_len

    dev_ids = {it.id for it in dev_items}
    test_ids = {it.id for it in test_items}
    all_ids = {it.id for it in items}

    # 1. Disjoint: no overlap between dev and test
    assert dev_ids.isdisjoint(test_ids), f"Overlapping IDs found: {dev_ids & test_ids}"

    # 2. Complete: union covers all original IDs
    assert dev_ids | test_ids == all_ids

    # 3. Ratio check (approx 40% dev, 60% test)
    assert len(dev_items) == 40
    assert len(test_items) == 66


def test_stratified_split_reproducibility() -> None:
    """Verify pinned seed ensures 100% deterministic reproducibility."""
    items = load_gold_items(DEFAULT_DATASET_PATH)

    dev_1, test_1 = stratified_split(items, dev_ratio=0.4, seed=42)
    dev_2, test_2 = stratified_split(items, dev_ratio=0.4, seed=42)

    assert [x.id for x in dev_1] == [x.id for x in dev_2]
    assert [x.id for x in test_1] == [x.id for x in test_2]

    # Different seed should yield different partitions
    dev_diff, _ = stratified_split(items, dev_ratio=0.4, seed=999)
    assert [x.id for x in dev_1] != [x.id for x in dev_diff]


def test_stratified_split_category_and_abstain_coverage() -> None:
    """Verify all 9 categories and abstention items are represented in both splits."""
    items = load_gold_items(DEFAULT_DATASET_PATH)
    dev_items, test_items = stratified_split(items, dev_ratio=0.4, seed=42)

    dev_cats = {it.category for it in dev_items}
    test_cats = {it.category for it in test_items}

    for cat in VALID_CATEGORIES:
        assert cat in dev_cats, f"Category '{cat}' missing from dev split"
        assert cat in test_cats, f"Category '{cat}' missing from test split"

    # Both splits must have should_abstain=True and should_abstain=False
    assert any(it.should_abstain for it in dev_items)
    assert any(not it.should_abstain for it in dev_items)
    assert any(it.should_abstain for it in test_items)
    assert any(not it.should_abstain for it in test_items)


def test_stratified_split_invalid_ratio() -> None:
    """Verify error on invalid ratio outside (0.0, 1.0)."""
    items = load_gold_items(DEFAULT_DATASET_PATH)
    with pytest.raises(ValueError, match="dev_ratio must be between 0.0 and 1.0"):
        stratified_split(items, dev_ratio=1.5)
    with pytest.raises(ValueError, match="dev_ratio must be between 0.0 and 1.0"):
        stratified_split(items, dev_ratio=0.0)


def test_generated_split_files_exist_and_validate() -> None:
    """Verify dev.jsonl and test.jsonl pass full schema and grounding validation."""
    assert DEV_PATH.exists(), f"Dev file missing at {DEV_PATH}"
    assert TEST_PATH.exists(), f"Test file missing at {TEST_PATH}"

    dev_report = validate_gold_dataset(DEV_PATH, min_items=35)
    assert dev_report.is_valid, f"Dev validation failed: {dev_report.errors}"
    assert dev_report.total_items == 40

    test_report = validate_gold_dataset(TEST_PATH, min_items=50)
    assert test_report.is_valid, f"Test validation failed: {test_report.errors}"
    assert test_report.total_items == 66


def test_split_dataset_function_execution(tmp_path: Path) -> None:
    """Verify split_dataset end-to-end execution with custom output directory."""
    dev_file, test_file = split_dataset(
        input_path=DEFAULT_DATASET_PATH,
        output_dir=tmp_path,
        dev_ratio=0.3,
        seed=123,
    )

    assert dev_file.exists()
    assert test_file.exists()

    with open(dev_file, encoding="utf-8") as f:
        dev_lines = f.readlines()
    with open(test_file, encoding="utf-8") as f:
        test_lines = f.readlines()

    assert len(dev_lines) + len(test_lines) == 106
    # Each line should be valid JSON matching GoldItem
    GoldItem.model_validate(json.loads(dev_lines[0]))
    GoldItem.model_validate(json.loads(test_lines[0]))
