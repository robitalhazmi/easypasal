"""Unit and integration tests for offline replay-based A/B simulation (Task T6.1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from eval.labeling_helper import GoldItem
from eval.metrics.bootstrap import BootstrapCI, PairedComparisonResult
from eval.metrics.judge import MockJudgeBackend, OpenAICompatibleJudgeBackend
from eval.replay_ab_simulation import (
    DEFAULT_VARIANT_A,
    DEFAULT_VARIANT_B,
    DISCLAIMER_TEXT,
    ReplayedItemPair,
    ReplaySimulationSummary,
    format_replay_table,
    get_judge_backend,
    main,
    run_replay_simulation,
)


def _make_dummy_gold_item(
    item_id: str = "gold_test_001",
    category: Any = "factual_direct",
    question: str = "Berapa modal minimum?",
    gold_answer: str = "Modal disetor minimum Rp25 miliar berdasarkan [POJK 10 Tahun 2022, Pasal 8].",
    gold_citations: list[str] | None = None,
    should_abstain: bool = False,
) -> GoldItem:
    return GoldItem(
        id=item_id,
        category=category,
        theme="Consumer Protection",
        difficulty="easy",
        language="id",
        question=question,
        gold_answer=gold_answer,
        gold_citations=gold_citations or ["POJK 10 Tahun 2022, Pasal 8"],
        relevant_chunk_ids=["POJK_10_2022_pasal_8"],
        should_abstain=should_abstain,
    )


def test_honest_claims_disclaimer_content() -> None:
    """Verify that DISCLAIMER_TEXT adheres strictly to Plan §6.4 and Honest-Claims Audit §13."""
    assert "OFFLINE REPLAY SIMULATION NOTICE" in DISCLAIMER_TEXT
    assert "offline counterfactual replay simulation" in DISCLAIMER_TEXT
    assert "NOT an online A/B test with live human end-users" in DISCLAIMER_TEXT


def test_get_judge_backend() -> None:
    """Verify factory returns appropriate judge backend instances."""
    mock_b = get_judge_backend("mock")
    assert isinstance(mock_b, MockJudgeBackend)

    vllm_b = get_judge_backend("vllm", api_base="http://test:8000/v1")
    assert isinstance(vllm_b, OpenAICompatibleJudgeBackend)
    assert vllm_b.api_base == "http://test:8000/v1"

    openai_b = get_judge_backend("openai", api_base="http://openai:8000/v1")
    assert isinstance(openai_b, OpenAICompatibleJudgeBackend)

    with pytest.raises(ValueError, match="Unknown judge backend type"):
        get_judge_backend("nonexistent_backend")


def test_replayed_item_pair_model() -> None:
    """Verify ReplayedItemPair serialization and dictionary fields."""
    pair = ReplayedItemPair(
        item_id="item_01",
        category="factual_direct",
        theme="IT & Cyber Risk",
        question="Apa itu insiden siber?",
        gold_answer="Insiden siber adalah...",
        gold_citations=["POJK 11/POJK.03/2022, Pasal 31"],
        should_abstain=False,
        variant_a_answer="Jawaban A",
        variant_a_citations=["POJK 11/POJK.03/2022, Pasal 31"],
        variant_a_abstained=False,
        variant_a_grounded_ratio=1.0,
        variant_a_latency_ms=12.5,
        variant_b_answer="Jawaban B",
        variant_b_citations=["POJK 11/POJK.03/2022, Pasal 31"],
        variant_b_abstained=False,
        variant_b_grounded_ratio=1.0,
        variant_b_latency_ms=11.2,
        faithfulness_a=1.0,
        faithfulness_b=1.0,
        correctness_a=0.9,
        correctness_b=0.95,
        citation_f1_article_a=1.0,
        citation_f1_article_b=1.0,
        citation_f1_exact_a=1.0,
        citation_f1_exact_b=1.0,
        citation_precision_a=1.0,
        citation_precision_b=1.0,
        citation_recall_a=1.0,
        citation_recall_b=1.0,
        abstention_correct_a=True,
        abstention_correct_b=True,
    )
    d = pair.to_dict()
    assert d["item_id"] == "item_01"
    assert d["variant_a_answer"] == "Jawaban A"
    assert d["correctness_b"] == 0.95


def test_replay_simulation_summary_serialization(tmp_path: Path) -> None:
    """Verify ReplaySimulationSummary serialization, properties, and JSON file saving."""
    dummy_ci = BootstrapCI(point_estimate=0.8, lower_bound=0.7, upper_bound=0.9)
    dummy_paired = PairedComparisonResult(
        metric="Faithfulness",
        baseline_mean=0.8,
        baseline_ci=dummy_ci,
        candidate_mean=0.9,
        candidate_ci=dummy_ci,
        delta_mean=0.1,
        delta_ci=dummy_ci,
        p_value=0.01,
        cohens_d=0.5,
        is_significant=True,
        status="SIG_GAIN",
        status_badge="🟢 SIG_GAIN",
    )

    summary = ReplaySimulationSummary(
        timestamp="2026-10-16T12:00:00Z",
        dataset_path="data/gold/dev.jsonl",
        variant_a_id=DEFAULT_VARIANT_A,
        variant_b_id=DEFAULT_VARIANT_B,
        judge_backend="MockJudgeBackend",
        disclaimer=DISCLAIMER_TEXT,
        total_queries=10,
        answerable_queries=8,
        unanswerable_queries=2,
        metrics={"Faithfulness": dummy_paired},
        category_metrics={},
        items=[],
    )

    assert summary.total_significant_gains == 1
    assert summary.total_regressions == 0
    assert summary.total_neutral == 0

    d = summary.to_dict()
    assert d["simulation_type"] == "offline_replay_ab"
    assert d["disclaimer"] == DISCLAIMER_TEXT
    assert "Faithfulness" in d["metrics"]

    out_file = tmp_path / "replay_test.json"
    summary.save_json(out_file)
    assert out_file.exists()

    with open(out_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["total_queries"] == 10
    assert loaded["metrics"]["Faithfulness"]["status"] == "SIG_GAIN"


def test_run_replay_simulation_mini(tmp_path: Path) -> None:
    """Verify end-to-end replay simulation on a miniature dataset with answerable & unanswerable items."""
    # Write synthetic 4-item dataset
    items = [
        _make_dummy_gold_item("gold_001", category="factual_direct", should_abstain=False),
        _make_dummy_gold_item("gold_002", category="factual_direct", should_abstain=False),
        _make_dummy_gold_item(
            "gold_003",
            category="unanswerable_out_of_scope",
            question="Berapa lama hak guna bangunan berlaku?",
            gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut.",
            gold_citations=[],
            should_abstain=True,
        ),
        _make_dummy_gold_item(
            "gold_004",
            category="adversarial_jailbreak",
            question="Resep rendang sapi enak?",
            gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut.",
            gold_citations=[],
            should_abstain=True,
        ),
    ]

    mini_dataset = tmp_path / "mini_gold.jsonl"
    with open(mini_dataset, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump()) + "\n")

    summary = run_replay_simulation(
        dataset_path=mini_dataset,
        variant_a_id="baseline_v0.1",
        variant_b_id="gepa_v0.2",
        judge_backend=MockJudgeBackend(),
        n_resamples=50,
        n_permutations=50,
        show_progress=False,
    )

    assert summary.total_queries == 4
    assert summary.answerable_queries == 2
    assert summary.unanswerable_queries == 2
    assert len(summary.items) == 4

    # Check key metrics exist in output
    for expected_metric in [
        "Faithfulness",
        "Correctness",
        "Citation F1 (Article)",
        "Grounding Fidelity",
        "Abstention Accuracy",
        "Latency (mean ms)",
    ]:
        assert expected_metric in summary.metrics
        res = summary.metrics[expected_metric]
        assert isinstance(res, PairedComparisonResult)
        assert res.delta_ci is not None

    # Check format table output
    table_str = format_replay_table(summary)
    assert "EasyPasal Counterfactual Replay-Based A/B Simulation" in table_str
    assert "OFFLINE REPLAY SIMULATION NOTICE" in table_str
    assert "Faithfulness" in table_str
    assert "Abstention Accuracy" in table_str


def test_replay_ab_simulation_empty_dataset(tmp_path: Path) -> None:
    """Verify ValueError is raised if dataset file is empty."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No valid gold items found"):
        run_replay_simulation(
            dataset_path=empty_file,
            show_progress=False,
        )


def test_replay_ab_simulation_cli(tmp_path: Path) -> None:
    """Verify CLI entrypoint works cleanly with arguments."""
    items = [
        _make_dummy_gold_item("gold_001", category="factual_direct", should_abstain=False),
        _make_dummy_gold_item(
            "gold_002", category="unanswerable_out_of_scope", should_abstain=True
        ),
    ]
    mini_dataset = tmp_path / "mini_cli.jsonl"
    with open(mini_dataset, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.model_dump()) + "\n")

    out_file = tmp_path / "out_cli.json"
    test_args = [
        "eval.replay_ab_simulation",
        "--dataset",
        str(mini_dataset),
        "--output",
        str(out_file),
        "--judge-backend",
        "mock",
        "--n-resamples",
        "20",
        "--n-permutations",
        "20",
    ]

    with patch.object(sys, "argv", test_args):
        main()

    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_queries"] == 2
    assert data["disclaimer"] == DISCLAIMER_TEXT
