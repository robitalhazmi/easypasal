"""Unit tests for the offline evaluation runner and MLflow tracking (T3.2)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from eval.labeling_helper import GoldItem
from eval.run_eval import (
    EvalReport,
    evaluate_retrieval,
    log_to_mlflow,
    print_eval_report,
)


class MockChunk:
    """Mock retrieved chunk."""

    def __init__(self, chunk_id: str, score: float = 0.5) -> None:
        self.chunk_id = chunk_id
        self.score = score


class MockRetrievalResult:
    """Mock retrieval result containing chunks."""

    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunks = [MockChunk(cid) for cid in chunk_ids]


class MockRetriever:
    """Deterministic mock retriever with lookup map."""

    def __init__(self, return_map: dict[str, list[str]]) -> None:
        self.return_map = return_map

    def retrieve(self, query: str, top_k: int = 10, mode: str = "hybrid") -> MockRetrievalResult:
        _ = top_k, mode
        return MockRetrievalResult(self.return_map.get(query, []))


@pytest.fixture
def sample_gold_items() -> list[GoldItem]:
    """Sample gold items covering answerable and unanswerable categories."""
    return [
        GoldItem(
            id="gold_001",
            category="factual_direct",
            theme="Consumer Protection",
            difficulty="easy",
            language="id",
            question="Pertanyaan 1?",
            gold_answer="Jawaban 1 [POJK 22 Tahun 2023, Pasal 75].",
            gold_citations=["POJK 22 Tahun 2023, Pasal 75"],
            relevant_chunk_ids=["chunk_1"],
            grounded_facts=["Fakta 1"],
            should_abstain=False,
        ),
        GoldItem(
            id="gold_002",
            category="factual_direct",
            theme="IT & Cyber Risk",
            difficulty="medium",
            language="id",
            question="Pertanyaan 2?",
            gold_answer="Jawaban 2 [POJK 11/POJK.03/2022, Pasal 31].",
            gold_citations=["POJK 11/POJK.03/2022, Pasal 31"],
            relevant_chunk_ids=["chunk_2"],
            grounded_facts=["Fakta 2"],
            should_abstain=False,
        ),
        GoldItem(
            id="gold_003",
            category="cross_article_synthesis",
            theme="Consumer Protection",
            difficulty="hard",
            language="id",
            question="Pertanyaan 3?",
            gold_answer="Jawaban 3 [POJK 22/2023, Pasal 75].",
            gold_citations=["POJK 22/2023, Pasal 75"],
            relevant_chunk_ids=["chunk_3a", "chunk_3b"],
            grounded_facts=["Fakta 3"],
            should_abstain=False,
        ),
        GoldItem(
            id="gold_004",
            category="unanswerable_out_of_scope",
            theme="General",
            difficulty="medium",
            language="id",
            question="Pertanyaan di luar korpus?",
            gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut.",
            gold_citations=[],
            relevant_chunk_ids=[],
            grounded_facts=["Tidak ada dasar hukum"],
            should_abstain=True,
            abstain_reason="out_of_corpus",
        ),
    ]


def test_evaluate_retrieval_metrics_aggregation(sample_gold_items: list[GoldItem]) -> None:
    """Verify evaluation runner aggregates metrics accurately over answerable items."""
    # Query 1: Perfect retrieval (chunk_1 at rank 1)
    # Query 2: Retrieved at rank 2 (chunk_other, chunk_2)
    # Query 3: Multi-relevant (chunk_3a at rank 1, chunk_3b at rank 3)
    # Query 4: Unanswerable (retrieved chunk_other)
    retriever_map = {
        "Pertanyaan 1?": ["chunk_1", "other_1", "other_2"],
        "Pertanyaan 2?": ["other_1", "chunk_2", "other_2"],
        "Pertanyaan 3?": ["chunk_3a", "other_1", "chunk_3b"],
        "Pertanyaan di luar korpus?": ["other_1", "other_2"],
    }
    mock_retriever = MockRetriever(retriever_map)

    report: EvalReport = evaluate_retrieval(
        retriever=mock_retriever,
        items=sample_gold_items,
        top_k=5,
        k_values=(1, 3, 5),
        dataset_path="test_dataset.jsonl",
    )

    assert report.total_items == 4
    assert report.answerable_items == 3
    assert report.unanswerable_items == 1
    assert report.top_k == 5
    assert report.k_values == [1, 3, 5]

    # Q1: Hit@1=1.0, Rec@1=1.0, MRR=1.0
    # Q2: Hit@1=0.0, Rec@1=0.0, Hit@3=1.0, Rec@3=1.0, MRR=0.5
    # Q3: Hit@1=1.0, Rec@1=0.5, Hit@3=1.0, Rec@3=1.0, MRR=1.0
    # Mean Hit@1: (1 + 0 + 1) / 3 = 2/3 ≈ 0.6667
    # Mean Hit@3: (1 + 1 + 1) / 3 = 1.0
    # Mean Rec@1: (1.0 + 0.0 + 0.5) / 3 = 1.5 / 3 = 0.50
    # Mean Rec@3: (1.0 + 1.0 + 1.0) / 3 = 1.0
    # Mean MRR: (1.0 + 0.5 + 1.0) / 3 = 2.5 / 3 ≈ 0.8333

    assert pytest.approx(report.mean_hit_at_k[1], rel=1e-4) == 2.0 / 3
    assert report.mean_hit_at_k[3] == 1.0
    assert report.mean_hit_at_k[5] == 1.0

    assert pytest.approx(report.mean_recall_at_k[1], rel=1e-4) == 0.50
    assert report.mean_recall_at_k[3] == 1.0
    assert report.mean_recall_at_k[5] == 1.0

    assert pytest.approx(report.mean_mrr_at_k[5], rel=1e-4) == 2.5 / 3
    assert report.gate_recall_pass is True  # 1.0 >= 0.80
    assert report.gate_mrr_pass is True  # 0.833 >= 0.65
    assert report.gate_overall_pass is True


def test_per_category_breakdown(sample_gold_items: list[GoldItem]) -> None:
    """Verify per-category breakdown dictionary and categorization."""
    mock_retriever = MockRetriever(
        {
            "Pertanyaan 1?": ["chunk_1"],
            "Pertanyaan 2?": ["chunk_2"],
            "Pertanyaan 3?": ["chunk_3a", "chunk_3b"],
            "Pertanyaan di luar korpus?": [],
        }
    )

    report = evaluate_retrieval(
        retriever=mock_retriever,
        items=sample_gold_items,
        top_k=5,
        k_values=(1, 5),
    )

    assert "factual_direct" in report.categories
    assert "cross_article_synthesis" in report.categories
    assert "unanswerable_out_of_scope" in report.categories

    cat_factual = report.categories["factual_direct"]
    assert cat_factual.total_items == 2
    assert cat_factual.answerable_items == 2
    assert cat_factual.recall_at_k[5] == 1.0

    cat_unans = report.categories["unanswerable_out_of_scope"]
    assert cat_unans.total_items == 1
    assert cat_unans.answerable_items == 0


def test_report_json_serialization(sample_gold_items: list[GoldItem], tmp_path: Path) -> None:
    """Verify saving report to JSON file and dictionary structure."""
    mock_retriever = MockRetriever({"Pertanyaan 1?": ["chunk_1"]})
    report = evaluate_retrieval(
        retriever=mock_retriever,
        items=sample_gold_items,
        top_k=5,
        k_values=(1, 5),
    )

    json_file = tmp_path / "report.json"
    saved = report.save_json(json_file)
    assert saved.exists()

    with open(saved, encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_items"] == 4
    assert data["answerable_items"] == 3
    assert "gates" in data
    assert "overall_metrics" in data
    assert "categories" in data
    assert len(data["item_results"]) == 4


def test_print_eval_report_output(
    sample_gold_items: list[GoldItem], capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify terminal formatting output prints expected banners and sections."""
    mock_retriever = MockRetriever({"Pertanyaan 1?": ["chunk_1"]})
    report = evaluate_retrieval(
        retriever=mock_retriever,
        items=sample_gold_items,
        top_k=5,
        k_values=(1, 5),
    )

    print_eval_report(report)
    captured = capsys.readouterr().out

    assert "EasyPasal Retrieval Evaluation Report (T3.2)" in captured
    assert "Retrieval Metric" in captured
    assert "Recall@5" in captured
    assert "MRR" in captured
    assert "Per-Category Retrieval Breakdown:" in captured


def test_log_to_mlflow_local_tracking(sample_gold_items: list[GoldItem], tmp_path: Path) -> None:
    """Verify logging metrics, parameters, and report artifact to local MLflow tracking URI."""
    mock_retriever = MockRetriever({"Pertanyaan 1?": ["chunk_1"]})
    report = evaluate_retrieval(
        retriever=mock_retriever,
        items=sample_gold_items,
        top_k=5,
        k_values=(1, 5),
    )

    mlruns_dir = tmp_path / "test_mlruns"
    tracking_uri = f"file:{mlruns_dir}"

    run_id = log_to_mlflow(
        report=report,
        experiment_name="test-experiment",
        run_name="unit-test-run",
        params={"strategy": "test_strat", "mode": "hybrid"},
        tracking_uri=tracking_uri,
    )

    assert run_id is not None
    assert isinstance(run_id, str)
    assert mlruns_dir.exists()


def test_main_cli_execution_with_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI main entrypoint executes with mocked HybridRetriever."""
    toy_dataset = tmp_path / "toy.jsonl"
    item = GoldItem(
        id="gold_001",
        category="factual_direct",
        theme="Consumer Protection",
        difficulty="easy",
        language="id",
        question="Pertanyaan CLI?",
        gold_answer="Jawaban CLI [POJK 22/2023, Pasal 75].",
        gold_citations=["POJK 22/2023, Pasal 75"],
        relevant_chunk_ids=["chunk_1"],
        grounded_facts=["Fakta 1"],
        should_abstain=False,
    )
    with open(toy_dataset, "w", encoding="utf-8") as f:
        f.write(item.model_dump_json() + "\n")

    output_json = tmp_path / "cli_out.json"

    # Mock HybridRetriever
    mock_retriever_instance = MockRetriever({"Pertanyaan CLI?": ["chunk_1"]})
    mock_hybrid_cls = MagicMock()
    mock_hybrid_cls.from_paths.return_value = mock_retriever_instance

    with patch("easypasal.retrieval.retriever.HybridRetriever", mock_hybrid_cls):
        test_args = [
            "eval.run_eval",
            "--dataset",
            str(toy_dataset),
            "--output-json",
            str(output_json),
            "--no-mlflow",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        from eval.run_eval import main

        main()

    assert output_json.exists()
