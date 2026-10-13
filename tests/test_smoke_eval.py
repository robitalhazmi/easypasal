"""Unit tests for CI smoke evaluation runner and regression gate (T3.6).

Demonstrates and verifies that:
1. Valid retrieval pipelines pass the smoke evaluation gate.
2. Deliberately broken retrieval changes (e.g. empty results, scrambled rankings,
   broken abstention, latency spikes) fail the CI gate and trigger failure exits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.labeling_helper import GoldItem
from eval.smoke_eval import (
    CIGateFailureError,
    FixtureSmokeRetriever,
    MockRetrievalResult,
    MockRetrievedChunk,
    SmokeGateResult,
    evaluate_smoke_gate,
    load_smoke_baseline,
    main,
)


class PerfectRetriever:
    """Mock retriever that perfectly returns relevant chunks for all answerable queries."""

    def __init__(self, items: list[GoldItem]) -> None:
        self.item_map = {it.question: it.relevant_chunk_ids for it in items}

    def retrieve(self, query: str, top_k: int = 10, mode: str = "hybrid") -> MockRetrievalResult:
        _ = top_k, mode
        gold_chunks = self.item_map.get(query, [])
        return MockRetrievalResult([MockRetrievedChunk(cid) for cid in gold_chunks])


class DeliberatelyBrokenRetriever:
    """Deliberately broken retriever: returns empty chunks or completely irrelevant garbage."""

    def __init__(self, mode: str = "empty") -> None:
        self.mode = mode

    def retrieve(self, query: str, top_k: int = 10, mode: str = "hybrid") -> MockRetrievalResult:
        _ = query, top_k, mode
        if self.mode == "empty":
            return MockRetrievalResult([])
        # Return garbage irrelevant chunks
        return MockRetrievalResult(
            [MockRetrievedChunk(f"garbage_unrelated_chunk_{i}") for i in range(top_k)]
        )


class BrokenAbstentionRetriever:
    """Retriever that fails abstention by claiming relevance on adversarial/unanswerable queries."""

    def __init__(self, items: list[GoldItem]) -> None:
        self.item_map = {it.question: it for it in items}

    def retrieve(self, query: str, top_k: int = 10, mode: str = "hybrid") -> MockRetrievalResult:
        _ = top_k, mode
        it = self.item_map.get(query)
        if it and it.should_abstain:
            # Improperly refuses to abstain on unanswerable/adversarial query
            return MockRetrievalResult(
                [MockRetrievedChunk("hallucinated_chunk_1")], should_abstain=False
            )
        if it:
            return MockRetrievalResult([MockRetrievedChunk(c) for c in it.relevant_chunk_ids])
        return MockRetrievalResult([])


class SlowRetriever:
    """Retriever that introduces an intentional artificial latency spike."""

    def __init__(self, delay_sec: float = 0.05) -> None:
        self.delay_sec = delay_sec

    def retrieve(self, query: str, top_k: int = 10, mode: str = "hybrid") -> MockRetrievalResult:
        import time

        _ = query, top_k, mode
        time.sleep(self.delay_sec)
        return MockRetrievalResult([MockRetrievedChunk("chunk_1")])


@pytest.fixture
def sample_smoke_items() -> list[GoldItem]:
    """Sample smoke items for unit testing."""
    return [
        GoldItem(
            id="gold_001",
            category="factual_direct",
            theme="Consumer Protection",
            difficulty="easy",
            language="id",
            question="Berapa batas waktu penanganan pengaduan konsumen?",
            gold_answer="Berdasarkan POJK 22/2023, batas waktu adalah 20 hari kerja.",
            gold_citations=["POJK 22 Tahun 2023, Pasal 75"],
            relevant_chunk_ids=["POJK_22_Tahun_2023_pasal_75_chunk_0"],
            grounded_facts=["Batas waktu 20 hari kerja"],
            should_abstain=False,
        ),
        GoldItem(
            id="gold_002",
            category="cross_article_synthesis",
            theme="IT & Cyber Risk",
            difficulty="medium",
            language="id",
            question="Apa kewajiban bank umum saat mengalami insiden siber kritis?",
            gold_answer="Bank wajib melaporkan kepada OJK paling lambat 1x24 jam.",
            gold_citations=["POJK 11/POJK.03/2022, Pasal 31"],
            relevant_chunk_ids=["POJK_11_POJK.03_2022_pasal_31_chunk_0"],
            grounded_facts=["Wajib lapor 1x24 jam"],
            should_abstain=False,
        ),
        GoldItem(
            id="gold_003",
            category="unanswerable_out_of_scope",
            theme="General",
            difficulty="medium",
            language="id",
            question="Berapa tarif PPh Pasal 21 atas dividen luar negeri?",
            gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut.",
            gold_citations=[],
            relevant_chunk_ids=[],
            grounded_facts=["Tidak ada dasar hukum perpajakan luar negeri"],
            should_abstain=True,
            abstain_reason="out_of_corpus",
        ),
    ]


@pytest.fixture
def mock_baseline_data() -> dict[str, object]:
    """Sample baseline metrics data."""
    return {
        "dataset_path": "data/gold/smoke.jsonl",
        "total_items": 30,
        "answerable_items": 25,
        "unanswerable_items": 5,
        "overall_metrics": {
            "recall_at_k": {"5": 0.24},
            "mrr_at_k": {"5": 0.1267},
            "ndcg_at_k": {"5": 0.1590},
            "latency_mean_ms": 259.5,
            "latency_p95_ms": 267.5,
        },
    }


def test_smoke_eval_normal_pass(
    sample_smoke_items: list[GoldItem], mock_baseline_data: dict[str, object]
) -> None:
    """Test that a healthy retriever satisfying baseline metrics passes all smoke gates."""
    retriever = PerfectRetriever(sample_smoke_items)
    result = evaluate_smoke_gate(
        retriever=retriever,
        items=sample_smoke_items,
        baseline_data=mock_baseline_data,
        tolerance=0.05,
    )

    assert isinstance(result, SmokeGateResult)
    assert result.passed is True
    assert len(result.failed_checks) == 0
    assert result.candidate_metrics["recall_at_5"] >= 0.24
    assert result.candidate_metrics["mrr"] >= 0.12

    # Check terminal formatting
    table = result.format_terminal_table()
    assert "✅ PASSED" in table
    assert "Recall@5 Regression Gate" in table


def test_smoke_eval_deliberately_broken_retrieval_fails_ci(
    sample_smoke_items: list[GoldItem], mock_baseline_data: dict[str, object]
) -> None:
    """ACCEPTANCE CRITERIA TEST: Verify deliberately broken retrieval changes fail the CI gate.

    Tests two broken scenarios:
    1. Empty retrieval results (Recall@5 = 0.0)
    2. Completely irrelevant chunks (Recall@5 = 0.0)
    Both must fail the gate and raise CIGateFailureError when requested.
    """
    # Scenario 1: Empty retrieval
    broken_empty = DeliberatelyBrokenRetriever(mode="empty")
    res_empty = evaluate_smoke_gate(
        retriever=broken_empty,
        items=sample_smoke_items,
        baseline_data=mock_baseline_data,
        tolerance=0.05,
        raise_on_failure=False,
    )

    assert res_empty.passed is False
    assert len(res_empty.failed_checks) >= 2
    failed_check_names = [c.name for c in res_empty.failed_checks]
    assert "Recall@5 Regression Gate" in failed_check_names
    assert "Recall@5 Absolute Floor" in failed_check_names
    assert res_empty.candidate_metrics["recall_at_5"] == 0.0

    # Verify raise_on_failure triggers CIGateFailureError
    with pytest.raises(CIGateFailureError, match="CI Smoke Gates failed"):
        evaluate_smoke_gate(
            retriever=broken_empty,
            items=sample_smoke_items,
            baseline_data=mock_baseline_data,
            raise_on_failure=True,
        )

    # Scenario 2: Irrelevant garbage chunks
    broken_garbage = DeliberatelyBrokenRetriever(mode="garbage")
    res_garbage = evaluate_smoke_gate(
        retriever=broken_garbage,
        items=sample_smoke_items,
        baseline_data=mock_baseline_data,
        tolerance=0.05,
    )
    assert res_garbage.passed is False
    assert "Recall@5 Regression Gate" in [c.name for c in res_garbage.failed_checks]


def test_smoke_eval_broken_abstention_fails_ci(
    sample_smoke_items: list[GoldItem], mock_baseline_data: dict[str, object]
) -> None:
    """Test that a retriever hallucinating on unanswerable queries fails the abstention gate."""
    retriever = BrokenAbstentionRetriever(sample_smoke_items)
    result = evaluate_smoke_gate(
        retriever=retriever,
        items=sample_smoke_items,
        baseline_data=mock_baseline_data,
        min_abstention_accuracy=0.90,
    )

    # Abstention failure should be detected
    failed_names = [c.name for c in result.failed_checks]
    assert "Abstention Accuracy Gate" in failed_names
    assert result.passed is False


def test_smoke_eval_latency_ceiling_gate(
    sample_smoke_items: list[GoldItem], mock_baseline_data: dict[str, object]
) -> None:
    """Test that severe latency regressions fail the p95 latency ceiling gate."""
    slow_retriever = SlowRetriever(delay_sec=0.03)  # 30ms per item
    # Set ceiling very tight to trigger gate
    result = evaluate_smoke_gate(
        retriever=slow_retriever,
        items=sample_smoke_items,
        baseline_data=mock_baseline_data,
        max_p95_latency_ms=10.0,  # 10ms ceiling
    )

    failed_names = [c.name for c in result.failed_checks]
    assert "p95 Latency Ceiling Gate" in failed_names
    assert result.passed is False


def test_smoke_eval_fixture_retriever(tmp_path: Path) -> None:
    """Test in-memory FixtureSmokeRetriever construction and retrieval."""
    sample_chunks = [
        {
            "chunk_id": "c1",
            "text": "Kewajiban pelaporan bank umum dalam 20 hari kerja.",
            "doc_id": "POJK_1",
            "doc_number": "POJK 1",
            "citation_label": "POJK 1, Pasal 1",
            "pasal": "1",
            "ayat": "1",
            "status": "berlaku",
        },
        {
            "chunk_id": "c2",
            "text": "Sanksi administratif bagi pelanggaran pelindungan konsumen.",
            "doc_id": "POJK_2",
            "doc_number": "POJK 2",
            "citation_label": "POJK 2, Pasal 2",
            "pasal": "2",
            "ayat": "1",
            "status": "berlaku",
        },
    ]
    fixture_file = tmp_path / "test_chunks.json"
    fixture_file.write_text(json.dumps(sample_chunks))

    retriever = FixtureSmokeRetriever(fixture_path=fixture_file)
    res = retriever.retrieve("pelaporan bank umum", top_k=5)
    assert len(res.chunks) > 0
    assert res.chunks[0].chunk_id == "c1"


def test_smoke_eval_load_baseline(tmp_path: Path) -> None:
    """Test loading smoke baseline JSON and missing file handling."""
    base_file = tmp_path / "baseline.json"
    base_file.write_text(json.dumps({"overall_metrics": {"recall_at_k": {"5": 0.25}}}))

    data = load_smoke_baseline(base_file)
    assert data["overall_metrics"]["recall_at_k"]["5"] == 0.25

    with pytest.raises(FileNotFoundError):
        load_smoke_baseline(tmp_path / "nonexistent.json")


def test_smoke_eval_cli_main_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test smoke_eval main() CLI entrypoint execution and exit codes."""
    smoke_dataset = tmp_path / "smoke.jsonl"
    smoke_baseline = tmp_path / "baseline.json"
    fixture_file = tmp_path / "smoke_chunks.json"

    # Write 1 item
    item = {
        "id": "gold_001",
        "category": "factual_direct",
        "theme": "General",
        "difficulty": "easy",
        "language": "id",
        "question": "Apa itu POJK?",
        "gold_answer": "Jawaban POJK",
        "gold_citations": ["POJK 1, Pasal 1"],
        "relevant_chunk_ids": ["c1"],
        "grounded_facts": ["Fakta POJK"],
        "should_abstain": False,
    }
    smoke_dataset.write_text(json.dumps(item) + "\n")

    chunk = {
        "chunk_id": "c1",
        "text": "Definisi POJK Peraturan OJK.",
        "doc_id": "POJK_1",
        "doc_number": "POJK 1",
        "citation_label": "POJK 1, Pasal 1",
        "pasal": "1",
        "ayat": "1",
        "status": "berlaku",
    }
    fixture_file.write_text(json.dumps([chunk]))

    baseline_data = {
        "overall_metrics": {
            "recall_at_k": {"5": 0.50},
            "mrr_at_k": {"5": 0.50},
            "ndcg_at_k": {"5": 0.50},
            "latency_mean_ms": 10.0,
            "latency_p95_ms": 10.0,
        }
    }
    smoke_baseline.write_text(json.dumps(baseline_data))

    out_file = tmp_path / "out_report.json"

    # 1. Success run: fixture BM25 will retrieve c1 for "Apa itu POJK?"
    monkeypatch.setattr(
        "sys.argv",
        [
            "smoke_eval",
            "--dataset",
            str(smoke_dataset),
            "--baseline",
            str(smoke_baseline),
            "--fixture",
            str(fixture_file),
            "--index-dir",
            str(tmp_path / "no_index"),
            "--tolerance",
            "0.50",
            "--output",
            str(out_file),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert out_file.exists()

    # 2. Failure run: tighten baseline so it fails
    tight_baseline = {
        "overall_metrics": {
            "recall_at_k": {"5": 1.0},
            "mrr_at_k": {"5": 1.0},
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
        }
    }
    smoke_baseline.write_text(json.dumps(tight_baseline))

    # Give irrelevant query item that scores 0 recall
    bad_item = dict(item)
    bad_item["question"] = "Query tanpa kecocokan istilah apapun xyz123"
    smoke_dataset.write_text(json.dumps(bad_item) + "\n")

    with pytest.raises(SystemExit) as exc2:
        main()
    assert exc2.value.code == 1
