"""Unit tests for judge calibration and human agreement analysis (T3.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.compute_judge_calibration import (
    CalibratedItem,
    compute_cohens_kappa,
    compute_mae,
    compute_pabak,
    compute_pearson_correlation,
    evaluate_calibration,
    load_calibrated_items,
    main,
    print_calibration_report,
)

# ==============================================================================
# 1. Statistical Formula Tests
# ==============================================================================


class TestCalibrationMath:
    """Tests for statistical agreement and error functions."""

    def test_compute_cohens_kappa_perfect(self) -> None:
        """Verify perfect agreement returns 1.0."""
        preds = [True, True, False, False]
        refs = [True, True, False, False]
        assert compute_cohens_kappa(preds, refs) == 1.0

    def test_compute_cohens_kappa_disagreement(self) -> None:
        """Verify complete disagreement returns -1.0."""
        preds = [True, True, False, False]
        refs = [False, False, True, True]
        assert compute_cohens_kappa(preds, refs) == -1.0

    def test_compute_cohens_kappa_empty_and_errors(self) -> None:
        """Verify empty sequence and length mismatch handling."""
        assert compute_cohens_kappa([], []) == 1.0
        with pytest.raises(ValueError, match="identical length"):
            compute_cohens_kappa([True], [True, False])

    def test_compute_pabak(self) -> None:
        """Verify Prevalence-Adjusted Bias-Adjusted Kappa calculations."""
        # 100% agreement
        assert compute_pabak([True, True], [True, True]) == 1.0

        # 80% agreement (4/5) -> 2*(0.8) - 1 = 0.6
        preds = [True, True, True, True, False]
        refs = [True, True, True, True, True]
        assert pytest.approx(compute_pabak(preds, refs), abs=1e-4) == 0.6000

        # Empty and error
        assert compute_pabak([], []) == 1.0
        with pytest.raises(ValueError, match="identical length"):
            compute_pabak([True], [])

    def test_compute_mae(self) -> None:
        """Verify Mean Absolute Error computation."""
        preds = [1.0, 0.8, 0.5]
        refs = [1.0, 1.0, 0.5]
        # Diff: 0.0 + 0.2 + 0.0 = 0.2 / 3 = 0.0667
        assert pytest.approx(compute_mae(preds, refs), abs=1e-3) == 0.0667
        assert compute_mae([], []) == 0.0
        with pytest.raises(ValueError, match="identical length"):
            compute_mae([1.0], [1.0, 0.5])

    def test_compute_pearson_correlation(self) -> None:
        """Verify Pearson correlation coefficient under regular and constant series."""
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        assert compute_pearson_correlation(x, y) == 1.0

        # Identical constant sequences
        assert compute_pearson_correlation([1.0, 1.0], [1.0, 1.0]) == 1.0

        # Single item fallback
        assert compute_pearson_correlation([1.0], [1.0]) == 1.0

        with pytest.raises(ValueError, match="identical length"):
            compute_pearson_correlation([1.0], [1.0, 2.0])


# ==============================================================================
# 2. Data Models and Loader Tests
# ==============================================================================


class TestDataModelsAndLoader:
    """Tests for CalibratedItem parsing, dataclass methods, and file loading."""

    def test_calibrated_item_valid(self) -> None:
        """Verify CalibratedItem creation from dictionary."""
        data = {
            "item_id": "gold_001",
            "category": "factual_direct",
            "theme": "Consumer Protection",
            "question": "Berapa lama batas waktu?",
            "context": "Konteks regulasi...",
            "generated_answer": "10 hari kerja.",
            "gold_answer": "10 hari kerja.",
            "should_abstain": False,
            "judge_faithfulness": 1.0,
            "judge_correctness": 1.0,
            "judge_faithfulness_reasoning": "Reason 1",
            "judge_correctness_reasoning": "Reason 2",
            "human_faithfulness_score": 1.0,
            "human_correctness_score": 1.0,
            "annotator_notes": "Verified against Pasal 75.",
        }
        item = CalibratedItem.from_dict(data)
        assert item.item_id == "gold_001"
        assert item.human_faithfulness_score == 1.0
        assert item.human_correctness_score == 1.0

    def test_calibrated_item_missing_human_scores(self) -> None:
        """Verify error is raised if human scores are missing/null."""
        data = {
            "item_id": "gold_001",
            "category": "factual_direct",
            "question": "Q?",
            "judge_faithfulness": 1.0,
            "judge_correctness": 1.0,
            "human_faithfulness_score": None,
            "human_correctness_score": 1.0,
        }
        with pytest.raises(ValueError, match="missing human_faithfulness_score"):
            CalibratedItem.from_dict(data)

    def test_load_calibrated_items_file(self, tmp_path: Path) -> None:
        """Verify loading items from a JSONL file."""
        target = tmp_path / "annotations.jsonl"
        valid_row = {
            "item_id": "gold_001",
            "category": "factual_direct",
            "question": "Q?",
            "judge_faithfulness": 1.0,
            "judge_correctness": 1.0,
            "human_faithfulness_score": 1.0,
            "human_correctness_score": 1.0,
        }
        target.write_text(json.dumps(valid_row) + "\n", encoding="utf-8")

        loaded = load_calibrated_items(target)
        assert len(loaded) == 1
        assert loaded[0].item_id == "gold_001"

        # Missing file error
        with pytest.raises(FileNotFoundError):
            load_calibrated_items(tmp_path / "non_existent.jsonl")


# ==============================================================================
# 3. Calibration Evaluation & Reporting Tests
# ==============================================================================


@pytest.fixture
def sample_calibrated_items() -> list[CalibratedItem]:
    """Sample annotated items with answerable, refusal, and discrepancy cases."""
    return [
        CalibratedItem(
            item_id="gold_001",
            category="factual_direct",
            theme="Consumer Protection",
            question="Pertanyaan A?",
            context="Konteks A",
            generated_answer="Jawaban A",
            gold_answer="Jawaban A",
            should_abstain=False,
            judge_faithfulness=1.0,
            judge_correctness=1.0,
            judge_faithfulness_reasoning="ok",
            judge_correctness_reasoning="ok",
            human_faithfulness_score=1.0,
            human_correctness_score=1.0,
            annotator_notes="Verified",
        ),
        CalibratedItem(
            item_id="gold_002",
            category="unanswerable_out_of_scope",
            theme="Taxation",
            question="Pertanyaan B?",
            context="Konteks B",
            generated_answer="Tidak ditemukan dasar hukum.",
            gold_answer="Tidak ditemukan dasar hukum.",
            should_abstain=True,
            judge_faithfulness=1.0,
            judge_correctness=1.0,
            judge_faithfulness_reasoning="refusal",
            judge_correctness_reasoning="refusal",
            human_faithfulness_score=1.0,
            human_correctness_score=1.0,
            annotator_notes="Proper refusal",
        ),
        CalibratedItem(
            item_id="gold_003",
            category="english_slice",
            theme="IT & Cyber",
            question="Question C?",
            context="Konteks C",
            generated_answer="Answer in English.",
            gold_answer="Answer in English.",
            should_abstain=False,
            judge_faithfulness=0.45,  # Judge penalizes cross-lingual
            judge_correctness=1.0,
            judge_faithfulness_reasoning="Lexical mismatch",
            judge_correctness_reasoning="ok",
            human_faithfulness_score=1.0,  # Human recognizes valid translation
            human_correctness_score=1.0,
            annotator_notes="Cross-lingual entailment verified",
        ),
    ]


class TestCalibrationEvaluation:
    """Tests for evaluate_calibration and report generation."""

    def test_evaluate_calibration_metrics(
        self, sample_calibrated_items: list[CalibratedItem]
    ) -> None:
        """Verify evaluation metrics calculation and discrepancy capture."""
        report = evaluate_calibration(sample_calibrated_items)
        assert report.total_items == 3

        # Faithfulness agreement: 2 of 3 pass threshold
        assert pytest.approx(report.faithfulness_agreement_rate, abs=1e-3) == 0.667
        assert report.correctness_agreement_rate == 1.0

        # Discrepancy detected on gold_003 (diff 0.55 >= 0.15)
        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].item_id == "gold_003"
        assert report.discrepancies[0].metric == "faithfulness"

        # Per-category summaries
        assert "english_slice" in report.category_summaries
        assert "factual_direct" in report.category_summaries
        assert report.category_summaries["factual_direct"].faithfulness_mae == 0.0

    def test_report_serialization_and_save(
        self, tmp_path: Path, sample_calibrated_items: list[CalibratedItem]
    ) -> None:
        """Verify report dictionary export and JSON file serialization."""
        report = evaluate_calibration(sample_calibrated_items)
        d = report.to_dict()
        assert "overall_metrics" in d
        assert "categories" in d
        assert len(d["discrepancies"]) == 1

        out_file = tmp_path / "report.json"
        saved = report.save_json(out_file)
        assert saved.exists()
        with open(saved, encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_items"] == 3

    def test_print_calibration_report(
        self, sample_calibrated_items: list[CalibratedItem], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify formatted terminal report printing."""
        report = evaluate_calibration(sample_calibrated_items)
        print_calibration_report(report)
        out = capsys.readouterr().out
        assert "EasyPasal LLM Judge Calibration & Human Agreement Report" in out
        assert "Faithfulness Agreement Rate" in out
        assert "PABAK" in out
        assert "Identified Discrepancies" in out

    def test_main_cli(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify main() CLI executes calibration and produces output file."""
        input_file = tmp_path / "annotations.jsonl"
        valid_row = {
            "item_id": "gold_001",
            "category": "factual_direct",
            "question": "Q?",
            "judge_faithfulness": 1.0,
            "judge_correctness": 1.0,
            "human_faithfulness_score": 1.0,
            "human_correctness_score": 1.0,
        }
        input_file.write_text(json.dumps(valid_row) + "\n", encoding="utf-8")
        out_report = tmp_path / "report.json"

        test_args = [
            "compute_judge_calibration.py",
            "--input",
            str(input_file),
            "--output",
            str(out_report),
        ]
        monkeypatch.setattr("sys.argv", test_args)
        main()

        assert out_report.exists()
