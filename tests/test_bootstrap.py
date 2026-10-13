"""Unit tests for bootstrap confidence intervals and paired variant comparisons (T3.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.compare_variants import (
    VariantComparisonReport,
    compare_judged_runs,
    compare_retrieval_runs,
    compare_variants,
    detect_report_type,
    format_comparison_table,
    load_report_json,
    main,
)
from eval.metrics.bootstrap import (
    BootstrapCI,
    PairedComparisonResult,
    bootstrap_ci,
    paired_bootstrap_comparison,
    paired_permutation_test,
)


def test_bootstrap_ci_basic_distribution() -> None:
    """Test bootstrap CI on a known synthetic distribution."""
    values = [0.2, 0.4, 0.6, 0.8, 1.0] * 8  # 40 items, mean = 0.6
    ci = bootstrap_ci(values, n_resamples=1000, confidence=0.95, seed=42)

    assert isinstance(ci, BootstrapCI)
    assert pytest.approx(ci.point_estimate, rel=1e-4) == 0.6
    assert ci.lower_bound < ci.point_estimate < ci.upper_bound
    assert 0.45 <= ci.lower_bound <= 0.58
    assert 0.62 <= ci.upper_bound <= 0.75
    assert ci.confidence_level == 0.95

    # Test dictionary and string formatting
    ci_dict = ci.to_dict()
    assert ci_dict["point_estimate"] == 0.6
    assert "formatted" in ci_dict
    assert str(ci).startswith("0.6000 [")


def test_bootstrap_ci_constant_and_edge_cases() -> None:
    """Test bootstrap CI on zero variance, single value, and empty list."""
    # Empty
    empty_ci = bootstrap_ci([])
    assert empty_ci.point_estimate == 0.0
    assert empty_ci.lower_bound == 0.0
    assert empty_ci.upper_bound == 0.0

    # Single item
    single_ci = bootstrap_ci([0.75])
    assert single_ci.point_estimate == 0.75
    assert single_ci.lower_bound == 0.75
    assert single_ci.upper_bound == 0.75

    # Constant array
    constant_ci = bootstrap_ci([0.4, 0.4, 0.4, 0.4])
    assert pytest.approx(constant_ci.point_estimate) == 0.4
    assert pytest.approx(constant_ci.lower_bound) == 0.4
    assert pytest.approx(constant_ci.upper_bound) == 0.4


def test_bootstrap_ci_seed_determinism() -> None:
    """Test that pinned random seed yields 100% deterministic CI bounds."""
    data = [0.1, 0.3, 0.5, 0.7, 0.9]
    ci1 = bootstrap_ci(data, n_resamples=500, seed=123)
    ci2 = bootstrap_ci(data, n_resamples=500, seed=123)
    ci3 = bootstrap_ci(data, n_resamples=500, seed=999)

    assert ci1.lower_bound == ci2.lower_bound
    assert ci1.upper_bound == ci2.upper_bound
    assert ci1.lower_bound != ci3.lower_bound


def test_paired_permutation_test_identical_and_separated() -> None:
    """Test paired permutation test with identical and separated distributions."""
    # Identical: delta is identically zero
    base = [0.5, 0.6, 0.7, 0.8]
    cand_same = [0.5, 0.6, 0.7, 0.8]
    p_same = paired_permutation_test(base, cand_same, n_permutations=1000)
    assert p_same == 1.0

    # Strongly separated: candidate is strictly higher
    cand_higher = [0.9, 0.95, 0.98, 0.99] * 5
    base_lower = [0.1, 0.15, 0.2, 0.25] * 5
    p_separated = paired_permutation_test(base_lower, cand_higher, n_permutations=2000, seed=42)
    assert p_separated < 0.01

    # Length mismatch error
    with pytest.raises(ValueError, match="equal length"):
        paired_permutation_test([0.1, 0.2], [0.1])

    # Empty sequence error
    with pytest.raises(ValueError, match="empty sequences"):
        paired_permutation_test([], [])


def test_paired_bootstrap_comparison_significant_gain() -> None:
    """Test paired comparison detecting significant positive gain."""
    base = [0.2, 0.3, 0.2, 0.4, 0.1, 0.3, 0.2, 0.4] * 4  # 32 items, mean ~ 0.2625
    cand = [0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.5, 0.8] * 4  # 32 items, mean ~ 0.6500

    result = paired_bootstrap_comparison(
        baseline=base,
        candidate=cand,
        metric_name="Recall@5",
        n_resamples=1000,
        n_permutations=2000,
        seed=42,
    )

    assert isinstance(result, PairedComparisonResult)
    assert result.delta_mean > 0.3
    assert result.delta_ci.lower_bound > 0.0
    assert result.p_value < 0.01
    assert result.is_significant is True
    assert result.status == "SIG_GAIN"
    assert result.status_badge == "✅ SIG GAIN"
    assert result.cohens_d > 1.0

    res_dict = result.to_dict()
    assert res_dict["status"] == "SIG_GAIN"
    assert res_dict["is_significant"] is True


def test_paired_bootstrap_comparison_significant_regression() -> None:
    """Test paired comparison detecting significant regression."""
    base = [0.8, 0.9, 0.7, 0.85] * 8
    cand = [0.2, 0.3, 0.25, 0.3] * 8

    result = paired_bootstrap_comparison(
        baseline=base,
        candidate=cand,
        metric_name="Recall@5",
        n_resamples=1000,
        n_permutations=2000,
        seed=42,
    )

    assert result.delta_mean < -0.4
    assert result.delta_ci.upper_bound < 0.0
    assert result.p_value < 0.01
    assert result.is_significant is True
    assert result.status == "REGRESSION"
    assert result.status_badge == "❌ REGRESSION"
    assert result.cohens_d < -1.0


def test_paired_bootstrap_comparison_neutral() -> None:
    """Test paired comparison on near-identical scores resulting in neutral status."""
    base = [0.5, 0.52, 0.48, 0.51, 0.49] * 6
    cand = [0.51, 0.49, 0.50, 0.52, 0.48] * 6

    result = paired_bootstrap_comparison(
        baseline=base,
        candidate=cand,
        metric_name="MRR",
        n_resamples=1000,
        n_permutations=2000,
        seed=42,
    )

    assert result.is_significant is False
    assert result.status == "NEUTRAL"
    assert result.status_badge == "ℹ️ NEUTRAL"
    assert result.delta_ci.lower_bound <= 0.0 <= result.delta_ci.upper_bound


def test_paired_bootstrap_comparison_latency_lower_is_better() -> None:
    """Test latency metric where delta < 0 is significant gain and delta > 0 is regression."""
    # Faster candidate (latency decreased)
    base_lat = [60.0, 70.0, 65.0, 80.0] * 5
    cand_lat_fast = [25.0, 30.0, 28.0, 35.0] * 5

    res_fast = paired_bootstrap_comparison(
        baseline=base_lat,
        candidate=cand_lat_fast,
        metric_name="Latency",
        n_resamples=500,
        n_permutations=1000,
        higher_is_better=False,
    )
    assert res_fast.status == "SIG_GAIN"
    assert res_fast.status_badge == "✅ SIG GAIN"

    # Slower candidate (latency increased)
    cand_lat_slow = [120.0, 130.0, 125.0, 140.0] * 5
    res_slow = paired_bootstrap_comparison(
        baseline=base_lat,
        candidate=cand_lat_slow,
        metric_name="Latency",
        n_resamples=500,
        n_permutations=1000,
        higher_is_better=False,
    )
    assert res_slow.status == "REGRESSION"
    assert res_slow.status_badge == "❌ REGRESSION"


def test_compare_retrieval_runs_mock(tmp_path: Path) -> None:
    """Test compare_retrieval_runs with paired retrieval reports."""
    mock_base = {
        "dataset_path": "data/gold/dev.jsonl",
        "item_results": [
            {
                "id": "q1",
                "category": "factual_direct",
                "should_abstain": False,
                "latency_ms": 50.0,
                "metrics": {
                    "recall@1": 0.0,
                    "recall@5": 0.2,
                    "mrr@5": 0.2,
                    "ndcg@5": 0.2,
                    "hit@1": 0.0,
                    "hit@5": 1.0,
                },
            },
            {
                "id": "q2",
                "category": "factual_direct",
                "should_abstain": False,
                "latency_ms": 60.0,
                "metrics": {
                    "recall@1": 0.0,
                    "recall@5": 0.4,
                    "mrr@5": 0.3,
                    "ndcg@5": 0.35,
                    "hit@1": 0.0,
                    "hit@5": 1.0,
                },
            },
            {
                "id": "q3",
                "category": "multi_hop_deductive",
                "should_abstain": False,
                "latency_ms": 45.0,
                "metrics": {
                    "recall@1": 0.0,
                    "recall@5": 0.0,
                    "mrr@5": 0.0,
                    "ndcg@5": 0.0,
                    "hit@1": 0.0,
                    "hit@5": 0.0,
                },
            },
            {
                "id": "q4",
                "category": "adversarial_jailbreak",
                "should_abstain": True,
                "latency_ms": 30.0,
                "metrics": {
                    "recall@1": 0.0,
                    "recall@5": 0.0,
                    "mrr@5": 0.0,
                    "ndcg@5": 0.0,
                    "hit@1": 0.0,
                    "hit@5": 0.0,
                },
            },
        ],
    }

    mock_cand = {
        "dataset_path": "data/gold/dev.jsonl",
        "item_results": [
            {
                "id": "q1",
                "category": "factual_direct",
                "should_abstain": False,
                "latency_ms": 48.0,
                "metrics": {
                    "recall@1": 1.0,
                    "recall@5": 0.8,
                    "mrr@5": 1.0,
                    "ndcg@5": 0.9,
                    "hit@1": 1.0,
                    "hit@5": 1.0,
                },
            },
            {
                "id": "q2",
                "category": "factual_direct",
                "should_abstain": False,
                "latency_ms": 52.0,
                "metrics": {
                    "recall@1": 1.0,
                    "recall@5": 0.9,
                    "mrr@5": 1.0,
                    "ndcg@5": 0.95,
                    "hit@1": 1.0,
                    "hit@5": 1.0,
                },
            },
            {
                "id": "q3",
                "category": "multi_hop_deductive",
                "should_abstain": False,
                "latency_ms": 40.0,
                "metrics": {
                    "recall@1": 0.0,
                    "recall@5": 0.5,
                    "mrr@5": 0.5,
                    "ndcg@5": 0.5,
                    "hit@1": 0.0,
                    "hit@5": 1.0,
                },
            },
            {
                "id": "q4",
                "category": "adversarial_jailbreak",
                "should_abstain": True,
                "latency_ms": 28.0,
                "metrics": {
                    "recall@1": 0.0,
                    "recall@5": 0.0,
                    "mrr@5": 0.0,
                    "ndcg@5": 0.0,
                    "hit@1": 0.0,
                    "hit@5": 0.0,
                },
            },
        ],
    }

    report = compare_retrieval_runs(
        baseline_data=mock_base,
        candidate_data=mock_cand,
        baseline_path="base.json",
        candidate_path="cand.json",
        n_resamples=500,
        n_permutations=1000,
    )

    assert isinstance(report, VariantComparisonReport)
    assert report.report_type == "retrieval"
    assert report.paired_items_count == 4
    assert report.answerable_items_count == 3
    assert "Recall@5" in report.metrics
    assert "MRR@5" in report.metrics
    assert "Latency (mean ms)" in report.metrics
    assert report.metrics["Recall@5"].delta_mean > 0.0

    # Test serialization and table formatting
    out_file = tmp_path / "comparison.json"
    saved_path = report.save_json(out_file)
    assert saved_path.exists()
    assert report.total_significant_gains >= 0

    table = format_comparison_table(report)
    assert "📊 EasyPasal Variant Comparison & Significance Report" in table
    assert "Recall@5" in table


def test_compare_judged_runs_mock() -> None:
    """Test compare_judged_runs with paired LLM judged reports."""
    mock_base = {
        "records": [
            {
                "id": "item1",
                "category": "definitional",
                "should_abstain": False,
                "faithfulness": {"score": 0.6},
                "correctness": {"score": 0.7},
            },
            {
                "id": "item2",
                "category": "definitional",
                "should_abstain": False,
                "faithfulness": {"score": 0.7},
                "correctness": {"score": 0.8},
            },
        ],
    }
    mock_cand = {
        "records": [
            {
                "id": "item1",
                "category": "definitional",
                "should_abstain": False,
                "faithfulness": {"score": 0.95},
                "correctness": {"score": 1.0},
            },
            {
                "id": "item2",
                "category": "definitional",
                "should_abstain": False,
                "faithfulness": {"score": 1.0},
                "correctness": {"score": 1.0},
            },
        ],
    }

    report = compare_judged_runs(
        baseline_data=mock_base,
        candidate_data=mock_cand,
        n_resamples=500,
        n_permutations=1000,
    )

    assert report.report_type == "judged"
    assert "Faithfulness" in report.metrics
    assert "Correctness" in report.metrics
    assert report.metrics["Faithfulness"].delta_mean > 0.3


def test_compare_variants_detection_and_exceptions(tmp_path: Path) -> None:
    """Test report auto-detection, type mismatch, and CLI invocation."""
    base_retrieval = tmp_path / "base_retrieval.json"
    cand_retrieval = tmp_path / "cand_retrieval.json"
    judged_file = tmp_path / "judged.json"
    invalid_file = tmp_path / "invalid.json"

    base_retrieval.write_text(
        json.dumps({"item_results": [{"id": "q1", "metrics": {}, "latency_ms": 10.0}]})
    )
    cand_retrieval.write_text(
        json.dumps({"item_results": [{"id": "q1", "metrics": {}, "latency_ms": 12.0}]})
    )
    judged_file.write_text(
        json.dumps(
            {
                "records": [
                    {"id": "q1", "faithfulness": {"score": 1.0}, "correctness": {"score": 1.0}}
                ]
            }
        )
    )
    invalid_file.write_text(json.dumps({"unknown_key": []}))

    assert detect_report_type(load_report_json(base_retrieval)) == "retrieval"
    assert detect_report_type(load_report_json(judged_file)) == "judged"
    assert detect_report_type(load_report_json(invalid_file)) == "unknown"

    # Mismatched types
    with pytest.raises(ValueError, match="Mismatched report types"):
        compare_variants(base_retrieval, judged_file)

    # Unknown report structure
    with pytest.raises(ValueError, match="Unable to recognize"):
        compare_variants(invalid_file, invalid_file)

    # Missing file
    with pytest.raises(FileNotFoundError):
        compare_variants(tmp_path / "nonexistent.json", base_retrieval)


def test_compare_variants_cli_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI main entrypoint."""
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    out = tmp_path / "out.json"

    item_b = {"id": "1", "metrics": {"recall@5": 0.5}, "latency_ms": 50.0}
    item_c = {"id": "1", "metrics": {"recall@5": 0.8}, "latency_ms": 40.0}

    base.write_text(json.dumps({"item_results": [item_b]}))
    cand.write_text(json.dumps({"item_results": [item_c]}))

    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_variants",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--output",
            str(out),
            "--n-resamples",
            "100",
            "--n-permutations",
            "200",
        ],
    )
    main()

    captured = capsys.readouterr()
    assert "📊 EasyPasal Variant Comparison" in captured.out
    assert out.exists()
