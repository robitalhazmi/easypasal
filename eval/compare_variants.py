"""CLI utility and statistical comparator for RAG pipeline variants and experiments.

Ingests two evaluation reports (retrieval EvalReport or LLM-judged JudgedEvalReport),
matches items by query ID, computes 95% Bootstrap Confidence Intervals and paired
significance permutation tests, and renders formatted terminal comparison tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.metrics.bootstrap import (
    PairedComparisonResult,
    paired_bootstrap_comparison,
)

logger = logging.getLogger("eval.compare_variants")


@dataclass
class VariantComparisonReport:
    """Comprehensive comparison report between baseline and candidate variant."""

    baseline_path: str
    candidate_path: str
    report_type: str  # "retrieval" or "judged"
    paired_items_count: int
    answerable_items_count: int
    metrics: dict[str, PairedComparisonResult]
    category_metrics: dict[str, dict[str, PairedComparisonResult]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def total_significant_gains(self) -> int:
        """Count of metrics with statistically significant gain."""
        return sum(1 for m in self.metrics.values() if m.status == "SIG_GAIN")

    @property
    def total_regressions(self) -> int:
        """Count of metrics with statistically significant regression."""
        return sum(1 for m in self.metrics.values() if m.status == "REGRESSION")

    @property
    def total_neutral(self) -> int:
        """Count of metrics with no statistically significant change."""
        return sum(1 for m in self.metrics.values() if m.status == "NEUTRAL")

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "report_type": self.report_type,
            "baseline_path": self.baseline_path,
            "candidate_path": self.candidate_path,
            "paired_items_count": self.paired_items_count,
            "answerable_items_count": self.answerable_items_count,
            "summary": {
                "significant_gains": self.total_significant_gains,
                "regressions": self.total_regressions,
                "neutral": self.total_neutral,
            },
            "metrics": {name: res.to_dict() for name, res in self.metrics.items()},
            "category_metrics": {
                cat: {name: res.to_dict() for name, res in cat_res.items()}
                for cat, cat_res in self.category_metrics.items()
            },
        }

    def save_json(self, output_path: Path | str) -> Path:
        """Save comparison report to JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return target


def load_report_json(path: Path | str) -> dict[str, Any]:
    """Load JSON evaluation report from file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Evaluation report file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid report structure in {p}: expected dictionary at root")
    return data


def detect_report_type(data: dict[str, Any]) -> str:
    """Detect whether report JSON is retrieval, judged, or unknown."""
    if "item_results" in data:
        return "retrieval"
    if "records" in data:
        return "judged"
    return "unknown"


def compare_retrieval_runs(
    baseline_data: dict[str, Any],
    candidate_data: dict[str, Any],
    baseline_path: str = "baseline",
    candidate_path: str = "candidate",
    n_resamples: int = 2000,
    n_permutations: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> VariantComparisonReport:
    """Compare two retrieval evaluation reports on paired query items.

    Args:
        baseline_data: Baseline EvalReport JSON dict.
        candidate_data: Candidate EvalReport JSON dict.
        baseline_path: Path/label of baseline run.
        candidate_path: Path/label of candidate run.
        n_resamples: Bootstrap resamples count.
        n_permutations: Permutation iterations count.
        confidence: Desired CI confidence level.
        seed: Random seed for exact reproducibility.

    Returns:
        VariantComparisonReport with paired retrieval metric comparisons.
    """
    base_items = {
        item["id"]: item for item in baseline_data.get("item_results", []) if "id" in item
    }
    cand_items = {
        item["id"]: item for item in candidate_data.get("item_results", []) if "id" in item
    }

    common_ids = [cid for cid in base_items if cid in cand_items]
    if not common_ids:
        raise ValueError("No common query IDs found between baseline and candidate reports.")

    # Filter answerable queries for retrieval relevance metrics
    answerable_ids = [cid for cid in common_ids if not base_items[cid].get("should_abstain", False)]
    if not answerable_ids:
        raise ValueError("No common answerable queries found between runs to evaluate retrieval.")

    # Metrics specifications: (key_name, source_path, higher_is_better)
    retrieval_metric_specs = [
        ("Recall@1", "recall@1", True),
        ("Recall@3", "recall@3", True),
        ("Recall@5", "recall@5", True),
        ("Recall@10", "recall@10", True),
        ("MRR@5", "mrr@5", True),
        ("MRR@10", "mrr@10", True),
        ("nDCG@5", "ndcg@5", True),
        ("nDCG@10", "ndcg@10", True),
        ("Hit@1", "hit@1", True),
        ("Hit@5", "hit@5", True),
    ]

    metrics_comparison: dict[str, PairedComparisonResult] = {}
    current_seed = seed

    for metric_label, dict_key, higher_is_better in retrieval_metric_specs:
        base_vals: list[float] = []
        cand_vals: list[float] = []

        for cid in answerable_ids:
            b_m = base_items[cid].get("metrics", {})
            c_m = cand_items[cid].get("metrics", {})
            if dict_key in b_m and dict_key in c_m:
                base_vals.append(float(b_m[dict_key]))
                cand_vals.append(float(c_m[dict_key]))

        if base_vals and len(base_vals) == len(cand_vals):
            res = paired_bootstrap_comparison(
                baseline=base_vals,
                candidate=cand_vals,
                metric_name=metric_label,
                n_resamples=n_resamples,
                n_permutations=n_permutations,
                confidence=confidence,
                seed=current_seed,
                higher_is_better=higher_is_better,
            )
            metrics_comparison[metric_label] = res
            current_seed += 10

    # Latency metric across all paired queries (lower is better)
    base_latencies: list[float] = [
        float(base_items[cid].get("latency_ms", 0.0)) for cid in common_ids
    ]
    cand_latencies: list[float] = [
        float(cand_items[cid].get("latency_ms", 0.0)) for cid in common_ids
    ]
    if base_latencies:
        metrics_comparison["Latency (mean ms)"] = paired_bootstrap_comparison(
            baseline=base_latencies,
            candidate=cand_latencies,
            metric_name="Latency (mean ms)",
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            confidence=confidence,
            seed=current_seed,
            higher_is_better=False,
        )
        current_seed += 10

    # Category breakdowns on key benchmark metric: Recall@5
    category_metrics: dict[str, dict[str, PairedComparisonResult]] = defaultdict(dict)
    cat_to_ids: dict[str, list[str]] = defaultdict(list)
    for cid in answerable_ids:
        cat = base_items[cid].get("category", "unknown")
        cat_to_ids[cat].append(cid)

    for cat, cat_ids in cat_to_ids.items():
        if len(cat_ids) >= 2:
            base_cat = [
                float(base_items[cid].get("metrics", {}).get("recall@5", 0.0)) for cid in cat_ids
            ]
            cand_cat = [
                float(cand_items[cid].get("metrics", {}).get("recall@5", 0.0)) for cid in cat_ids
            ]
            category_metrics[cat]["Recall@5"] = paired_bootstrap_comparison(
                baseline=base_cat,
                candidate=cand_cat,
                metric_name=f"{cat} Recall@5",
                n_resamples=n_resamples,
                n_permutations=n_permutations,
                confidence=confidence,
                seed=current_seed,
                higher_is_better=True,
            )
            current_seed += 10

    return VariantComparisonReport(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        report_type="retrieval",
        paired_items_count=len(common_ids),
        answerable_items_count=len(answerable_ids),
        metrics=metrics_comparison,
        category_metrics=dict(category_metrics),
    )


def compare_judged_runs(
    baseline_data: dict[str, Any],
    candidate_data: dict[str, Any],
    baseline_path: str = "baseline",
    candidate_path: str = "candidate",
    n_resamples: int = 2000,
    n_permutations: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> VariantComparisonReport:
    """Compare two LLM-judged evaluation reports on paired query items.

    Args:
        baseline_data: Baseline JudgedEvalReport JSON dict.
        candidate_data: Candidate JudgedEvalReport JSON dict.
        baseline_path: Path/label of baseline run.
        candidate_path: Path/label of candidate run.
        n_resamples: Bootstrap resamples count.
        n_permutations: Permutation iterations count.
        confidence: Desired CI confidence level.
        seed: Random seed for exact reproducibility.

    Returns:
        VariantComparisonReport with paired faithfulness and correctness comparisons.
    """
    base_records = {rec["id"]: rec for rec in baseline_data.get("records", []) if "id" in rec}
    cand_records = {rec["id"]: rec for rec in candidate_data.get("records", []) if "id" in rec}

    common_ids = [cid for cid in base_records if cid in cand_records]
    if not common_ids:
        raise ValueError("No common query IDs found between baseline and candidate judged reports.")

    # Faithfulness: answerable items
    answerable_ids = [
        cid for cid in common_ids if not base_records[cid].get("should_abstain", False)
    ]
    eval_ids = answerable_ids if answerable_ids else common_ids

    metrics_comparison: dict[str, PairedComparisonResult] = {}
    current_seed = seed

    # Faithfulness
    base_faith = [
        float(base_records[cid].get("faithfulness", {}).get("score", 0.0)) for cid in eval_ids
    ]
    cand_faith = [
        float(cand_records[cid].get("faithfulness", {}).get("score", 0.0)) for cid in eval_ids
    ]
    if base_faith:
        metrics_comparison["Faithfulness"] = paired_bootstrap_comparison(
            baseline=base_faith,
            candidate=cand_faith,
            metric_name="Faithfulness",
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            confidence=confidence,
            seed=current_seed,
            higher_is_better=True,
        )
        current_seed += 10

    # Correctness
    base_corr = [
        float(base_records[cid].get("correctness", {}).get("score", 0.0)) for cid in eval_ids
    ]
    cand_corr = [
        float(cand_records[cid].get("correctness", {}).get("score", 0.0)) for cid in eval_ids
    ]
    if base_corr:
        metrics_comparison["Correctness"] = paired_bootstrap_comparison(
            baseline=base_corr,
            candidate=cand_corr,
            metric_name="Correctness",
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            confidence=confidence,
            seed=current_seed,
            higher_is_better=True,
        )
        current_seed += 10

    # Category breakdowns for Faithfulness
    category_metrics: dict[str, dict[str, PairedComparisonResult]] = defaultdict(dict)
    cat_to_ids: dict[str, list[str]] = defaultdict(list)
    for cid in eval_ids:
        cat = base_records[cid].get("category", "unknown")
        cat_to_ids[cat].append(cid)

    for cat, cat_ids in cat_to_ids.items():
        if len(cat_ids) >= 2:
            bf = [
                float(base_records[cid].get("faithfulness", {}).get("score", 0.0))
                for cid in cat_ids
            ]
            cf = [
                float(cand_records[cid].get("faithfulness", {}).get("score", 0.0))
                for cid in cat_ids
            ]
            category_metrics[cat]["Faithfulness"] = paired_bootstrap_comparison(
                baseline=bf,
                candidate=cf,
                metric_name=f"{cat} Faithfulness",
                n_resamples=n_resamples,
                n_permutations=n_permutations,
                confidence=confidence,
                seed=current_seed,
                higher_is_better=True,
            )
            current_seed += 10

    return VariantComparisonReport(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        report_type="judged",
        paired_items_count=len(common_ids),
        answerable_items_count=len(eval_ids),
        metrics=metrics_comparison,
        category_metrics=dict(category_metrics),
    )


def compare_variants(
    baseline_path: Path | str,
    candidate_path: Path | str,
    output_path: Path | str | None = None,
    n_resamples: int = 2000,
    n_permutations: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> VariantComparisonReport:
    """Load and compare two variant evaluation runs, saving output if requested.

    Args:
        baseline_path: Path to baseline eval JSON file.
        candidate_path: Path to candidate eval JSON file.
        output_path: Optional path to save comparison JSON report.
        n_resamples: Number of bootstrap iterations.
        n_permutations: Number of permutation sign flips.
        confidence: Confidence level for CIs.
        seed: Random seed.

    Returns:
        VariantComparisonReport.
    """
    base_data = load_report_json(baseline_path)
    cand_data = load_report_json(candidate_path)

    base_type = detect_report_type(base_data)
    cand_type = detect_report_type(cand_data)

    if base_type == "unknown" or cand_type == "unknown":
        raise ValueError(
            f"Unable to recognize evaluation report format. "
            f"Baseline: {base_type}, Candidate: {cand_type}. "
            f"Files must contain either 'item_results' (retrieval) or 'records' (judged)."
        )

    if base_type != cand_type:
        raise ValueError(
            f"Mismatched report types cannot be paired: "
            f"baseline is '{base_type}' while candidate is '{cand_type}'."
        )

    if base_type == "retrieval":
        report = compare_retrieval_runs(
            baseline_data=base_data,
            candidate_data=cand_data,
            baseline_path=str(baseline_path),
            candidate_path=str(candidate_path),
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            confidence=confidence,
            seed=seed,
        )
    else:
        report = compare_judged_runs(
            baseline_data=base_data,
            candidate_data=cand_data,
            baseline_path=str(baseline_path),
            candidate_path=str(candidate_path),
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            confidence=confidence,
            seed=seed,
        )

    if output_path:
        report.save_json(output_path)
        logger.info("Saved variant comparison report to %s", output_path)

    return report


def format_comparison_table(report: VariantComparisonReport) -> str:
    """Format variant comparison report into an attractive ASCII terminal table."""
    lines: list[str] = []
    w = 118
    lines.append("=" * w)
    lines.append("📊 EasyPasal Variant Comparison & Significance Report (T3.5)".center(w))
    lines.append("=" * w)
    lines.append(f"Baseline:  {report.baseline_path}")
    lines.append(f"Candidate: {report.candidate_path}")
    report_type_str = "Retrieval" if report.report_type == "retrieval" else "LLM-Judged"
    lines.append(
        f"Evaluation Type: {report_type_str} | "
        f"Total Paired: {report.paired_items_count} | "
        f"Answerable Paired: {report.answerable_items_count} items"
    )
    lines.append("Confidence Level: 95% Bootstrap CI (B=2000, P=5000)")
    lines.append("-" * w)

    # Column header
    header = (
        f"{'Metric':<20} | {'Baseline [95% CI]':<26} | {'Candidate [95% CI]':<26} | "
        f"{'Delta [95% CI]':<26} | {'p-val':<7} | {'Cohen d':<7} | {'Status'}"
    )
    lines.append(header)
    lines.append("-" * w)

    for name, res in report.metrics.items():
        base_ci_str = f"{res.baseline_mean:.4f} [{res.baseline_ci.lower_bound:.4f}, {res.baseline_ci.upper_bound:.4f}]"
        cand_ci_str = f"{res.candidate_mean:.4f} [{res.candidate_ci.lower_bound:.4f}, {res.candidate_ci.upper_bound:.4f}]"
        delta_ci_str = f"{res.delta_mean:+.4f} [{res.delta_ci.lower_bound:+.4f}, {res.delta_ci.upper_bound:+.4f}]"
        pval_str = f"{res.p_value:.4f}" if res.p_value >= 0.0001 else "<0.0001"
        cd_str = f"{res.cohens_d:+.2f}"

        row = (
            f"{name:<20} | {base_ci_str:<26} | {cand_ci_str:<26} | "
            f"{delta_ci_str:<26} | {pval_str:<7} | {cd_str:<7} | {res.status_badge}"
        )
        lines.append(row)

    lines.append("-" * w)
    lines.append(
        f"Summary: {report.total_significant_gains} significant gain(s), "
        f"{report.total_regressions} regression(s), "
        f"{report.total_neutral} neutral"
    )

    # Category breakdown if available
    if report.category_metrics:
        lines.append("-" * w)
        lines.append("📁 Per-Category Key Metric Breakdown:")
        cat_header = (
            f"{'Category':<30} | {'Metric':<14} | {'Delta [95% CI]':<26} | "
            f"{'p-val':<7} | {'Status'}"
        )
        lines.append(cat_header)
        lines.append("-" * w)
        for cat, cat_res in sorted(report.category_metrics.items()):
            for m_key, res in cat_res.items():
                delta_str = f"{res.delta_mean:+.4f} [{res.delta_ci.lower_bound:+.4f}, {res.delta_ci.upper_bound:+.4f}]"
                pval_str = f"{res.p_value:.4f}" if res.p_value >= 0.0001 else "<0.0001"
                cat_row = f"{cat:<30} | {m_key:<14} | {delta_str:<26} | {pval_str:<7} | {res.status_badge}"
                lines.append(cat_row)
        lines.append("-" * w)

    lines.append("=" * w)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for compare_variants."""
    parser = argparse.ArgumentParser(
        description="EasyPasal Variant Comparison & Statistical Significance Utility (T3.5)"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Path to baseline evaluation JSON file (retrieval or judged).",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Path to candidate evaluation JSON file (retrieval or judged).",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Optional path to save comparison report JSON.",
    )
    parser.add_argument(
        "--n-resamples",
        default=2000,
        type=int,
        help="Bootstrap resamples count (default: 2000).",
    )
    parser.add_argument(
        "--n-permutations",
        default=5000,
        type=int,
        help="Permutation test iterations (default: 5000).",
    )
    parser.add_argument(
        "--confidence",
        default=0.95,
        type=float,
        help="Confidence interval level (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        report = compare_variants(
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            output_path=args.output,
            n_resamples=args.n_resamples,
            n_permutations=args.n_permutations,
            confidence=args.confidence,
            seed=args.seed,
        )
        print(format_comparison_table(report))
        if report.total_regressions > 0:
            logger.warning(
                "Detected %d statistically significant regression(s)!", report.total_regressions
            )
    except Exception as e:
        logger.error("Variant comparison failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
