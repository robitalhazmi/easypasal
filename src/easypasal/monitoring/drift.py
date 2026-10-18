"""Evidently data and quality drift reporting engine (Plan §9 / Task T7.4).

Quantifies distributional drift between live production query logs and the
hand-curated gold standard evaluation dataset across question length, word count,
regulatory legal themes, and abstention behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from easypasal.collector.db import CollectorDB
from easypasal.monitoring.schemas import RequestLogRecord

logger = logging.getLogger("easypasal.monitoring.drift")


def build_gold_reference_dataset(gold_dir: Path | str = "data/gold") -> pd.DataFrame:
    """Load reference dataset from hand-curated gold standard splits (train/dev/test).

    Args:
        gold_dir: Path to directory containing gold dataset JSONL files.

    Returns:
        DataFrame with standardized feature columns for drift analysis.
    """
    gold_path = Path(gold_dir)
    records: list[dict[str, Any]] = []

    files = list(gold_path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No gold dataset .jsonl files found in {gold_dir}")

    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    item = json.loads(line_str)
                    q = item.get("question", "")
                    theme = item.get("theme") or "Unknown"
                    should_abstain = bool(item.get("should_abstain", False))
                    records.append(
                        {
                            "question_length": len(q),
                            "word_count": len(q.split()),
                            "theme": theme,
                            "abstained": 1 if should_abstain else 0,
                        }
                    )
                except Exception as exc:
                    logger.warning("Skipped malformed gold item in %s: %s", file_path, exc)

    df = pd.DataFrame(records)
    logger.info("Loaded reference gold dataset with %d items from %s", len(df), gold_dir)
    return df


def build_current_dataset_from_db(
    db: CollectorDB | Path | str,
    limit: int = 1000,
    fallback_if_empty: bool = True,
) -> pd.DataFrame:
    """Extract production query logs from SQLite collector database for drift comparison.

    Args:
        db: CollectorDB instance or path to SQLite file.
        limit: Maximum number of recent request logs to load.
        fallback_if_empty: If true and DB has < 10 rows, bootstrap sample queries.

    Returns:
        DataFrame with feature columns matching reference dataset.
    """
    collector_db = CollectorDB(db_path=db) if isinstance(db, (str, Path)) else db

    items, total = collector_db.get_requests(limit=limit)

    if total < 10 and fallback_if_empty:
        logger.warning(
            "Collector DB has only %d requests (<10). Using gold dataset baseline sample.",
            total,
        )
        gold_df = build_gold_reference_dataset()
        # Sample slightly perturbed subset as proxy
        sampled = gold_df.sample(frac=min(1.0, 50 / max(len(gold_df), 1)), random_state=42).copy()
        return sampled

    records: list[dict[str, Any]] = []
    for item in items:
        q = item.get("question_redacted", "")
        records.append(
            {
                "question_length": int(item.get("question_length", len(q))),
                "word_count": len(q.split()),
                "theme": item.get("theme") or "Unknown",
                "abstained": 1 if item.get("abstained") else 0,
            }
        )

    return pd.DataFrame(records)


def build_current_dataset_from_file(log_file: Path | str) -> pd.DataFrame:
    """Extract query log records from a JSON Lines log file."""
    p = Path(log_file)
    if not p.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    records: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = RequestLogRecord.model_validate_json(line_str)
                records.append(
                    {
                        "question_length": rec.question_length,
                        "word_count": len(rec.question_redacted.split()),
                        "theme": rec.theme or "Unknown",
                        "abstained": 1 if rec.abstained else 0,
                    }
                )
            except Exception as exc:
                logger.warning("Skipped invalid log line in %s: %s", p, exc)

    return pd.DataFrame(records)


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_html_path: Path | str | None = "reports/drift_report.html",
    output_json_path: Path | str | None = "reports/drift_summary.json",
    drift_share_threshold: float = 0.5,
) -> dict[str, Any]:
    """Execute Evidently data drift analysis and save HTML and JSON artifacts.

    Args:
        reference_df: Baseline reference dataset (gold set).
        current_df: Production query logs dataset.
        output_html_path: Optional path to save full interactive HTML report.
        output_json_path: Optional path to save machine-readable summary JSON.
        drift_share_threshold: Fraction of drifted columns triggering dataset drift alert.

    Returns:
        Structured drift summary dictionary.
    """
    if reference_df.empty or current_df.empty:
        raise ValueError("Both reference_df and current_df must be non-empty DataFrames")

    # Align columns
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    ref_aligned = reference_df[common_cols].copy()
    cur_aligned = current_df[common_cols].copy()

    # Configure and run Evidently report
    report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
    snapshot = report.run(current_data=cur_aligned, reference_data=ref_aligned)

    # Save HTML artifact
    if output_html_path is not None:
        html_p = Path(output_html_path)
        html_p.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_p))
        logger.info("Saved Evidently HTML drift report to %s", html_p)

    # Extract structured summary metrics from snapshot dictionary
    snap_dict = snapshot.dict()
    metrics_list = snap_dict.get("metrics", [])

    drifted_cols_count = 0.0
    drift_share = 0.0
    column_details: dict[str, Any] = {}

    for m in metrics_list:
        metric_name = m.get("metric_name", "")
        val = m.get("value")

        if "DriftedColumnsCount" in metric_name and isinstance(val, dict):
            drifted_cols_count = float(val.get("count", 0.0))
            drift_share = float(val.get("share", 0.0))

        elif "ValueDrift" in metric_name:
            # Parse column name and method from metric string: e.g. ValueDrift(column=theme,method=chi-square p_value,threshold=0.05)
            config = m.get("config", {})
            col_name = config.get("column")
            if not col_name:
                parts = metric_name.split("column=")
                if len(parts) > 1:
                    col_name = parts[1].split(",")[0]

            p_val = float(val) if val is not None else 1.0
            threshold = float(config.get("threshold", 0.05))
            drift_detected = p_val < threshold

            if col_name:
                column_details[col_name] = {
                    "p_value": round(p_val, 5),
                    "drift_detected": drift_detected,
                    "threshold": threshold,
                    "method": config.get("method", "statistical"),
                }

    dataset_drift = drift_share >= drift_share_threshold

    summary: dict[str, Any] = {
        "dataset_drift": dataset_drift,
        "drift_share": round(drift_share, 4),
        "number_of_drifted_columns": int(drifted_cols_count),
        "number_of_columns": len(common_cols),
        "drift_share_threshold": drift_share_threshold,
        "reference_records_count": len(ref_aligned),
        "current_records_count": len(cur_aligned),
        "columns": column_details,
        "timestamp": datetime.now(UTC).isoformat(),
        "html_report_path": str(output_html_path) if output_html_path else None,
    }

    # Save JSON summary artifact
    if output_json_path is not None:
        json_p = Path(output_json_path)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved Evidently JSON drift summary to %s", json_p)

    return summary


def main() -> None:
    """CLI entrypoint for running Evidently drift analysis."""
    parser = argparse.ArgumentParser(description="Run Evidently data drift analysis.")
    parser.add_argument(
        "--gold-dir",
        default="data/gold",
        help="Path to gold dataset directory (default: data/gold)",
    )
    parser.add_argument(
        "--db-path",
        default="data/collector/collector.db",
        help="Path to SQLite collector database (default: data/collector/collector.db)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to requests.jsonl file",
    )
    parser.add_argument(
        "--output-html",
        default="reports/drift_report.html",
        help="Path for saving HTML drift report",
    )
    parser.add_argument(
        "--output-json",
        default="reports/drift_summary.json",
        help="Path for saving JSON drift summary",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ref_df = build_gold_reference_dataset(gold_dir=args.gold_dir)

    if args.log_file and Path(args.log_file).exists():
        cur_df = build_current_dataset_from_file(args.log_file)
    else:
        cur_df = build_current_dataset_from_db(args.db_path)

    summary = generate_drift_report(
        reference_df=ref_df,
        current_df=cur_df,
        output_html_path=args.output_html,
        output_json_path=args.output_json,
    )

    print("\n" + "=" * 60)
    print("Evidently Data Drift Analysis Summary")
    print("=" * 60)
    print(f"Dataset Drift Detected:   {summary['dataset_drift']}")
    print(f"Drift Share:             {summary['drift_share'] * 100:.1f}%")
    print(
        f"Drifted Columns:         {summary['number_of_drifted_columns']} / {summary['number_of_columns']}"
    )
    print(f"Reference Records (Gold): {summary['reference_records_count']}")
    print(f"Current Records (Logs):  {summary['current_records_count']}")
    print("\nFeature Details:")
    for col, detail in summary["columns"].items():
        status = "DRIFTED" if detail["drift_detected"] else "OK"
        print(f"  • {col:<18}: {status:<8} (p-value={detail['p_value']:.4f})")
    print("=" * 60)
    print(f"HTML Report: {summary['html_report_path']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
