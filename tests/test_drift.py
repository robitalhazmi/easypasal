"""Unit and integration tests for Evidently drift reporting engine (Plan §9 / Task T7.4).

Validates gold dataset reference loading, production log extraction,
distributional drift detection, HTML report generation, and collector API integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from easypasal.collector.app import create_collector_app
from easypasal.collector.config import CollectorConfig
from easypasal.collector.db import CollectorDB
from easypasal.monitoring.drift import (
    build_current_dataset_from_db,
    build_current_dataset_from_file,
    build_gold_reference_dataset,
    generate_drift_report,
)
from easypasal.monitoring.schemas import RequestLogRecord

AUTH_TOKEN = "test-drift-collector-secret"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary SQLite database path."""
    return tmp_path / "drift_test.db"


@pytest.fixture
def collector_config(temp_db_path: Path) -> CollectorConfig:
    """Fixture returning a test collector configuration."""
    return CollectorConfig(
        host="127.0.0.1",
        port=8080,
        db_path=temp_db_path,
        auth_token=AUTH_TOKEN,
    )


@pytest.fixture
def client(collector_config: CollectorConfig) -> TestClient:
    """Fixture providing a TestClient for collector app."""
    app = create_collector_app(config=collector_config, is_test=True)
    return TestClient(app)


def test_build_gold_reference_dataset() -> None:
    """Test loading gold standard dataset splits as reference baseline."""
    df = build_gold_reference_dataset(gold_dir="data/gold")
    assert not df.empty
    assert len(df) >= 100
    assert set(df.columns) == {"question_length", "word_count", "theme", "abstained"}
    assert df["question_length"].dtype in ("int64", "int32")
    assert df["word_count"].dtype in ("int64", "int32")
    assert df["abstained"].isin([0, 1]).all()


def test_build_gold_reference_dataset_not_found(tmp_path: Path) -> None:
    """Test FileNotFoundError when gold directory is empty or missing."""
    empty_dir = tmp_path / "empty_gold"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_gold_reference_dataset(gold_dir=empty_dir)


def test_build_current_dataset_from_db_with_records(temp_db_path: Path) -> None:
    """Test extracting query logs from CollectorDB with populated records."""
    db = CollectorDB(db_path=temp_db_path)

    # Insert 15 dummy requests
    for i in range(15):
        rec = RequestLogRecord(
            request_id=f"drift-req-{i}",
            variant="gepa_v0.2",
            language="id",
            question_length=40 + i,
            question_redacted="Berapa modal minimum BPR menurut peraturan?",
            retrieval_score_top1=0.85,
            abstained=(i % 5 == 0),
            routed=True,
            theme="Perbankan",
            total_latency_ms=115.0,
        )
        db.insert_request(rec)

    cur_df = build_current_dataset_from_db(db=db, limit=50)
    assert len(cur_df) == 15
    assert set(cur_df.columns) == {"question_length", "word_count", "theme", "abstained"}
    assert cur_df["theme"].iloc[0] == "Perbankan"


def test_build_current_dataset_from_db_empty_fallback(temp_db_path: Path) -> None:
    """Test graceful fallback to gold sample when DB has fewer than 10 records."""
    db = CollectorDB(db_path=temp_db_path)
    cur_df = build_current_dataset_from_db(db=db, fallback_if_empty=True)
    assert not cur_df.empty
    assert set(cur_df.columns) == {"question_length", "word_count", "theme", "abstained"}


def test_build_current_dataset_from_file(tmp_path: Path) -> None:
    """Test loading current dataset from a JSONL log file."""
    log_file = tmp_path / "requests.jsonl"
    rec = RequestLogRecord(
        request_id="file-req-1",
        variant="baseline_v0.1",
        language="id",
        question_length=35,
        question_redacted="Apa syarat izin usaha fintech?",
        retrieval_score_top1=0.81,
        abstained=False,
        routed=True,
        theme="Fintech",
        total_latency_ms=95.0,
    )
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(rec.model_dump_json() + "\n")

    cur_df = build_current_dataset_from_file(log_file)
    assert len(cur_df) == 1
    assert cur_df["theme"].iloc[0] == "Fintech"


def test_generate_drift_report_and_artifacts(tmp_path: Path) -> None:
    """Test generating Evidently drift report and validating output JSON/HTML."""
    # Synthetic reference and current datasets
    ref_data: dict[str, list[Any]] = {
        "question_length": [30, 45, 50, 60, 40, 55, 65, 70, 35, 48, 52, 63, 44, 38, 56],
        "word_count": [5, 7, 8, 10, 6, 9, 11, 12, 6, 8, 9, 10, 7, 6, 9],
        "theme": [
            "Perbankan",
            "Pasar Modal",
            "Perbankan",
            "Fintech",
            "Perbankan",
            "Fintech",
            "Pasar Modal",
            "Perbankan",
            "Fintech",
            "Pasar Modal",
            "Perbankan",
            "Fintech",
            "Perbankan",
            "Fintech",
            "Pasar Modal",
        ],
        "abstained": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    }
    cur_data: dict[str, list[Any]] = {
        "question_length": [32, 44, 48, 65, 42, 58, 62, 72, 38, 50, 54, 60, 46, 40, 58],
        "word_count": [5, 7, 8, 11, 7, 9, 10, 12, 6, 8, 9, 10, 8, 6, 9],
        "theme": [
            "Perbankan",
            "Pasar Modal",
            "Perbankan",
            "Fintech",
            "Perbankan",
            "Fintech",
            "Pasar Modal",
            "Perbankan",
            "Fintech",
            "Pasar Modal",
            "Perbankan",
            "Fintech",
            "Perbankan",
            "Fintech",
            "Pasar Modal",
        ],
        "abstained": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    }
    ref_df = pd.DataFrame(ref_data)
    cur_df = pd.DataFrame(cur_data)

    out_html = tmp_path / "drift.html"
    out_json = tmp_path / "drift.json"

    summary = generate_drift_report(
        reference_df=ref_df,
        current_df=cur_df,
        output_html_path=out_html,
        output_json_path=out_json,
    )

    assert "dataset_drift" in summary
    assert "drift_share" in summary
    assert summary["number_of_columns"] == 4
    assert "question_length" in summary["columns"]
    assert "theme" in summary["columns"]

    # Verify artifacts on disk
    assert out_html.exists()
    assert out_html.stat().st_size > 50000
    assert out_json.exists()

    with open(out_json, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["number_of_columns"] == 4


def test_generate_drift_report_empty_df_raises() -> None:
    """Test ValueError is raised when either DataFrame is empty."""
    empty = pd.DataFrame()
    valid = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Both reference_df and current_df must be non-empty"):
        generate_drift_report(reference_df=empty, current_df=valid)


def test_build_current_dataset_from_file_not_found() -> None:
    """Test FileNotFoundError is raised when log file does not exist."""
    with pytest.raises(FileNotFoundError, match="Log file not found"):
        build_current_dataset_from_file(Path("non_existent_dir/requests.jsonl"))


def test_build_current_dataset_from_file_with_malformed_and_empty_lines(tmp_path: Path) -> None:
    """Test handling of empty and malformed JSONL lines in log file."""
    log_file = tmp_path / "corrupted_requests.jsonl"
    rec = RequestLogRecord(
        request_id="req-valid-1",
        variant="baseline_v0.1",
        language="id",
        question_length=30,
        question_redacted="Valid question text",
        retrieval_score_top1=0.75,
        abstained=False,
        routed=False,
        theme="Perbankan",
        total_latency_ms=100.0,
    )
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n")  # Empty line
        f.write("{invalid_json}\n")  # Corrupted line
        f.write(rec.model_dump_json() + "\n")  # Valid line

    df = build_current_dataset_from_file(log_file)
    assert len(df) == 1
    assert df["theme"].iloc[0] == "Perbankan"


def test_collector_api_drift_endpoints(client: TestClient) -> None:
    """Test GET /v1/reports/drift and GET /reports/drift endpoints."""
    # Authenticated summary endpoint
    res = client.get(
        "/v1/reports/drift",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "dataset_drift" in data
    assert "drift_share" in data
    assert "columns" in data

    # Unauthenticated /reports/drift HTML endpoint
    res_html = client.get("/reports/drift")
    assert res_html.status_code == 200
    assert "text/html" in res_html.headers["content-type"]


def test_collector_api_drift_html_regenerates_when_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test GET /reports/drift dynamically generates report when file does not exist."""
    fake_report = tmp_path / "missing_dir" / "drift_report.html"
    monkeypatch.setattr(
        "easypasal.collector.routes.Path",
        lambda p: fake_report if p == "reports/drift_report.html" else Path(p),
    )

    res_html = client.get("/reports/drift")
    assert res_html.status_code == 200
    assert "text/html" in res_html.headers["content-type"]
    assert fake_report.exists()


def test_drift_cli_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI main() execution in easypasal.monitoring.drift."""
    from easypasal.monitoring.drift import main

    out_html = tmp_path / "cli_drift.html"
    out_json = tmp_path / "cli_drift.json"

    test_args = [
        "drift.py",
        "--gold-dir",
        "data/gold",
        "--db-path",
        str(tmp_path / "cli_db.sqlite"),
        "--output-html",
        str(out_html),
        "--output-json",
        str(out_json),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    # Invoke CLI entrypoint
    main()

    assert out_html.exists()
    assert out_json.exists()
