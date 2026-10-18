"""Tests for real usage traffic generator and monitoring data population (Task T7.9)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.run_real_usage_session import SAMPLE_QUERIES, main, run_session


def test_sample_queries_dataset_structure() -> None:
    """Validate sample queries volume, domain coverage, and feedback structure."""
    # AC: ask 30+ varied questions
    assert len(SAMPLE_QUERIES) >= 30, f"Expected >= 30 questions, found {len(SAMPLE_QUERIES)}"

    domains = {q["domain"] for q in SAMPLE_QUERIES}
    assert "Banking" in domains
    assert "Capital Markets" in domains
    assert "Fintech P2P Lending" in domains
    assert "Consumer Protection" in domains
    assert "IT & Cyber Risk" in domains
    assert "AML / CFT" in domains
    assert "Out of Scope" in domains
    assert "Prompt Injection" in domains

    # Verify both production variants are exercised
    variants = {q["variant"] for q in SAMPLE_QUERIES}
    assert "gepa_v0.2" in variants
    assert "baseline_v0.1" in variants

    # Verify non-trivial feedback entries exist
    feedback_entries = [q for q in SAMPLE_QUERIES if q.get("feedback") is not None]
    assert len(feedback_entries) >= 10, "Expected at least 10 feedback entries"


def test_run_session_miniature(tmp_path: Path) -> None:
    """Test running session driver on isolated temporary SQLite DB and JSONL files."""
    db_path = tmp_path / "test_collector.db"
    req_file = tmp_path / "test_requests.jsonl"
    fb_file = tmp_path / "test_feedback.jsonl"

    mini_queries = SAMPLE_QUERIES[:5]

    with patch("scripts.run_real_usage_session.SAMPLE_QUERIES", mini_queries):
        report = run_session(
            db_path=db_path,
            log_file=req_file,
            feedback_file=fb_file,
            provider="mock",
        )

    assert report["total_queries_asked"] == 5
    assert db_path.exists()
    assert req_file.exists()

    # Validate JSON Lines structure
    with open(req_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 5

    record = json.loads(lines[0])
    assert "request_id" in record
    assert "total_latency_ms" in record
    assert "variant" in record


def test_real_usage_main_cli(tmp_path: Path) -> None:
    """Test CLI invocation of real usage session driver."""
    db_path = tmp_path / "cli_collector.db"
    req_file = tmp_path / "cli_requests.jsonl"
    fb_file = tmp_path / "cli_feedback.jsonl"

    mini_queries = SAMPLE_QUERIES[:2]
    cli_args = [
        "run_real_usage_session.py",
        "--db-path",
        str(db_path),
        "--log-file",
        str(req_file),
        "--feedback-file",
        str(fb_file),
        "--provider",
        "mock",
    ]

    with (
        patch("scripts.run_real_usage_session.SAMPLE_QUERIES", mini_queries),
        patch.object(sys, "argv", cli_args),
    ):
        main()

    assert db_path.exists()
    assert req_file.exists()
