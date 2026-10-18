"""Unit and integration tests for EasyPasal operational rollback rehearsal (Task T7.8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.rehearse_rollback import main, run_rehearsal


def test_run_rehearsal_in_process(tmp_path: Path) -> None:
    """Test full in-process rollback rehearsal with simulated incident."""
    out_file = tmp_path / "test_rehearsal.json"

    report = run_rehearsal(
        mode="in-process",
        from_variant="gepa_v0.2",
        to_variant="baseline_v0.1",
        provider="mock",
        output_path=out_file,
    )

    assert report["status"] == "SUCCESS"
    assert report["from_variant"] == "gepa_v0.2"
    assert report["to_variant"] == "baseline_v0.1"
    assert report["method"] == "hot-swap"
    assert report["total_rehearsal_duration_ms"] > 0

    phases = report["phases"]
    assert "phase1_pre_rollback" in phases
    assert phases["phase1_pre_rollback"]["status"] == "PASS"
    assert phases["phase1_pre_rollback"]["active_variant"] == "gepa_v0.2"
    assert phases["phase1_pre_rollback"]["response_variant"] == "gepa_v0.2"

    assert "phase2_incident_trigger" in phases
    assert phases["phase2_incident_trigger"]["severity"] == "CRITICAL"
    assert "INC-SIM-" in phases["phase2_incident_trigger"]["incident_id"]

    assert "phase3_rollback_execution" in phases
    assert phases["phase3_rollback_execution"]["status"] == "PASS"
    assert phases["phase3_rollback_execution"]["target_variant"] == "baseline_v0.1"
    assert phases["phase3_rollback_execution"]["rollback_duration_ms"] >= 0

    assert "phase4_post_rollback" in phases
    assert phases["phase4_post_rollback"]["status"] == "PASS"
    assert phases["phase4_post_rollback"]["active_variant"] == "baseline_v0.1"
    assert phases["phase4_post_rollback"]["response_variant"] == "baseline_v0.1"
    assert phases["phase4_post_rollback"]["header_variant"] == "baseline_v0.1"

    assert "phase5_index_recovery" in phases
    assert phases["phase5_index_recovery"]["status"] == "PASS"
    assert phases["phase5_index_recovery"]["index_ready"] is True
    assert phases["phase5_index_recovery"]["qdrant_present"] is True
    assert phases["phase5_index_recovery"]["bm25_present"] is True

    # Validate file persistence
    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["status"] == "SUCCESS"
    assert loaded["phases"]["phase4_post_rollback"]["active_variant"] == "baseline_v0.1"


def test_cli_rehearsal_main(tmp_path: Path) -> None:
    """Test CLI entrypoint execution for rollback rehearsal."""
    out_file = tmp_path / "cli_rehearsal.json"
    test_args = [
        "rehearse_rollback.py",
        "--from-variant",
        "gepa_v0.2",
        "--to-variant",
        "baseline_v0.1",
        "--provider",
        "mock",
        "--output",
        str(out_file),
    ]

    with patch.object(sys, "argv", test_args):
        main()

    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "SUCCESS"


def test_rehearsal_index_failure(tmp_path: Path) -> None:
    """Test rollback rehearsal failure when index directory is missing required stores."""
    import pytest

    empty_idx = tmp_path / "empty_index"
    empty_idx.mkdir()
    out_file = tmp_path / "failed_rehearsal.json"

    with (
        patch(
            "easypasal.index.download.download_prebuilt_index",
            side_effect=RuntimeError("Index download disabled for test"),
        ),
        pytest.raises((FileNotFoundError, AssertionError, RuntimeError)),
    ):
        run_rehearsal(
            index_dir=empty_idx,
            output_path=out_file,
            provider="mock",
        )
