"""Unit and integration tests for EasyPasal threshold alerting and drills (Plan §9 / Task T7.5).

Tests operational threshold evaluation (abstain rate, p95 latency, low evidence),
flagged event persistence in SQLite CollectorDB and JSONL logs, automated alert drills,
webhook notifications, and collector REST API endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from easypasal.collector.app import create_collector_app
from easypasal.collector.config import CollectorConfig
from easypasal.collector.db import CollectorDB
from easypasal.monitoring.alerts import (
    dispatch_webhook,
    evaluate_threshold_alerts,
    main,
    run_alert_drill,
)
from easypasal.monitoring.schemas import (
    AlertCheckResult,
    AlertEvent,
    AlertThresholds,
    RequestLogRecord,
)

AUTH_TOKEN = "test-alerts-secret-token"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary SQLite database path."""
    return tmp_path / "alerts_test.db"


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
    """Fixture providing a TestClient for testing collector alert endpoints."""
    app = create_collector_app(config=collector_config, is_test=True)
    return TestClient(app)


def _populate_requests(
    db: CollectorDB,
    count: int = 15,
    abstain_count: int = 1,
    latency_ms: float = 150.0,
    top1_score: float = 0.82,
) -> None:
    """Helper to seed request logs with parameterized distributions."""
    for idx in range(count):
        is_abstain = idx < abstain_count
        rec = RequestLogRecord(
            request_id=f"alert-req-{idx}",
            variant="baseline_v0.1",
            language="id",
            question_length=35,
            question_redacted=f"Sample query {idx} for alert testing",
            retrieval_score_top1=top1_score,
            retrieval_scores_topk=[top1_score, top1_score - 0.1],
            retrieved_doc_ids=["POJK-1-2023"],
            abstained=is_abstain,
            routed=False,
            theme="Perbankan",
            total_latency_ms=latency_ms + (idx * 10.0),
        )
        db.insert_request(rec)


def test_alert_schemas() -> None:
    """Verify serialization and validation of AlertEvent and AlertThresholds."""
    alert = AlertEvent(
        metric_name="abstain_rate",
        severity="WARNING",
        current_value=0.42,
        threshold_value=0.35,
        condition=">",
        sample_size=100,
        message="Abstain rate exceeded limit.",
        is_drill=False,
    )
    assert alert.metric_name == "abstain_rate"
    assert alert.current_value == 0.42
    assert not alert.is_drill

    # JSON roundtrip
    dumped = alert.model_dump_json()
    loaded = AlertEvent.model_validate_json(dumped)
    assert loaded.alert_id == alert.alert_id

    # Thresholds defaults
    thresh = AlertThresholds()
    assert thresh.abstain_rate_max == 0.35
    assert thresh.p95_latency_ms_max == 5000.0
    assert thresh.min_sample_size == 10

    # Result model
    res = AlertCheckResult(
        sample_size=50,
        status="OK",
        alerts_triggered=[],
        summary={"total_requests": 50},
    )
    assert res.status == "OK"


def test_collector_db_alert_persistence(temp_db_path: Path) -> None:
    """Verify inserting, querying, and filtering alert events in CollectorDB."""
    db = CollectorDB(db_path=temp_db_path)

    alert_live = AlertEvent(
        metric_name="abstain_rate",
        severity="WARNING",
        current_value=0.40,
        threshold_value=0.35,
        condition=">",
        sample_size=50,
        message="Live alert: abstain rate high.",
        is_drill=False,
    )
    alert_drill = AlertEvent(
        metric_name="p95_latency_ms",
        severity="CRITICAL",
        current_value=8500.0,
        threshold_value=5000.0,
        condition=">",
        sample_size=50,
        message="Drill alert: latency spike.",
        is_drill=True,
    )

    db.insert_alert(alert_live)
    db.insert_alert(alert_drill)

    # Query all
    items, total = db.get_alerts()
    assert total == 2
    assert len(items) == 2

    # Filter by is_drill
    drill_items, drill_total = db.get_alerts(is_drill=True)
    assert drill_total == 1
    assert drill_items[0]["metric_name"] == "p95_latency_ms"
    assert drill_items[0]["is_drill"] is True

    # Filter by metric_name
    abstain_items, _ = db.get_alerts(metric_name="abstain_rate")
    assert len(abstain_items) == 1
    assert abstain_items[0]["severity"] == "WARNING"

    # Active alerts in last 60 minutes
    active = db.get_active_alerts(lookback_minutes=60)
    assert len(active) == 2


def test_evaluate_threshold_alerts_insufficient_data(temp_db_path: Path) -> None:
    """Verify that fewer than min_sample_size records skips alerting."""
    db = CollectorDB(db_path=temp_db_path)
    _populate_requests(db, count=5, abstain_count=4)  # 5 requests (< 10 threshold)

    res = evaluate_threshold_alerts(db=db)
    assert res.status == "INSUFFICIENT_DATA"
    assert res.sample_size == 5
    assert len(res.alerts_triggered) == 0


def test_evaluate_threshold_alerts_normal(temp_db_path: Path) -> None:
    """Verify that healthy metrics produce status OK and zero alerts."""
    db = CollectorDB(db_path=temp_db_path)
    # 20 requests, 1 abstain (5% < 35%), latency ~150ms (< 5000ms), score 0.85
    _populate_requests(db, count=20, abstain_count=1, latency_ms=150.0, top1_score=0.85)

    res = evaluate_threshold_alerts(db=db)
    assert res.status == "OK"
    assert len(res.alerts_triggered) == 0
    assert res.sample_size == 20


def test_evaluate_threshold_alerts_abstain_rate_breach(temp_db_path: Path, tmp_path: Path) -> None:
    """Verify that abstention rate > 35% triggers a flagged alert event."""
    db = CollectorDB(db_path=temp_db_path)
    log_sink = tmp_path / "alerts.jsonl"
    # 20 requests, 10 abstains (50% >= 50% -> CRITICAL)
    _populate_requests(db, count=20, abstain_count=10, latency_ms=150.0)

    res = evaluate_threshold_alerts(db=db, alerts_log_path=log_sink)
    assert res.status == "ALERT"
    assert len(res.alerts_triggered) >= 1

    abstain_alert = next(a for a in res.alerts_triggered if a.metric_name == "abstain_rate")
    assert abstain_alert.severity == "CRITICAL"
    assert abstain_alert.current_value == 0.50
    assert abstain_alert.threshold_value == 0.35
    assert not abstain_alert.is_drill

    # Verify event stored in DB
    items, total = db.get_alerts(metric_name="abstain_rate")
    assert total >= 1

    # Verify event written to log sink
    assert log_sink.exists()
    with open(log_sink, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) >= 1
    logged_data = json.loads(lines[0])
    assert logged_data["metric_name"] == "abstain_rate"


def test_evaluate_threshold_alerts_p95_latency_breach(temp_db_path: Path) -> None:
    """Verify that p95 latency > 5000ms triggers a flagged alert event."""
    db = CollectorDB(db_path=temp_db_path)
    # 15 requests, 0 abstains, latency ~6000ms
    _populate_requests(db, count=15, abstain_count=0, latency_ms=6000.0)

    res = evaluate_threshold_alerts(db=db)
    assert res.status == "ALERT"
    lat_alert = next(a for a in res.alerts_triggered if a.metric_name == "p95_latency_ms")
    assert lat_alert.severity == "WARNING"
    assert lat_alert.current_value >= 5000.0


def test_evaluate_threshold_alerts_low_evidence_breach(temp_db_path: Path) -> None:
    """Verify that low-evidence query rate (>30% with score < 0.60) triggers alert."""
    db = CollectorDB(db_path=temp_db_path)
    # 15 requests, top-1 score 0.45 (< 0.60), so 100% low-evidence > 30%
    _populate_requests(db, count=15, abstain_count=0, latency_ms=150.0, top1_score=0.45)

    res = evaluate_threshold_alerts(db=db)
    assert res.status == "ALERT"
    ev_alert = next(a for a in res.alerts_triggered if a.metric_name == "low_evidence_rate")
    assert ev_alert.severity == "WARNING"
    assert ev_alert.current_value == 1.0


def test_run_alert_drill(temp_db_path: Path, tmp_path: Path) -> None:
    """Verify automated alert drill execution (Plan §9 / Task T7.5 AC)."""
    db = CollectorDB(db_path=temp_db_path)
    log_sink = tmp_path / "drill_alerts.jsonl"

    # Drill 1: abstain_rate
    alert_abstain = run_alert_drill(
        db=db,
        metric_to_test="abstain_rate",
        alerts_log_path=log_sink,
    )
    assert alert_abstain.is_drill is True
    assert alert_abstain.metric_name == "abstain_rate"
    assert alert_abstain.current_value > alert_abstain.threshold_value
    assert "DRILL ALERT" in alert_abstain.message

    # Drill 2: p95_latency_ms
    alert_lat = run_alert_drill(
        db=db,
        metric_to_test="p95_latency_ms",
        alerts_log_path=log_sink,
    )
    assert alert_lat.is_drill is True
    assert alert_lat.metric_name == "p95_latency_ms"

    # Drill 3: low_evidence_rate
    alert_ev = run_alert_drill(
        db=db,
        metric_to_test="low_evidence_rate",
        alerts_log_path=log_sink,
    )
    assert alert_ev.is_drill is True

    # Verify both in DB with is_drill=True
    drills, total = db.get_alerts(is_drill=True)
    assert total == 3
    assert len(drills) == 3

    # Verify log sink file exists and has 3 records
    assert log_sink.exists()
    with open(log_sink, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 3


def test_dispatch_webhook() -> None:
    """Verify webhook notification dispatching and graceful failure handling."""
    alert = AlertEvent(
        metric_name="abstain_rate",
        severity="WARNING",
        current_value=0.45,
        threshold_value=0.35,
        condition=">",
        sample_size=100,
        message="Test alert",
        is_drill=True,
    )

    # 1. Success case (HTTP 200)
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        success = dispatch_webhook("https://webhook.site/mock-test", alert)
        assert success is True

    # 2. Unexpected status code (HTTP 500)
    mock_resp_err = MagicMock()
    mock_resp_err.getcode.return_value = 500
    mock_resp_err.__enter__.return_value = mock_resp_err

    with patch("urllib.request.urlopen", return_value=mock_resp_err):
        failure_status = dispatch_webhook("https://webhook.site/mock-test", alert)
        assert failure_status is False

    # 3. Connection error case
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        failure_exc = dispatch_webhook("https://webhook.site/mock-test", alert)
        assert failure_exc is False


def test_collector_api_alerts_endpoints(client: TestClient) -> None:
    """Verify REST API endpoints for checking, drilling, and querying alerts."""
    auth_headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}

    # 1. Unauthorized access should be rejected
    res_unauth = client.post("/v1/alerts/check")
    assert res_unauth.status_code in (401, 403)

    # 2. Check alerts (empty DB -> INSUFFICIENT_DATA)
    res_check = client.post("/v1/alerts/check", headers=auth_headers)
    assert res_check.status_code == 200
    data_check = res_check.json()
    assert data_check["status"] == "INSUFFICIENT_DATA"

    # 3. Execute alert drill
    res_drill = client.post(
        "/v1/alerts/drill?metric=abstain_rate",
        headers=auth_headers,
    )
    assert res_drill.status_code == 200
    drill_data = res_drill.json()
    assert drill_data["is_drill"] is True
    assert drill_data["metric_name"] == "abstain_rate"

    # 4. Query alerts
    res_list = client.get("/v1/alerts", headers=auth_headers)
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert list_data["total"] >= 1
    assert len(list_data["items"]) >= 1

    # 5. Query active alerts
    res_active = client.get("/v1/alerts/active?lookback_minutes=60", headers=auth_headers)
    assert res_active.status_code == 200
    active_data = res_active.json()
    assert active_data["count"] >= 1


def test_alerts_cli_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify alerts CLI entrypoint with check, drill, and list actions."""
    db_file = tmp_path / "cli_alerts.db"
    log_file = tmp_path / "cli_alerts.jsonl"

    # 1. CLI drill
    monkeypatch.setattr(
        "sys.argv",
        [
            "alerts.py",
            "--action",
            "drill",
            "--metric",
            "abstain_rate",
            "--db-path",
            str(db_file),
            "--alerts-log",
            str(log_file),
        ],
    )
    main()
    assert db_file.exists()
    assert log_file.exists()

    # 2. CLI list
    monkeypatch.setattr(
        "sys.argv",
        [
            "alerts.py",
            "--action",
            "list",
            "--db-path",
            str(db_file),
        ],
    )
    main()

    # 3. CLI check
    monkeypatch.setattr(
        "sys.argv",
        [
            "alerts.py",
            "--action",
            "check",
            "--db-path",
            str(db_file),
            "--alerts-log",
            str(log_file),
        ],
    )
    main()
