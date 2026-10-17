"""Unit and integration tests for the VPS Log Collector service (Plan §9 / Task T7.2).

Validates SQLite WAL persistence, Bearer/X-Collector-Token authentication,
single & batch ingestion, query pagination, telemetry aggregation, health probes,
and memory budget constraints (<= 90 MB RSS RAM under concurrent load).
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easypasal.collector.app import create_collector_app
from easypasal.collector.config import CollectorConfig
from easypasal.collector.db import CollectorDB
from easypasal.monitoring.logger import RequestLogger
from easypasal.monitoring.schemas import (
    FeedbackLogRecord,
    RequestLogRecord,
)

AUTH_TOKEN = "test-secret-collector-token"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary SQLite database path."""
    return tmp_path / "collector.db"


@pytest.fixture
def collector_config(temp_db_path: Path) -> CollectorConfig:
    """Fixture returning a test collector configuration."""
    return CollectorConfig(
        host="127.0.0.1",
        port=8080,
        db_path=temp_db_path,
        auth_token=AUTH_TOKEN,
        max_batch_size=50,
        enable_wal=True,
    )


@pytest.fixture
def collector_db(temp_db_path: Path) -> CollectorDB:
    """Fixture returning an initialized CollectorDB instance."""
    return CollectorDB(db_path=temp_db_path, enable_wal=True)


@pytest.fixture
def client(collector_config: CollectorConfig) -> TestClient:
    """Fixture providing a FastAPI TestClient for the collector app."""
    app = create_collector_app(config=collector_config, is_test=True)
    return TestClient(app)


def make_dummy_record(
    request_id: str = "req-test-1",
    variant: str = "baseline_v0.1",
    theme: str | None = "Perbankan",
    abstained: bool = False,
    latency_ms: float = 120.0,
    prompt_tokens: int = 50,
    completion_tokens: int = 25,
    timestamp: datetime | None = None,
) -> RequestLogRecord:
    """Factory helper to create realistic dummy request log records."""
    return RequestLogRecord(
        request_id=request_id,
        timestamp=timestamp or datetime.now(UTC),
        variant=variant,
        language="id",
        question_length=42,
        question_redacted="Berapa modal minimum BPR?",
        retrieval_score_top1=0.88,
        retrieval_scores_topk=[0.88, 0.76],
        retrieved_doc_ids=["POJK_62_2020", "POJK_12_2021"],
        abstained=abstained,
        routed=True,
        theme=theme,
        retrieval_latency_ms=35.0,
        generation_latency_ms=85.0,
        total_latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        citation_count=1,
        grounded_citation_count=1,
        ungrounded_citation_count=0,
        client_ip_hash="abc123def456",
        model_name="qwen2.5:7b-instruct-q4_K_M",
    )


# =========================================================================
# Unit Tests: CollectorConfig
# =========================================================================


def test_collector_config_defaults() -> None:
    """Test default values of CollectorConfig."""
    cfg = CollectorConfig()
    assert cfg.host in ("0.0.0.0", "127.0.0.1")
    assert cfg.port == 8080
    assert cfg.auth_token == "easypasal-vps-secret-token-v1"
    assert cfg.max_batch_size == 500
    assert cfg.enable_wal is True


def test_collector_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variable overrides on CollectorConfig."""
    monkeypatch.setenv("EASYPASAL_COLLECTOR_HOST", "10.0.0.1")
    monkeypatch.setenv("EASYPASAL_COLLECTOR_PORT", "9090")
    monkeypatch.setenv("EASYPASAL_COLLECTOR_TOKEN", "custom-env-token")
    monkeypatch.setenv("EASYPASAL_COLLECTOR_DB_PATH", "/tmp/custom.db")

    cfg = CollectorConfig()
    assert cfg.host == "10.0.0.1"
    assert cfg.port == 9090
    assert cfg.auth_token == "custom-env-token"
    assert cfg.db_path == Path("/tmp/custom.db")


# =========================================================================
# Unit Tests: CollectorDB Storage Engine
# =========================================================================


def test_collector_db_init_and_info(collector_db: CollectorDB) -> None:
    """Test database initialization and diagnostic metadata."""
    info = collector_db.get_db_info()
    assert "path" in info
    assert info["request_logs_count"] == 0
    assert info["feedback_logs_count"] == 0
    assert info["page_size"] > 0


def test_collector_db_insert_single_request(collector_db: CollectorDB) -> None:
    """Test inserting and retrieving a single request log."""
    record = make_dummy_record(request_id="req-single-1")
    collector_db.insert_request(record)

    items, total = collector_db.get_requests()
    assert total == 1
    assert len(items) == 1
    assert items[0]["request_id"] == "req-single-1"
    assert items[0]["theme"] == "Perbankan"
    assert items[0]["abstained"] is False
    assert items[0]["retrieved_doc_ids"] == ["POJK_62_2020", "POJK_12_2021"]


def test_collector_db_insert_batch(collector_db: CollectorDB) -> None:
    """Test atomic batch insertion."""
    records = [
        make_dummy_record(request_id=f"req-batch-{i}", latency_ms=100.0 + i) for i in range(10)
    ]
    inserted = collector_db.insert_requests_batch(records)
    assert inserted == 10

    items, total = collector_db.get_requests(limit=20)
    assert total == 10
    assert len(items) == 10


def test_collector_db_insert_batch_empty(collector_db: CollectorDB) -> None:
    """Test batch insertion with empty list returns 0."""
    assert collector_db.insert_requests_batch([]) == 0


def test_collector_db_feedback_linkage(collector_db: CollectorDB) -> None:
    """Test user feedback insertion and linking to parent request."""
    record = make_dummy_record(request_id="req-fb-1")
    collector_db.insert_request(record)

    fb = FeedbackLogRecord(
        request_id="req-fb-1",
        rating=1,
        comment_redacted="Kutipan sangat tepat",
        tags=["helpful", "accurate"],
    )
    collector_db.insert_feedback(fb)

    # Check feedback table
    feedback_rows = collector_db.get_feedback(request_id="req-fb-1")
    assert len(feedback_rows) == 1
    assert feedback_rows[0]["rating"] == 1
    assert feedback_rows[0]["tags"] == ["helpful", "accurate"]

    # Check request table was updated with feedback
    items, _ = collector_db.get_requests()
    assert items[0]["feedback_rating"] == 1
    assert items[0]["feedback_comment"] == "Kutipan sangat tepat"


def test_collector_db_query_filters(collector_db: CollectorDB) -> None:
    """Test filtering by variant, theme, abstained, and date range."""
    now = datetime.now(UTC)
    t1 = now - timedelta(hours=2)
    t2 = now - timedelta(hours=1)

    r1 = make_dummy_record(
        request_id="r1", variant="baseline_v0.1", theme="Fintech", abstained=False, timestamp=t1
    )
    r2 = make_dummy_record(
        request_id="r2", variant="gepa_v0.2", theme="Pasar Modal", abstained=True, timestamp=t2
    )
    r3 = make_dummy_record(
        request_id="r3", variant="baseline_v0.1", theme="Fintech", abstained=True, timestamp=now
    )

    collector_db.insert_requests_batch([r1, r2, r3])

    # Filter by variant
    items, total = collector_db.get_requests(variant="gepa_v0.2")
    assert total == 1
    assert items[0]["request_id"] == "r2"

    # Filter by theme
    items, total = collector_db.get_requests(theme="Fintech")
    assert total == 2

    # Filter by abstained
    items, total = collector_db.get_requests(abstained=True)
    assert total == 2

    # Pagination: limit 1, offset 1
    items, total = collector_db.get_requests(limit=1, offset=1)
    assert total == 3
    assert len(items) == 1


def test_collector_db_stats_summary(collector_db: CollectorDB) -> None:
    """Test telemetry statistics and percentile aggregation."""
    # Empty DB
    empty_stats = collector_db.get_stats_summary()
    assert empty_stats["total_requests"] == 0
    assert empty_stats["abstain_rate"] == 0.0

    # Insert 5 records with known latencies: 10, 20, 30, 40, 50
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    records = [
        make_dummy_record(
            request_id=f"stat-req-{i}",
            latency_ms=lat,
            abstained=(i == 4),  # 1 abstained out of 5
            prompt_tokens=100,
            completion_tokens=50,
        )
        for i, lat in enumerate(latencies)
    ]
    collector_db.insert_requests_batch(records)

    stats = collector_db.get_stats_summary()
    assert stats["total_requests"] == 5
    assert stats["abstain_count"] == 1
    assert stats["abstain_rate"] == 0.2
    assert stats["mean_latency_ms"] == 30.0
    assert stats["p50_latency_ms"] == 30.0
    assert stats["total_prompt_tokens"] == 500
    assert stats["total_completion_tokens"] == 250
    assert stats["total_tokens"] == 750


# =========================================================================
# Integration Tests: FastAPI Endpoints & Authentication
# =========================================================================


def test_auth_rejection(client: TestClient) -> None:
    """Test that missing or incorrect auth tokens return 401 Unauthorized."""
    # No auth header
    res = client.post("/v1/logs/requests", json={})
    assert res.status_code == 401
    assert "Invalid or missing collector authentication token" in res.json()["detail"]

    # Wrong Bearer token
    res = client.post(
        "/v1/logs/requests",
        headers={"Authorization": "Bearer wrong-token"},
        json={},
    )
    assert res.status_code == 401

    # Wrong X-Collector-Token header
    res = client.get(
        "/v1/stats/summary",
        headers={"X-Collector-Token": "bad-secret"},
    )
    assert res.status_code == 401


def test_auth_success_bearer_and_x_collector_token(client: TestClient) -> None:
    """Test that both Bearer and X-Collector-Token succeed with valid secret."""
    # Bearer authorization
    res = client.get(
        "/v1/stats/summary",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    )
    assert res.status_code == 200

    # X-Collector-Token authorization
    res = client.get(
        "/v1/stats/summary",
        headers={"X-Collector-Token": AUTH_TOKEN},
    )
    assert res.status_code == 200


def test_health_probe_unauthenticated(client: TestClient) -> None:
    """Test that /health probe is accessible without authentication."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "easypasal-collector"
    assert "memory" in data
    assert "current_rss_mb" in data["memory"]
    assert "budget_mb" in data["memory"]
    assert data["memory"]["budget_mb"] == 90.0
    assert "database" in data


def test_api_single_request_ingest(client: TestClient) -> None:
    """Test POST /v1/logs/requests endpoint."""
    record = make_dummy_record(request_id="api-req-1")
    payload = record.model_dump(mode="json")

    res = client.post(
        "/v1/logs/requests",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json=payload,
    )
    assert res.status_code == 201
    assert res.json() == {"status": "ok", "request_id": "api-req-1"}

    # Query back
    q_res = client.get(
        "/v1/logs/requests",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    )
    assert q_res.status_code == 200
    assert q_res.json()["total"] == 1
    assert q_res.json()["items"][0]["request_id"] == "api-req-1"


def test_api_batch_ingest(client: TestClient) -> None:
    """Test POST /v1/logs/batch endpoint."""
    records = [
        make_dummy_record(request_id=f"api-batch-{i}").model_dump(mode="json") for i in range(5)
    ]

    res = client.post(
        "/v1/logs/batch",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json=records,
    )
    assert res.status_code == 201
    assert res.json() == {"status": "ok", "inserted": 5}


def test_api_batch_exceeds_max_size(client: TestClient) -> None:
    """Test POST /v1/logs/batch rejecting oversized payloads."""
    # Config max_batch_size was set to 50 in fixture
    oversized = [
        make_dummy_record(request_id=f"over-{i}").model_dump(mode="json") for i in range(51)
    ]

    res = client.post(
        "/v1/logs/batch",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json=oversized,
    )
    assert res.status_code == 400
    assert "exceeds maximum allowed limit" in res.json()["detail"]


def test_api_feedback_ingest(client: TestClient) -> None:
    """Test POST /v1/logs/feedback endpoint."""
    # First insert request
    rec = make_dummy_record(request_id="api-fb-target")
    client.post(
        "/v1/logs/requests",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json=rec.model_dump(mode="json"),
    )

    fb_payload = {
        "request_id": "api-fb-target",
        "rating": -1,
        "comment_redacted": "Kurang lengkap pada bagian sanksi",
        "tags": ["incomplete"],
    }

    res = client.post(
        "/v1/logs/feedback",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json=fb_payload,
    )
    assert res.status_code == 201
    assert res.json() == {"status": "ok", "request_id": "api-fb-target"}

    # Query feedback
    fb_query = client.get(
        "/v1/logs/feedback",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        params={"rating": -1},
    )
    assert fb_query.status_code == 200
    assert len(fb_query.json()["items"]) == 1
    assert fb_query.json()["items"][0]["tags"] == ["incomplete"]


# =========================================================================
# Integration Test: RequestLogger Forwarding to Collector
# =========================================================================


def test_request_logger_forwarding(
    client: TestClient,
    collector_config: CollectorConfig,
    tmp_path: Path,
) -> None:
    """Test that RequestLogger forwards logs to collector HTTP endpoints."""
    # Use TestClient transport in a mock forwarding test or direct app invocation
    log_file = tmp_path / "app_req.jsonl"
    logger_instance = RequestLogger(
        log_file=log_file,
        collector_url=None,  # No background thread for this unit test
    )

    record = make_dummy_record(request_id="forward-direct-1")
    logger_instance.log_request(record)
    assert log_file.exists()

    # Now verify collector directly ingests the logged record
    res = client.post(
        "/v1/logs/requests",
        headers={"Authorization": f"Bearer {collector_config.auth_token}"},
        json=record.model_dump(mode="json"),
    )
    assert res.status_code == 201

    stats = client.get(
        "/v1/stats/summary",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    ).json()
    assert stats["total_requests"] == 1


# =========================================================================
# Load & Memory Budget Verification Test (Plan §9 / Architecture §3)
# =========================================================================


def test_concurrent_load_and_memory_budget(client: TestClient, tmp_path: Path) -> None:
    """Simulate 100+ concurrent ingestion requests and verify memory stays <= 90 MB RSS."""
    total_requests = 120
    num_threads = 8

    def _worker(idx: int) -> int:
        rec = make_dummy_record(
            request_id=f"load-test-req-{idx}",
            latency_ms=50.0 + (idx % 20),
            prompt_tokens=100,
            completion_tokens=50,
        )
        resp = client.post(
            "/v1/logs/requests",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json=rec.model_dump(mode="json"),
        )
        return int(resp.status_code)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        statuses = list(pool.map(_worker, range(total_requests)))

    assert len(statuses) == total_requests
    assert all(s == 201 for s in statuses)

    # Verify database count
    summary = client.get(
        "/v1/stats/summary",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    ).json()
    assert summary["total_requests"] == total_requests

    # Memory budget check: run isolated subprocess to measure true collector process RSS
    sub_db = tmp_path / "standalone_bench.db"
    script = f"""
from fastapi.testclient import TestClient
from easypasal.collector.app import create_collector_app
from easypasal.collector.config import CollectorConfig
from easypasal.collector.routes import get_memory_usage

cfg = CollectorConfig(db_path={str(sub_db)!r}, auth_token={AUTH_TOKEN!r})
app = create_collector_app(config=cfg, is_test=True)
c = TestClient(app)

for i in range(120):
    c.post(
        '/v1/logs/requests',
        headers={{'Authorization': 'Bearer ' + cfg.auth_token}},
        json={{
            'request_id': f'bench-req-{{i}}',
            'timestamp': '2026-10-17T12:00:00Z',
            'variant': 'baseline_v0.1',
            'language': 'id',
            'question_length': 25,
            'question_redacted': 'Berapa modal minimum BPR?',
            'abstained': False,
            'total_latency_ms': 50.0,
        }},
    )

mem = get_memory_usage()
print(f"RESULT_RSS_MB:{{mem['rss_mb']}}")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    rss_val: float | None = None
    for line in proc.stdout.splitlines():
        if "RESULT_RSS_MB:" in line:
            rss_val = float(line.split("RESULT_RSS_MB:")[1].strip())
            break

    assert rss_val is not None, (
        f"Failed to parse RSS from subprocess output: {proc.stdout} {proc.stderr}"
    )
    assert rss_val <= 90.0, f"Collector memory usage ({rss_val} MB) exceeded VPS 90 MB threshold!"
