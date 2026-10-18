"""Unit and integration tests for the VPS Log Collector Dashboard (Plan §9 / Task T7.3).

Validates HTML dashboard rendering, Chart.js canvas elements, KPI metrics,
root URL redirect, and automated screenshot generation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easypasal.collector.app import create_collector_app
from easypasal.collector.config import CollectorConfig
from easypasal.collector.dashboard import get_dashboard_html
from easypasal.collector.db import CollectorDB


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary SQLite database path."""
    return tmp_path / "dashboard_test.db"


@pytest.fixture
def collector_config(temp_db_path: Path) -> CollectorConfig:
    """Fixture returning a test collector configuration."""
    return CollectorConfig(
        host="127.0.0.1",
        port=8080,
        db_path=temp_db_path,
        auth_token="test-dashboard-secret",
    )


@pytest.fixture
def client(collector_config: CollectorConfig) -> TestClient:
    """Fixture providing a TestClient configured with dashboard router."""
    app = create_collector_app(config=collector_config, is_test=True)
    return TestClient(app)


def test_get_dashboard_html_structure() -> None:
    """Verify that dashboard HTML contains required KPI cards, canvas elements, and scripts."""
    html = get_dashboard_html()
    assert "<!DOCTYPE html>" in html
    assert "EasyPasal Log Collector Dashboard" in html
    assert 'id="latencyChart"' in html
    assert 'id="variantChart"' in html
    assert 'id="retrievalChart"' in html
    assert 'id="themeChart"' in html
    assert 'id="kpiTotalRequests"' in html
    assert 'id="kpiAbstainRate"' in html
    assert 'id="kpiP50Latency"' in html
    assert 'id="kpiP95Latency"' in html
    assert 'id="kpiMemoryRss"' in html
    assert 'id="alertBanner"' in html
    assert 'id="alertDrillBtn"' in html
    assert "chart.umd.min.js" in html


def test_dashboard_endpoint(client: TestClient) -> None:
    """Test GET /dashboard returns HTTP 200 with text/html content-type."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "EasyPasal" in res.text


def test_root_redirect_to_dashboard(client: TestClient) -> None:
    """Test that accessing root / redirects to /dashboard."""
    res = client.get("/", follow_redirects=False)
    assert res.status_code in (302, 307)
    assert res.headers["location"] == "/dashboard"


def test_generate_dashboard_preview_script(tmp_path: Path) -> None:
    """Test that scripts/generate_dashboard_preview.py creates valid image artifacts."""
    out_dir = tmp_path / "images"
    db_file = tmp_path / "test_preview.db"

    # Initialize empty DB
    _ = CollectorDB(db_path=db_file)

    cmd = [
        sys.executable,
        "scripts/generate_dashboard_preview.py",
        "--db-path",
        str(db_file),
        "--output-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    assert "Generated dashboard infographic" in proc.stdout

    img_path = out_dir / "dashboard_monitoring.png"
    assert img_path.exists()
    assert img_path.stat().st_size > 10000  # Non-trivial image size
