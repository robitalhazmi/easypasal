"""Unit and integration tests for EasyPasal Docker & Docker Compose setup (Plan §10 / Task T7.7)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
ROOT_DOCKERFILE = REPO_ROOT / "Dockerfile"
COLLECTOR_DOCKERFILE = REPO_ROOT / "docker" / "collector" / "Dockerfile"
SPACE_DOCKERFILE = REPO_ROOT / "docker" / "space" / "Dockerfile"


def test_compose_file_exists_and_parses() -> None:
    """Verify docker-compose.yml exists and parses as valid YAML."""
    assert COMPOSE_FILE.exists(), f"Missing docker-compose.yml at {COMPOSE_FILE}"
    content = COMPOSE_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    assert isinstance(data, dict), "docker-compose.yml must parse to a dictionary"
    assert "services" in data, "docker-compose.yml must define 'services'"
    assert "volumes" in data, "docker-compose.yml must define 'volumes'"


def test_compose_services_declared() -> None:
    """Verify all required production services (collector, app, ui) are configured."""
    data: dict[str, Any] = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    services = data["services"]

    assert "collector" in services, "Service 'collector' must be defined"
    assert "app" in services, "Service 'app' must be defined"
    assert "ui" in services, "Service 'ui' must be defined"


def test_compose_service_ports_and_networks() -> None:
    """Verify standard port exposures across services."""
    data: dict[str, Any] = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    services = data["services"]

    # Collector on port 8080
    assert "8080:8080" in services["collector"]["ports"]
    # REST API on port 8000
    assert "8000:8000" in services["app"]["ports"]
    # Gradio Web UI on port 7860
    assert "7860:7860" in services["ui"]["ports"]


def test_compose_service_dependencies_and_healthchecks() -> None:
    """Verify healthchecks and inter-service dependencies (app -> collector, ui -> app & collector)."""
    data: dict[str, Any] = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    services = data["services"]

    # All three services must define healthchecks
    for name in ("collector", "app", "ui"):
        assert "healthcheck" in services[name], f"Service '{name}' must have a healthcheck"
        assert "test" in services[name]["healthcheck"], f"Service '{name}' healthcheck needs a test"

    # app depends on collector being healthy
    app_deps = services["app"].get("depends_on", {})
    assert "collector" in app_deps
    assert app_deps["collector"].get("condition") == "service_healthy"

    # ui depends on app and collector being healthy
    ui_deps = services["ui"].get("depends_on", {})
    assert "app" in ui_deps
    assert ui_deps["app"].get("condition") == "service_healthy"
    assert "collector" in ui_deps
    assert ui_deps["collector"].get("condition") == "service_healthy"


def test_compose_volume_persisted() -> None:
    """Verify persistent SQLite data volume and model cache volume are declared."""
    data: dict[str, Any] = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert "collector_data" in data.get("volumes", {})
    assert "model_cache" in data.get("volumes", {})
    collector_vols = data["services"]["collector"].get("volumes", [])
    assert any("collector_data:/data" in v for v in collector_vols)
    app_vols = data["services"]["app"].get("volumes", [])
    assert any("model_cache:/home/user/.cache" in v for v in app_vols)


def test_root_dockerfile_structure() -> None:
    """Verify root Dockerfile contains necessary multi-stage build, indices, and security directives."""
    assert ROOT_DOCKERFILE.exists(), f"Missing root Dockerfile at {ROOT_DOCKERFILE}"
    text = ROOT_DOCKERFILE.read_text(encoding="utf-8")

    # Multi-stage build
    assert "AS builder" in text
    assert "AS runner" in text
    assert "python:3.12-slim" in text

    # Non-root user
    assert "useradd" in text
    assert "USER user" in text

    # Embedded index & assets
    assert "COPY --chown=user:user data/index/ /app/data/index/" in text
    assert "COPY --chown=user:user configs/ /app/configs/" in text
    assert "COPY --chown=user:user data/corpus_manifest.csv /app/data/corpus_manifest.csv" in text

    # Port exposure & healthcheck
    assert "EXPOSE 8000 7860" in text
    assert "HEALTHCHECK" in text


def test_collector_dockerfile_structure() -> None:
    """Verify docker/collector/Dockerfile security and volume directives."""
    assert COLLECTOR_DOCKERFILE.exists(), f"Missing collector Dockerfile at {COLLECTOR_DOCKERFILE}"
    text = COLLECTOR_DOCKERFILE.read_text(encoding="utf-8")

    assert "AS builder" in text
    assert "AS runner" in text
    assert 'VOLUME ["/data"]' in text
    assert "EXPOSE 8080" in text
    assert "HEALTHCHECK" in text
    assert "easypasal.collector.app" in text


def test_docker_compose_config_cli() -> None:
    """Execute 'docker compose config' if docker CLI is available to validate full specification."""
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is not installed in the current environment")

    res = subprocess.run(
        ["docker", "compose", "config"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 and "Cannot connect to the Docker daemon" in res.stderr:
        pytest.skip("Docker daemon not running; skipping live CLI validation")

    assert res.returncode == 0, f"docker compose config failed:\n{res.stderr}"
    assert "easypasal-app" in res.stdout
    assert "easypasal-collector" in res.stdout
    assert "easypasal-ui" in res.stdout
