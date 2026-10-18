"""Configuration schema for the VPS Log Collector service (Plan §9 / Task T7.2)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class CollectorConfig(BaseModel):
    """Configuration settings for the lightweight VPS Log Collector microservice."""

    host: str = Field(
        default_factory=lambda: os.getenv("EASYPASAL_COLLECTOR_HOST", "0.0.0.0"),
        description="Network interface for the collector HTTP server.",
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("EASYPASAL_COLLECTOR_PORT", "8080")),
        description="Port for the collector HTTP server.",
    )
    db_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("EASYPASAL_COLLECTOR_DB_PATH", "data/collector/collector.db")
        ),
        description="Path to the SQLite database storage file.",
    )
    auth_token: str = Field(
        default_factory=lambda: os.getenv(
            "EASYPASAL_COLLECTOR_TOKEN", "easypasal-vps-secret-token-v1"
        ),
        description="Shared bearer token securing ingestion and query endpoints.",
    )
    max_batch_size: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum allowed request log entries in a single batch POST payload.",
    )
    enable_wal: bool = Field(
        default=True,
        description="Enable SQLite Write-Ahead Logging (WAL) for high concurrency and performance.",
    )
