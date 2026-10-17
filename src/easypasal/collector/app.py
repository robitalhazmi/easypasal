"""Application factory and CLI server entrypoint for VPS Log Collector (Plan §9 / Task T7.2)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from easypasal.collector.config import CollectorConfig
from easypasal.collector.dashboard import dashboard_router
from easypasal.collector.db import CollectorDB
from easypasal.collector.routes import router

logger = logging.getLogger("easypasal.collector.app")


def create_collector_app(
    config: CollectorConfig | None = None,
    is_test: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI Log Collector service instance.

    Args:
        config: Optional configuration instance. Defaults to environment/CollectorConfig.
        is_test: Flag indicating automated test environment.

    Returns:
        Configured FastAPI application instance.
    """
    effective_config = config or CollectorConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup: Ensure SQLite DB and tables are initialized
        if getattr(app.state, "db", None) is None:
            app.state.db = CollectorDB(
                db_path=effective_config.db_path,
                enable_wal=effective_config.enable_wal,
            )
            logger.info(
                "Collector SQLite storage loaded at %s (WAL=%s)",
                effective_config.db_path,
                effective_config.enable_wal,
            )
        yield
        # Shutdown
        logger.info("Collector service shutting down.")

    app = FastAPI(
        title="EasyPasal Log Collector",
        version="0.1.0",
        description="Lightweight SQLite log ingestion and telemetry microservice for VPS deployment.",
        lifespan=lifespan,
    )

    # Attach configuration and test state
    app.state.collector_config = effective_config
    app.state.is_test = is_test
    if getattr(app.state, "db", None) is None:
        app.state.db = CollectorDB(
            db_path=effective_config.db_path,
            enable_wal=effective_config.enable_wal,
        )

    # Enable CORS for dashboards / Streamlit integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API endpoints
    app.include_router(router)
    app.include_router(dashboard_router)

    @app.get("/", include_in_schema=False)
    def root_redirect() -> Any:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/dashboard")

    return app


def main() -> None:
    """CLI entrypoint for running the VPS Log Collector daemon."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Run EasyPasal VPS Log Collector server.")
    parser.add_argument("--host", default=None, help="Binding host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: 8080)")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to SQLite database file (default: data/collector/collector.db)",
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Shared bearer authentication secret",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Collector logging verbosity level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = CollectorConfig()
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.db_path:
        cfg.db_path = args.db_path
    if args.auth_token:
        cfg.auth_token = args.auth_token

    logger.info(
        "Starting EasyPasal Log Collector on %s:%d (db_path=%s)",
        cfg.host,
        cfg.port,
        cfg.db_path,
    )

    app_instance = create_collector_app(config=cfg)

    uvicorn.run(
        app_instance,
        host=cfg.host,
        port=cfg.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
