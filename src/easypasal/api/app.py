"""Application factory and CLI server entrypoint for EasyPasal API."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from easypasal.api.middleware import RequestIDMiddleware
from easypasal.api.routes import api_router
from easypasal.generation.pipeline import RAGPipeline

if TYPE_CHECKING:
    from easypasal.monitoring.logger import RequestLogger

logger = logging.getLogger("easypasal.api.app")


def create_app(
    pipeline: RAGPipeline | None = None,
    title: str = "EasyPasal API",
    version: str = "0.1.0",
    is_test: bool = False,
    config_path: Path | str | None = None,
    variant: str | None = None,
    traffic_split: bool | None = None,
    request_logger: RequestLogger | None = None,
    provider: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        pipeline: Optional pre-configured RAGPipeline instance (useful for mocking/testing).
        title: API title metadata.
        version: API version string.
        is_test: Flag indicating automated test environment.
        config_path: Optional path to YAML configuration file.
        variant: Optional default variant override ('baseline_v0.1' or 'gepa_v0.2').
        traffic_split: Optional boolean flag to enable or disable traffic splitting.
        request_logger: Optional pre-configured RequestLogger instance.
        provider: Optional LLM backend provider override ('auto', 'mock', 'llamacpp', etc.).

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if getattr(app.state, "pipeline", None) is None and not getattr(
            app.state, "is_test", False
        ):
            try:
                # If variant is specified and config_path not explicitly provided, map to variant yaml
                eff_config_path = config_path
                if variant and not eff_config_path:
                    cand_path = Path(f"configs/{variant}.yaml")
                    if cand_path.exists():
                        eff_config_path = cand_path

                eff_provider: str = provider or os.getenv("EASYPASAL_LLM_PROVIDER") or "auto"
                logger.info(
                    "Initializing default on-disk EasyPasal RAGPipeline on startup (provider=%s)...",
                    eff_provider,
                )
                app.state.pipeline = RAGPipeline.from_defaults(
                    config_path=eff_config_path, provider=eff_provider
                )
                logger.info("EasyPasal RAGPipeline loaded and ready.")
            except Exception as exc:
                logger.warning(
                    "Default RAGPipeline initialization deferred or failed: %s. "
                    "Endpoints requiring inference may return 503 until configured.",
                    exc,
                )

        # Set or override active variant and traffic split state
        pip = getattr(app.state, "pipeline", None)
        pip_cfg = getattr(pip, "config", None) if pip else None
        active_var = variant
        if (
            not active_var
            and pip_cfg is not None
            and isinstance(getattr(pip_cfg, "variant_id", None), str)
        ):
            active_var = pip_cfg.variant_id
        app.state.active_variant = active_var if isinstance(active_var, str) else "baseline_v0.1"

        traffic_cfg = None
        if pip_cfg is not None and hasattr(pip_cfg, "traffic_split"):
            traffic_cfg = pip_cfg.traffic_split.model_copy()
        if traffic_cfg is not None and traffic_split is not None:
            traffic_cfg.enabled = traffic_split
        app.state.traffic_split_config = traffic_cfg

        yield

    app = FastAPI(
        title=title,
        version=version,
        description=(
            "EasyPasal: Citation-grounded RAG API for Indonesian Financial Regulations "
            "(Otoritas Jasa Keuangan & Bank Indonesia)."
        ),
        lifespan=lifespan,
    )

    # State configuration
    app.state.pipeline = pipeline
    app.state.is_test = is_test
    pip_cfg = getattr(pipeline, "config", None) if pipeline else None
    if isinstance(variant, str):
        app.state.active_variant = variant
    elif pip_cfg and isinstance(getattr(pip_cfg, "variant_id", None), str):
        app.state.active_variant = pip_cfg.variant_id
    else:
        app.state.active_variant = "baseline_v0.1"

    if pip_cfg and hasattr(pip_cfg, "traffic_split"):
        cfg = pip_cfg.traffic_split.model_copy()
        if traffic_split is not None:
            cfg.enabled = traffic_split
        app.state.traffic_split_config = cfg
    else:
        app.state.traffic_split_config = None

    # RequestLogger configuration (Plan §9 / Task T7.1)
    if request_logger is not None:
        app.state.request_logger = request_logger
    else:
        monitoring_cfg = getattr(pip_cfg, "monitoring", None) if pip_cfg else None
        if monitoring_cfg is None:
            from easypasal.config import MonitoringConfig

            monitoring_cfg = MonitoringConfig()

        from easypasal.monitoring.logger import RequestLogger

        active_log_file = None if is_test else monitoring_cfg.log_file
        active_fb_file = None if is_test else monitoring_cfg.feedback_file

        app.state.request_logger = RequestLogger(
            log_file=active_log_file if monitoring_cfg.enabled else None,
            feedback_file=active_fb_file if monitoring_cfg.enabled else None,
            redact_pii=monitoring_cfg.redact_pii,
            salt=monitoring_cfg.salt,
            collector_url=monitoring_cfg.collector_url,
            collector_token=monitoring_cfg.collector_token,
        )

    # Middleware registration (executed in reverse order of addition)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Route inclusion
    app.include_router(api_router)

    return app


def main() -> None:
    """CLI entrypoint for running the EasyPasal REST server."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the EasyPasal FastAPI server.")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for local development"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "mock", "ollama", "llamacpp"],
        default=os.getenv("EASYPASAL_LLM_PROVIDER", "auto"),
        help="LLM provider for RAGPipeline (default: auto or EASYPASAL_LLM_PROVIDER)",
    )
    parser.add_argument(
        "--variant",
        choices=["baseline_v0.1", "gepa_v0.2", "A", "B", "baseline", "gepa"],
        default=None,
        help="Active default pipeline variant (default: from config)",
    )
    parser.add_argument(
        "--traffic-split",
        action="store_true",
        help="Enable deterministic traffic splitting across variants",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Server log level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Starting EasyPasal REST API server on %s:%d (provider=%s, variant=%s, traffic_split=%s, config=%s)",
        args.host,
        args.port,
        args.provider,
        args.variant,
        args.traffic_split,
        args.config,
    )

    # If provider or config is explicitly specified other than auto, initialize upfront
    configured_pipeline: RAGPipeline | None = None
    if args.provider != "auto" or args.config is not None or args.variant is not None:
        eff_cfg = args.config
        if not eff_cfg and args.variant:
            cand = Path(f"configs/{args.variant}.yaml")
            if cand.exists():
                eff_cfg = cand
        try:
            configured_pipeline = RAGPipeline.from_defaults(
                provider=args.provider,
                config_path=eff_cfg,
            )
        except Exception as exc:
            logger.warning(
                "Could not initialize on-disk index with provider %s: %s", args.provider, exc
            )

    app_instance = create_app(
        pipeline=configured_pipeline,
        config_path=args.config,
        variant=args.variant,
        traffic_split=True if args.traffic_split else None,
        provider=args.provider,
    )

    uvicorn.run(
        app_instance,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
