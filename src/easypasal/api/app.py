"""Application factory and CLI server entrypoint for EasyPasal API."""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from easypasal.api.middleware import RequestIDMiddleware
from easypasal.api.routes import api_router
from easypasal.generation.pipeline import RAGPipeline

logger = logging.getLogger("easypasal.api.app")


def create_app(
    pipeline: RAGPipeline | None = None,
    title: str = "EasyPasal API",
    version: str = "0.1.0",
    is_test: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        pipeline: Optional pre-configured RAGPipeline instance (useful for mocking/testing).
        title: API title metadata.
        version: API version string.
        is_test: Flag indicating automated test environment.

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if getattr(app.state, "pipeline", None) is None and not getattr(
            app.state, "is_test", False
        ):
            try:
                logger.info("Initializing default on-disk EasyPasal RAGPipeline on startup...")
                app.state.pipeline = RAGPipeline.from_defaults()
                logger.info("EasyPasal RAGPipeline loaded and ready.")
            except Exception as exc:
                logger.warning(
                    "Default RAGPipeline initialization deferred or failed: %s. "
                    "Endpoints requiring inference may return 503 until configured.",
                    exc,
                )
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
        "--provider",
        choices=["auto", "mock", "ollama", "llamacpp"],
        default="auto",
        help="LLM provider for RAGPipeline (default: auto)",
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
        "Starting EasyPasal REST API server on %s:%d (provider=%s)",
        args.host,
        args.port,
        args.provider,
    )

    # If provider is explicitly specified other than auto, initialize upfront
    configured_pipeline: RAGPipeline | None = None
    if args.provider != "auto":
        try:
            configured_pipeline = RAGPipeline.from_defaults(provider=args.provider)
        except Exception as exc:
            logger.warning(
                "Could not initialize on-disk index with provider %s: %s", args.provider, exc
            )

    app_instance = create_app(pipeline=configured_pipeline)

    uvicorn.run(
        app_instance,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
