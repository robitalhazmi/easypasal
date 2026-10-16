"""API routes for EasyPasal: health checks, legal Q&A, and root metadata."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from easypasal.api.schemas import (
    AskRequest,
    AskResponse,
    CitationVerificationResponse,
    ErrorResponse,
    HealthResponse,
    RetrievedChunkResponse,
)
from easypasal.api.traffic import VARIANT_BASELINE_V0_1, resolve_variant
from easypasal.generation.pipeline import RAGPipeline

logger = logging.getLogger("easypasal.api.routes")
api_router = APIRouter()


def get_pipeline(request: Request) -> RAGPipeline:
    """Dependency that resolves the initialized RAGPipeline from application state."""
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="EasyPasal RAG pipeline is not initialized or still starting up.",
        )
    return pipeline


@api_router.get("/", tags=["General"])
async def root() -> dict[str, Any]:
    """Root metadata providing service information and endpoint documentation."""
    return {
        "service": "EasyPasal RAG API",
        "version": "0.1.0",
        "status": "online",
        "description": "Evaluated, cited answers on Indonesian financial regulations.",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json",
        },
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
        },
    }


@api_router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health(request: Request) -> HealthResponse:
    """Readiness probe checking index availability and generation backend status."""
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    index_ready = False
    vector_count = 0
    bm25_count = 0
    provider_name: str | None = None

    if pipeline is not None:
        if pipeline.generator is not None:
            provider_name = getattr(
                pipeline.generator, "model_name", pipeline.generator.__class__.__name__
            )

        if pipeline.retriever is not None:
            index_ready = True
            try:
                retriever = pipeline.retriever
                if hasattr(retriever, "qdrant_store") and hasattr(retriever, "collection_name"):
                    vector_count = retriever.qdrant_store.count(retriever.collection_name)
            except Exception:
                pass

            if hasattr(pipeline.retriever, "bm25_index") and hasattr(
                pipeline.retriever.bm25_index, "corpus"
            ):
                bm25_count = len(pipeline.retriever.bm25_index.corpus)

    environment = "test" if getattr(request.app.state, "is_test", False) else "development"

    active_variant = getattr(request.app.state, "active_variant", None)
    if not isinstance(active_variant, str):
        pipe_cfg = getattr(pipeline, "config", None) if pipeline else None
        if pipe_cfg is not None and isinstance(getattr(pipe_cfg, "variant_id", None), str):
            active_variant = pipe_cfg.variant_id
        else:
            active_variant = VARIANT_BASELINE_V0_1

    traffic_cfg = getattr(request.app.state, "traffic_split_config", None)
    if not hasattr(traffic_cfg, "enabled") and pipeline is not None:
        pipe_cfg = getattr(pipeline, "config", None)
        if pipe_cfg is not None and hasattr(pipe_cfg, "traffic_split"):
            traffic_cfg = pipe_cfg.traffic_split
        else:
            traffic_cfg = None
    traffic_split_enabled = bool(getattr(traffic_cfg, "enabled", False))

    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=environment,
        index_ready=index_ready,
        vector_count=vector_count,
        bm25_doc_count=bm25_count,
        backend_provider=provider_name,
        active_variant=active_variant,
        traffic_split_enabled=traffic_split_enabled,
    )


@api_router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal pipeline error"},
        503: {"model": ErrorResponse, "description": "Pipeline service unavailable"},
    },
    tags=["Inference"],
)
async def ask(
    payload: AskRequest,
    request: Request,
    response: Response,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> AskResponse:
    """Answer a regulatory question with exact statutory citations and grounding verification."""
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Resolve variant via payload override > header override > traffic split > default
    header_variant = request.headers.get("X-EasyPasal-Variant") or request.headers.get("X-Variant")
    client_key = (
        request.headers.get("X-Forwarded-For")
        or (request.client.host if request.client else None)
        or payload.question
    )

    traffic_cfg = getattr(request.app.state, "traffic_split_config", None)
    if not hasattr(traffic_cfg, "enabled") and pipeline is not None:
        pipe_cfg = getattr(pipeline, "config", None)
        if pipe_cfg is not None and hasattr(pipe_cfg, "traffic_split"):
            traffic_cfg = pipe_cfg.traffic_split
        else:
            traffic_cfg = None

    try:
        resolved_variant = resolve_variant(
            payload_variant=payload.variant,
            header_variant=header_variant,
            client_key=client_key,
            traffic_config=traffic_cfg,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid variant configuration: {exc}",
        ) from exc

    # Set on request state for access log middleware and response header
    request.state.variant = resolved_variant
    response.headers["X-EasyPasal-Variant"] = resolved_variant

    try:
        output = pipeline.ask(
            question=payload.question,
            language=payload.language,
            top_k=payload.top_k,
            strict_grounding=payload.strict_grounding,
            variant=resolved_variant,
        )
    except Exception as exc:
        logger.exception(
            "Error executing RAG pipeline for request %s (variant=%s): %s",
            request_id,
            resolved_variant,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Inference failure: {exc}",
        ) from exc

    retrieved_chunks = [
        RetrievedChunkResponse(
            chunk_id=c.chunk_id,
            doc_number=c.doc_number,
            citation_label=c.citation_label,
            pasal=c.pasal,
            ayat=c.ayat,
            score=round(c.score, 6),
            text=c.text,
        )
        for c in output.retrieved_chunks
    ]

    verification_resp: CitationVerificationResponse | None = None
    v_dict = output.metadata.get("citation_verification")
    if v_dict:
        verification_resp = CitationVerificationResponse(
            is_valid=bool(v_dict.get("is_valid", True)),
            total_citations=int(v_dict.get("total_citations", 0)),
            grounded_count=int(v_dict.get("grounded_count", 0)),
            ungrounded_count=int(v_dict.get("ungrounded_count", 0)),
            checks=[],
            ungrounded_citations=list(v_dict.get("ungrounded_citations", [])),
            hallucinated_pasal=list(v_dict.get("hallucinated_pasal", [])),
        )

    return AskResponse(
        request_id=request_id,
        question=output.question,
        answer=output.answer,
        citations=output.citations,
        abstained=output.abstained,
        language=output.language,
        model=output.model_name,
        variant=resolved_variant,
        latency_ms=output.latency_ms,
        retrieved_chunks=retrieved_chunks,
        verification=verification_resp,
    )
