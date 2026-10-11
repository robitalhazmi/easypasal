"""API routes for EasyPasal: health checks, legal Q&A, and root metadata."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from easypasal.api.schemas import (
    AskRequest,
    AskResponse,
    CitationVerificationResponse,
    ErrorResponse,
    HealthResponse,
    RetrievedChunkResponse,
)
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

    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=environment,
        index_ready=index_ready,
        vector_count=vector_count,
        bm25_doc_count=bm25_count,
        backend_provider=provider_name,
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
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> AskResponse:
    """Answer a regulatory question with exact statutory citations and grounding verification."""
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))

    try:
        output = pipeline.ask(
            question=payload.question,
            language=payload.language,
            top_k=payload.top_k,
            strict_grounding=payload.strict_grounding,
        )
    except Exception as exc:
        logger.exception("Error executing RAG pipeline for request %s: %s", request_id, exc)
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
        latency_ms=output.latency_ms,
        retrieved_chunks=retrieved_chunks,
        verification=verification_resp,
    )
