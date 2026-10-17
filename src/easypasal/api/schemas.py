"""Pydantic schemas for the EasyPasal REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from easypasal.monitoring.schemas import FeedbackPayload


class AskRequest(BaseModel):
    """Request payload for legal question answering."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Legal inquiry regarding Indonesian financial regulations.",
        examples=["Berapa batas waktu penanganan pengaduan konsumen menurut POJK 22 Tahun 2023?"],
    )
    language: Literal["id", "en"] | None = Field(
        default=None,
        description="Optional language override ('id' for Indonesian, 'en' for English).",
        examples=["id"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of candidate statutory passages to retrieve via hybrid search.",
        examples=[5],
    )
    provider: str | None = Field(
        default=None,
        description="Optional generator backend override (e.g. 'mock', 'ollama', 'llamacpp').",
        examples=["mock"],
    )
    strict_grounding: bool = Field(
        default=False,
        description="If True, forces an abstention refusal if any cited statutory article is ungrounded.",
        examples=[False],
    )
    variant: str | None = Field(
        default=None,
        description="Optional pipeline variant override (e.g. 'baseline_v0.1', 'gepa_v0.2', 'A', 'B').",
        examples=["baseline_v0.1"],
    )


class RetrievedChunkResponse(BaseModel):
    """Summary of a retrieved statutory passage supporting the answer."""

    chunk_id: str
    doc_number: str
    citation_label: str
    pasal: str | None = None
    ayat: str | None = None
    score: float
    text: str


class CitationCheckResponse(BaseModel):
    """Detailed verification verdict for an individual citation."""

    citation: str
    is_grounded: bool
    doc_match: bool
    pasal_match: bool
    ayat_match: bool | None = None
    matched_chunk_id: str | None = None
    reason: str


class CitationVerificationResponse(BaseModel):
    """Aggregate post-hoc statutory verification metrics and audits."""

    is_valid: bool
    total_citations: int
    grounded_count: int
    ungrounded_count: int
    checks: list[CitationCheckResponse] = Field(default_factory=list)
    ungrounded_citations: list[str] = Field(default_factory=list)
    hallucinated_pasal: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    """Complete answer response from the EasyPasal RAG pipeline."""

    request_id: str = Field(
        description="Unique request tracking identifier (UUID or client-supplied)."
    )
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    abstained: bool
    language: str
    model: str
    variant: str = Field(
        default="baseline_v0.1",
        description="Active pipeline variant that processed this request (e.g. 'baseline_v0.1', 'gepa_v0.2').",
        examples=["baseline_v0.1"],
    )
    latency_ms: float
    retrieved_chunks: list[RetrievedChunkResponse] = Field(default_factory=list)
    verification: CitationVerificationResponse | None = None


class HealthResponse(BaseModel):
    """Readiness and health status of the EasyPasal service."""

    status: str = Field(default="ok", description="Liveness status ('ok').")
    version: str = Field(description="Service package version.")
    environment: str = Field(default="development", description="Runtime environment name.")
    index_ready: bool = Field(
        description="Whether the vector and sparse indexes are loaded and queryable."
    )
    vector_count: int = Field(default=0, description="Number of indexed vectors in Qdrant.")
    bm25_doc_count: int = Field(default=0, description="Number of indexed documents in BM25.")
    backend_provider: str | None = Field(default=None, description="Active LLM generation backend.")
    active_variant: str = Field(
        default="baseline_v0.1",
        description="Default or active pipeline variant ID.",
    )
    traffic_split_enabled: bool = Field(
        default=False,
        description="Whether deterministic traffic splitting is active across variants.",
    )


class ErrorResponse(BaseModel):
    """Standardized error structure."""

    detail: str
    request_id: str | None = None


class FeedbackResponse(BaseModel):
    """Response returned upon successfully recording user feedback."""

    status: str = Field(default="ok")
    request_id: str
    rating: int
    message: str = Field(default="Feedback recorded successfully.")


__all__ = [
    "AskRequest",
    "AskResponse",
    "CitationCheckResponse",
    "CitationVerificationResponse",
    "ErrorResponse",
    "FeedbackPayload",
    "FeedbackResponse",
    "HealthResponse",
    "RetrievedChunkResponse",
]
