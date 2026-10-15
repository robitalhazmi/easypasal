"""Data models for retrieval results and candidate chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with fused scores and full legal metadata."""

    chunk_id: str
    score: float
    citation_label: str
    doc_id: str
    doc_number: str
    doc_title: str = ""
    year: int | None = None
    issuer: str = ""
    status: str = "berlaku"
    source_url: str = ""
    bab: str | None = None
    bab_title: str | None = None
    bagian: str | None = None
    paragraf: str | None = None
    pasal: str | None = None
    ayat: str | None = None
    text: str = ""
    raw_content: str = ""
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        chunk_id: str,
        score: float,
        payload: dict[str, Any],
        dense_score: float | None = None,
        dense_rank: int | None = None,
        bm25_score: float | None = None,
        bm25_rank: int | None = None,
        rerank_score: float | None = None,
        rerank_rank: int | None = None,
    ) -> RetrievedChunk:
        """Create a RetrievedChunk from a chunk dictionary or search result payload."""
        return cls(
            chunk_id=chunk_id,
            score=score,
            citation_label=payload.get("citation_label", ""),
            doc_id=payload.get("doc_id", ""),
            doc_number=payload.get("doc_number", ""),
            doc_title=payload.get("doc_title", ""),
            year=payload.get("year"),
            issuer=payload.get("issuer", ""),
            status=payload.get("status", "berlaku"),
            source_url=payload.get("source_url", ""),
            bab=payload.get("bab"),
            bab_title=payload.get("bab_title"),
            bagian=payload.get("bagian"),
            paragraf=payload.get("paragraf"),
            pasal=payload.get("pasal"),
            ayat=payload.get("ayat"),
            text=payload.get("text", ""),
            raw_content=payload.get("raw_content", ""),
            dense_score=dense_score,
            dense_rank=dense_rank,
            bm25_score=bm25_score,
            bm25_rank=bm25_rank,
            rerank_score=rerank_score,
            rerank_rank=rerank_rank,
            metadata=payload,
        )


@dataclass
class RetrievalResult:
    """Encapsulates the complete response of a retrieval query."""

    query: str
    mode: str
    chunks: list[RetrievedChunk]
    total_candidates: int
    dense_count: int
    bm25_count: int
    duration_ms: float
