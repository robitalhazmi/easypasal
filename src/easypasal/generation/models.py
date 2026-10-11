"""Data models for text generation, citation tracking, and LLM outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from easypasal.retrieval.models import RetrievedChunk


@dataclass
class GenerationInput:
    """Input payload for regulatory question answering."""

    question: str
    retrieved_chunks: list[RetrievedChunk]
    temperature: float = 0.1
    max_tokens: int = 512
    system_prompt_override: str | None = None
    language: str | None = None  # auto-detected if None ("id" or "en")


@dataclass
class GenerationOutput:
    """Output result produced by the RAG generation pipeline."""

    question: str
    answer: str
    language: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
