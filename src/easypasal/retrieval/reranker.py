"""Cross-encoder reranking for refining retrieved legal passages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from easypasal.retrieval.models import RetrievedChunk

logger = logging.getLogger("easypasal.retrieval.reranker")

DEFAULT_RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"


class BaseReranker(ABC):
    """Abstract base class for cross-encoder rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Rerank candidate chunks according to cross-encoder query-passage relevance.

        Args:
            query: Input natural language question or statutory search.
            chunks: List of candidate RetrievedChunk items (typically from hybrid RRF).
            top_k: Maximum number of reranked chunks to return.

        Returns:
            Reranked list of RetrievedChunk instances sorted by descending cross-encoder score.
        """


class FlashRankReranker(BaseReranker):
    """Lightweight cross-encoder reranker powered by FlashRank (quantized ONNX on CPU)."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._ranker: Any = None

    def _ensure_model(self) -> Any:
        """Lazy-load the underlying FlashRank model."""
        if self._ranker is None:
            try:
                from flashrank import Ranker  # type: ignore[import-untyped]

                logger.info("Initializing FlashRank reranker (model=%r)...", self.model_name)
                kwargs: dict[str, Any] = {"model_name": self.model_name}
                if self.cache_dir:
                    kwargs["cache_dir"] = self.cache_dir
                self._ranker = Ranker(**kwargs)
                logger.info("FlashRank reranker %r loaded successfully.", self.model_name)
            except Exception as e:
                logger.error("Failed to load FlashRank reranker %r: %s", self.model_name, e)
                raise
        return self._ranker

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Rerank candidate chunks using FlashRank cross-encoder."""
        if not chunks or top_k <= 0:
            return []

        query_clean = query.strip()
        if not query_clean:
            return chunks[:top_k]

        ranker = self._ensure_model()
        from flashrank import RerankRequest

        chunk_by_id = {c.chunk_id: c for c in chunks}
        passages = [{"id": c.chunk_id, "text": c.text[:800]} for c in chunks]

        rerank_request = RerankRequest(query=query_clean, passages=passages)
        results = ranker.rerank(rerank_request)

        reranked_chunks: list[RetrievedChunk] = []
        for rank, res in enumerate(results[:top_k], start=1):
            cid = res["id"]
            orig_chunk = chunk_by_id.get(cid)
            if orig_chunk is None:
                continue

            r_score = float(res.get("score", 0.0))
            reranked_chunk = RetrievedChunk(
                chunk_id=orig_chunk.chunk_id,
                score=r_score,
                citation_label=orig_chunk.citation_label,
                doc_id=orig_chunk.doc_id,
                doc_number=orig_chunk.doc_number,
                doc_title=orig_chunk.doc_title,
                year=orig_chunk.year,
                issuer=orig_chunk.issuer,
                status=orig_chunk.status,
                source_url=orig_chunk.source_url,
                bab=orig_chunk.bab,
                bab_title=orig_chunk.bab_title,
                bagian=orig_chunk.bagian,
                paragraf=orig_chunk.paragraf,
                pasal=orig_chunk.pasal,
                ayat=orig_chunk.ayat,
                text=orig_chunk.text,
                raw_content=orig_chunk.raw_content,
                dense_score=orig_chunk.dense_score,
                dense_rank=orig_chunk.dense_rank,
                bm25_score=orig_chunk.bm25_score,
                bm25_rank=orig_chunk.bm25_rank,
                rerank_score=r_score,
                rerank_rank=rank,
                metadata=orig_chunk.metadata,
            )
            reranked_chunks.append(reranked_chunk)

        return reranked_chunks
