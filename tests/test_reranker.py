"""Unit tests for cross-encoder reranker interfaces and FlashRank integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from easypasal.retrieval.models import RetrievedChunk
from easypasal.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    BaseReranker,
    FlashRankReranker,
)


def _make_dummy_chunk(chunk_id: str, text: str, dense_score: float = 0.5) -> RetrievedChunk:
    """Helper to construct dummy RetrievedChunk instances for testing."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=dense_score,
        citation_label=f"Doc {chunk_id}",
        doc_id="doc1",
        doc_number="UU 1/2023",
        text=text,
        dense_score=dense_score,
        dense_rank=1,
        bm25_score=10.0,
        bm25_rank=1,
    )


class TestBaseReranker:
    """Test BaseReranker interface."""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]


class TestFlashRankReranker:
    """Test FlashRankReranker behavior and edge cases."""

    def test_init_defaults(self) -> None:
        reranker = FlashRankReranker()
        assert reranker.model_name == DEFAULT_RERANKER_MODEL
        assert reranker.cache_dir is None
        assert reranker._ranker is None

    def test_empty_chunks_returns_empty_list(self) -> None:
        reranker = FlashRankReranker()
        assert reranker.rerank(query="bunga pinjaman", chunks=[], top_k=5) == []

    def test_zero_or_negative_top_k_returns_empty(self) -> None:
        reranker = FlashRankReranker()
        chunk = _make_dummy_chunk("c1", "Ketentuan modal minimum")
        assert reranker.rerank(query="modal", chunks=[chunk], top_k=0) == []
        assert reranker.rerank(query="modal", chunks=[chunk], top_k=-1) == []

    def test_empty_query_returns_original_chunks_truncated(self) -> None:
        reranker = FlashRankReranker()
        c1 = _make_dummy_chunk("c1", "Teks 1")
        c2 = _make_dummy_chunk("c2", "Teks 2")
        result = reranker.rerank(query="   ", chunks=[c1, c2], top_k=1)
        assert len(result) == 1
        assert result[0].chunk_id == "c1"

    def test_rerank_with_mock_ranker_preserves_provenance_and_sorts(self) -> None:
        reranker = FlashRankReranker(model_name="test-model")

        # Mock underlying ranker
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": "c2", "score": 0.95},
            {"id": "c1", "score": 0.12},
        ]
        reranker._ranker = mock_ranker

        c1 = _make_dummy_chunk("c1", "Ketentuan umum", dense_score=0.88)
        c2 = _make_dummy_chunk("c2", "Modal inti minimum bank", dense_score=0.60)

        results = reranker.rerank(
            query="Berapa modal minimum?",
            chunks=[c1, c2],
            top_k=5,
        )

        assert len(results) == 2
        # c2 must be ranked first due to higher cross-encoder score
        assert results[0].chunk_id == "c2"
        assert results[0].score == pytest.approx(0.95)
        assert results[0].rerank_score == pytest.approx(0.95)
        assert results[0].rerank_rank == 1
        assert results[0].dense_score == pytest.approx(0.60)
        assert results[0].dense_rank == 1
        assert results[0].bm25_score == pytest.approx(10.0)

        # c1 must be ranked second
        assert results[1].chunk_id == "c1"
        assert results[1].score == pytest.approx(0.12)
        assert results[1].rerank_score == pytest.approx(0.12)
        assert results[1].rerank_rank == 2
        assert results[1].dense_score == pytest.approx(0.88)

    def test_ensure_model_failure_raises(self) -> None:
        reranker = FlashRankReranker(model_name="nonexistent-model")
        with (
            patch.dict("sys.modules", {"flashrank": None}),
            pytest.raises((ModuleNotFoundError, ImportError)),
        ):
            reranker._ensure_model()
