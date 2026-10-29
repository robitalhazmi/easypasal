"""Tests for hybrid retrieval, Reciprocal Rank Fusion (RRF), and candidate ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from easypasal.index.bm25 import BM25Index, BM25SearchResult
from easypasal.index.embeddings import EmbeddingBackend
from easypasal.index.vector_store import QdrantVectorStore, VectorSearchResult
from easypasal.retrieval.fusion import reciprocal_rank_fusion
from easypasal.retrieval.models import RetrievalResult, RetrievedChunk
from easypasal.retrieval.retriever import HybridRetriever


class DummyEmbeddingBackend(EmbeddingBackend):
    """Deterministic embedding backend for unit testing without downloading models."""

    def __init__(self, dimension: int = 4) -> None:
        self._dim = dimension

    def embed_documents(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = True
    ) -> list[list[float]]:
        # Map words to simple deterministic vector components
        _ = (batch_size, show_progress)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        # Simple bag-of-words coordinate mapper for testing
        q = query.lower()
        vec = [0.1, 0.1, 0.1, 0.1]
        if "konsumen" in q:
            vec[0] = 0.9
        if "modal" in q:
            vec[1] = 0.9
        if "bank" in q:
            vec[2] = 0.8
        return vec

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "dummy-test-embed"


def _make_vector_result(
    chunk_id: str,
    score: float,
    status: str = "berlaku",
    pasal: str = "1",
    text: str | None = None,
    citation_label: str | None = None,
) -> VectorSearchResult:
    cit_label = citation_label or f"POJK 1 Tahun 2024, Pasal {pasal}"
    doc_id = "POJK_1_Tahun_2024"
    doc_number = "POJK 1 Tahun 2024"
    ayat = "1"
    content_text = text if text is not None else f"Isi pengujian untuk pasal {pasal}."
    payload: dict[str, Any] = {
        "chunk_id": chunk_id,
        "citation_label": cit_label,
        "doc_id": doc_id,
        "doc_number": doc_number,
        "doc_title": "Peraturan Uji Coba",
        "year": 2024,
        "issuer": "OJK",
        "status": status,
        "source_url": "https://example.com/pojk1.pdf",
        "pasal": pasal,
        "ayat": ayat,
        "text": content_text,
        "raw_content": f"Isi pasal {pasal}.",
    }
    return VectorSearchResult(
        chunk_id=chunk_id,
        point_id="00000000-0000-0000-0000-000000000001",
        score=score,
        citation_label=cit_label,
        doc_id=doc_id,
        doc_number=doc_number,
        pasal=pasal,
        ayat=ayat,
        status=status,
        text=content_text,
        payload=payload,
    )


def _make_bm25_result(
    chunk_id: str,
    score: float,
    status: str = "berlaku",
    pasal: str = "1",
) -> BM25SearchResult:
    citation_label = f"POJK 1 Tahun 2024, Pasal {pasal}"
    doc_id = "POJK_1_Tahun_2024"
    doc_number = "POJK 1 Tahun 2024"
    ayat = "1"
    text = f"Isi pengujian untuk pasal {pasal}."
    payload: dict[str, Any] = {
        "chunk_id": chunk_id,
        "citation_label": citation_label,
        "doc_id": doc_id,
        "doc_number": doc_number,
        "doc_title": "Peraturan Uji Coba",
        "year": 2024,
        "issuer": "OJK",
        "status": status,
        "source_url": "https://example.com/pojk1.pdf",
        "pasal": pasal,
        "ayat": ayat,
        "text": text,
        "raw_content": f"Isi pasal {pasal}.",
    }
    return BM25SearchResult(
        chunk_id=chunk_id,
        score=score,
        citation_label=citation_label,
        doc_id=doc_id,
        doc_number=doc_number,
        pasal=pasal,
        ayat=ayat,
        status=status,
        text=text,
        payload=payload,
    )


def test_rrf_mathematical_precision() -> None:
    """Verify exact analytical RRF score calculation against theoretical values."""
    # chunk_A is Rank 1 in Dense, Rank 2 in BM25
    dense_res = [_make_vector_result("chunk_A", 0.95)]
    bm25_res = [_make_bm25_result("chunk_B", 15.0), _make_bm25_result("chunk_A", 12.0)]

    # Expected for chunk_A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ~= 0.01639344 + 0.01612903 = 0.03252247
    # Expected for chunk_B: 1/(60+1) = 1/61 ~= 0.01639344
    fused = reciprocal_rank_fusion(
        dense_results=dense_res,
        bm25_results=bm25_res,
        k_rrf=60,
        dense_weight=1.0,
        bm25_weight=1.0,
        top_k=5,
    )

    assert len(fused) == 2
    assert fused[0].chunk_id == "chunk_A"
    assert fused[0].score == pytest.approx(0.032522, abs=1e-5)
    assert fused[0].dense_rank == 1
    assert fused[0].dense_score == 0.95
    assert fused[0].bm25_rank == 2
    assert fused[0].bm25_score == 12.0

    assert fused[1].chunk_id == "chunk_B"
    assert fused[1].score == pytest.approx(0.016393, abs=1e-5)
    assert fused[1].dense_rank is None
    assert fused[1].bm25_rank == 1


def test_rrf_weights_and_ranking() -> None:
    """Verify that weighting adjusts relative rank priority between channels."""
    dense_res = [_make_vector_result("dense_star", 0.90)]
    bm25_res = [_make_bm25_result("sparse_star", 20.0)]

    # If bm25_weight is 3x dense_weight, sparse_star should rank higher despite identical rank 1
    fused = reciprocal_rank_fusion(
        dense_results=dense_res,
        bm25_results=bm25_res,
        k_rrf=60,
        dense_weight=1.0,
        bm25_weight=3.0,
    )

    assert len(fused) == 2
    assert fused[0].chunk_id == "sparse_star"
    assert fused[1].chunk_id == "dense_star"


def test_rrf_edge_cases() -> None:
    """Verify edge cases such as empty lists, top_k limits, and bad parameters."""
    # Both empty
    assert reciprocal_rank_fusion([], []) == []

    # top_k <= 0
    res = [_make_vector_result("c1", 0.8)]
    assert reciprocal_rank_fusion(res, [], top_k=0) == []
    assert reciprocal_rank_fusion(res, [], top_k=-1) == []

    # Invalid k_rrf
    with pytest.raises(ValueError, match="k_rrf must be positive"):
        reciprocal_rank_fusion(res, [], k_rrf=0)


def test_rrf_normative_salience_boost() -> None:
    """Verify that query imperative action predicate concordance boosts primary norm chunks."""
    # Chunk A: Ancillary general clause (dense rank 1)
    res_ancillary = _make_vector_result(
        "ancillary",
        0.95,
        text="Bank wajib memastikan sistem kelangsungan usaha tetap berjalan.",
        citation_label="POJK 11/2022, Pasal 37",
    )
    # Chunk B: Direct imperative action clause matching query (dense rank 2)
    res_primary = _make_vector_result(
        "primary",
        0.90,
        text="Bank wajib menempatkan Sistem Elektronik pada Pusat Data di wilayah Indonesia.",
        citation_label="POJK 11/2022, Pasal 35 ayat (1)",
    )

    query = "Di mana Bank Umum wajib menempatkan Pusat Data?"

    # Without query: dense rank 1 (ancillary) beats dense rank 2 (primary)
    fused_unboosted = reciprocal_rank_fusion(
        dense_results=[res_ancillary, res_primary],
        bm25_results=[],
        query=None,
    )
    assert fused_unboosted[0].chunk_id == "ancillary"

    # With query: primary chunk receives imperative action concordance boost ("wajib menempatkan" + "di mana")
    fused_boosted = reciprocal_rank_fusion(
        dense_results=[res_ancillary, res_primary],
        bm25_results=[],
        query=query,
    )
    assert fused_boosted[0].chunk_id == "primary"
    assert fused_boosted[0].score > fused_boosted[1].score


def test_detect_normative_intent_and_remedial_downweighting() -> None:
    """Verify normative intent detection and supervisory remedial penalty under obligation queries."""
    from easypasal.retrieval.fusion import (
        _compute_normative_salience_boost,
        detect_normative_intent,
    )

    assert (
        detect_normative_intent("Di mana Bank Umum wajib menempatkan Pusat Data?") == "obligation"
    )
    assert (
        detect_normative_intent("Kapan OJK dapat meminta Bank memindahkan Pusat Data?")
        == "sanction_enforcement"
    )
    assert (
        detect_normative_intent(
            "Kapan Bank dapat menempatkan Pusat Data di luar negeri sepanjang izin?"
        )
        == "permission_exemption"
    )

    # For obligation query:
    q_obl = "Di mana Bank Umum wajib menempatkan Pusat Data?"
    payload_primary = {
        "text": "Bank wajib menempatkan Sistem Elektronik pada Pusat Data di wilayah Indonesia."
    }
    payload_remedial = {
        "text": "Otoritas Jasa Keuangan dapat meminta Bank untuk menempatkan Sistem Elektronik di wilayah Indonesia."
    }

    boost_primary = _compute_normative_salience_boost(q_obl, payload_primary)
    boost_remedial = _compute_normative_salience_boost(q_obl, payload_remedial)

    assert boost_primary > 0.20
    assert boost_remedial < 0.0


def test_hybrid_retriever_in_memory() -> None:
    """Verify HybridRetriever operations across all modes using in-memory components."""
    qdrant = QdrantVectorStore(path=":memory:")
    backend = DummyEmbeddingBackend(dimension=4)
    collection = "easypasal_test"

    chunks: list[dict[str, Any]] = [
        {
            "chunk_id": "test_chunk_1",
            "citation_label": "POJK 1 Tahun 2024, Pasal 1",
            "doc_id": "POJK_1_2024",
            "doc_number": "POJK 1 Tahun 2024",
            "doc_title": "Pelindungan Konsumen",
            "year": 2024,
            "issuer": "OJK",
            "status": "berlaku",
            "source_url": "https://example.com/1.pdf",
            "pasal": "1",
            "ayat": "1",
            "text": "PUJK wajib menjamin pelindungan konsumen jasa keuangan.",
            "raw_content": "PUJK wajib menjamin pelindungan konsumen.",
        },
        {
            "chunk_id": "test_chunk_2",
            "citation_label": "POJK 2 Tahun 2024, Pasal 5",
            "doc_id": "POJK_2_2024",
            "doc_number": "POJK 2 Tahun 2024",
            "doc_title": "Permodalan Bank",
            "year": 2024,
            "issuer": "OJK",
            "status": "berlaku",
            "source_url": "https://example.com/2.pdf",
            "pasal": "5",
            "ayat": "1",
            "text": "Modal inti minimum bank umum sebesar Rp3 triliun.",
            "raw_content": "Modal inti minimum bank umum sebesar Rp3 triliun.",
        },
        {
            "chunk_id": "test_chunk_3",
            "citation_label": "POJK 3 Tahun 2020, Pasal 10",
            "doc_id": "POJK_3_2020",
            "doc_number": "POJK 3 Tahun 2020",
            "doc_title": "Ketentuan Lama Dicabut",
            "year": 2020,
            "issuer": "OJK",
            "status": "dicabut",
            "source_url": "https://example.com/3.pdf",
            "pasal": "10",
            "ayat": "1",
            "text": "Ketentuan pencabutan modal dan perlindungan masa lalu.",
            "raw_content": "Ketentuan masa lalu.",
        },
    ]

    # 1. Populate in-memory Qdrant
    qdrant.init_collection(collection_name=collection, dimension=backend.dimension)
    vectors = backend.embed_documents([c["text"] for c in chunks])
    qdrant.upsert_chunks(collection_name=collection, chunks=chunks, vectors=vectors)

    # 2. Populate BM25
    bm25 = BM25Index.build_from_chunks(chunks)

    # 3. Instantiate HybridRetriever
    retriever = HybridRetriever(
        qdrant_store=qdrant,
        bm25_index=bm25,
        embedding_backend=backend,
        collection_name=collection,
    )

    # A. Test Hybrid Mode
    res_hybrid = retriever.retrieve("pelindungan konsumen PUJK", top_k=2, mode="hybrid")
    assert isinstance(res_hybrid, RetrievalResult)
    assert res_hybrid.mode == "hybrid"
    assert len(res_hybrid.chunks) == 2
    assert res_hybrid.chunks[0].chunk_id == "test_chunk_1"
    assert res_hybrid.chunks[0].citation_label == "POJK 1 Tahun 2024, Pasal 1"
    assert res_hybrid.duration_ms >= 0.0

    # B. Test Dense Only Mode
    res_dense = retriever.retrieve("modal minimum bank", top_k=2, mode="dense")
    assert res_dense.mode == "dense"
    assert res_dense.chunks[0].chunk_id == "test_chunk_2"
    assert res_dense.chunks[0].dense_rank == 1

    # C. Test BM25 Only Mode
    res_bm25 = retriever.retrieve("konsumen jasa keuangan", top_k=2, mode="bm25")
    assert res_bm25.mode == "bm25"
    assert res_bm25.chunks[0].chunk_id == "test_chunk_1"
    assert res_bm25.chunks[0].bm25_rank == 1

    # D. Test Status Filter ('berlaku' should exclude test_chunk_3)
    res_filtered = retriever.retrieve(
        "modal perlindungan masa lalu",
        top_k=5,
        mode="hybrid",
        status_filter="berlaku",
    )
    returned_ids = {c.chunk_id for c in res_filtered.chunks}
    assert "test_chunk_3" not in returned_ids

    # E. Test Empty Query
    res_empty = retriever.retrieve("   ", top_k=5)
    assert res_empty.chunks == []
    assert res_empty.total_candidates == 0

    # F. Test Unsupported Mode
    with pytest.raises(ValueError, match="Unsupported retrieval mode"):
        retriever.retrieve("query", mode="unknown_mode")


def test_package_exports() -> None:
    """Verify that easypasal.retrieval properly exports all symbols."""
    import easypasal.retrieval as ret

    assert hasattr(ret, "HybridRetriever")
    assert hasattr(ret, "RetrievedChunk")
    assert hasattr(ret, "RetrievalResult")
    assert hasattr(ret, "reciprocal_rank_fusion")
    assert hasattr(ret, "DEFAULT_K_RRF")


def test_retriever_from_paths_not_found(tmp_path: Path) -> None:
    """Verify FileNotFoundError when indices are missing."""
    empty_dir = tmp_path / "empty_index"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Qdrant directory does not exist"):
        HybridRetriever.from_paths(index_dir=empty_dir)


def test_retriever_collection_name_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test collection name resolution in HybridRetriever.from_paths."""
    q_dir = tmp_path / "qdrant"
    q_dir.mkdir()
    b_dir = tmp_path / "bm25"
    b_dir.mkdir()

    monkeypatch.setattr(QdrantVectorStore, "__init__", lambda _self, **_kwargs: None)
    monkeypatch.setattr(BM25Index, "load", lambda **_kwargs: None)
    monkeypatch.setattr(
        "easypasal.retrieval.retriever.get_embedding_backend",
        lambda **_kwargs: DummyEmbeddingBackend(),
    )

    # Default model
    r1 = HybridRetriever.from_paths(index_dir=tmp_path, strategy="structure_aware")
    assert r1.collection_name == "easypasal_structure_aware"

    # Non-default model
    r2 = HybridRetriever.from_paths(
        index_dir=tmp_path,
        strategy="structure_aware",
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    assert r2.collection_name == "easypasal_structure_aware_paraphrase_multilingual_mpnet_base_v2"

    # Explicit override
    r3 = HybridRetriever.from_paths(
        index_dir=tmp_path,
        strategy="structure_aware",
        collection_name="custom_collection",
    )
    assert r3.collection_name == "custom_collection"


def test_real_index_retrieval_if_available() -> None:
    """Integration test against actual on-disk index if already built."""
    qdrant_dir = Path("data/index/qdrant")
    bm25_file = Path("data/index/bm25/structure_aware_model.pkl")

    if not qdrant_dir.exists() or not bm25_file.exists():
        pytest.skip("Full on-disk indices not present; skipping real index probe.")

    retriever = HybridRetriever.from_paths(index_dir="data/index", strategy="structure_aware")
    try:
        # Warm-up inference loads ONNX runtime session
        _ = retriever.retrieve("warmup", top_k=1, mode="hybrid")
        res = retriever.retrieve(
            "pelindungan konsumen dan penanganan pengaduan", top_k=3, mode="hybrid"
        )

        assert len(res.chunks) == 3
        assert res.total_candidates > 0
        assert res.duration_ms < 500.0  # Warm hybrid search should be sub-500ms on CPU
        for chunk in res.chunks:
            assert chunk.citation_label
            assert chunk.status in ("berlaku", "diubah", "dicabut")
            assert chunk.score > 0.0
    finally:
        retriever.close()


def test_retriever_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that retriever CLI executes and prints formatted results."""
    import sys

    from easypasal.retrieval.models import RetrievedChunk
    from easypasal.retrieval.retriever import main

    test_chunk = RetrievedChunk(
        chunk_id="test_c1",
        score=0.032,
        citation_label="POJK 1 Tahun 2024, Pasal 1",
        doc_id="POJK_1_2024",
        doc_number="POJK 1 Tahun 2024",
        status="berlaku",
        text="Isi peraturan uji coba.",
        dense_rank=1,
        dense_score=0.9,
        bm25_rank=1,
        bm25_score=15.0,
    )
    mock_result = RetrievalResult(
        query="konsumen",
        mode="hybrid",
        chunks=[test_chunk],
        total_candidates=1,
        dense_count=1,
        bm25_count=1,
        duration_ms=12.5,
    )

    class MockRetriever:
        def retrieve(self, *_args: Any, **_kwargs: Any) -> RetrievalResult:
            return mock_result

    monkeypatch.setattr(
        HybridRetriever, "from_paths", classmethod(lambda _cls, **_kwargs: MockRetriever())
    )
    monkeypatch.setattr(
        sys, "argv", ["retriever.py", "--query", "konsumen", "--mode", "hybrid", "--top-k", "1"]
    )

    main()
    captured = capsys.readouterr()
    assert "QUERY: konsumen" in captured.out
    assert "POJK 1 Tahun 2024, Pasal 1" in captured.out
    assert "Dense #1" in captured.out
    assert "BM25 #1" in captured.out


def test_hybrid_retriever_hybrid_rerank_mode() -> None:
    """Verify that mode='hybrid+rerank' invokes the configured reranker."""
    qdrant = QdrantVectorStore(path=":memory:")
    backend = DummyEmbeddingBackend(dimension=4)
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Perlindungan konsumen perbankan",
            "citation_label": "POJK 1, Pasal 1",
            "doc_id": "d1",
            "doc_number": "POJK 1",
            "status": "berlaku",
        },
        {
            "chunk_id": "c2",
            "text": "Modal minimum bank umum",
            "citation_label": "POJK 2, Pasal 1",
            "doc_id": "d2",
            "doc_number": "POJK 2",
            "status": "berlaku",
        },
    ]
    qdrant.init_collection(collection_name="test_col", dimension=4)
    vectors = backend.embed_documents([c["text"] for c in chunks])
    qdrant.upsert_chunks(collection_name="test_col", chunks=chunks, vectors=vectors)
    bm25 = BM25Index.build_from_chunks(chunks)

    mock_reranker = MagicMock()

    def mock_rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        _ = query
        for rank, c in enumerate(reversed(chunks[:top_k]), start=1):
            c.rerank_score = 0.99 / rank
            c.rerank_rank = rank
            c.score = 0.99 / rank
        return list(reversed(chunks[:top_k]))

    mock_reranker.rerank.side_effect = mock_rerank

    retriever = HybridRetriever(
        qdrant_store=qdrant,
        bm25_index=bm25,
        embedding_backend=backend,
        collection_name="test_col",
        reranker=mock_reranker,
    )

    res = retriever.retrieve(query="perlindungan konsumen", top_k=2, mode="hybrid+rerank")
    assert len(res.chunks) == 2
    assert mock_reranker.rerank.called
    assert res.chunks[0].rerank_score is not None
    assert res.chunks[0].rerank_rank == 1


def test_hybrid_retriever_hybrid_rerank_fallback_when_none() -> None:
    """Verify fallback to Hybrid RRF when mode='hybrid+rerank' but reranker is None."""
    qdrant = QdrantVectorStore(path=":memory:")
    backend = DummyEmbeddingBackend(dimension=4)
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Perlindungan konsumen perbankan",
            "citation_label": "POJK 1, Pasal 1",
            "doc_id": "d1",
            "doc_number": "POJK 1",
            "status": "berlaku",
        },
    ]
    qdrant.init_collection(collection_name="test_col", dimension=4)
    vectors = backend.embed_documents([c["text"] for c in chunks])
    qdrant.upsert_chunks(collection_name="test_col", chunks=chunks, vectors=vectors)
    bm25 = BM25Index.build_from_chunks(chunks)

    retriever = HybridRetriever(
        qdrant_store=qdrant,
        bm25_index=bm25,
        embedding_backend=backend,
        collection_name="test_col",
        reranker=None,
    )

    res = retriever.retrieve(query="perlindungan konsumen", top_k=1, mode="hybrid+rerank")
    assert len(res.chunks) == 1
    assert res.chunks[0].score > 0.0


def test_expand_primary_statutory_norms() -> None:
    """Verify structural and cross-reference primary obligation expansion when only remedial/exception chunks were retrieved."""
    from easypasal.retrieval.retriever import _expand_primary_statutory_norms

    corpus_payloads = [
        {
            "chunk_id": "pojk11_35_1",
            "citation_label": "POJK 11/POJK.03/2022, Pasal 35 ayat (1)",
            "doc_id": "POJK_11_POJK.03_2022",
            "doc_number": "POJK 11/POJK.03/2022",
            "pasal": "35",
            "ayat": "1",
            "status": "berlaku",
            "text": (
                "[POJK 11/POJK.03/2022 | Pasal 35 ayat (1)]\n"
                "(1) Bank wajib menempatkan Sistem Elektronik pada Pusat Data dan Pusat Pemulihan Bencana di wilayah Indonesia."
            ),
        },
        {
            "chunk_id": "pojk11_36_4",
            "citation_label": "POJK 11/POJK.03/2022, Pasal 36 ayat (4)",
            "doc_id": "POJK_11_POJK.03_2022",
            "doc_number": "POJK 11/POJK.03/2022",
            "pasal": "36",
            "ayat": "4",
            "status": "berlaku",
            "text": (
                "[POJK 11/POJK.03/2022 | Pasal 36 ayat (4)]\n"
                "(4) Dalam hal berdasarkan penilaian Otoritas Jasa\nKeuangan penempatan Sistem Elektronik di luar wilayah Indonesia berisiko, "
                "Otoritas Jasa Keuangan dapat meminta Bank untuk menempatkan Sistem Elektronik pada Pusat Data di wilayah Indonesia."
            ),
        },
        {
            "chunk_id": "pojk11_36_3",
            "citation_label": "POJK 11/POJK.03/2022, Pasal 36 ayat (3)",
            "doc_id": "POJK_11_POJK.03_2022",
            "doc_number": "POJK 11/POJK.03/2022",
            "pasal": "36",
            "ayat": "3",
            "status": "berlaku",
            "text": (
                "[POJK 11/POJK.03/2022 | Pasal 36 ayat (3)]\n"
                "(3) Bank wajib memastikan bahwa data yang digunakan dalam Sistem Elektronik di luar wilayah Indonesia "
                "tidak digunakan selain sebagaimana dimaksud dalam Pasal 35 ayat (3)."
            ),
        },
    ]

    bm25 = BM25Index.build_from_chunks(corpus_payloads)

    # Simulate a retrieval result that missed Pasal 35 ayat (1) and only returned Pasal 36 ayat (4) and Pasal 36 ayat (3)
    initial_chunks = [
        RetrievedChunk.from_payload("pojk11_36_4", 0.88, corpus_payloads[1]),
        RetrievedChunk.from_payload("pojk11_36_3", 0.86, corpus_payloads[2]),
    ]

    query = "Di mana Bank Umum wajib menempatkan Pusat Data dan Pusat Pemulihan Bencana menurut regulasi OJK?"
    expanded = _expand_primary_statutory_norms(
        query=query,
        chunks=initial_chunks,
        bm25_index=bm25,
        status_filter="berlaku",
        top_k=5,
    )

    assert expanded[0].chunk_id == "pojk11_35_1"
    assert expanded[0].citation_label == "POJK 11/POJK.03/2022, Pasal 35 ayat (1)"
    assert expanded[0].score > 0.88


def test_expand_sibling_statutory_norms() -> None:
    """Verify sibling proximity expansion bundles complementary modalities (e.g. oral vs written complaints)."""
    from easypasal.retrieval.retriever import _expand_primary_statutory_norms

    corpus_payloads = [
        {
            "chunk_id": "pojk22_74_1",
            "citation_label": "POJK 22 Tahun 2023, Pasal 74 ayat (1)",
            "doc_id": "POJK_22_Tahun_2023",
            "doc_number": "POJK 22 Tahun 2023",
            "pasal": "74",
            "ayat": "1",
            "status": "berlaku",
            "text": (
                "[POJK 22 Tahun 2023 | Pasal 74 ayat (1)]\n"
                "(1) PUJK wajib melakukan tindak lanjut dan menyelesaikan Pengaduan secara lisan paling lama 5 (lima) hari kerja sejak Pengaduan diterima PUJK."
            ),
        },
        {
            "chunk_id": "pojk22_75_1",
            "citation_label": "POJK 22 Tahun 2023, Pasal 75 ayat (1)",
            "doc_id": "POJK_22_Tahun_2023",
            "doc_number": "POJK 22 Tahun 2023",
            "pasal": "75",
            "ayat": "1",
            "status": "berlaku",
            "text": (
                "[POJK 22 Tahun 2023 | Pasal 75 ayat (1)]\n"
                "(1) PUJK wajib melakukan tindak lanjut dan melakukan penyelesaian Pengaduan secara tertulis paling lama 10 (sepuluh) hari kerja sejak dokumen sebagaimana dimaksud dalam Pasal 71 ayat (2) dan/atau ayat (3) diterima secara lengkap."
            ),
        },
    ]

    bm25 = BM25Index.build_from_chunks(corpus_payloads)

    # Initial retrieval only returned Pasal 74 (oral)
    initial_chunks = [
        RetrievedChunk.from_payload("pojk22_74_1", 0.92, corpus_payloads[0]),
    ]

    query = "Bagaimana kewajiban Lembaga Jasa Keuangan dalam menangani dan menyelesaikan pengaduan konsumen?"
    expanded = _expand_primary_statutory_norms(
        query=query,
        chunks=initial_chunks,
        bm25_index=bm25,
        status_filter="berlaku",
        top_k=5,
    )

    expanded_ids = [c.chunk_id for c in expanded]
    assert "pojk22_74_1" in expanded_ids
    assert "pojk22_75_1" in expanded_ids
    # Both oral and written complaint resolution primary articles are retained in top ranks
    assert expanded[0].chunk_id == "pojk22_74_1"
    assert expanded[1].chunk_id == "pojk22_75_1"
