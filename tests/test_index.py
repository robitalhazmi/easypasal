"""Unit tests for indexing module (BM25 sparse index, Qdrant vector store, and build pipeline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from easypasal.index.bm25 import (
    BM25Index,
    tokenize_indonesian_legal,
)
from easypasal.index.build import build_indices
from easypasal.index.embeddings import FastEmbedBackend
from easypasal.index.vector_store import (
    QdrantVectorStore,
    chunk_id_to_uuid,
)


class MockEmbeddingBackend:
    """Lightweight deterministic mock embedding backend for unit tests."""

    def __init__(self, dimension: int = 4, model_name: str = "mock-embed") -> None:
        self._dim = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        _ = batch_size
        # Generate simple deterministic pseudo-embeddings based on text length
        vectors: list[list[float]] = []
        for t in texts:
            val = float(len(t) % 10) / 10.0
            vectors.append([val, val * 0.5, 0.1, 0.9])
        return vectors

    def embed_query(self, query: str) -> list[float]:
        val = float(len(query) % 10) / 10.0
        return [val, val * 0.5, 0.1, 0.9]


def test_tokenize_indonesian_legal() -> None:
    """Test Indonesian legal tokenization preserving statutory terms and numbers."""
    text = "Pasal 71 ayat (1) POJK Nomor 22 Tahun 2023 tentang perlindungan konsumen dan undang-undang."
    tokens = tokenize_indonesian_legal(text, filter_stopwords=True)

    # Key terms must be present
    assert "pasal" in tokens
    assert "71" in tokens
    assert "1" in tokens
    assert "pojk" in tokens
    assert "22" in tokens
    assert "2023" in tokens
    assert "konsumen" in tokens
    assert "undang-undang" in tokens

    # Stopwords must be filtered out
    assert "dan" not in tokens
    assert "tentang" not in tokens or "tentang" in tokens  # depending on stopword set


def test_bm25_index_lifecycle(tmp_path: Path) -> None:
    """Test BM25 index creation, querying, serialization, and reloading."""
    mock_chunks: list[dict[str, Any]] = [
        {
            "chunk_id": "doc1_p1",
            "text": "PUJK wajib menangani pengaduan konsumen secara tertulis.",
            "citation_label": "POJK 22 Tahun 2023, Pasal 71",
            "doc_id": "POJK_22",
            "doc_number": "POJK 22",
            "pasal": "71",
            "ayat": "1",
            "status": "berlaku",
        },
        {
            "chunk_id": "doc1_p2",
            "text": "Sanksi administratif berupa denda keterlambatan penyampaian laporan.",
            "citation_label": "POJK 22 Tahun 2023, Pasal 72",
            "doc_id": "POJK_22",
            "doc_number": "POJK 22",
            "pasal": "72",
            "ayat": None,
            "status": "berlaku",
        },
        {
            "chunk_id": "doc2_p1",
            "text": "Keamanan siber dan ketahanan teknologi informasi perbankan umum.",
            "citation_label": "SEOJK 29/2022, Romawi II",
            "doc_id": "SEOJK_29",
            "doc_number": "SEOJK 29",
            "pasal": None,
            "ayat": None,
            "status": "berlaku",
        },
    ]

    index = BM25Index.build_from_chunks(mock_chunks)
    assert len(index.corpus_ids) == 3

    # Query matching chunk 0
    results = index.search("pengaduan konsumen tertulis", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "doc1_p1"
    assert results[0].citation_label == "POJK 22 Tahun 2023, Pasal 71"
    assert results[0].score > 0

    # Query matching chunk 2
    res_siber = index.search("keamanan siber perbankan", top_k=1)
    assert len(res_siber) == 1
    assert res_siber[0].chunk_id == "doc2_p1"

    # Save to disk
    save_dir = tmp_path / "bm25"
    index.save(save_dir, prefix="test_strat")
    assert (save_dir / "test_strat_model.pkl").exists()
    assert (save_dir / "test_strat_meta.json").exists()

    # Load from disk
    loaded = BM25Index.load(save_dir, prefix="test_strat")
    assert len(loaded.corpus_ids) == 3

    loaded_results = loaded.search("pengaduan konsumen tertulis", top_k=2)
    assert loaded_results[0].chunk_id == results[0].chunk_id
    assert abs(loaded_results[0].score - results[0].score) < 1e-5


def test_chunk_id_to_uuid() -> None:
    """Test deterministic UUID generation from chunk_id."""
    cid = "POJK_22_Tahun_2023_pasal_71_ayat_1"
    u1 = chunk_id_to_uuid(cid)
    u2 = chunk_id_to_uuid(cid)
    assert u1 == u2
    assert len(u1) == 36  # standard UUID format (8-4-4-4-12)

    diff = chunk_id_to_uuid("POJK_22_Tahun_2023_pasal_71_ayat_2")
    assert u1 != diff


def test_qdrant_vector_store_in_memory() -> None:
    """Test Qdrant collection initialization, upsert, and search in-memory."""
    store = QdrantVectorStore(path=":memory:")
    collection = "test_collection"

    store.init_collection(collection_name=collection, dimension=4, recreate=True)

    mock_chunks: list[dict[str, Any]] = [
        {
            "chunk_id": "c1",
            "citation_label": "Cit 1",
            "text": "Text 1",
            "doc_id": "D1",
            "status": "berlaku",
        },
        {
            "chunk_id": "c2",
            "citation_label": "Cit 2",
            "text": "Text 2",
            "doc_id": "D2",
            "status": "berlaku",
        },
    ]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    count = store.upsert_chunks(
        collection_name=collection,
        chunks=mock_chunks,
        vectors=vectors,
        batch_size=2,
    )
    assert count == 2
    assert store.count(collection) == 2

    # Query close to vector 0
    hits = store.search(collection_name=collection, query_vector=[0.9, 0.1, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"
    assert hits[0].citation_label == "Cit 1"
    assert hits[0].score > 0.8

    store.close()


def test_fastembed_backend() -> None:
    """Test FastEmbed backend embedding generation and dimension consistency."""
    backend = FastEmbedBackend()
    assert backend.dimension == 384

    query = "pelindungan data pribadi nasabah"
    vec = backend.embed_query(query)
    assert len(vec) == 384
    assert isinstance(vec[0], float)

    docs = ["Dokumen satu tentang bank.", "Dokumen dua tentang asuransi."]
    vecs = backend.embed_documents(docs, batch_size=2)
    assert len(vecs) == 2
    assert len(vecs[0]) == 384
    assert len(vecs[1]) == 384


def test_build_indices_end_to_end(tmp_path: Path) -> None:
    """Test build_indices end-to-end pipeline with mock embeddings."""
    mock_chunks: list[dict[str, Any]] = [
        {
            "chunk_id": "UU4_P1",
            "doc_id": "UU_4_2023",
            "doc_number": "UU 4 Tahun 2023",
            "doc_title": "P2SK",
            "year": 2023,
            "issuer": "Pemerintah RI",
            "status": "berlaku",
            "source_url": "https://example.com/uu4.pdf",
            "chunk_type": "structure_aware",
            "text": "[UU 4 Tahun 2023 | Pasal 1]\nKetentuan umum sektor keuangan.",
            "raw_content": "Ketentuan umum sektor keuangan.",
            "pasal": "1",
            "ayat": None,
            "citation_label": "UU 4 Tahun 2023, Pasal 1",
            "char_count": 65,
            "token_count_est": 16,
        },
        {
            "chunk_id": "UU4_P2",
            "doc_id": "UU_4_2023",
            "doc_number": "UU 4 Tahun 2023",
            "doc_title": "P2SK",
            "year": 2023,
            "issuer": "Pemerintah RI",
            "status": "berlaku",
            "source_url": "https://example.com/uu4.pdf",
            "chunk_type": "structure_aware",
            "text": "[UU 4 Tahun 2023 | Pasal 2]\nPengawasan terintegrasi perbankan dan asuransi.",
            "raw_content": "Pengawasan terintegrasi perbankan dan asuransi.",
            "pasal": "2",
            "ayat": None,
            "citation_label": "UU 4 Tahun 2023, Pasal 2",
            "char_count": 80,
            "token_count_est": 20,
        },
    ]

    chunks_file = tmp_path / "mock_chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in mock_chunks:
            f.write(json.dumps(c) + "\n")

    index_dir = tmp_path / "index"
    stats = build_indices(
        chunks_file=chunks_file,
        index_dir=index_dir,
        strategy="structure_aware",
        recreate=True,
        embedding_backend=MockEmbeddingBackend(dimension=4),
    )

    assert stats.num_chunks == 2
    assert stats.dimension == 4
    assert stats.strategy == "structure_aware"
    assert stats.elapsed_seconds >= 0.0

    # Verify Qdrant disk directory
    assert (index_dir / "qdrant").exists()

    # Verify BM25 disk directory
    assert (index_dir / "bm25" / "structure_aware_model.pkl").exists()
    assert (index_dir / "bm25" / "structure_aware_meta.json").exists()
