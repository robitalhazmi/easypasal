"""Indexing module: vector embeddings, Qdrant store, and BM25 sparse index."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.index.bm25 import (
        BM25Index,
        BM25SearchResult,
        tokenize_indonesian_legal,
    )
    from easypasal.index.build import IndexStats, build_indices
    from easypasal.index.download import download_prebuilt_index
    from easypasal.index.embeddings import (
        DEFAULT_EMBEDDING_MODEL,
        EmbeddingBackend,
        FastEmbedBackend,
        get_embedding_backend,
        get_model_slug,
    )
    from easypasal.index.vector_store import (
        QdrantVectorStore,
        VectorSearchResult,
        chunk_id_to_uuid,
    )

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "BM25Index",
    "BM25SearchResult",
    "EmbeddingBackend",
    "FastEmbedBackend",
    "IndexStats",
    "QdrantVectorStore",
    "VectorSearchResult",
    "build_indices",
    "chunk_id_to_uuid",
    "download_prebuilt_index",
    "get_embedding_backend",
    "get_model_slug",
    "tokenize_indonesian_legal",
]

_BM25_SYMBOLS = {
    "BM25Index",
    "BM25SearchResult",
    "tokenize_indonesian_legal",
}

_EMBEDDING_SYMBOLS = {
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingBackend",
    "FastEmbedBackend",
    "get_embedding_backend",
    "get_model_slug",
}

_VECTOR_SYMBOLS = {
    "QdrantVectorStore",
    "VectorSearchResult",
    "chunk_id_to_uuid",
}

_BUILD_SYMBOLS = {
    "IndexStats",
    "build_indices",
}

_DOWNLOAD_SYMBOLS = {
    "download_prebuilt_index",
}


def __getattr__(name: str) -> object:
    if name in _BM25_SYMBOLS:
        import easypasal.index.bm25 as _bm25

        return getattr(_bm25, name)
    if name in _EMBEDDING_SYMBOLS:
        import easypasal.index.embeddings as _embeddings

        return getattr(_embeddings, name)
    if name in _VECTOR_SYMBOLS:
        import easypasal.index.vector_store as _vector_store

        return getattr(_vector_store, name)
    if name in _BUILD_SYMBOLS:
        import easypasal.index.build as _build

        return getattr(_build, name)
    if name in _DOWNLOAD_SYMBOLS:
        import easypasal.index.download as _download

        return getattr(_download, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
