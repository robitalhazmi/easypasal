"""Ingestion module: document downloading, PDF parsing, and structure-aware chunking."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.ingest.chunker import (
        Chunk,
        FixedSizeChunker,
        StructureAwareChunker,
        chunk_corpus,
        chunk_document,
    )
    from easypasal.ingest.download import (
        ManifestRecord,
        compute_sha256,
        download_corpus,
        download_single_file,
        load_manifest,
    )
    from easypasal.ingest.parser import (
        ParsedDocument,
        ParsedPage,
        clean_page_text,
        dehyphenate_text,
        parse_corpus,
        parse_pdf,
        unwrap_legal_text,
    )

__all__ = [
    "Chunk",
    "FixedSizeChunker",
    "ManifestRecord",
    "ParsedDocument",
    "ParsedPage",
    "StructureAwareChunker",
    "chunk_corpus",
    "chunk_document",
    "clean_page_text",
    "compute_sha256",
    "dehyphenate_text",
    "download_corpus",
    "download_single_file",
    "load_manifest",
    "parse_corpus",
    "parse_pdf",
    "unwrap_legal_text",
]

_PARSER_SYMBOLS = {
    "ParsedDocument",
    "ParsedPage",
    "clean_page_text",
    "dehyphenate_text",
    "parse_corpus",
    "parse_pdf",
    "unwrap_legal_text",
}

_CHUNKER_SYMBOLS = {
    "Chunk",
    "FixedSizeChunker",
    "StructureAwareChunker",
    "chunk_corpus",
    "chunk_document",
}


def __getattr__(name: str) -> object:
    if name in _PARSER_SYMBOLS:
        import easypasal.ingest.parser as _parser

        return getattr(_parser, name)
    if name in _CHUNKER_SYMBOLS:
        import easypasal.ingest.chunker as _chunker

        return getattr(_chunker, name)
    if name in __all__:
        import easypasal.ingest.download as _download

        return getattr(_download, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
