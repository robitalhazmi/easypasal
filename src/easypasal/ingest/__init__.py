"""Ingestion module: document downloading, PDF parsing, and structure-aware chunking."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.ingest.download import (
        ManifestRecord,
        compute_sha256,
        download_corpus,
        download_single_file,
        load_manifest,
    )

__all__ = [
    "ManifestRecord",
    "compute_sha256",
    "download_corpus",
    "download_single_file",
    "load_manifest",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        import easypasal.ingest.download as _download

        return getattr(_download, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
