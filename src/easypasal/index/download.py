"""Automated downloader for prebuilt regulatory indexes (Plan §10 / Task T7.7).

Allows developers and interviewers running on a clean laptop to download the prebuilt
Qdrant vector store and BM25 index from Hugging Face Space in seconds without
running full offline PDF ingestion and embedding builds.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("easypasal.index.download")

DEFAULT_SPACE_REPO = "robitalhazmi/easypasal"


def download_prebuilt_index(
    repo_id: str = DEFAULT_SPACE_REPO,
    target_dir: Path | str = "data/index",
    force: bool = False,
) -> Path:
    """Download prebuilt Qdrant and BM25 indexes from Hugging Face Space.

    Args:
        repo_id: Hugging Face Space repository ID.
        target_dir: Destination directory for local index files.
        force: If True, re-download even if target directory already contains index files.

    Returns:
        Path to local index directory.
    """
    out_dir = Path(target_dir).resolve()
    qdrant_dir = out_dir / "qdrant"
    bm25_dir = out_dir / "bm25"

    if (
        not force
        and qdrant_dir.exists()
        and any(qdrant_dir.iterdir())
        and bm25_dir.exists()
        and any(bm25_dir.iterdir())
    ):
        logger.info(
            "Prebuilt indexes already present in %s. Skipping download (use force=True to overwrite).",
            out_dir,
        )
        return out_dir

    logger.info(
        "Downloading prebuilt indexes from Hugging Face Space %r into %s...", repo_id, out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        # Target repo root where data/index is mirrored
        repo_root = out_dir.parent.parent
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="space",
            allow_patterns=["data/index/**"],
            local_dir=str(repo_root),
        )
        logger.info(
            "Successfully fetched prebuilt indexes into %s (cached at %s)", out_dir, downloaded_path
        )
    except Exception as exc:
        logger.warning("Failed to download prebuilt indexes from %s: %s", repo_id, exc)
        raise

    return out_dir
