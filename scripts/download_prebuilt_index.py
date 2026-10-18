"""CLI script to download prebuilt indexes from Hugging Face Space (Plan §10 / Task T7.7)."""

from __future__ import annotations

import argparse
import logging
import sys

from easypasal.index.download import DEFAULT_SPACE_REPO, download_prebuilt_index

logger = logging.getLogger("easypasal.download_index")


def main() -> None:
    """CLI entrypoint for downloading prebuilt index."""
    parser = argparse.ArgumentParser(
        description="Download prebuilt EasyPasal Qdrant and BM25 indexes from Hugging Face Space."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_SPACE_REPO,
        help=f"Hugging Face Space repository ID (default: {DEFAULT_SPACE_REPO})",
    )
    parser.add_argument(
        "--target-dir",
        default="data/index",
        help="Local target directory for indexes (default: data/index)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if index files exist locally",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        download_prebuilt_index(
            repo_id=args.repo_id,
            target_dir=args.target_dir,
            force=args.force,
        )
    except Exception as exc:
        logger.error("Error downloading indexes: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
