"""Corpus downloader module for EasyPasal.

Fetches regulatory documents from official sources declared in corpus_manifest.csv,
verifies/calculates SHA-256 hashes, and caches raw PDFs locally in data/raw/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.ingest.download")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 EasyPasal/0.1"
)
DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
}


def ensure_session_cookies(client: httpx.Client, url: str, timeout: float = 15.0) -> None:
    """Acquire required session cookies for sites with anti-bot/WAF protection."""
    if "ojk.go.id" in url.lower():
        has_ojk = any("ojk" in c.name.lower() for c in client.cookies.jar)
        if not has_ojk:
            try:
                client.get("https://www.ojk.go.id/", timeout=timeout, follow_redirects=True)
            except Exception as exc:
                logger.debug("OJK session warm-up note: %s", exc)


@dataclass
class ManifestRecord:
    """Represents a single document record in corpus_manifest.csv."""

    title: str
    doc_number: str
    year: int
    issuer: str
    url: str
    retrieval_date: str
    status: str
    language: str
    sha256: str

    @property
    def filename(self) -> str:
        """Derive a safe filesystem filename from document number."""
        clean_num = re.sub(r"[^\w\.-]", "_", self.doc_number.strip())
        clean_num = re.sub(r"_+", "_", clean_num)
        return f"{clean_num}.pdf"


def sanitize_filename(doc_number: str) -> str:
    """Sanitize doc_number to create safe filesystem filename."""
    clean_num = re.sub(r"[^\w\.-]", "_", doc_number.strip())
    clean_num = re.sub(r"_+", "_", clean_num)
    return f"{clean_num}.pdf"


def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 hash of a file by streaming in chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path) -> list[ManifestRecord]:
    """Parse corpus manifest CSV into ManifestRecord instances."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Corpus manifest not found at: {manifest_path}")

    records: list[ManifestRecord] = []
    with open(manifest_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            try:
                record = ManifestRecord(
                    title=row["title"].strip(),
                    doc_number=row["doc_number"].strip(),
                    year=int(row["year"].strip()),
                    issuer=row["issuer"].strip(),
                    url=row["url"].strip(),
                    retrieval_date=row["retrieval_date"].strip(),
                    status=row["status"].strip(),
                    language=row.get("language", "id").strip(),
                    sha256=row.get("sha256", "TBD").strip(),
                )
                records.append(record)
            except (KeyError, ValueError) as err:
                logger.warning("Skipping invalid row %d in %s: %s", idx, manifest_path, err)
    return records


def save_manifest(manifest_path: Path, records: list[ManifestRecord]) -> None:
    """Write updated records back to corpus manifest CSV."""
    fieldnames = [
        "title",
        "doc_number",
        "year",
        "issuer",
        "url",
        "retrieval_date",
        "status",
        "language",
        "sha256",
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "title": rec.title,
                    "doc_number": rec.doc_number,
                    "year": rec.year,
                    "issuer": rec.issuer,
                    "url": rec.url,
                    "retrieval_date": rec.retrieval_date,
                    "status": rec.status,
                    "language": rec.language,
                    "sha256": rec.sha256,
                }
            )


def download_single_file(
    url: str,
    dest_path: Path,
    client: httpx.Client,
    max_retries: int = 3,
    timeout: float = 30.0,
) -> bool:
    """Download a file via streaming HTTP GET with retries and exponential backoff."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            ensure_session_cookies(client, url, timeout=timeout)
            with client.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
                if resp.status_code == 404:
                    logger.error("HTTP 404 Not Found for %s", url)
                    return False
                resp.raise_for_status()

                content_type = resp.headers.get("Content-Type", "")
                if "html" in content_type.lower() and not url.lower().endswith(".html"):
                    logger.warning(
                        "URL returned HTML instead of PDF binary (%s): %s",
                        content_type,
                        url,
                    )

                is_first_chunk = True
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=32768):
                        if not chunk:
                            continue
                        if is_first_chunk:
                            if dest_path.suffix.lower() == ".pdf" and not chunk.startswith(b"%PDF"):
                                logger.error(
                                    "Download returned non-PDF content (magic bytes mismatch) for %s",
                                    url,
                                )
                                if temp_path.exists():
                                    temp_path.unlink(missing_ok=True)
                                return False
                            is_first_chunk = False
                        f.write(chunk)

            # Atomic move from temp to destination
            temp_path.replace(dest_path)
            return True

        except (httpx.HTTPError, TimeoutError, OSError) as exc:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            wait_time = 2**attempt
            logger.warning(
                "Attempt %d/%d failed for %s (%s). Retrying in %ds...",
                attempt,
                max_retries,
                url,
                exc,
                wait_time,
            )
            time.sleep(wait_time)

    return False


def download_corpus(
    manifest_path: Path,
    output_dir: Path,
    update_hashes: bool = False,
    max_retries: int = 3,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Orchestrate downloading and hash calculation for all corpus documents."""
    records = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "total": len(records),
        "downloaded": 0,
        "cached": 0,
        "failed": 0,
        "verified_hashes": 0,
        "errors": [],
    }

    close_client = False
    if client is None:
        client = httpx.Client(headers=DEFAULT_HEADERS, timeout=timeout)
        close_client = True

    manifest_modified = False

    try:
        for rec in records:
            target_file = output_dir / rec.filename

            # Check if file already exists with matching hash
            if target_file.is_file() and target_file.stat().st_size > 0:
                current_hash = compute_sha256(target_file)
                if rec.sha256 != "TBD" and current_hash == rec.sha256:
                    logger.info("CACHED [MATCH]: %s (%s)", rec.doc_number, rec.filename)
                    summary["cached"] += 1
                    summary["verified_hashes"] += 1
                    continue
                elif rec.sha256 == "TBD":
                    logger.info("EXISTS [COMPUTING HASH]: %s -> %s", rec.doc_number, current_hash)
                    summary["cached"] += 1
                    if update_hashes:
                        rec.sha256 = current_hash
                        manifest_modified = True
                    continue

            logger.info("DOWNLOADING: %s -> %s from %s", rec.doc_number, rec.filename, rec.url)
            success = download_single_file(
                url=rec.url,
                dest_path=target_file,
                client=client,
                max_retries=max_retries,
                timeout=timeout,
            )

            if success and target_file.exists():
                computed_hash = compute_sha256(target_file)
                summary["downloaded"] += 1
                logger.info("SAVED: %s (SHA-256: %s...)", target_file.name, computed_hash[:12])

                if rec.sha256 == "TBD" or update_hashes:
                    rec.sha256 = computed_hash
                    manifest_modified = True
                elif rec.sha256 != computed_hash:
                    err_msg = (
                        f"Hash mismatch for {rec.doc_number}: "
                        f"expected {rec.sha256}, got {computed_hash}"
                    )
                    logger.error(err_msg)
                    summary["errors"].append(err_msg)
                else:
                    summary["verified_hashes"] += 1
            else:
                summary["failed"] += 1
                err_msg = f"Failed to download: {rec.doc_number} ({rec.url})"
                summary["errors"].append(err_msg)

        if manifest_modified and update_hashes:
            save_manifest(manifest_path, records)
            logger.info("Updated %s with new SHA-256 hashes.", manifest_path)

    finally:
        if close_client:
            client.close()

    return summary


def main() -> None:
    """CLI entrypoint for corpus ingestion download."""
    parser = argparse.ArgumentParser(
        description="Download regulatory corpus PDFs defined in manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/corpus_manifest.csv"),
        help="Path to corpus_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for raw PDFs",
    )
    parser.add_argument(
        "--update-hashes",
        action="store_true",
        help="Update manifest with newly computed SHA-256 hashes",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Download timeout in seconds",
    )
    args = parser.parse_args()

    logger.info("Starting corpus ingestion from %s", args.manifest)
    summary = download_corpus(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        update_hashes=args.update_hashes,
        timeout=args.timeout,
    )

    logger.info(
        "Ingestion complete: %d total, %d downloaded, %d cached, %d failed",
        summary["total"],
        summary["downloaded"],
        summary["cached"],
        summary["failed"],
    )

    if summary["failed"] > 0:
        logger.warning("Encountered %d errors during ingestion.", summary["failed"])
        sys.exit(1)


if __name__ == "__main__":
    main()
