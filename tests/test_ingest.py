"""Unit tests for corpus download, hashing, and manifest management."""

from __future__ import annotations

import csv
import hashlib
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from easypasal.ingest.download import (
    compute_sha256,
    download_corpus,
    download_single_file,
    ensure_session_cookies,
    load_manifest,
    main,
    sanitize_filename,
    save_manifest,
)


def test_sanitize_filename() -> None:
    """Verify filename sanitization converts special chars safely."""
    assert sanitize_filename("POJK 22 Tahun 2023") == "POJK_22_Tahun_2023.pdf"
    assert sanitize_filename("POJK 11/POJK.03/2022") == "POJK_11_POJK.03_2022.pdf"
    assert sanitize_filename("UU No. 8/2010") == "UU_No._8_2010.pdf"


def test_compute_sha256(tmp_path: Path) -> None:
    """Verify SHA-256 calculation matches known digest."""
    test_file = tmp_path / "sample.pdf"
    content = b"%PDF-1.4 sample regulatory test content"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert compute_sha256(test_file) == expected_hash


def test_manifest_load_and_save(tmp_path: Path) -> None:
    """Verify loading from and saving to manifest CSV."""
    manifest_csv = tmp_path / "test_manifest.csv"
    manifest_csv.write_text(
        "title,doc_number,year,issuer,url,retrieval_date,status,language,sha256\n"
        '"Test Title",POJK 1/2024,2024,OJK,https://example.com/test.pdf,2026-10-10,berlaku,id,TBD\n'
        '"Invalid Row",BadYear,not_int,OJK,https://example.com/bad.pdf,2026-10-10,berlaku,id,TBD\n',
        encoding="utf-8",
    )

    records = load_manifest(manifest_csv)
    assert len(records) == 1
    assert records[0].doc_number == "POJK 1/2024"
    assert records[0].year == 2024
    assert records[0].filename == "POJK_1_2024.pdf"

    # Test updating and saving
    records[0].sha256 = "dummy_sha256_hash_12345"
    save_manifest(manifest_csv, records)

    reloaded = load_manifest(manifest_csv)
    assert len(reloaded) == 1
    assert reloaded[0].sha256 == "dummy_sha256_hash_12345"


def test_load_manifest_nonexistent() -> None:
    """Verify load_manifest raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError, match="Corpus manifest not found"):
        load_manifest(Path("nonexistent/manifest.csv"))


def test_ensure_session_cookies_ojk() -> None:
    """Verify session cookie warm-up for ojk.go.id domains."""
    warmup_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal warmup_called
        if "ojk.go.id" in str(request.url):
            warmup_called = True
            return httpx.Response(status_code=200, headers={"Set-Cookie": "ojk_session=123"})
        return httpx.Response(status_code=404)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        ensure_session_cookies(client, "https://www.ojk.go.id/id/regulasi/doc.pdf")
        assert warmup_called is True

    # Test error handling during cookie warm-up
    def error_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection failed")

    err_transport = httpx.MockTransport(error_handler)
    with httpx.Client(transport=err_transport) as client:
        # Should catch and log without raising exception
        ensure_session_cookies(client, "https://www.ojk.go.id/id/regulasi/doc.pdf")


def test_download_single_file_success(tmp_path: Path) -> None:
    """Test successful single file download via mock transport."""
    dest = tmp_path / "test_doc.pdf"
    pdf_content = b"%PDF-1.5 test document stream"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/doc.pdf"
        return httpx.Response(
            status_code=200,
            content=pdf_content,
            headers={"Content-Type": "application/pdf"},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        success = download_single_file(
            url="https://example.com/doc.pdf",
            dest_path=dest,
            client=client,
            max_retries=1,
        )

    assert success is True
    assert dest.exists()
    assert dest.read_bytes() == pdf_content


def test_download_single_file_404_failure(tmp_path: Path) -> None:
    """Test 404 response handling during download."""
    dest = tmp_path / "missing_doc.pdf"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=404)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        success = download_single_file(
            url="https://example.com/notfound.pdf",
            dest_path=dest,
            client=client,
            max_retries=1,
        )

    assert success is False
    assert not dest.exists()


def test_download_single_file_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify download retry logic on network errors."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    dest = tmp_path / "network_fail.pdf"

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network unreachable")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        success = download_single_file(
            url="https://example.com/fail.pdf",
            dest_path=dest,
            client=client,
            max_retries=2,
        )

    assert success is False
    assert not dest.exists()


def test_download_corpus_orchestration(tmp_path: Path) -> None:
    """Test full download_corpus flow including caching and hash update."""
    manifest_csv = tmp_path / "manifest.csv"
    raw_dir = tmp_path / "raw"

    content_a = b"%PDF-1.4 Document A"
    content_b = b"%PDF-1.4 Document B"
    hash_a = hashlib.sha256(content_a).hexdigest()

    # Pre-create file A to test caching behavior
    (raw_dir).mkdir(parents=True)
    file_a = raw_dir / "POJK_1_2024.pdf"
    file_a.write_bytes(content_a)

    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "doc_number",
                "year",
                "issuer",
                "url",
                "retrieval_date",
                "status",
                "language",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Doc A",
                "doc_number": "POJK 1/2024",
                "year": "2024",
                "issuer": "OJK",
                "url": "https://example.com/a.pdf",
                "retrieval_date": "2026-10-10",
                "status": "berlaku",
                "language": "id",
                "sha256": hash_a,
            }
        )
        writer.writerow(
            {
                "title": "Doc B",
                "doc_number": "POJK 2/2024",
                "year": "2024",
                "issuer": "OJK",
                "url": "https://example.com/b.pdf",
                "retrieval_date": "2026-10-10",
                "status": "berlaku",
                "language": "id",
                "sha256": "TBD",
            }
        )

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == "https://example.com/b.pdf":
            return httpx.Response(status_code=200, content=content_b)
        return httpx.Response(status_code=404)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as mock_client:
        summary = download_corpus(
            manifest_path=manifest_csv,
            output_dir=raw_dir,
            update_hashes=True,
            client=mock_client,
        )

    assert summary["total"] == 2
    assert summary["cached"] == 1
    assert summary["downloaded"] == 1
    assert summary["failed"] == 0

    # Verify Doc B file was created and manifest hash was updated
    file_b = raw_dir / "POJK_2_2024.pdf"
    assert file_b.exists()
    assert file_b.read_bytes() == content_b

    updated_records = load_manifest(manifest_csv)
    hash_b = hashlib.sha256(content_b).hexdigest()
    assert updated_records[1].sha256 == hash_b


def test_download_corpus_hash_mismatch(tmp_path: Path) -> None:
    """Verify detection of SHA-256 hash mismatch when update_hashes=False."""
    manifest_csv = tmp_path / "manifest.csv"
    raw_dir = tmp_path / "raw"
    pdf_content = b"%PDF-1.4 Mock content mismatch"

    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "doc_number",
                "year",
                "issuer",
                "url",
                "retrieval_date",
                "status",
                "language",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Mismatch Doc",
                "doc_number": "POJK 99/2024",
                "year": "2024",
                "issuer": "OJK",
                "url": "https://example.com/mismatch.pdf",
                "retrieval_date": "2026-10-10",
                "status": "berlaku",
                "language": "id",
                "sha256": "expected_but_different_sha256_hash",
            }
        )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, content=pdf_content)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as mock_client:
        summary = download_corpus(
            manifest_path=manifest_csv,
            output_dir=raw_dir,
            update_hashes=False,
            client=mock_client,
        )

    assert summary["total"] == 1
    assert summary["downloaded"] == 1
    assert len(summary["errors"]) == 1
    assert "Hash mismatch" in summary["errors"][0]


def test_download_corpus_existing_tbd_and_default_client(tmp_path: Path) -> None:
    """Verify existing file with TBD hash is updated and default client is closed."""
    manifest_csv = tmp_path / "manifest.csv"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    pdf_content = b"%PDF-1.4 Existing cached file content"
    target_file = raw_dir / "POJK_50_2024.pdf"
    target_file.write_bytes(pdf_content)
    expected_hash = hashlib.sha256(pdf_content).hexdigest()

    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "doc_number",
                "year",
                "issuer",
                "url",
                "retrieval_date",
                "status",
                "language",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Cached Doc",
                "doc_number": "POJK 50/2024",
                "year": "2024",
                "issuer": "OJK",
                "url": "https://example.com/cached.pdf",
                "retrieval_date": "2026-10-10",
                "status": "berlaku",
                "language": "id",
                "sha256": "TBD",
            }
        )

    # client=None to test default httpx.Client creation and closing
    summary = download_corpus(
        manifest_path=manifest_csv,
        output_dir=raw_dir,
        update_hashes=True,
        client=None,
    )

    assert summary["total"] == 1
    assert summary["cached"] == 1
    records = load_manifest(manifest_csv)
    assert records[0].sha256 == expected_hash


def test_download_corpus_failed_download_tracked(tmp_path: Path) -> None:
    """Verify summary tracks failed downloads properly."""
    manifest_csv = tmp_path / "manifest.csv"
    raw_dir = tmp_path / "raw"

    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "doc_number",
                "year",
                "issuer",
                "url",
                "retrieval_date",
                "status",
                "language",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Fail Doc",
                "doc_number": "POJK 77/2024",
                "year": "2024",
                "issuer": "OJK",
                "url": "https://example.com/fail.pdf",
                "retrieval_date": "2026-10-10",
                "status": "berlaku",
                "language": "id",
                "sha256": "TBD",
            }
        )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=404)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as mock_client:
        summary = download_corpus(
            manifest_path=manifest_csv,
            output_dir=raw_dir,
            update_hashes=False,
            client=mock_client,
        )

    assert summary["total"] == 1
    assert summary["failed"] == 1
    assert len(summary["errors"]) == 1


def test_main_cli_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify CLI main entrypoint succeeds when no failures occur."""
    dummy_summary = {
        "total": 1,
        "downloaded": 1,
        "cached": 0,
        "failed": 0,
        "verified_hashes": 1,
        "errors": [],
    }
    manifest_file = tmp_path / "manifest.csv"
    output_dir = tmp_path / "raw"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download.py",
            "--manifest",
            str(manifest_file),
            "--output-dir",
            str(output_dir),
            "--update-hashes",
            "--timeout",
            "10.0",
        ],
    )
    with patch("easypasal.ingest.download.download_corpus", return_value=dummy_summary):
        main()


def test_main_cli_failure_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify CLI main entrypoint exits with code 1 when failures occur."""
    dummy_summary = {
        "total": 1,
        "downloaded": 0,
        "cached": 0,
        "failed": 1,
        "verified_hashes": 0,
        "errors": ["Failed to download doc"],
    }
    manifest_file = tmp_path / "manifest.csv"
    output_dir = tmp_path / "raw"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download.py",
            "--manifest",
            str(manifest_file),
            "--output-dir",
            str(output_dir),
        ],
    )
    with patch("easypasal.ingest.download.download_corpus", return_value=dummy_summary):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_download_single_file_non_pdf_rejection(tmp_path: Path) -> None:
    """Test rejection when server returns HTML instead of PDF magic bytes."""
    dest = tmp_path / "fake.pdf"
    html_content = b"<html><body>Not a PDF</body></html>"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=html_content,
            headers={"Content-Type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        success = download_single_file(
            url="https://example.com/fake.pdf",
            dest_path=dest,
            client=client,
            max_retries=1,
        )

    assert success is False
    assert not dest.exists()


def test_package_exports() -> None:
    """Verify public symbols accessible via easypasal.ingest."""
    import easypasal.ingest as ingest

    assert hasattr(ingest, "ManifestRecord")
    assert hasattr(ingest, "compute_sha256")
    assert hasattr(ingest, "download_corpus")
    assert hasattr(ingest, "download_single_file")
    assert hasattr(ingest, "load_manifest")
