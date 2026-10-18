"""Unit tests for PyMuPDF PDF parsing, header/footer stripping, and dehyphenation."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from easypasal.ingest.parser import (
    ParsedPage,
    dehyphenate_text,
    is_footer_block,
    is_header_block,
    normalize_whitespace,
    parse_corpus,
    parse_pdf,
    spot_check_documents,
    unwrap_legal_text,
)


def test_dehyphenate_text() -> None:
    """Verify Indonesian dehyphenation rules for compounds and single-word splits."""
    # Reduplications
    assert dehyphenate_text("undang-\nundang") == "undang-undang"
    assert dehyphenate_text("masing-\nmasing") == "masing-masing"
    assert dehyphenate_text("prinsip-\nprinsip") == "prinsip-prinsip"

    # Known legal compounds
    assert dehyphenate_text("perundang-\nundangan") == "perundang-undangan"
    assert dehyphenate_text("tata-\ntertib") == "tata-tertib"

    # Regular single-word line-wrap splits
    assert dehyphenate_text("keuang-\nan") == "keuangan"
    assert dehyphenate_text("pelak-\nsanaan") == "pelaksanaan"
    assert dehyphenate_text("menyelenggara-\nkan") == "menyelenggarakan"


def test_normalize_whitespace() -> None:
    """Verify non-breaking space replacement and multiple space reduction."""
    raw = "Kata\u00a0pertama   dan    kedua.\n\nBaris   baru."
    normalized = normalize_whitespace(raw)
    assert "\u00a0" not in normalized
    assert "Kata pertama dan kedua." in normalized
    assert "Baris baru." in normalized


def test_unwrap_legal_text() -> None:
    """Verify unwrapping lines into sentences while strictly preserving headings."""
    raw = (
        "BAB I\n"
        "KETENTUAN UMUM\n"
        "Pasal 1\n"
        "Dalam Peraturan Otoritas Jasa Keuangan\n"
        "ini yang dimaksud dengan:\n"
        "1. Konsumen adalah setiap orang\n"
        "yang menggunakan produk."
    )
    unwrapped = unwrap_legal_text(raw)
    lines = unwrapped.splitlines()

    assert "BAB I" in lines
    assert "Pasal 1" in lines
    assert "Dalam Peraturan Otoritas Jasa Keuangan ini yang dimaksud dengan:" in lines
    assert "1. Konsumen adalah setiap orang yang menggunakan produk." in lines


def test_header_footer_detection() -> None:
    """Verify running headers, page numbers, security numbers, and footers are identified."""
    page_height = 1000.0

    # Header checks (top margin <= 220)
    assert is_header_block("- 2 -", y1=50.0, page_height=page_height, page_idx=1) is True
    assert is_header_block("12", y1=50.0, page_height=page_height, page_idx=1) is True
    assert (
        is_header_block("2022, No.5/OJK \n-2-", y1=60.0, page_height=page_height, page_idx=1)
        is True
    )
    assert is_header_block("SALINAN", y1=70.0, page_height=page_height, page_idx=1) is True

    # First page headers should not be stripped
    assert is_header_block("PERATURAN OJK", y1=150.0, page_height=page_height, page_idx=0) is False

    # Footer checks (bottom margin >= 850)
    assert is_footer_block("- 2 -", y0=950.0, page_height=page_height) is True
    assert is_footer_block("SK No 017000 A", y0=960.0, page_height=page_height) is True
    assert is_footer_block("www.ojk.go.id", y0=970.0, page_height=page_height) is True
    assert is_footer_block("5. Prosesor . . .", y0=940.0, page_height=page_height) is True

    # Normal body content in middle of page should not be detected as header/footer
    assert is_header_block("Pasal 1", y1=300.0, page_height=page_height, page_idx=1) is False
    assert is_footer_block("Pasal 1", y0=700.0, page_height=page_height) is False


def create_synthetic_pdf(output_path: Path) -> None:
    """Helper to generate a multi-page test PDF document."""
    doc = pymupdf.open()

    # Page 1
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        pymupdf.Point(50, 100),
        "PERATURAN OTORITAS JASA KEUANGAN\nNOMOR 99 TAHUN 2024",
        fontsize=14,
    )
    p1.insert_text(
        pymupdf.Point(50, 200),
        "Pasal 1\nDalam Peraturan Otoritas Jasa Keuangan ini yang diatur.",
        fontsize=11,
    )

    # Page 2
    p2 = doc.new_page(width=595, height=842)
    # Header block (at y=40, y1 < 0.22 * 842)
    p2.insert_text(pymupdf.Point(280, 40), "- 2 -", fontsize=10)
    # Body with hyphen split
    p2.insert_text(
        pymupdf.Point(50, 150),
        "Pasal 2\n(1) Penyelenggaraan jasa keuang-\nan wajib mematuhi aturan.",
        fontsize=11,
    )
    # Footer block (at y=800, y0 > 0.85 * 842)
    p2.insert_text(pymupdf.Point(250, 800), "www.ojk.go.id", fontsize=9)

    doc.save(str(output_path))
    doc.close()


def test_parse_pdf_end_to_end(tmp_path: Path) -> None:
    """Test full PDF parsing, verifying headers/footers stripped and dehyphenation applied."""
    test_pdf = tmp_path / "test_regulation.pdf"
    create_synthetic_pdf(test_pdf)

    parsed = parse_pdf(test_pdf)

    assert parsed.doc_id == "test_regulation"
    assert parsed.total_pages == 2
    assert len(parsed.pages) == 2

    # Verify header "- 2 -" is stripped from clean text
    assert "- 2 -" not in parsed.clean_text
    # Verify footer "www.ojk.go.id" is stripped
    assert "www.ojk.go.id" not in parsed.clean_text
    # Verify dehyphenation worked: "keuang-\nan" -> "keuangan"
    assert "keuangan" in parsed.clean_text

    # Verify structural items
    assert "Pasal 1" in parsed.clean_text
    assert "Pasal 2" in parsed.clean_text

    # Serialization test
    doc_dict = parsed.to_dict()
    assert doc_dict["doc_id"] == "test_regulation"
    assert doc_dict["total_pages"] == 2
    assert len(doc_dict["pages"]) == 2


def test_parse_pdf_file_not_found() -> None:
    """Verify FileNotFoundError is raised for non-existent PDF."""
    with pytest.raises(FileNotFoundError):
        parse_pdf(Path("non_existent_file.pdf"))


def test_parse_corpus_and_spot_check(tmp_path: Path) -> None:
    """Test parse_corpus batch processor and spot_check_documents utility."""
    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    raw_dir.mkdir()

    pdf_a = raw_dir / "DOC_A.pdf"
    create_synthetic_pdf(pdf_a)

    docs = parse_corpus(raw_dir=raw_dir, output_dir=parsed_dir)
    assert len(docs) == 1
    assert (parsed_dir / "DOC_A.json").exists()

    # Test spot-check
    results = spot_check_documents(["DOC_A.pdf"], raw_dir=raw_dir, output_dir=parsed_dir)
    assert "DOC_A.pdf" in results
    assert results["DOC_A.pdf"]["status"] == "success"
    assert results["DOC_A.pdf"]["pages"] == 2
    assert results["DOC_A.pdf"]["has_pasal"] is True
    assert results["DOC_A.pdf"]["header_stripped"] is True


def test_package_exports() -> None:
    """Verify parser public symbols accessible via easypasal.ingest."""
    import easypasal.ingest as ingest

    assert hasattr(ingest, "ParsedDocument")
    assert hasattr(ingest, "ParsedPage")
    assert hasattr(ingest, "parse_pdf")
    assert hasattr(ingest, "parse_corpus")
    assert hasattr(ingest, "dehyphenate_text")
    assert hasattr(ingest, "clean_page_text")
    assert hasattr(ingest, "unwrap_legal_text")


def test_parsed_page_to_dict() -> None:
    """Verify ParsedPage serializes correctly to dictionary."""
    page = ParsedPage(
        page_number=1,
        raw_text="raw",
        clean_text="clean",
        char_count=5,
        block_count=1,
    )
    d = page.to_dict()
    assert d["page_number"] == 1
    assert d["clean_text"] == "clean"
    assert d["char_count"] == 5


def test_empty_header_and_footer() -> None:
    """Verify empty string blocks in header/footer regions return True."""
    assert is_header_block("", y1=10.0, page_height=1000.0, page_idx=1) is True
    assert is_footer_block("", y0=900.0, page_height=1000.0) is True


def test_parse_corpus_with_manifest_and_limit(tmp_path: Path) -> None:
    """Verify parse_corpus attaches manifest metadata and respects document limits."""
    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    manifest_csv = tmp_path / "manifest.csv"
    raw_dir.mkdir()

    pdf_1 = raw_dir / "POJK_1_2024.pdf"
    pdf_2 = raw_dir / "POJK_2_2024.pdf"
    create_synthetic_pdf(pdf_1)
    create_synthetic_pdf(pdf_2)

    manifest_csv.write_text(
        "title,doc_number,year,issuer,url,retrieval_date,status,language,sha256\n"
        '"Doc 1",POJK 1/2024,2024,OJK,https://example.com/1.pdf,2026-10-10,berlaku,id,hash1\n',
        encoding="utf-8",
    )

    docs = parse_corpus(
        raw_dir=raw_dir,
        output_dir=parsed_dir,
        manifest_path=manifest_csv,
        limit=1,
    )
    assert len(docs) == 1
    assert docs[0].metadata.get("title") == "Doc 1"


def test_spot_check_missing_file(tmp_path: Path) -> None:
    """Verify spot_check_documents gracefully marks missing files."""
    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    raw_dir.mkdir()
    res = spot_check_documents(["nonexistent.pdf"], raw_dir=raw_dir, output_dir=parsed_dir)
    assert res["nonexistent.pdf"]["status"] == "missing"


def test_main_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify parser CLI handles batch and spot-check options."""
    from unittest.mock import patch

    from easypasal.ingest.parser import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "parser.py",
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--limit",
            "2",
        ],
    )
    with patch("easypasal.ingest.parser.parse_corpus", return_value=[]):
        main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "parser.py",
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--spot-check",
        ],
    )
    with patch(
        "easypasal.ingest.parser.spot_check_documents",
        return_value={"test.pdf": {"status": "ok"}},
    ):
        main()
