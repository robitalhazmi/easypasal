"""Unit tests for structure-aware and baseline fixed-size chunkers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easypasal.ingest.chunker import (
    Chunk,
    FixedSizeChunker,
    StructureAwareChunker,
    build_breadcrumb,
    chunk_corpus,
    chunk_document,
    estimate_tokens,
    is_catchword,
    main,
    split_oversized,
)
from easypasal.ingest.parser import ParsedDocument


@pytest.fixture
def sample_statute_doc() -> ParsedDocument:
    """Creates a sample statutory regulation with hierarchy, clauses, and definitions."""
    text = (
        "SALINAN\n"
        "PERATURAN OTORITAS JASA KEUANGAN\n"
        "NOMOR 22 TAHUN 2023\n"
        "TENTANG PELINDUNGAN KONSUMEN\n"
        "Menimbang : a. bahwa ...\n"
        "Mengingat : 1. UU ...\n"
        "MEMUTUSKAN:\n"
        "Menetapkan : PERATURAN ...\n"
        "BAB I\n"
        "KETENTUAN UMUM\n"
        "Pasal 1\n"
        "Dalam Peraturan ini yang dimaksud dengan:\n"
        "1. Lembaga Jasa Keuangan adalah lembaga perbankan dan pasar modal.\n"
        "2. Konsumen adalah pihak yang memanfaatkan produk keuangan.\n"
        "Pasal 2\n"
        "PUJK berhak mendapatkan informasi yang akurat dari calon konsumen.\n"
        "BAB II\n"
        "HAK DAN KEWAJIBAN\n"
        "Bagian Kesatu\n"
        "Prinsip Pelindungan\n"
        "Paragraf 1\n"
        "Umum\n"
        "Pasal 3\n"
        "(1) PUJK wajib menerapkan prinsip transparansi.\n"
        "(2) Transparansi sebagaimana dimaksud pada ayat (1) meliputi:\n"
        "a. informasi produk;\n"
        "b. syarat dan ketentuan biaya.\n"
        "PENJELASAN\n"
        "ATAS PERATURAN OTORITAS JASA KEUANGAN NOMOR 22 TAHUN 2023\n"
        "I. UMUM\n"
        "Sektor jasa keuangan memegang peranan krusial bagi perekonomian nasional.\n"
        "II. PASAL DEMI PASAL\n"
        "Pasal 1\n"
        "Cukup jelas.\n"
        "Pasal 2\n"
        "Cukup jelas.\n"
        "Pasal 3\n"
        "Ayat (1)\n"
        "Yang dimaksud dengan prinsip transparansi adalah keterbukaan informasi.\n"
        "Ayat (2)\n"
        "Cukup jelas.\n"
    )
    return ParsedDocument(
        doc_id="POJK_22_Tahun_2023",
        filename="POJK_22_Tahun_2023.pdf",
        total_pages=5,
        clean_text=text,
        metadata={
            "doc_number": "POJK 22 Tahun 2023",
            "title": "Pelindungan Konsumen dan Masyarakat",
            "year": "2023",
            "issuer": "OJK",
            "status": "berlaku",
            "url": "https://example.com/pojk22.pdf",
        },
    )


@pytest.fixture
def sample_seojk_doc() -> ParsedDocument:
    """Creates a sample circular letter (SEOJK) without articles."""
    text = (
        "SURAT EDARAN OTORITAS JASA KEUANGAN\n"
        "NOMOR 2/SEOJK.07/2014\n"
        "TENTANG PELAYANAN PENGADUAN\n"
        "Sehubungan dengan POJK Nomor 1/2013, diatur ketentuan sebagai berikut:\n"
        "I. KETENTUAN UMUM\n"
        "1. Penanganan Pengaduan adalah pelayanan pengaduan konsumen.\n"
        "2. PUJK wajib menyediakan unit pengaduan.\n"
        "II. MEKANISME PELAYANAN\n"
        "1. Pengaduan dapat disampaikan secara lisan maupun tertulis.\n"
        "2. PUJK wajib menindaklanjuti pengaduan paling lambat 20 hari kerja.\n"
        "III. PENUTUP\n"
        "Surat Edaran ini mulai berlaku pada tanggal ditetapkan.\n"
    )
    return ParsedDocument(
        doc_id="SEOJK_2_SEOJK.07_2014",
        filename="SEOJK_2_SEOJK.07_2014.pdf",
        total_pages=2,
        clean_text=text,
        metadata={
            "doc_number": "SEOJK 2/SEOJK.07/2014",
            "title": "Pelayanan dan Penyelesaian Pengaduan",
            "year": "2014",
            "issuer": "OJK",
            "status": "berlaku",
            "url": "https://example.com/seojk2.pdf",
        },
    )


def test_chunk_dataclass_serialization() -> None:
    """Test Chunk serialization to and from dictionary."""
    chunk = Chunk(
        chunk_id="test_doc_pasal_1_ayat_1",
        doc_id="test_doc",
        doc_number="UU 1 Tahun 2024",
        doc_title="Uji Coba",
        year=2024,
        issuer="Pemerintah RI",
        status="berlaku",
        source_url="https://example.com/test.pdf",
        chunk_type="structure_aware",
        text="[UU 1 Tahun 2024 | Pasal 1 ayat (1)]\n(1) Ketentuan...",
        raw_content="(1) Ketentuan...",
        bab="BAB I",
        bab_title="KETENTUAN UMUM",
        pasal="1",
        ayat="1",
        citation_label="UU 1 Tahun 2024, Pasal 1 ayat (1)",
        char_count=52,
        token_count_est=13,
    )

    d = chunk.to_dict()
    assert d["chunk_id"] == "test_doc_pasal_1_ayat_1"
    assert d["pasal"] == "1"
    assert d["ayat"] == "1"
    assert d["citation_label"] == "UU 1 Tahun 2024, Pasal 1 ayat (1)"

    restored = Chunk.from_dict(d)
    assert restored == chunk


def test_estimate_tokens() -> None:
    """Test token estimation heuristic."""
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_build_breadcrumb() -> None:
    """Test breadcrumb string formation."""
    bc = build_breadcrumb(
        doc_number="POJK 22 Tahun 2023",
        bab="BAB II",
        bab_title="KETENTUAN UMUM",
        bagian="Bagian Kesatu",
        paragraf="Paragraf 1",
        pasal="3",
        ayat="1",
    )
    assert bc == (
        "POJK 22 Tahun 2023 | BAB II KETENTUAN UMUM | Bagian Kesatu | Paragraf 1 | Pasal 3 ayat (1)"
    )

    # Penjelasan breadcrumb
    bc_penj = build_breadcrumb(
        doc_number="POJK 22 Tahun 2023",
        pasal="3",
        ayat="2",
        is_penjelasan=True,
    )
    assert bc_penj == "Penjelasan POJK 22 Tahun 2023 | Pasal 3 ayat (2)"


def test_catchword_detection() -> None:
    """Test detection of page-edge catchwords."""
    assert is_catchword("(3) Rencana...")
    assert is_catchword(" (2) Lembaga... ")
    assert not is_catchword("(3) Rencana kerja dan anggaran tahunan wajib disusun.")


def test_structure_aware_statute_chunking(sample_statute_doc: ParsedDocument) -> None:
    """Test structure-aware chunking on multi-article regulation."""
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(sample_statute_doc)

    assert len(chunks) > 0

    # Every chunk must fulfill acceptance criteria fields
    for c in chunks:
        assert c.doc_id == "POJK_22_Tahun_2023"
        assert c.status == "berlaku"
        assert c.source_url == "https://example.com/pojk22.pdf"
        assert c.citation_label
        assert c.chunk_type == "structure_aware"
        assert c.char_count == len(c.text)

    # 1. Preamble chunk
    preamble = [c for c in chunks if c.chunk_id.endswith("pembukaan")]
    assert len(preamble) == 1
    assert "Pembukaan" in preamble[0].citation_label

    # 2. Definition item chunks (Pasal 1)
    p1_chunks = [
        c
        for c in chunks
        if c.pasal == "1" and not c.chunk_id.startswith("POJK_22_Tahun_2023_penjelasan")
    ]
    assert len(p1_chunks) == 2
    assert p1_chunks[0].citation_label == "POJK 22 Tahun 2023, Pasal 1 angka 1"
    assert "Lembaga Jasa Keuangan" in p1_chunks[0].raw_content
    assert p1_chunks[1].citation_label == "POJK 22 Tahun 2023, Pasal 1 angka 2"

    # 3. Unnumbered single-clause chunk (Pasal 2)
    p2_chunks = [
        c
        for c in chunks
        if c.pasal == "2" and not c.chunk_id.startswith("POJK_22_Tahun_2023_penjelasan")
    ]
    assert len(p2_chunks) == 1
    assert p2_chunks[0].ayat is None
    assert p2_chunks[0].citation_label == "POJK 22 Tahun 2023, Pasal 2"

    # 4. Multi-clause chunk with Ayat (Pasal 3)
    p3_chunks = [
        c
        for c in chunks
        if c.pasal == "3" and not c.chunk_id.startswith("POJK_22_Tahun_2023_penjelasan")
    ]
    assert len(p3_chunks) == 2
    assert p3_chunks[0].ayat == "1"
    assert p3_chunks[0].citation_label == "POJK 22 Tahun 2023, Pasal 3 ayat (1)"
    assert p3_chunks[0].bab == "BAB II"
    assert p3_chunks[0].bagian == "Bagian Kesatu"
    assert p3_chunks[0].paragraf == "Paragraf 1"
    assert p3_chunks[1].ayat == "2"
    assert p3_chunks[1].citation_label == "POJK 22 Tahun 2023, Pasal 3 ayat (2)"
    assert "Transparansi sebagaimana dimaksud" in p3_chunks[1].raw_content

    # 5. Penjelasan chunks
    penj_chunks = [c for c in chunks if "penjelasan" in c.chunk_id]
    assert len(penj_chunks) >= 4
    penj_umum = [c for c in penj_chunks if c.chunk_id == "POJK_22_Tahun_2023_penjelasan_umum"]
    assert len(penj_umum) == 1
    assert penj_umum[0].citation_label == "Penjelasan POJK 22 Tahun 2023, I. Umum"

    penj_p3_a1 = [
        c for c in penj_chunks if c.chunk_id == "POJK_22_Tahun_2023_penjelasan_pasal_3_ayat_1"
    ]
    assert len(penj_p3_a1) == 1
    assert penj_p3_a1[0].citation_label == "Penjelasan POJK 22 Tahun 2023, Pasal 3 ayat (1)"
    assert "keterbukaan informasi" in penj_p3_a1[0].raw_content


def test_structure_aware_seojk_chunking(sample_seojk_doc: ParsedDocument) -> None:
    """Test structure-aware chunking on circular letter (SEOJK)."""
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(sample_seojk_doc)

    assert len(chunks) > 0

    # Check Roman sections
    rom1 = [c for c in chunks if c.bab == "Romawi I"]
    assert len(rom1) == 2
    assert rom1[0].citation_label == "SEOJK 2/SEOJK.07/2014, Romawi I angka 1"
    assert rom1[0].pasal is None
    assert rom1[0].ayat is None

    rom2 = [c for c in chunks if c.bab == "Romawi II"]
    assert len(rom2) == 2
    assert rom2[1].citation_label == "SEOJK 2/SEOJK.07/2014, Romawi II angka 2"
    assert "20 hari kerja" in rom2[1].raw_content


def test_split_oversized() -> None:
    """Test splitting text exceeding maximum character thresholds."""
    short_text = "Singkat saja."
    assert split_oversized(short_text, max_chars=100) == [short_text]

    # Paragraph-based split
    p1 = "Paragraf satu " * 10  # ~140 chars
    p2 = "Paragraf dua " * 10  # ~130 chars
    long_text = f"{p1}\n\n{p2}"
    parts = split_oversized(long_text, max_chars=160, overlap=20)
    assert len(parts) == 2
    assert parts[0] == p1.strip()
    assert parts[1] == p2.strip()

    # Giant single block without paragraphs
    giant = "A" * 500
    parts_giant = split_oversized(giant, max_chars=200, overlap=50)
    assert len(parts_giant) == 3
    assert len(parts_giant[0]) == 200
    assert len(parts_giant[1]) == 200
    assert len(parts_giant[2]) == 200


def test_fixed_size_chunker() -> None:
    """Test fixed-size sliding window baseline chunker."""
    text = "Kalimat pertama. Kalimat kedua. Kalimat ketiga. Kalimat keempat. Kalimat kelima."
    doc = ParsedDocument(
        doc_id="test_doc",
        filename="test.pdf",
        total_pages=1,
        clean_text=text,
        metadata={"doc_number": "Test Doc 2024", "url": "https://example.com/test.pdf"},
    )

    chunker = FixedSizeChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_type == "fixed_size"
        assert c.chunk_id == f"test_doc_fix_{i + 1:04d}"
        assert c.citation_label == f"Test Doc 2024, Chunk #{i + 1}"
        assert c.char_count <= 40

    # Overlap validation error
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        FixedSizeChunker(chunk_size=50, chunk_overlap=50)


def test_chunk_document_dispatch(sample_statute_doc: ParsedDocument) -> None:
    """Test chunk_document helper function with both strategies."""
    struct_chunks = chunk_document(sample_statute_doc, strategy="structure_aware")
    assert all(c.chunk_type == "structure_aware" for c in struct_chunks)

    fix_chunks = chunk_document(sample_statute_doc, strategy="fixed_size", chunk_size=300)
    assert all(c.chunk_type == "fixed_size" for c in fix_chunks)

    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        chunk_document(sample_statute_doc, strategy="invalid_strategy")


def test_chunk_corpus_jsonl(tmp_path: Path, sample_statute_doc: ParsedDocument) -> None:
    """Test chunk_corpus end-to-end execution and JSONL output creation."""
    parsed_dir = tmp_path / "parsed"
    chunks_dir = tmp_path / "chunks"
    parsed_dir.mkdir()

    # Write sample parsed document
    sample_file = parsed_dir / f"{sample_statute_doc.doc_id}.json"
    sample_file.write_text(json.dumps(sample_statute_doc.to_dict()), encoding="utf-8")

    # Run structure_aware
    chunks_s = chunk_corpus(parsed_dir, chunks_dir, strategy="structure_aware")
    assert len(chunks_s) > 0
    s_file = chunks_dir / "structure_aware.jsonl"
    assert s_file.exists()

    with open(s_file, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == len(chunks_s)
    assert lines[0]["chunk_id"] == chunks_s[0].chunk_id

    # Run fixed_size
    chunks_f = chunk_corpus(parsed_dir, chunks_dir, strategy="fixed_size")
    assert len(chunks_f) > 0
    f_file = chunks_dir / "fixed_size.jsonl"
    assert f_file.exists()

    # Run structure_aware_no_header
    chunks_nh = chunk_corpus(parsed_dir, chunks_dir, strategy="structure_aware_no_header")
    assert len(chunks_nh) == len(chunks_s)
    nh_file = chunks_dir / "structure_aware_no_header.jsonl"
    assert nh_file.exists()


def test_structure_aware_chunker_no_breadcrumbs(sample_statute_doc: ParsedDocument) -> None:
    """Test that include_breadcrumbs=False strips breadcrumb prefixes from text."""
    chunker_with = StructureAwareChunker(include_breadcrumbs=True)
    chunks_with = chunker_with.chunk(sample_statute_doc)

    chunker_no = StructureAwareChunker(include_breadcrumbs=False)
    chunks_no = chunker_no.chunk(sample_statute_doc)

    assert len(chunks_with) == len(chunks_no)

    for c_with, c_no in zip(chunks_with, chunks_no, strict=True):
        assert c_with.chunk_id == c_no.chunk_id
        assert c_with.raw_content == c_no.raw_content
        # Breadcrumbs format: "[...]\n"
        assert c_with.text.startswith("[")
        assert not c_no.text.startswith("[")
        assert c_no.text == c_no.raw_content


def test_chunk_document_structure_aware_no_header(sample_statute_doc: ParsedDocument) -> None:
    """Test chunk_document dispatch with structure_aware_no_header."""
    chunks = chunk_document(sample_statute_doc, strategy="structure_aware_no_header")
    assert len(chunks) > 0
    assert all(c.chunk_type == "structure_aware" for c in chunks)
    assert not chunks[0].text.startswith("[")


def test_chunker_empty_document() -> None:
    """Verify empty document clean_text returns empty chunk list."""
    empty_doc = ParsedDocument(
        doc_id="empty",
        filename="empty.pdf",
        total_pages=1,
        clean_text="",
        metadata={"year": "invalid_year"},
    )
    s_chunker = StructureAwareChunker()
    assert s_chunker.chunk(empty_doc) == []

    f_chunker = FixedSizeChunker()
    assert f_chunker.chunk(empty_doc) == []


def test_chunker_duplicate_id_handling() -> None:
    """Verify duplicate chunk IDs receive sequential count suffixes."""
    text = (
        "\nPasal 1\n(1) Aturan pertama.\n"
        "\nPasal 1\n(1) Aturan kedua yang memiliki penomoran identik.\n"
    )
    doc = ParsedDocument(
        doc_id="dup_doc",
        filename="dup.pdf",
        total_pages=1,
        clean_text=text,
        metadata={"year": 2024},
    )
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "dup_doc_pasal_1_ayat_1"
    assert chunks[1].chunk_id == "dup_doc_pasal_1_ayat_1_2"


def test_chunk_corpus_empty_and_error(tmp_path: Path) -> None:
    """Verify chunk_corpus handles empty directories and malformed JSON files."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "out"
    assert chunk_corpus(empty_dir, out_dir) == []

    # Corrupt JSON file
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "bad.json").write_text("invalid json syntax", encoding="utf-8")
    assert chunk_corpus(bad_dir, out_dir) == []


def test_main_cli_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify CLI runner handles arguments for all strategies."""
    from unittest.mock import patch

    monkeypatch.setattr(
        "sys.argv",
        [
            "chunker.py",
            "--strategy",
            "all",
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "chunks"),
        ],
    )
    with patch("easypasal.ingest.chunker.chunk_corpus") as mock_chunk:
        main()
        assert mock_chunk.call_count == 3


def test_amendment_quoted_pasal_chunking() -> None:
    """Verify that amendment laws with quoted Pasal and amendment transitions chunk accurately."""
    amendment_text = (
        "Pasal I\n"
        "Mengubah beberapa ketentuan sebagai berikut:\n"
        "16. Ketentuan Pasal 22 diubah sehingga menjadi berbunyi sebagai berikut:\n"
        '"Pasal 22\n'
        "(1) Bank Umum hanya dapat didirikan oleh WNI.\n"
        '(2) Ketentuan pendirian diatur oleh Bank Indonesia."\n'
        "17. Ketentuan Pasal 26 diubah sebagai berikut:\n"
        '"Pasal 26\n'
        '(1) Bank dapat melakukan emisi saham."\n'
        "Pasal II\n"
        "Undang-Undang ini mulai berlaku pada tanggal diundangkan.\n"
    )
    doc = ParsedDocument(
        doc_id="UU_Amend_1998",
        filename="UU_Amend_1998.pdf",
        total_pages=2,
        clean_text=amendment_text,
        metadata={"doc_number": "UU Amend 1998", "year": 1998},
    )
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc)

    p22_chunks = [c for c in chunks if c.pasal == "22"]
    assert len(p22_chunks) == 2
    assert p22_chunks[0].chunk_id == "UU_Amend_1998_pasal_22_ayat_1"
    assert p22_chunks[0].citation_label == "UU Amend 1998, Pasal 22 ayat (1)"
    assert p22_chunks[0].raw_content == "(1) Bank Umum hanya dapat didirikan oleh WNI."
    assert p22_chunks[0].bab is None  # Legitimate null metadata

    p26_chunks = [c for c in chunks if c.pasal == "26"]
    assert len(p26_chunks) == 1
    assert p26_chunks[0].chunk_id == "UU_Amend_1998_pasal_26_ayat_1"
    assert "emisi saham" in p26_chunks[0].raw_content


def test_ocr_clause_normalization() -> None:
    """Verify that OCR artifacts like '(21 ' without newline and 'Ayat (21' in explanations are normalized into atomic clauses."""
    statute_text = (
        "BAB I\n"
        "KETENTUAN UMUM\n"
        "Pasal 245\n"
        "(1) PUSK wajib memiliki dan melaksanakan mekanisme penanganan pengaduan yang disampaikan oleh Konsumen. "
        "(21 Dalam hal tidak terdapat kesepakatan terhadap hasil penanganan pengaduan, Konsumen dapat:\n"
        "a. menyampaikan pengaduan kepada otoritas;\n"
        "b. mengajukan sengketa kepada badan penyelesaian sengketa.\n"
        "\nPENJELASAN ATAS UNDANG-UNDANG\n"
        "Pasal 245\n"
        "Ayat (1)\n"
        "Cukup jelas.\n"
        "Ayat (21\n"
        "Yang dimaksud dengan sengketa adalah sengketa perdata.\n"
    )
    doc = ParsedDocument(
        doc_id="UU_Test_2023",
        filename="UU_Test_2023.pdf",
        total_pages=2,
        clean_text=statute_text,
        metadata={"doc_number": "UU Nomor 4 Tahun 2023", "year": 2023},
    )
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc)

    c_ayat1 = [c for c in chunks if c.chunk_id == "UU_Test_2023_pasal_245_ayat_1"]
    c_ayat2 = [c for c in chunks if c.chunk_id == "UU_Test_2023_pasal_245_ayat_2"]
    assert len(c_ayat1) == 1
    assert len(c_ayat2) == 1
    assert "(21" not in c_ayat1[0].raw_content
    assert c_ayat2[0].raw_content.startswith("(2) Dalam hal tidak terdapat kesepakatan")

    c_penj2 = [c for c in chunks if c.chunk_id == "UU_Test_2023_penjelasan_pasal_245_ayat_2"]
    assert len(c_penj2) == 1
    assert "Yang dimaksud dengan sengketa" in c_penj2[0].raw_content
