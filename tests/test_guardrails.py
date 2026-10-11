"""Tests for post-hoc citation verification, legal attribute parsing, and grounding checks."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from easypasal.generation.backends import MockGeneratorBackend
from easypasal.generation.pipeline import RAGPipeline
from easypasal.guardrails.citation_verifier import (
    ParsedCitation,
    VerificationResult,
    parse_citation_details,
    verify_citations,
    verify_single_citation,
)
from easypasal.retrieval.models import RetrievalResult, RetrievedChunk


def _make_chunk(
    chunk_id: str,
    doc_number: str,
    citation_label: str,
    pasal: str | None = None,
    ayat: str | None = None,
    text: str = "",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=0.035,
        citation_label=citation_label,
        doc_id=doc_number.replace(" ", "_"),
        doc_number=doc_number,
        pasal=pasal,
        ayat=ayat,
        text=text or f"Isi dari {citation_label}.",
    )


# ==============================================================================
# 1. PARSER UNIT TESTS
# ==============================================================================


def test_parse_citation_details() -> None:
    """Verify granular statutory extraction across various citation formats."""
    # Format A: POJK with Pasal and Ayat in brackets
    c1 = parse_citation_details("[POJK 22 Tahun 2023, Pasal 71 ayat (1)]")
    assert c1.doc_type == "POJK"
    assert c1.doc_number == "22 Tahun 2023"
    assert c1.pasal == "71"
    assert c1.ayat == "1"

    # Format B: UU with Pasal only
    c2 = parse_citation_details("UU Nomor 4 Tahun 2023, Pasal 243")
    assert c2.doc_type == "UU"
    assert c2.doc_number == "4 Tahun 2023"
    assert c2.pasal == "243"
    assert c2.ayat is None

    # Format C: Slash-formatted POJK
    c3 = parse_citation_details("POJK 12/POJK.03/2020, Pasal 3")
    assert c3.doc_type == "POJK"
    assert c3.doc_number == "12/POJK.03/2020"
    assert c3.pasal == "3"

    # Format D: Pasal with Angka
    c4 = parse_citation_details("POJK 8 Tahun 2023 Pasal 1 angka 4")
    assert c4.doc_type == "POJK"
    assert c4.doc_number == "8 Tahun 2023"
    assert c4.pasal == "1"
    assert isinstance(c1, ParsedCitation)
    assert c4.angka == "4"


def test_verify_single_citation_direct() -> None:
    """Directly test verify_single_citation function with match and mismatch."""
    chunk = _make_chunk(
        chunk_id="chunk_1",
        doc_number="POJK 22 Tahun 2023",
        citation_label="POJK 22 Tahun 2023, Pasal 71",
        pasal="71",
    )
    check = verify_single_citation("[POJK 22 Tahun 2023, Pasal 71]", [chunk])
    assert check.is_grounded is True
    assert check.matched_chunk_id == "chunk_1"

    check_bad = verify_single_citation("[POJK 99 Tahun 2099, Pasal 1]", [chunk])
    assert check_bad.is_grounded is False
    assert check_bad.doc_match is False


# ==============================================================================
# 2. PASS CASES
# ==============================================================================


def test_verify_citations_exact_match_pass() -> None:
    """PASS CASE: Exact match of regulation, pasal, and ayat."""
    chunk = _make_chunk(
        chunk_id="c_pojk22_71_1",
        doc_number="POJK 22 Tahun 2023",
        citation_label="POJK 22 Tahun 2023, Pasal 71 ayat (1)",
        pasal="71",
        ayat="1",
        text="(1) PUJK wajib menyelesaikan pengaduan konsumen paling lambat 10 hari kerja.",
    )

    answer = (
        "Berdasarkan [POJK 22 Tahun 2023, Pasal 71 ayat (1)], PUJK wajib menyelesaikan pengaduan."
    )
    res = verify_citations(answer=answer, retrieved_chunks=[chunk])

    assert isinstance(res, VerificationResult)
    assert res.is_valid is True
    assert res.total_citations == 1
    assert res.grounded_count == 1
    assert res.ungrounded_count == 0
    assert len(res.hallucinated_pasal) == 0
    assert res.checks[0].doc_match is True
    assert res.checks[0].pasal_match is True
    assert res.checks[0].ayat_match is True
    assert res.checks[0].matched_chunk_id == "c_pojk22_71_1"


def test_verify_citations_pasal_only_pass() -> None:
    """PASS CASE: Model cites Pasal without Ayat, matching chunk with that Pasal."""
    chunk = _make_chunk(
        chunk_id="c_pojk12_3",
        doc_number="POJK 12/POJK.03/2020",
        citation_label="POJK 12/POJK.03/2020, Pasal 3",
        pasal="3",
        text="Bank Umum wajib memenuhi Modal Inti minimum paling sedikit Rp3 triliun.",
    )

    answer = "Kewajiban modal diatur dalam [POJK 12/POJK.03/2020, Pasal 3]."
    res = verify_citations(answer=answer, retrieved_chunks=[chunk])

    assert res.is_valid is True
    assert res.checks[0].is_grounded is True
    assert res.checks[0].pasal_match is True
    assert res.checks[0].ayat_match is None


def test_verify_citations_multiple_grounded_pass() -> None:
    """PASS CASE: Multiple distinct citations all grounded in retrieved chunks."""
    c1 = _make_chunk("c1", "POJK 22 Tahun 2023", "POJK 22 Tahun 2023, Pasal 71", pasal="71")
    c2 = _make_chunk("c2", "UU Nomor 4 Tahun 2023", "UU Nomor 4 Tahun 2023, Pasal 243", pasal="243")

    answer = (
        "Berdasarkan [POJK 22 Tahun 2023, Pasal 71], PUJK menangani pengaduan. "
        "Hal ini ditegaskan dalam [UU Nomor 4 Tahun 2023, Pasal 243]."
    )
    res = verify_citations(answer=answer, retrieved_chunks=[c1, c2])

    assert res.is_valid is True
    assert res.total_citations == 2
    assert res.grounded_count == 2
    assert res.ungrounded_count == 0


def test_verify_citations_valid_abstention_pass() -> None:
    """PASS CASE: Model explicitly abstained without citing regulations."""
    chunk = _make_chunk("c1", "POJK 22 Tahun 2023", "POJK 22 Tahun 2023, Pasal 71", pasal="71")
    answer = (
        "Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut."
    )

    res = verify_citations(answer=answer, retrieved_chunks=[chunk])
    assert res.is_valid is True
    assert res.total_citations == 0
    assert res.ungrounded_count == 0


# ==============================================================================
# 3. FAIL CASES (HALLUCINATIONS)
# ==============================================================================


def test_verify_citations_hallucinated_pasal_fail() -> None:
    """FAIL CASE: Regulation is present, but the cited Pasal does NOT exist in retrieved chunks."""
    chunk = _make_chunk(
        chunk_id="c_pojk22_71",
        doc_number="POJK 22 Tahun 2023",
        citation_label="POJK 22 Tahun 2023, Pasal 71",
        pasal="71",
        text="PUJK wajib menangani pengaduan.",
    )

    # Model hallucinates Pasal 999
    answer = "Sanksi diatur secara tegas dalam [POJK 22 Tahun 2023, Pasal 999]."
    res = verify_citations(answer=answer, retrieved_chunks=[chunk])

    assert res.is_valid is False
    assert res.total_citations == 1
    assert res.grounded_count == 0
    assert res.ungrounded_count == 1
    assert res.checks[0].doc_match is True
    assert res.checks[0].pasal_match is False
    assert "POJK 22 Tahun 2023, Pasal 999" in res.ungrounded_citations
    assert any("Pasal 999" in h for h in res.hallucinated_pasal)


def test_verify_citations_hallucinated_regulation_fail() -> None:
    """FAIL CASE: Cited regulation is completely absent from retrieved chunks."""
    chunk = _make_chunk(
        chunk_id="c_pojk22_71",
        doc_number="POJK 22 Tahun 2023",
        citation_label="POJK 22 Tahun 2023, Pasal 71",
        pasal="71",
    )

    # Model hallucinates UU 99 Tahun 2099
    answer = "Aturan pidana terdapat pada [UU Nomor 99 Tahun 2099, Pasal 10]."
    res = verify_citations(answer=answer, retrieved_chunks=[chunk])

    assert res.is_valid is False
    assert res.checks[0].doc_match is False
    assert res.ungrounded_count == 1
    assert "UU Nomor 99 Tahun 2099, Pasal 10" in res.ungrounded_citations


def test_verify_citations_hallucinated_ayat_fail() -> None:
    """FAIL CASE: Document and Pasal match, but cited Ayat is absent from chunk."""
    chunk = _make_chunk(
        chunk_id="c_pojk22_71_1",
        doc_number="POJK 22 Tahun 2023",
        citation_label="POJK 22 Tahun 2023, Pasal 71 ayat (1)",
        pasal="71",
        ayat="1",
        text="(1) PUJK wajib menangani pengaduan.",
    )

    # Model cites ayat (99) which does not exist
    answer = "Prosedur banding diatur pada [POJK 22 Tahun 2023, Pasal 71 ayat (99)]."
    res = verify_citations(answer=answer, retrieved_chunks=[chunk])

    assert res.is_valid is False
    assert res.checks[0].doc_match is True
    assert res.checks[0].pasal_match is True
    assert res.checks[0].ayat_match is False
    assert res.ungrounded_count == 1


def test_verify_citations_mixed_grounded_and_hallucinated_fail() -> None:
    """FAIL CASE: One citation is grounded, but another is hallucinated."""
    c1 = _make_chunk("c1", "POJK 22 Tahun 2023", "POJK 22 Tahun 2023, Pasal 71", pasal="71")

    # POJK 22 Pasal 71 is valid; UU 50 Tahun 2050 is hallucinated
    answer = (
        "Berdasarkan [POJK 22 Tahun 2023, Pasal 71], PUJK menangani pengaduan. "
        "Hal ini juga didukung oleh [UU Nomor 50 Tahun 2050, Pasal 5]."
    )
    res = verify_citations(answer=answer, retrieved_chunks=[c1])

    assert res.is_valid is False
    assert res.total_citations == 2
    assert res.grounded_count == 1
    assert res.ungrounded_count == 1
    assert res.checks[0].is_grounded is True
    assert res.checks[1].is_grounded is False


# ==============================================================================
# 4. PIPELINE INTEGRATION TESTS
# ==============================================================================


def test_pipeline_citation_verification_grounded() -> None:
    """Verify that RAGPipeline records valid citation verification metadata on grounded answers."""
    chunk = _make_chunk(
        "c1",
        "POJK 22 Tahun 2023",
        "POJK 22 Tahun 2023, Pasal 71",
        pasal="71",
        text="PUJK wajib melayani pengaduan konsumen.",
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="pengaduan",
        mode="hybrid",
        chunks=[chunk],
        total_candidates=1,
        dense_count=1,
        bm25_count=1,
        duration_ms=2.0,
    )

    mock_gen = MockGeneratorBackend()
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_gen)

    output = pipeline.ask("Berapa batas waktu penanganan pengaduan?")
    v_info = output.metadata.get("citation_verification")

    assert v_info is not None
    assert v_info["is_valid"] is True
    assert v_info["grounded_count"] > 0
    assert v_info["ungrounded_count"] == 0


def test_pipeline_strict_grounding_triggers_abstention() -> None:
    """Verify that strict_grounding=True forces abstention when model hallucinates a citation."""
    chunk = _make_chunk(
        "c1",
        "POJK 22 Tahun 2023",
        "POJK 22 Tahun 2023, Pasal 71",
        pasal="71",
        text="PUJK wajib melayani pengaduan konsumen.",
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="pengaduan",
        mode="hybrid",
        chunks=[chunk],
        total_candidates=1,
        dense_count=1,
        bm25_count=1,
        duration_ms=2.0,
    )

    # Generator simulates a hallucination by citing unretrieved Pasal 999
    mock_gen = MagicMock()
    mock_gen.model_name = "hallucinating-generator"
    mock_gen.generate.return_value = MagicMock(
        text="Kewajiban tersebut diatur dalam [POJK 22 Tahun 2023, Pasal 999].",
        prompt_tokens=10,
        completion_tokens=10,
        latency_ms=5.0,
    )

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_gen)

    # Run with strict_grounding=True
    output = pipeline.ask("Pengaduan?", strict_grounding=True)

    # Should have reverted answer to abstention
    assert output.abstained is True
    assert "tidak ditemukan dasar hukum" in output.answer
    assert output.citations == []
    v_info = output.metadata.get("citation_verification")
    assert v_info is not None
    assert v_info["is_valid"] is False
    assert len(v_info["hallucinated_pasal"]) > 0


# ==============================================================================
# 5. PACKAGE EXPORTS TEST
# ==============================================================================


def test_guardrails_package_exports() -> None:
    """Verify that easypasal.guardrails exports verifier symbols cleanly."""
    import easypasal.guardrails as grd

    assert hasattr(grd, "CitationCheck")
    assert hasattr(grd, "ParsedCitation")
    assert hasattr(grd, "VerificationResult")
    assert hasattr(grd, "parse_citation_details")
    assert hasattr(grd, "verify_citations")
    assert hasattr(grd, "verify_single_citation")

    with pytest.raises(AttributeError, match="has no attribute 'unknown'"):
        _ = grd.unknown
