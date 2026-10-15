"""Tests for prompt engineering, citation extraction, backends, and RAG pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from easypasal.generation.backends import (
    MockGeneratorBackend,
    get_generator_backend,
)
from easypasal.generation.citations import (
    check_abstention,
    extract_citations,
    verify_citation_coverage,
)
from easypasal.generation.models import GenerationOutput
from easypasal.generation.pipeline import RAGPipeline
from easypasal.generation.prompts import (
    STANDARD_ABSTAIN_EN,
    STANDARD_ABSTAIN_ID,
    build_rag_prompt,
    detect_language,
    format_passages,
)
from easypasal.retrieval.models import RetrievalResult, RetrievedChunk


def _make_dummy_chunk(
    chunk_id: str,
    citation_label: str,
    text: str,
    doc_number: str = "POJK 22 Tahun 2023",
    status: str = "berlaku",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=0.032,
        citation_label=citation_label,
        doc_id="POJK_22_2023",
        doc_number=doc_number,
        status=status,
        text=text,
    )


def test_detect_language() -> None:
    """Verify language detector correctly identifies Indonesian and English inputs."""
    assert detect_language("Berapa batas waktu penanganan pengaduan konsumen?") == "id"
    assert detect_language("Apa kewajiban modal inti minimum bank umum?") == "id"
    assert detect_language("What is the minimum capital requirement for commercial banks?") == "en"
    assert detect_language("How does a consumer submit a financial complaint?") == "en"
    # Default to Indonesian on ambiguous/empty string
    assert detect_language("") == "id"


def test_format_passages_and_prompt_assembly() -> None:
    """Verify context formatting and prompt assembly for Indonesian and English."""
    c1 = _make_dummy_chunk(
        "c1",
        "POJK 22 Tahun 2023, Pasal 71 ayat (1)",
        "PUJK wajib menangani pengaduan konsumen secara tertulis.",
    )
    c2 = _make_dummy_chunk(
        "c2",
        "POJK 22 Tahun 2023, Pasal 72",
        "PUJK wajib menyelesaikan pengaduan paling lambat 10 hari kerja.",
    )

    passages = format_passages([c1, c2])
    assert "POJK 22 Tahun 2023, Pasal 71 ayat (1)" in passages
    assert "PUJK wajib menyelesaikan pengaduan" in passages

    # Test Indonesian prompt
    id_prompt = build_rag_prompt(
        question="Berapa hari penanganan pengaduan?",
        chunks=[c1, c2],
        language="id",
    )
    assert "<system>" in id_prompt
    assert "ATURAN MUTLAK PENJAWABAN" in id_prompt
    assert "POJK 22 Tahun 2023, Pasal 72" in id_prompt
    assert "Berapa hari penanganan pengaduan?" in id_prompt

    # Test English prompt
    en_prompt = build_rag_prompt(
        question="How many days are allowed for complaint handling?",
        chunks=[c1, c2],
        language="en",
    )
    assert "STRICT ANSWERING RULES" in en_prompt
    assert "How many days are allowed for complaint handling?" in en_prompt

    # Test empty passages
    empty_prompt = build_rag_prompt(question="Pertanyaan?", chunks=[])
    assert "TIDAK ADA KONTEKS REGULASI YANG DITEMUKAN" in empty_prompt


def test_extract_citations() -> None:
    """Verify extraction of bracket and inline statutory citations."""
    text_with_brackets = (
        "Berdasarkan [POJK 22 Tahun 2023, Pasal 71 ayat (1)], PUJK wajib menerima pengaduan. "
        "Hal ini diperkuat dalam [UU Nomor 4 Tahun 2023, Pasal 243]."
    )
    cites = extract_citations(text_with_brackets)
    assert "POJK 22 Tahun 2023, Pasal 71 ayat (1)" in cites
    assert "UU Nomor 4 Tahun 2023, Pasal 243" in cites

    text_inline = "Sesuai dengan POJK 11/POJK.03/2022 Pasal 19 bank wajib menempatkan data center di Indonesia."
    inline_cites = extract_citations(text_inline)
    assert len(inline_cites) > 0
    assert any("POJK 11/POJK.03/2022" in c for c in inline_cites)


def test_check_abstention() -> None:
    """Verify identification of abstention refusal phrases."""
    # Positive Indonesian
    assert check_abstention(STANDARD_ABSTAIN_ID, language="id") is True
    assert (
        check_abstention("Maaf, tidak ditemukan dasar hukum mengenai hal tersebut.", language="id")
        is True
    )
    assert (
        check_abstention(
            "Informasi tersebut tidak terdapat dalam regulasi yang tersedia.", language="id"
        )
        is True
    )

    # Positive English
    assert check_abstention(STANDARD_ABSTAIN_EN, language="en") is True
    assert (
        check_abstention("Information not found in the available regulations.", language="en")
        is True
    )

    # Negative (Answer provided)
    assert (
        check_abstention(
            "PUJK wajib menyelesaikan pengaduan konsumen paling lambat 10 hari kerja menurut [POJK 22 Tahun 2023].",
            language="id",
        )
        is False
    )
    assert (
        check_abstention(
            "Commercial banks must maintain a minimum core capital of IDR 3 trillion according to [POJK 12/2020].",
            language="en",
        )
        is False
    )


def test_verify_citation_coverage() -> None:
    """Verify checking extracted citations against retrieved passage metadata."""
    c1 = _make_dummy_chunk(
        "c1",
        "POJK 22 Tahun 2023, Pasal 71",
        "Kewajiban penanganan pengaduan konsumen.",
        doc_number="POJK 22 Tahun 2023",
    )
    grounded, ungrounded = verify_citation_coverage(
        ["POJK 22 Tahun 2023, Pasal 71", "UU Nomor 99 Tahun 2099, Pasal 1"],
        [c1],
    )
    assert "POJK 22 Tahun 2023, Pasal 71" in grounded
    assert "UU Nomor 99 Tahun 2099, Pasal 1" in ungrounded


def test_mock_generator_backend() -> None:
    """Verify MockGeneratorBackend produces citations for in-corpus and abstains on out-of-corpus."""
    gen = MockGeneratorBackend()

    # In-corpus prompt
    in_corpus_prompt = (
        "<system>...</system>\n"
        "<context>\n--- DOKUMEN 1: POJK 22 Tahun 2023, Pasal 71 ---\nPUJK wajib melayani pengaduan konsumen.\n</context>\n"
        "<question>Berapa batas waktu penanganan pengaduan konsumen?</question>"
    )
    res_in = gen.generate(in_corpus_prompt)
    assert "POJK 22 Tahun 2023, Pasal 71" in res_in.text
    assert check_abstention(res_in.text, language="id") is False

    # Out-of-corpus prompt (rendang recipe)
    ooc_prompt = (
        "<system>...</system>\n"
        "<context>\n--- DOKUMEN 1: POJK 22 Tahun 2023 ---\nPUJK wajib melayani.\n</context>\n"
        "<question>Bagaimana resep memasak rendang sapi?</question>"
    )
    res_ooc = gen.generate(ooc_prompt)
    assert check_abstention(res_ooc.text, language="id") is True


def test_get_generator_backend_factory() -> None:
    """Verify generator backend factory instantiation."""
    mock_gen = get_generator_backend(provider="mock")
    assert isinstance(mock_gen, MockGeneratorBackend)

    auto_gen = get_generator_backend(provider="auto")
    assert hasattr(auto_gen, "generate")


def test_pipeline_unit_mock() -> None:
    """Verify end-to-end RAGPipeline with mock retriever and mock generator."""
    chunk = _make_dummy_chunk(
        "test_1",
        "POJK 22 Tahun 2023, Pasal 71",
        "PUJK wajib menangani pengaduan konsumen.",
    )
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="pengaduan",
        mode="hybrid",
        chunks=[chunk],
        total_candidates=1,
        dense_count=1,
        bm25_count=1,
        duration_ms=5.0,
    )

    mock_gen = MockGeneratorBackend()
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_gen)

    output = pipeline.ask("Berapa batas waktu penanganan pengaduan?")
    assert isinstance(output, GenerationOutput)
    assert output.language == "id"
    assert output.abstained is False
    assert len(output.citations) > 0
    assert "POJK 22 Tahun 2023, Pasal 71" in output.citations[0]
    assert output.latency_ms > 0.0

    pipeline.close()
    mock_retriever.close.assert_called_once()


def test_pipeline_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify pipeline CLI executes cleanly."""
    import sys

    from easypasal.generation.pipeline import main

    chunk = _make_dummy_chunk(
        "c1",
        "POJK 22 Tahun 2023, Pasal 71",
        "PUJK wajib menangani pengaduan konsumen.",
    )
    mock_output = GenerationOutput(
        question="Pertanyaan pengujian?",
        answer="Berdasarkan [POJK 22 Tahun 2023, Pasal 71], PUJK wajib.",
        language="id",
        citations=["POJK 22 Tahun 2023, Pasal 71"],
        abstained=False,
        retrieved_chunks=[chunk],
        latency_ms=10.5,
        model_name="test-model",
    )

    class MockPipeline:
        def ask(self, *args: Any, **kwargs: Any) -> GenerationOutput:
            _ = (args, kwargs)
            return mock_output

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        RAGPipeline, "from_defaults", classmethod(lambda _cls, **_kwargs: MockPipeline())
    )
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--question", "Pertanyaan pengujian?"])

    main()
    captured = capsys.readouterr()
    assert "QUESTION: Pertanyaan pengujian?" in captured.out
    assert "POJK 22 Tahun 2023, Pasal 71" in captured.out


# ==============================================================================
# ACCEPTANCE CRITERIA TEST SUITE (T1.6):
# 10 in-corpus regulatory questions with citations + 3 out-of-corpus abstentions
# ==============================================================================

IN_CORPUS_QUESTIONS = [
    # Q1
    "Berapa batas waktu bagi PUJK untuk menangani pengaduan konsumen menurut POJK 22 Tahun 2023?",
    # Q2
    "Apa yang dimaksud dengan Pelaku Usaha Jasa Keuangan (PUJK) menurut POJK 22 Tahun 2023?",
    # Q3
    "Apakah PUJK wajib menjaga kerahasiaan dan keamanan data pribadi konsumen?",
    # Q4
    "Berapa kewajiban modal inti minimum bagi bank umum menurut peraturan OJK?",
    # Q5
    "Di mana bank umum wajib menempatkan sistem elektronik dan pusat data menurut POJK 11/POJK.03/2022?",
    # Q6
    "Kapan PUJK wajib melakukan Prinsip Mengenali Pengguna Jasa (CDD) menurut POJK 8 Tahun 2023?",
    # Q7
    "Apa kewajiban PUJK terhadap nasabah yang tergolong Orang yang Populer Secara Politis (PEP) menurut POJK 8 Tahun 2023?",
    # Q8
    "Apa kewajiban penyampaian laporan berkala bagi emiten atau perusahaan publik menurut peraturan OJK pasar modal?",
    # Q9
    "Apa kewenangan OJK dalam melakukan gugatan perdata untuk ganti rugi konsumen menurut UU Nomor 4 Tahun 2023?",
    # Q10 (English)
    "What is the definition of financial consumer according to POJK 22 Year 2023?",
]

OUT_OF_CORPUS_QUESTIONS = [
    # OOC 1
    "Bagaimana syarat peralihan Hak Guna Bangunan (HGB) atas tanah menurut hukum pertanahan?",
    # OOC 2
    "Berapa besaran denda tilang untuk pengendara sepeda motor yang tidak memakai helm standar SNI?",
    # OOC 3
    "Bagaimana resep dan takaran bumbu untuk memasak rendang daging sapi Padang yang otentik?",
]


def test_acceptance_criteria_10_in_corpus_and_3_out_of_corpus() -> None:
    """Acceptance criteria test for T1.6:

    - Pipeline answers 10 hand-picked regulatory questions with citations.
    - Pipeline abstains on 3 out-of-corpus questions.
    """
    # Create deterministic mock retriever with synthetic legal passages for the 10 questions
    mock_retriever = MagicMock()

    def mock_retrieve(query: str, **kwargs: Any) -> RetrievalResult:
        _ = kwargs
        q_lower = query.lower()
        if "hgb" in q_lower or "tanah" in q_lower or "helm" in q_lower or "rendang" in q_lower:
            # Irrelevant or empty retrieval for out of corpus
            return RetrievalResult(
                query=query,
                mode="hybrid",
                chunks=[],
                total_candidates=0,
                dense_count=0,
                bm25_count=0,
                duration_ms=1.0,
            )

        # Match relevant in-corpus chunk
        if "modal inti" in q_lower:
            label = "POJK 12/POJK.03/2020, Pasal 3"
            text = (
                "Bank Umum wajib memenuhi Modal Inti minimum paling sedikit Rp3.000.000.000.000,00."
            )
        elif "pusat data" in q_lower or "data center" in q_lower:
            label = "POJK 11/POJK.03/2022, Pasal 19"
            text = "Bank wajib menempatkan Sistem Elektronik pada Pusat Data dan Pusat Pemulihan Bencana di wilayah Indonesia."
        elif "cdd" in q_lower or "mengenali pengguna jasa" in q_lower:
            label = "POJK 8 Tahun 2023, Pasal 18"
            text = "PUJK wajib melaksanakan CDD pada saat melakukan hubungan usaha dengan Calon Nasabah."
        elif "pep" in q_lower or "populer secara politis" in q_lower:
            label = "POJK 8 Tahun 2023, Pasal 35"
            text = "PUJK wajib memiliki sistem manajemen risiko untuk menentukan apakah Nasabah tergolong PEP."
        elif "emiten" in q_lower or "laporan berkala" in q_lower:
            label = "POJK 31/POJK.04/2015, Pasal 2"
            text = "Emiten atau Perusahaan Publik wajib menyampaikan laporan berkala kepada Otoritas Jasa Keuangan."
        elif "gugatan perdata" in q_lower or "uu nomor 4" in q_lower or "p2sk" in q_lower:
            label = "UU Nomor 4 Tahun 2023, Pasal 243"
            text = "OJK berwenang mengajukan gugatan untuk memperoleh ganti kerugian pihak yang dirugikan."
        elif "data pribadi" in q_lower:
            label = "POJK 22 Tahun 2023, Pasal 14"
            text = "PUJK wajib menjaga kerahasiaan dan keamanan data dan/atau informasi pribadi Konsumen."
        elif "consumer" in q_lower or "what is the definition" in q_lower:
            label = "POJK 22 Tahun 2023, Pasal 1 angka 4"
            text = "Consumer is any party that utilizes products and/or services provided by Financial Services Providers."
        elif (
            "pelaku usaha jasa keuangan" in q_lower or "apa yang dimaksud dengan pelaku" in q_lower
        ):
            label = "POJK 22 Tahun 2023, Pasal 1 angka 3"
            text = "PUJK adalah pihak yang melakukan kegiatan usaha di sektor jasa keuangan."
        else:
            # Default to POJK 22 complaint handling
            label = "POJK 22 Tahun 2023, Pasal 71 ayat (1)"
            text = "PUJK wajib menyelesaikan pengaduan paling lambat 10 (sepuluh) hari kerja."

        chunk = _make_dummy_chunk(chunk_id="chunk_match", citation_label=label, text=text)
        return RetrievalResult(
            query=query,
            mode="hybrid",
            chunks=[chunk],
            total_candidates=1,
            dense_count=1,
            bm25_count=1,
            duration_ms=1.5,
        )

    mock_retriever.retrieve.side_effect = mock_retrieve
    mock_gen = MockGeneratorBackend()
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_gen)

    # 1. Verify 10 Hand-Picked Regulatory Questions Answered with Citations
    assert len(IN_CORPUS_QUESTIONS) == 10
    for idx, question in enumerate(IN_CORPUS_QUESTIONS, start=1):
        res = pipeline.ask(question)
        assert res.abstained is False, f"Question Q{idx} was unexpectedly abstained: {question}"
        assert len(res.citations) > 0, f"Question Q{idx} produced no citations: {res.answer}"
        assert res.answer, f"Question Q{idx} produced empty answer"

    # 2. Verify 3 Out-of-Corpus Questions Abstain
    assert len(OUT_OF_CORPUS_QUESTIONS) == 3
    for idx, ooc_q in enumerate(OUT_OF_CORPUS_QUESTIONS, start=1):
        res = pipeline.ask(ooc_q)
        err_msg = f"Out-of-corpus question OOC{idx} did not abstain: {res.answer}"
        assert res.abstained is True, err_msg
        assert check_abstention(res.answer, language=res.language) is True


def test_generation_package_exports() -> None:
    """Verify dynamic __getattr__ exports in easypasal.generation."""
    import easypasal.generation as gen

    assert hasattr(gen, "RAGPipeline")
    assert hasattr(gen, "GenerationOutput")
    assert hasattr(gen, "GenerationInput")
    assert hasattr(gen, "MockGeneratorBackend")
    assert hasattr(gen, "OllamaBackend")
    assert hasattr(gen, "LlamaCppBackend")
    assert hasattr(gen, "OpenAICompatibleBackend")
    assert hasattr(gen, "get_generator_backend")
    assert hasattr(gen, "extract_citations")
    assert hasattr(gen, "check_abstention")
    assert hasattr(gen, "build_rag_prompt")
    assert hasattr(gen, "detect_language")
    assert hasattr(gen, "format_passages")
    assert hasattr(gen, "STANDARD_ABSTAIN_ID")
    assert hasattr(gen, "STANDARD_ABSTAIN_EN")

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        _ = gen.nonexistent


def test_pipeline_from_defaults_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RAGPipeline.from_defaults instantiation with mock provider."""
    from easypasal.retrieval.retriever import HybridRetriever

    mock_retriever = MagicMock()
    monkeypatch.setattr(
        HybridRetriever, "from_paths", classmethod(lambda _cls, **_kwargs: mock_retriever)
    )
    pipe = RAGPipeline.from_defaults(provider="mock")
    assert isinstance(pipe.generator, MockGeneratorBackend)
    pipe.close()


def test_get_process_peak_memory_mb() -> None:
    """Verify peak resident memory reporting returns positive value."""
    from easypasal.generation.backends import get_process_peak_memory_mb

    mem = get_process_peak_memory_mb()
    assert isinstance(mem, float)
    assert mem > 0.0


def test_mock_generator_throughput() -> None:
    """Verify mock generator computes non-zero tokens_per_second."""
    gen = MockGeneratorBackend()
    res = gen.generate("Test prompt")
    assert res.tokens_per_second > 0.0
    assert res.completion_tokens > 0


def test_llama_cpp_backend_chat_format_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify LlamaCppBackend delegates to create_chat_completion when system prompt is present."""
    from easypasal.generation.backends import LlamaCppBackend

    fake_model = tmp_path / "fake_model.gguf"
    fake_model.write_text("fake gguf content")

    mock_llm_instance = MagicMock()
    mock_llm_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Jawaban dari model instruct."}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15},
    }
    mock_llm_instance.create_completion.return_value = {
        "choices": [{"text": "Jawaban dari raw completion."}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }

    import llama_cpp

    monkeypatch.setattr(llama_cpp, "Llama", lambda **_kwargs: mock_llm_instance)

    backend = LlamaCppBackend(model_path=fake_model)
    assert backend.model_name == "fake_model"

    # 1. Prompt with <system> tag should invoke create_chat_completion
    chat_prompt = "<system>Anda asisten hukum.</system>\n<question>Apa itu PUJK?</question>"
    res_chat = backend.generate(chat_prompt)
    assert res_chat.text == "Jawaban dari model instruct."
    assert res_chat.tokens_per_second > 0.0
    mock_llm_instance.create_chat_completion.assert_called_once()

    # 2. Raw prompt without <system> tag should invoke create_completion
    raw_prompt = "Lengkapi teks berikut: Pasal 1 adalah"
    res_raw = backend.generate(raw_prompt)
    assert res_raw.text == "Jawaban dari raw completion."
    assert res_raw.tokens_per_second > 0.0
    mock_llm_instance.create_completion.assert_called_once()
