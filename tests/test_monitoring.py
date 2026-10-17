"""Comprehensive tests for structured request logging, PII redaction, and feedback (Task T7.1)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from easypasal.api.app import create_app
from easypasal.generation.backends import MockGeneratorBackend
from easypasal.generation.models import GenerationOutput
from easypasal.generation.pipeline import RAGPipeline
from easypasal.monitoring.logger import RequestLogger
from easypasal.monitoring.redaction import has_pii, redact_pii
from easypasal.monitoring.schemas import (
    FeedbackPayload,
    RequestLogRecord,
)
from easypasal.retrieval.models import RetrievedChunk

# ==============================================================================
# 1. PII REDACTION UNIT TESTS
# ==============================================================================


def test_redact_pii_indonesian_nik() -> None:
    """Verify 16-digit NIK is replaced with [REDACTED_NIK]."""
    text = "Nasabah dengan NIK 3171012345670001 mengajukan komplain bunga pinjaman."
    redacted = redact_pii(text)
    assert "3171012345670001" not in redacted
    assert "[REDACTED_NIK]" in redacted
    assert has_pii(text) is True


def test_redact_pii_email() -> None:
    """Verify email addresses are replaced with [REDACTED_EMAIL]."""
    text = "Silakan hubungi customer service kami di support.fintech@example.co.id untuk info."
    redacted = redact_pii(text)
    assert "support.fintech@example.co.id" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert has_pii(text) is True


def test_redact_pii_indonesian_phone() -> None:
    """Verify Indonesian phone numbers are replaced with [REDACTED_PHONE]."""
    cases = [
        "Hubungi +6281234567890 sekarang",
        "Kontak darurat di 081234567892 untuk pelaporan",
        "Nomor kantor: 6281987654321",
    ]
    for text in cases:
        redacted = redact_pii(text)
        assert "[REDACTED_PHONE]" in redacted
        assert has_pii(text) is True


def test_redact_pii_credit_cards() -> None:
    """Verify credit and debit card numbers are replaced with [REDACTED_CARD]."""
    cases = [
        "Kartu kredit 4111-2222-3333-4444 terkena fraud.",
        "Nomor kartu: 4111 2222 3333 4444.",
        "Kartu debit 4111222233334444 diblokir.",
    ]
    for text in cases:
        redacted = redact_pii(text)
        assert "4111" not in redacted or "[REDACTED_CARD]" in redacted
        assert "[REDACTED_CARD]" in redacted
        assert has_pii(text) is True


def test_redact_pii_npwp() -> None:
    """Verify formatted and raw Indonesian NPWP numbers are redacted."""
    text_formatted = "NPWP wajib pajak adalah 01.234.567.8-901.000."
    assert "[REDACTED_NPWP]" in redact_pii(text_formatted)

    text_raw = "NPWP: 012345678901000 telah diverifikasi."
    assert "[REDACTED_NPWP]" in redact_pii(text_raw)


def test_redact_pii_bank_account() -> None:
    """Verify bank account numbers preceded by keywords are redacted."""
    text = "Tolong transfer ke no rekening: 1234567890123 atas nama nasabah."
    redacted = redact_pii(text)
    assert "1234567890123" not in redacted
    assert "[REDACTED_ACCOUNT]" in redacted
    assert has_pii(text) is True


def test_redact_pii_preserves_statutory_citations() -> None:
    """CRITICAL: Verify statutory articles, years, and numbers are NOT corrupted by redaction."""
    legal_text = (
        "Berdasarkan POJK Nomor 22/POJK.07/2023, Pasal 71 ayat (1) dan "
        "UU Nomor 4 Tahun 2023 Pasal 243 ayat (2) serta PBI 9/15/PBI/2007, "
        "PUJK wajib menyelesaikan pengaduan paling lambat 20 hari kerja."
    )
    redacted = redact_pii(legal_text)
    assert redacted == legal_text
    assert has_pii(legal_text) is False


def test_redact_pii_handles_empty_and_none() -> None:
    """Verify safe handling of empty or None inputs."""
    assert redact_pii("") == ""
    assert redact_pii(None) == ""
    assert has_pii("") is False
    assert has_pii(None) is False


# ==============================================================================
# 2. SCHEMA VALIDATION TESTS (Plan §9)
# ==============================================================================


def test_request_log_record_valid_schema() -> None:
    """Verify RequestLogRecord validates and serializes all Plan §9 required fields."""
    record = RequestLogRecord(
        request_id="req-uuid-1234",
        variant="baseline_v0.1",
        language="id",
        question_length=50,
        question_redacted="Berapa batas waktu pengaduan nasabah?",
        retrieval_score_top1=0.8842,
        retrieval_scores_topk=[0.8842, 0.7612, 0.7100],
        retrieved_doc_ids=["POJK_22_2023", "UU_4_2023"],
        abstained=False,
        routed=True,
        theme="Consumer Protection",
        retrieval_latency_ms=12.4,
        generation_latency_ms=185.0,
        total_latency_ms=198.5,
        prompt_tokens=150,
        completion_tokens=45,
        citation_count=2,
        grounded_citation_count=2,
        ungrounded_citation_count=0,
        feedback_rating=1,
        feedback_comment="Sangat membantu",
        client_ip_hash="a1b2c3d4e5f60718",
        model_name="mock-generator",
    )

    # Check total_tokens validator computed prompt + completion
    assert record.total_tokens == 195

    # Check serialization roundtrip
    dumped_json = record.model_dump_json()
    parsed = RequestLogRecord.model_validate_json(dumped_json)
    assert parsed.request_id == "req-uuid-1234"
    assert parsed.retrieval_scores_topk == [0.8842, 0.7612, 0.7100]
    assert parsed.theme == "Consumer Protection"


def test_request_log_record_missing_required_fields_raises() -> None:
    """Verify validation error when required fields are missing."""
    with pytest.raises(ValidationError):
        # Missing total_latency_ms and question_redacted
        RequestLogRecord(  # type: ignore[call-arg]
            request_id="req-1",
            variant="baseline_v0.1",
            language="id",
            question_length=10,
            abstained=False,
        )


def test_request_log_record_forbids_extra_fields() -> None:
    """Verify extra fields are strictly forbidden by schema config."""
    with pytest.raises(ValidationError):
        RequestLogRecord(
            request_id="req-1",
            variant="baseline_v0.1",
            language="id",
            question_length=10,
            question_redacted="Pertanyaan?",
            abstained=False,
            total_latency_ms=10.0,
            unexpected_field="disallowed",  # type: ignore[call-arg]
        )


def test_feedback_payload_validation() -> None:
    """Verify FeedbackPayload schema constraints."""
    # Valid positive rating
    p1 = FeedbackPayload(request_id="req-1", rating=1, comment="Bagus", tags=["accurate"])
    assert p1.rating == 1

    # Valid negative rating
    p2 = FeedbackPayload(request_id="req-2", rating=-1, comment="Salah pasal")
    assert p2.rating == -1

    # Invalid rating raises
    with pytest.raises(ValidationError):
        FeedbackPayload(request_id="req-3", rating=0)  # type: ignore[arg-type]


# ==============================================================================
# 3. REQUEST LOGGER STORAGE & THREAD SAFETY TESTS
# ==============================================================================


def test_request_logger_file_persistence(tmp_path: Path) -> None:
    """Verify RequestLogger appends records to file and reads them back."""
    log_file = tmp_path / "sub" / "requests.jsonl"
    fb_file = tmp_path / "sub" / "feedback.jsonl"

    logger = RequestLogger(log_file=log_file, feedback_file=fb_file)

    rec1 = RequestLogRecord(
        request_id="r1",
        variant="baseline_v0.1",
        language="id",
        question_length=20,
        question_redacted="Pertanyaan 1",
        abstained=False,
        total_latency_ms=15.2,
    )
    rec2 = RequestLogRecord(
        request_id="r2",
        variant="gepa_v0.2",
        language="en",
        question_length=25,
        question_redacted="Question 2",
        abstained=True,
        total_latency_ms=25.0,
    )

    logger.log_request(rec1)
    logger.log_request(rec2)

    # Read back all records
    records = logger.read_logs()
    assert len(records) == 2
    assert records[0].request_id == "r1"
    assert records[1].request_id == "r2"

    # Test limit option
    assert len(logger.read_logs(limit=1)) == 1
    assert logger.read_logs(limit=1)[0].request_id == "r2"


def test_request_logger_feedback_persistence(tmp_path: Path) -> None:
    """Verify RequestLogger logs user feedback to feedback_file."""
    fb_file = tmp_path / "feedback.jsonl"
    logger = RequestLogger(log_file=None, feedback_file=fb_file)

    payload = FeedbackPayload(request_id="r1", rating=1, comment="Akurat", tags=["helpful"])
    logger.log_feedback(payload)

    feedback_entries = logger.read_feedback()
    assert len(feedback_entries) == 1
    assert feedback_entries[0].request_id == "r1"
    assert feedback_entries[0].rating == 1
    assert feedback_entries[0].comment_redacted == "Akurat"


def test_request_logger_corrupt_line_resilience(tmp_path: Path) -> None:
    """Verify RequestLogger gracefully skips corrupt lines when reading logs."""
    log_file = tmp_path / "corrupt.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            '{"request_id": "good_1", "variant": "A", "language": "id", "question_length": 5, "question_redacted": "q", "abstained": false, "total_latency_ms": 1.0}\n'
        )
        f.write("NOT_A_VALID_JSON_STRING\n")
        f.write(
            '{"request_id": "good_2", "variant": "B", "language": "id", "question_length": 5, "question_redacted": "q", "abstained": true, "total_latency_ms": 2.0}\n'
        )

    logger = RequestLogger(log_file=log_file)
    records = logger.read_logs()
    assert len(records) == 2
    assert records[0].request_id == "good_1"
    assert records[1].request_id == "good_2"


def test_request_logger_thread_safety(tmp_path: Path) -> None:
    """Verify RequestLogger handles concurrent log writes from multiple threads safely."""
    log_file = tmp_path / "concurrent.jsonl"
    logger = RequestLogger(log_file=log_file)

    def write_worker(idx: int) -> None:
        rec = RequestLogRecord(
            request_id=f"thread-{idx}",
            variant="baseline_v0.1",
            language="id",
            question_length=15,
            question_redacted=f"Query {idx}",
            abstained=False,
            total_latency_ms=10.0,
        )
        logger.log_request(rec)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_worker, range(50)))

    records = logger.read_logs()
    assert len(records) == 50
    ids = {r.request_id for r in records}
    assert len(ids) == 50


def test_hash_client_ip() -> None:
    """Verify hash_client_ip produces deterministic, privacy-preserving salted hashes."""
    logger = RequestLogger(salt="custom-salt-123")

    h1 = logger.hash_client_ip("192.168.1.100")
    h2 = logger.hash_client_ip("192.168.1.100")
    h3 = logger.hash_client_ip("10.0.0.1")

    assert h1 is not None
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16
    assert "192" not in h1

    # Empty and unknown IPs return None
    assert logger.hash_client_ip(None) is None
    assert logger.hash_client_ip("") is None
    assert logger.hash_client_ip("unknown") is None


# ==============================================================================
# 4. API INTEGRATION TESTS (POST /ask, POST /feedback, GET /)
# ==============================================================================


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Create a mock RAGPipeline for API testing."""
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.generator = MockGeneratorBackend()
    pipeline.retriever = MagicMock()
    pipeline.retriever.collection_name = "test_col"
    pipeline.retriever.qdrant_store.count.return_value = 100
    pipeline.retriever.bm25_index.corpus = ["c1"]

    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        score=0.045,
        citation_label="POJK 22 Tahun 2023, Pasal 71",
        doc_id="POJK_22_2023",
        doc_number="POJK 22 Tahun 2023",
        pasal="71",
        ayat="1",
        text="PUJK wajib melayani pengaduan konsumen.",
    )

    def mock_ask(question: str, **_kwargs: object) -> GenerationOutput:
        return GenerationOutput(
            question=question,
            answer="Kewajiban diatur dalam [POJK 22 Tahun 2023, Pasal 71].",
            language="id",
            citations=["POJK 22 Tahun 2023, Pasal 71"],
            abstained=False,
            retrieved_chunks=[chunk],
            prompt_tokens=30,
            completion_tokens=15,
            latency_ms=25.0,
            model_name="mock-model",
            metadata={
                "retrieval_ms": 5.0,
                "generation_ms": 20.0,
                "routed": True,
                "theme": "Consumer Protection",
                "citation_verification": {
                    "is_valid": True,
                    "total_citations": 1,
                    "grounded_count": 1,
                    "ungrounded_count": 0,
                    "ungrounded_citations": [],
                    "hallucinated_pasal": [],
                },
            },
        )

    pipeline.ask.side_effect = mock_ask
    return pipeline


def test_api_ask_records_structured_log(mock_pipeline: MagicMock, tmp_path: Path) -> None:
    """Verify /ask automatically records a validated RequestLogRecord with PII redaction."""
    log_file = tmp_path / "requests.jsonl"
    logger = RequestLogger(log_file=log_file)

    app = create_app(pipeline=mock_pipeline, is_test=True, request_logger=logger)
    client = TestClient(app)

    # Submit query with PII: NIK and email
    query_with_pii = (
        "Nasabah dengan NIK 3171012345670001 dan email budi@example.com "
        "bertanya bagaimana mekanisme pengaduan di PUJK?"
    )

    resp = client.post("/ask", json={"question": query_with_pii})
    assert resp.status_code == 200
    req_id = resp.json()["request_id"]

    # Verify log entry persisted and validated
    logged_records = logger.read_logs()
    assert len(logged_records) == 1
    rec = logged_records[0]

    assert rec.request_id == req_id
    assert rec.language == "id"
    assert rec.variant == "baseline_v0.1"
    assert rec.abstained is False
    assert rec.routed is True
    assert rec.theme == "Consumer Protection"
    assert rec.retrieval_score_top1 == 0.045
    assert rec.retrieved_doc_ids == ["POJK_22_2023"]
    assert rec.total_tokens == 45
    assert rec.citation_count == 1
    assert rec.grounded_citation_count == 1
    assert rec.ungrounded_citation_count == 0

    # Verify PII was redacted in stored log
    assert "3171012345670001" not in rec.question_redacted
    assert "budi@example.com" not in rec.question_redacted
    assert "[REDACTED_NIK]" in rec.question_redacted
    assert "[REDACTED_EMAIL]" in rec.question_redacted


def test_api_feedback_endpoint(mock_pipeline: MagicMock, tmp_path: Path) -> None:
    """Verify POST /feedback records user ratings and redacted comments."""
    fb_file = tmp_path / "feedback.jsonl"
    logger = RequestLogger(log_file=None, feedback_file=fb_file)

    app = create_app(pipeline=mock_pipeline, is_test=True, request_logger=logger)
    client = TestClient(app)

    payload = {
        "request_id": "test-req-1234",
        "rating": 1,
        "comment": "Hubungi saya di 081234567890 jika ada pembaruan.",
        "tags": ["helpful", "accurate"],
    }

    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["request_id"] == "test-req-1234"
    assert data["rating"] == 1

    # Verify feedback was written with comment PII redacted
    entries = logger.read_feedback()
    assert len(entries) == 1
    assert entries[0].request_id == "test-req-1234"
    assert entries[0].rating == 1
    assert "081234567890" not in (entries[0].comment_redacted or "")
    assert "[REDACTED_PHONE]" in (entries[0].comment_redacted or "")


def test_api_feedback_validation_error(mock_pipeline: MagicMock) -> None:
    """Verify POST /feedback returns 422 on invalid rating."""
    app = create_app(pipeline=mock_pipeline, is_test=True)
    client = TestClient(app)

    # Invalid rating (0 is not in Literal[1, -1])
    resp = client.post("/feedback", json={"request_id": "r1", "rating": 0})
    assert resp.status_code == 422


def test_api_root_metadata_includes_privacy_notice(mock_pipeline: MagicMock) -> None:
    """Verify GET / metadata returns privacy notice and feedback endpoint."""
    app = create_app(pipeline=mock_pipeline, is_test=True)
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()

    assert "privacy_notice" in data
    assert "redacted" in data["privacy_notice"].lower()
    assert data["endpoints"]["feedback"] == "/feedback"


def test_monitoring_package_exports() -> None:
    """Verify PEP 562 dynamic attribute access on easypasal.monitoring."""
    import easypasal.monitoring as mon

    assert hasattr(mon, "RequestLogger")
    assert hasattr(mon, "redact_pii")
    assert hasattr(mon, "has_pii")
    assert hasattr(mon, "RequestLogRecord")
    assert hasattr(mon, "FeedbackPayload")
    assert hasattr(mon, "FeedbackLogRecord")

    with pytest.raises(AttributeError, match="has no attribute 'unknown'"):
        _ = mon.unknown
