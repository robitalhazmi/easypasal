"""Tests for the EasyPasal REST API (FastAPI service, endpoints, middleware, and request-ID tracking)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

import easypasal.api as api_pkg
from easypasal.api.app import create_app
from easypasal.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrievedChunkResponse,
)
from easypasal.generation.backends import MockGeneratorBackend
from easypasal.generation.models import GenerationOutput
from easypasal.generation.pipeline import RAGPipeline
from easypasal.retrieval.models import RetrievedChunk


def _make_sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="pojk22_pasal_71",
        score=0.038,
        citation_label="POJK 22 Tahun 2023, Pasal 71 ayat (1)",
        doc_id="POJK_22_Tahun_2023",
        doc_number="POJK 22 Tahun 2023",
        pasal="71",
        ayat="1",
        text="(1) PUJK wajib menyelesaikan penanganan pengaduan.",
    )


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Create a mock RAGPipeline for testing API endpoints."""
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.generator = MockGeneratorBackend()
    pipeline.retriever = MagicMock()
    pipeline.retriever.collection_name = "test_col"
    pipeline.retriever.qdrant_store.count.return_value = 142
    pipeline.retriever.bm25_index.corpus = ["doc1", "doc2", "doc3"]

    chunk = _make_sample_chunk()

    def mock_ask(
        question: str,
        language: str | None = None,
        top_k: int = 5,
        strict_grounding: bool = False,
        **_kwargs: object,
    ) -> GenerationOutput:
        _ = top_k
        # Check for abstention queries
        if "rendang" in question.lower() or "denda" in question.lower():
            return GenerationOutput(
                question=question,
                answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut.",
                language="id",
                citations=[],
                abstained=True,
                retrieved_chunks=[],
                prompt_tokens=50,
                completion_tokens=20,
                latency_ms=12.5,
                model_name="mock-generator",
                metadata={"citation_verification": {"is_valid": True, "total_citations": 0}},
            )

        if strict_grounding and "hallucinate" in question.lower():
            return GenerationOutput(
                question=question,
                answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut.",
                language="id",
                citations=[],
                abstained=True,
                retrieved_chunks=[chunk],
                prompt_tokens=50,
                completion_tokens=20,
                latency_ms=15.0,
                model_name="mock-generator",
                metadata={
                    "citation_verification": {
                        "is_valid": False,
                        "total_citations": 1,
                        "grounded_count": 0,
                        "ungrounded_count": 1,
                        "ungrounded_citations": ["POJK 99 Tahun 2099, Pasal 1"],
                    }
                },
            )

        return GenerationOutput(
            question=question,
            answer="Berdasarkan [POJK 22 Tahun 2023, Pasal 71 ayat (1)], PUJK wajib menyelesaikan pengaduan.",
            language=language or "id",
            citations=["POJK 22 Tahun 2023, Pasal 71 ayat (1)"],
            abstained=False,
            retrieved_chunks=[chunk],
            prompt_tokens=100,
            completion_tokens=30,
            latency_ms=18.4,
            model_name="mock-generator",
            metadata={
                "citation_verification": {
                    "is_valid": True,
                    "total_citations": 1,
                    "grounded_count": 1,
                    "ungrounded_count": 0,
                    "ungrounded_citations": [],
                    "hallucinated_pasal": [],
                }
            },
        )

    pipeline.ask.side_effect = mock_ask
    return pipeline


@pytest.fixture
def client(mock_pipeline: MagicMock) -> TestClient:
    app = create_app(pipeline=mock_pipeline, is_test=True)
    return TestClient(app)


# ==============================================================================
# 1. ROOT & HEALTH ENDPOINT TESTS
# ==============================================================================


def test_root_endpoint(client: TestClient) -> None:
    """Verify root GET / returns metadata and documentation links."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "EasyPasal RAG API"
    assert data["version"] == "0.1.0"
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]["swagger_ui"]


def test_health_endpoint_healthy(client: TestClient) -> None:
    """Verify GET /health returns readiness indicators with loaded indexes."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["environment"] == "test"
    assert data["index_ready"] is True
    assert data["vector_count"] == 142
    assert data["bm25_doc_count"] == 3
    assert data["backend_provider"] == "mock-generator"


def test_health_endpoint_uninitialized_pipeline() -> None:
    """Verify GET /health handles absent pipeline gracefully."""
    app = create_app(pipeline=None, is_test=True)
    uninitialized_client = TestClient(app)

    resp = uninitialized_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["index_ready"] is False
    assert data["vector_count"] == 0
    assert data["bm25_doc_count"] == 0
    assert data["backend_provider"] is None


# ==============================================================================
# 2. INFERENCE ENDPOINT (/ask) TESTS
# ==============================================================================


def test_ask_endpoint_grounded_answer(client: TestClient) -> None:
    """Verify POST /ask returns grounded answer with exact citations and chunks."""
    payload = {
        "question": "Berapa batas waktu penanganan pengaduan konsumen menurut POJK 22 Tahun 2023?",
        "top_k": 5,
    }
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["question"] == payload["question"]
    assert "POJK 22 Tahun 2023, Pasal 71 ayat (1)" in data["answer"]
    assert data["abstained"] is False
    assert len(data["citations"]) == 1
    assert data["citations"][0] == "POJK 22 Tahun 2023, Pasal 71 ayat (1)"
    assert len(data["retrieved_chunks"]) == 1
    assert data["retrieved_chunks"][0]["doc_number"] == "POJK 22 Tahun 2023"
    assert data["retrieved_chunks"][0]["pasal"] == "71"
    assert data["verification"] is not None
    assert data["verification"]["is_valid"] is True
    assert data["verification"]["grounded_count"] == 1

    # Validate Request-ID presence in headers and body
    header_req_id = resp.headers.get("X-Request-ID")
    assert header_req_id is not None
    assert data["request_id"] == header_req_id


def test_ask_endpoint_abstention(client: TestClient) -> None:
    """Verify POST /ask handles out-of-corpus queries with explicit abstention."""
    payload = {"question": "Resep memasak rendang daging sapi Padang?"}
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["abstained"] is True
    assert len(data["citations"]) == 0
    assert "tidak ditemukan dasar hukum" in data["answer"]


def test_ask_endpoint_strict_grounding_fallback(client: TestClient) -> None:
    """Verify strict_grounding=True forces abstention when hallucinated citations occur."""
    payload = {
        "question": "Apa sanksi berat menurut undang-undang? (hallucinate)",
        "strict_grounding": True,
    }
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["abstained"] is True
    assert "tidak ditemukan dasar hukum" in data["answer"]
    assert data["verification"] is not None
    assert data["verification"]["is_valid"] is False


def test_ask_endpoint_custom_request_id_propagated(client: TestClient) -> None:
    """Verify custom X-Request-ID from client is preserved in headers and response body."""
    custom_id = "test-custom-client-req-999"
    payload = {"question": "Berapa modal disetor BPR?"}
    headers = {"X-Request-ID": custom_id}

    resp = client.post("/ask", json=payload, headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == custom_id
    assert resp.json()["request_id"] == custom_id


def test_ask_endpoint_validation_errors(client: TestClient) -> None:
    """Verify input validation fails with 422 for invalid payloads."""
    # 1. Question too short (min_length=3)
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 422

    # 2. Invalid top_k (ge=1, le=20)
    resp = client.post("/ask", json={"question": "Valid query", "top_k": 0})
    assert resp.status_code == 422

    resp = client.post("/ask", json={"question": "Valid query", "top_k": 50})
    assert resp.status_code == 422

    # 3. Extra forbidden field
    resp = client.post("/ask", json={"question": "Valid query", "unauthorized_key": True})
    assert resp.status_code == 422


def test_ask_endpoint_uninitialized_pipeline_returns_503() -> None:
    """Verify POST /ask returns 503 Service Unavailable if pipeline is missing."""
    app = create_app(pipeline=None, is_test=True)
    uninitialized_client = TestClient(app)

    resp = uninitialized_client.post("/ask", json={"question": "Pertanyaan regulasi?"})
    assert resp.status_code == 503
    assert "pipeline is not initialized" in resp.json()["detail"].lower()


def test_ask_endpoint_pipeline_error_returns_500(
    client: TestClient, mock_pipeline: MagicMock
) -> None:
    """Verify internal pipeline execution errors return 500."""
    mock_pipeline.ask.side_effect = RuntimeError("Simulated crash in retrieval")

    resp = client.post("/ask", json={"question": "Pertanyaan regulasi?"})
    assert resp.status_code == 500
    assert "Simulated crash in retrieval" in resp.json()["detail"]


# ==============================================================================
# 3. PACKAGE EXPORTS & SCHEMA TESTS
# ==============================================================================


def test_api_package_exports() -> None:
    """Verify PEP 562 dynamic attribute access on easypasal.api."""
    assert callable(api_pkg.create_app)
    assert issubclass(api_pkg.AskRequest, AskRequest)
    assert issubclass(api_pkg.AskResponse, AskResponse)
    assert issubclass(api_pkg.HealthResponse, HealthResponse)
    assert issubclass(api_pkg.RetrievedChunkResponse, RetrievedChunkResponse)
    assert issubclass(api_pkg.RequestIDMiddleware, BaseHTTPMiddleware)
    assert hasattr(api_pkg, "FeedbackPayload")
    assert hasattr(api_pkg, "FeedbackResponse")

    with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
        _ = api_pkg.NonExistent


def test_app_lifespan_initializes_pipeline() -> None:
    """Verify lifespan context manager attempts to initialize pipeline when absent."""
    with patch("easypasal.generation.pipeline.RAGPipeline.from_defaults") as mock_from_defaults:
        mock_pipeline = MagicMock(spec=RAGPipeline)
        mock_pipeline.generator = MockGeneratorBackend()
        mock_pipeline.retriever = None
        mock_from_defaults.return_value = mock_pipeline

        app = create_app(pipeline=None, is_test=False)
        with TestClient(app) as test_client:
            assert app.state.pipeline is not None
            resp = test_client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


def test_main_cli_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main() parses arguments and invokes uvicorn.run."""
    monkeypatch.setattr(
        "sys.argv",
        ["easypasal-api", "--host", "127.0.0.1", "--port", "8888", "--provider", "mock"],
    )
    with (
        patch("easypasal.generation.pipeline.RAGPipeline.from_defaults") as mock_from_defaults,
        patch("uvicorn.run") as mock_run,
    ):
        mock_from_defaults.return_value = MagicMock()
        from easypasal.api.app import main

        main()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8888
