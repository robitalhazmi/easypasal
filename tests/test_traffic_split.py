"""Unit and integration tests for EasyPasal traffic splitting and variant routing (Task T5.3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from easypasal.api.app import create_app
from easypasal.api.traffic import (
    VARIANT_BASELINE_V0_1,
    VARIANT_GEPA_V0_2,
    compute_deterministic_variant,
    map_variant_to_prompt,
    normalize_variant,
    resolve_variant,
)
from easypasal.config import TrafficSplitConfig
from easypasal.generation.backends import MockGeneratorBackend
from easypasal.generation.models import GenerationOutput
from easypasal.generation.pipeline import RAGPipeline
from easypasal.retrieval.models import RetrievedChunk

# ==============================================================================
# 1. UNIT TESTS: ALIAS NORMALIZATION & PROMPT MAPPING
# ==============================================================================


def test_normalize_variant_valid_aliases() -> None:
    """Verify alias mapping correctly resolves to canonical variant strings."""
    assert normalize_variant("A") == VARIANT_BASELINE_V0_1
    assert normalize_variant("a") == VARIANT_BASELINE_V0_1
    assert normalize_variant("baseline") == VARIANT_BASELINE_V0_1
    assert normalize_variant("baseline_v0.1") == VARIANT_BASELINE_V0_1

    assert normalize_variant("B") == VARIANT_GEPA_V0_2
    assert normalize_variant("b") == VARIANT_GEPA_V0_2
    assert normalize_variant("gepa") == VARIANT_GEPA_V0_2
    assert normalize_variant("gepa_v0.2") == VARIANT_GEPA_V0_2


def test_normalize_variant_none_or_empty() -> None:
    """Verify None or empty strings return None without error."""
    assert normalize_variant(None) is None
    assert normalize_variant("") is None
    assert normalize_variant("   ") is None


def test_normalize_variant_invalid_raises() -> None:
    """Verify unrecognized variant strings raise ValueError with helpful hint."""
    with pytest.raises(ValueError, match="Unknown variant 'invalid_v99'"):
        normalize_variant("invalid_v99")


def test_map_variant_to_prompt() -> None:
    """Verify canonical variants map accurately to low-level prompt variants."""
    assert map_variant_to_prompt(VARIANT_BASELINE_V0_1) == "baseline"
    assert map_variant_to_prompt(VARIANT_GEPA_V0_2) == "gepa"
    assert map_variant_to_prompt("unknown") == "baseline"


# ==============================================================================
# 2. UNIT TESTS: DETERMINISTIC HASHING & UNIFORMITY
# ==============================================================================


def test_deterministic_hashing_repeatability() -> None:
    """Verify identical input key deterministically maps to the exact same variant across runs."""
    key = "user_session_abc123:Berapa modal minimum BPR?"
    weights = {VARIANT_BASELINE_V0_1: 0.5, VARIANT_GEPA_V0_2: 0.5}

    initial_assignment = compute_deterministic_variant(key, weights)
    for _ in range(500):
        assert compute_deterministic_variant(key, weights) == initial_assignment


def test_deterministic_hashing_uniform_distribution() -> None:
    """Verify 1,000 unique keys distribute approximately 50/50 under equal weights."""
    weights = {VARIANT_BASELINE_V0_1: 0.5, VARIANT_GEPA_V0_2: 0.5}
    counts: dict[str, int] = {VARIANT_BASELINE_V0_1: 0, VARIANT_GEPA_V0_2: 0}

    num_samples = 1000
    for i in range(num_samples):
        key = f"sample_query_{i}_peraturan_ojk_perbankan"
        var = compute_deterministic_variant(key, weights)
        counts[var] += 1

    # In a uniform distribution of 1,000 samples, each variant should receive 42% - 58%
    ratio_a = counts[VARIANT_BASELINE_V0_1] / num_samples
    ratio_b = counts[VARIANT_GEPA_V0_2] / num_samples
    assert 0.42 <= ratio_a <= 0.58
    assert 0.42 <= ratio_b <= 0.58


def test_deterministic_hashing_skewed_weights() -> None:
    """Verify 90/10 skewed traffic split respects configured proportions."""
    weights = {VARIANT_BASELINE_V0_1: 0.9, VARIANT_GEPA_V0_2: 0.1}
    counts: dict[str, int] = {VARIANT_BASELINE_V0_1: 0, VARIANT_GEPA_V0_2: 0}

    num_samples = 1000
    for i in range(num_samples):
        key = f"skew_test_query_{i}"
        var = compute_deterministic_variant(key, weights)
        counts[var] += 1

    ratio_a = counts[VARIANT_BASELINE_V0_1] / num_samples
    assert 0.85 <= ratio_a <= 0.95


# ==============================================================================
# 3. UNIT TESTS: HIERARCHICAL PRECEDENCE RESOLUTION
# ==============================================================================


def test_resolve_variant_payload_override_precedence() -> None:
    """Verify payload override takes precedence over header and traffic config."""
    cfg = TrafficSplitConfig(
        enabled=True,
        default_variant=VARIANT_BASELINE_V0_1,
        weights={VARIANT_BASELINE_V0_1: 0.5, VARIANT_GEPA_V0_2: 0.5},
    )

    resolved = resolve_variant(
        payload_variant="B",
        header_variant="A",
        client_key="client_ip_1",
        traffic_config=cfg,
    )
    assert resolved == VARIANT_GEPA_V0_2


def test_resolve_variant_header_override_precedence() -> None:
    """Verify HTTP header override takes precedence over traffic config when payload is absent."""
    cfg = TrafficSplitConfig(
        enabled=True,
        default_variant=VARIANT_BASELINE_V0_1,
        weights={VARIANT_BASELINE_V0_1: 0.5, VARIANT_GEPA_V0_2: 0.5},
    )

    resolved = resolve_variant(
        payload_variant=None,
        header_variant="B",
        client_key="client_ip_1",
        traffic_config=cfg,
    )
    assert resolved == VARIANT_GEPA_V0_2


def test_resolve_variant_disabled_split_falls_back_to_default() -> None:
    """Verify disabled traffic split always returns the default variant."""
    cfg = TrafficSplitConfig(
        enabled=False,
        default_variant=VARIANT_GEPA_V0_2,
    )

    resolved = resolve_variant(
        payload_variant=None,
        header_variant=None,
        client_key="client_ip_1",
        traffic_config=cfg,
    )
    assert resolved == VARIANT_GEPA_V0_2


# ==============================================================================
# 4. INTEGRATION TESTS: FASTAPI ENDPOINTS & SIDE-BY-SIDE SERVING
# ==============================================================================


@pytest.fixture
def test_pipeline() -> MagicMock:
    """Create a mock pipeline that tracks which variant was passed to ask()."""
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.generator = MockGeneratorBackend()
    pipeline.retriever = MagicMock()
    pipeline.retriever.collection_name = "test_col"
    pipeline.retriever.qdrant_store.count.return_value = 100
    pipeline.retriever.bm25_index.corpus = ["doc1"]

    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        score=0.9,
        citation_label="POJK 22 Tahun 2023, Pasal 71",
        doc_id="POJK_22",
        doc_number="POJK 22 Tahun 2023",
        pasal="71",
        ayat="1",
        text="Teks regulasi penanganan pengaduan.",
    )

    def mock_ask(
        question: str,
        variant: str | None = None,
        **_kwargs: object,
    ) -> GenerationOutput:
        assigned_var = variant or VARIANT_BASELINE_V0_1
        return GenerationOutput(
            question=question,
            answer=f"Answer produced by {assigned_var}",
            language="id",
            citations=["POJK 22 Tahun 2023, Pasal 71"],
            abstained=False,
            retrieved_chunks=[chunk],
            prompt_tokens=50,
            completion_tokens=20,
            latency_ms=10.0,
            model_name="mock-generator",
            metadata={"variant": assigned_var},
        )

    pipeline.ask.side_effect = mock_ask
    return pipeline


def test_api_side_by_side_serving(test_pipeline: MagicMock) -> None:
    """Verify the API can serve Variant A and Variant B side by side with tracking headers."""
    app = create_app(pipeline=test_pipeline, is_test=True)
    client = TestClient(app)

    # 1. Request explicitly directed to Variant A
    resp_a = client.post("/ask", json={"question": "Kueri Variant A", "variant": "A"})
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert data_a["variant"] == VARIANT_BASELINE_V0_1
    assert "baseline_v0.1" in data_a["answer"]
    assert resp_a.headers.get("X-EasyPasal-Variant") == VARIANT_BASELINE_V0_1

    # 2. Request explicitly directed to Variant B
    resp_b = client.post("/ask", json={"question": "Kueri Variant B", "variant": "B"})
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert data_b["variant"] == VARIANT_GEPA_V0_2
    assert "gepa_v0.2" in data_b["answer"]
    assert resp_b.headers.get("X-EasyPasal-Variant") == VARIANT_GEPA_V0_2


def test_api_header_override(test_pipeline: MagicMock) -> None:
    """Verify HTTP header X-EasyPasal-Variant pins the requested variant."""
    app = create_app(pipeline=test_pipeline, is_test=True)
    client = TestClient(app)

    resp = client.post(
        "/ask",
        json={"question": "Kueri dengan header override"},
        headers={"X-EasyPasal-Variant": "gepa_v0.2"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["variant"] == VARIANT_GEPA_V0_2
    assert resp.headers.get("X-EasyPasal-Variant") == VARIANT_GEPA_V0_2


def test_api_invalid_variant_returns_422(test_pipeline: MagicMock) -> None:
    """Verify passing an unrecognized variant string returns HTTP 422 with validation detail."""
    app = create_app(pipeline=test_pipeline, is_test=True)
    client = TestClient(app)

    resp = client.post("/ask", json={"question": "Kueri invalid variant", "variant": "xyz_v9"})
    assert resp.status_code == 422
    assert "Unknown variant" in resp.json()["detail"]


def test_api_health_reports_active_variant(test_pipeline: MagicMock) -> None:
    """Verify GET /health reports current active_variant and traffic_split_enabled."""
    app = create_app(
        pipeline=test_pipeline,
        is_test=True,
        variant=VARIANT_GEPA_V0_2,
        traffic_split=True,
    )
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_variant"] == VARIANT_GEPA_V0_2
    assert data["traffic_split_enabled"] is True


def test_api_deterministic_traffic_split_repeatable(test_pipeline: MagicMock) -> None:
    """Verify deterministic traffic split consistently routes repeat queries from same client."""
    app = create_app(
        pipeline=test_pipeline,
        is_test=True,
        traffic_split=True,
    )
    client = TestClient(app)

    # First request
    resp1 = client.post(
        "/ask",
        json={"question": "Berapa modal minimum BPR menurut POJK?"},
        headers={"X-Forwarded-For": "192.168.1.100"},
    )
    assert resp1.status_code == 200
    v1 = resp1.json()["variant"]

    # Second request from the same client
    resp2 = client.post(
        "/ask",
        json={"question": "Berapa modal minimum BPR menurut POJK?"},
        headers={"X-Forwarded-For": "192.168.1.100"},
    )
    assert resp2.status_code == 200
    v2 = resp2.json()["variant"]

    assert v1 == v2
    assert resp1.headers.get("X-EasyPasal-Variant") == v1
