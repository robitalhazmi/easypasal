"""Unit and integration tests for EasyPasal Gradio Web UI and Space Deployment (Plan §8.3 / Task T7.6).

Tests UI layout construction, citation formatting, user interaction callbacks,
feedback collection, and Space staging bundle preparation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import gradio as gr
import pytest
from scripts.deploy_hf_space import (
    deploy_to_huggingface,
    measure_cold_start,
    prepare_space_bundle,
)
from scripts.deploy_hf_space import (
    main as deploy_main,
)

from easypasal.generation.models import GenerationOutput
from easypasal.monitoring.logger import RequestLogger
from easypasal.retrieval.models import RetrievedChunk
from easypasal.ui.app import build_ui, format_citations_markdown


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Fixture providing a mocked RAGPipeline returning structured answers."""
    pipeline = MagicMock()
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        score=0.88,
        citation_label="POJK-12-2021",
        doc_id="POJK-12-2021",
        doc_number="12/POJK.03/2021",
        text="Bank Umum konvensional wajib memiliki modal disetor minimum sebesar Rp10 triliun.",
        pasal="4",
        ayat="1",
        metadata={"theme": "Perbankan"},
    )
    output = GenerationOutput(
        question="Apa modal disetor bank?",
        answer="Berdasarkan [POJK-12-2021], modal disetor minimum adalah Rp10 triliun.",
        language="id",
        citations=["POJK-12-2021 Pasal 4 ayat (1)"],
        abstained=False,
        retrieved_chunks=[chunk],
        latency_ms=115.0,
        model_name="test_model",
    )
    pipeline.ask.return_value = output
    return pipeline


@pytest.fixture
def mock_request_logger(tmp_path: Path) -> RequestLogger:
    """Fixture providing a RequestLogger pointing to temporary sinks."""
    return RequestLogger(
        log_file=tmp_path / "requests.jsonl",
        feedback_file=tmp_path / "feedback.jsonl",
        redact_pii=True,
    )


def test_build_ui_structure(mock_pipeline: MagicMock, mock_request_logger: RequestLogger) -> None:
    """Verify that build_ui returns a valid gr.Blocks application with essential components."""
    demo = build_ui(pipeline=mock_pipeline, request_logger=mock_request_logger)
    assert isinstance(demo, gr.Blocks)
    assert "EasyPasal" in str(demo.title)


def test_example_queries_configuration(
    mock_pipeline: MagicMock, mock_request_logger: RequestLogger
) -> None:
    """Verify that example queries are fully visible, unmasked, and properly wired."""
    from easypasal.ui.app import DEFAULT_EXAMPLES

    assert len(DEFAULT_EXAMPLES) == 6
    for item in DEFAULT_EXAMPLES:
        assert len(item) == 3
        question, lang, variant = item
        assert isinstance(question, str) and len(question) > 10
        assert "..." not in question
        assert lang in ("id", "en")
        assert variant in ("baseline_v0.1", "gepa_v0.2")

    demo = build_ui(pipeline=mock_pipeline, request_logger=mock_request_logger)
    # Verify that all 6 default examples exist as clickable buttons with untruncated labels
    buttons = [comp for comp in demo.blocks.values() if isinstance(comp, gr.Button)]
    button_labels = [getattr(b, "value", "") for b in buttons]
    for ex_q, _, _ in DEFAULT_EXAMPLES:
        assert ex_q in button_labels
        assert "..." not in ex_q


def test_format_citations_markdown() -> None:
    """Verify that format_citations_markdown produces properly formatted legal citations."""
    # 1. Output with chunks
    chunk = RetrievedChunk(
        chunk_id="c1",
        score=0.795,
        citation_label="POJK-10-2022",
        doc_id="POJK-10-2022",
        doc_number="10/POJK.05/2022",
        text="Penyelenggara fintech wajib menerapkan batasan bunga.",
        pasal="12",
        ayat="2",
        metadata={},
    )
    output = GenerationOutput(
        question="q",
        answer="a",
        language="id",
        retrieved_chunks=[chunk],
    )
    md = format_citations_markdown(output)
    assert "[POJK-10-2022]" in md
    assert "Pasal 12 ayat (2)" in md
    assert "0.795" in md
    assert "Penyelenggara fintech wajib" in md

    # 2. Output without chunks
    empty_output = GenerationOutput(
        question="q",
        answer="a",
        language="id",
        retrieved_chunks=[],
    )
    empty_md = format_citations_markdown(empty_output)
    assert "Tidak ada pasal rujukan" in empty_md


def test_prepare_space_bundle(tmp_path: Path) -> None:
    """Verify that prepare_space_bundle stages all required assets for Docker deployment."""
    bundle_dir = tmp_path / "test_bundle"
    prepared = prepare_space_bundle(output_dir=bundle_dir)

    assert prepared.exists()
    assert (prepared / "Dockerfile").exists()
    assert (prepared / "README.md").exists()
    assert (prepared / "pyproject.toml").exists()
    assert (prepared / "src").is_dir()
    assert (prepared / "configs").is_dir()
    assert (prepared / "data").is_dir()

    # Verify HF Space frontmatter metadata in staged README
    readme_content = (prepared / "README.md").read_text(encoding="utf-8")
    assert "sdk: gradio" in readme_content
    assert "app_file: app.py" in readme_content

    # Verify requirements.txt and app.py for ZeroGPU / Transformers support
    requirements_content = (prepared / "requirements.txt").read_text(encoding="utf-8")
    assert "torch" in requirements_content
    assert "transformers" in requirements_content
    assert "accelerate" in requirements_content

    app_py_content = (prepared / "app.py").read_text(encoding="utf-8")
    assert "import spaces" in app_py_content
    assert "RAGPipeline.from_defaults" in app_py_content


def test_measure_cold_start_unreachable() -> None:
    """Verify measure_cold_start returns None on unreachable endpoint without hanging."""
    # Probe a port that does not listen
    result = measure_cold_start("http://127.0.0.1:59999", timeout=0.5)
    assert result is None


def test_deploy_to_huggingface_no_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify deploy_to_huggingface gracefully aborts when no HF token is available."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    success = deploy_to_huggingface(
        space_id="test/space",
        staging_dir=tmp_path,
        token=None,
    )
    assert success is False


def test_deploy_script_main_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify deploy_hf_space CLI main execution in dry-run mode."""
    bundle_dir = tmp_path / "dry_run_bundle"
    monkeypatch.setattr(
        "sys.argv",
        [
            "deploy_hf_space.py",
            "--space-id",
            "robitalhazmi/easypasal",
            "--staging-dir",
            str(bundle_dir),
            "--dry-run",
        ],
    )
    deploy_main()
    assert bundle_dir.exists()
    assert (bundle_dir / "Dockerfile").exists()
