"""Unit tests for EasyPasal configuration management and baseline freezing (T4.5)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from easypasal.config import (
    BASELINE_CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    ChunkingConfig,
    CorpusConfig,
    EasyPasalConfig,
    EmbeddingConfig,
    GeneratorConfig,
    GuardrailsConfig,
    RetrievalConfig,
)
from easypasal.generation.models import GenerationOutput
from easypasal.generation.pipeline import RAGPipeline
from easypasal.retrieval.models import RetrievalResult, RetrievedChunk


def test_default_config_instantiation() -> None:
    """Verify default EasyPasalConfig instantiates with validated production defaults."""
    config = EasyPasalConfig()

    assert config.version == "0.1"
    assert config.variant_id == "baseline_v0.1"
    assert isinstance(config.corpus, CorpusConfig)
    assert config.corpus.total_documents == 21
    assert len(config.corpus.themes) == 5

    assert isinstance(config.chunking, ChunkingConfig)
    assert config.chunking.strategy == "structure_aware"
    assert config.chunking.include_breadcrumbs is True
    assert config.chunking.target_chunk_size == 512
    assert config.chunking.overlap == 64

    assert isinstance(config.embeddings, EmbeddingConfig)
    assert config.embeddings.model_name == "intfloat/multilingual-e5-large"
    assert config.embeddings.dimension == 1024
    assert config.embeddings.backend == "fastembed"
    assert config.embeddings.prefix_query == "query: "
    assert config.embeddings.prefix_passage == "passage: "

    assert isinstance(config.retrieval, RetrievalConfig)
    assert config.retrieval.mode == "dense"
    assert config.retrieval.top_k == 5
    assert config.retrieval.vector_store == "qdrant"
    assert config.retrieval.collection_name == "easypasal_structure_aware_multilingual_e5_large"

    assert isinstance(config.generator, GeneratorConfig)
    assert config.generator.cpu_default.provider == "llamacpp"
    assert config.generator.cpu_default.repo_id == "bartowski/Llama-3.2-1B-Instruct-GGUF"
    assert config.generator.cpu_default.filename == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    assert config.generator.cpu_default.n_ctx == 8192
    assert config.generator.cpu_default.max_tokens == 160
    assert config.generator.cpu_default.temperature == 0.1
    assert "<|eot_id|>" in config.generator.cpu_default.stop_tokens

    assert config.generator.gpu_target is not None
    assert config.generator.gpu_target.provider == "vllm"
    assert "Qwen2.5-3B" in config.generator.gpu_target.repo_id

    assert isinstance(config.guardrails, GuardrailsConfig)
    assert config.guardrails.verify_citations is True
    assert config.guardrails.strict_grounding is False
    assert config.guardrails.status_filter == "berlaku"


def test_load_baseline_yaml() -> None:
    """Verify loading frozen configs/baseline_v0.1.yaml from repository disk."""
    assert BASELINE_CONFIG_PATH.exists(), f"{BASELINE_CONFIG_PATH} must exist"
    config = EasyPasalConfig.from_yaml(BASELINE_CONFIG_PATH)

    assert config.version == "0.1"
    assert config.variant_id == "baseline_v0.1"
    assert config.chunking.strategy == "structure_aware"
    assert config.chunking.include_breadcrumbs is True
    assert config.embeddings.model_name == "intfloat/multilingual-e5-large"
    assert config.embeddings.dimension == 1024
    assert config.retrieval.mode == "dense"
    assert config.retrieval.top_k == 5
    assert config.generator.cpu_default.provider == "llamacpp"
    assert config.generator.cpu_default.filename == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"


def test_load_default_yaml() -> None:
    """Verify loading configs/default.yaml mirrors baseline v0.1 parameters."""
    assert DEFAULT_CONFIG_PATH.exists(), f"{DEFAULT_CONFIG_PATH} must exist"
    config = EasyPasalConfig.load_default()

    assert config.version == "0.1"
    assert config.variant_id == "baseline_v0.1"
    assert config.retrieval.mode == "dense"
    assert config.embeddings.model_name == "intfloat/multilingual-e5-large"


def test_yaml_roundtrip(tmp_path: Path) -> None:
    """Verify serialization to YAML and deserialization preserves identical state."""
    original = EasyPasalConfig()
    out_file = tmp_path / "test_config.yaml"

    yaml_str = original.to_yaml(out_file)
    assert out_file.exists()
    assert "baseline_v0.1" in yaml_str

    reloaded = EasyPasalConfig.from_yaml(out_file)
    assert original.model_dump() == reloaded.model_dump()


def test_from_yaml_file_not_found() -> None:
    """Verify from_yaml raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        EasyPasalConfig.from_yaml("non_existent_config.yaml")


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    """Verify ValueError is raised on invalid YAML syntax."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed list", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML parsing error"):
        EasyPasalConfig.from_yaml(bad_yaml)


def test_invalid_yaml_structure(tmp_path: Path) -> None:
    """Verify ValueError is raised when root YAML structure is not a dict."""
    not_dict = tmp_path / "list.yaml"
    not_dict.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected dictionary"):
        EasyPasalConfig.from_yaml(not_dict)


def test_chunking_validation_overlap_exceeds_target() -> None:
    """Verify validation rejects overlap greater than or equal to target_chunk_size."""
    with pytest.raises(ValidationError, match="overlap"):
        ChunkingConfig(target_chunk_size=256, overlap=256)


def test_retrieval_validation_invalid_mode() -> None:
    """Verify validation rejects unknown retrieval mode."""
    with pytest.raises(ValidationError):
        RetrievalConfig(mode="unsupported_mode")  # type: ignore[arg-type]


def test_generator_validation_temperature_range() -> None:
    """Verify validation enforces temperature bounds [0.0, 2.0]."""
    with pytest.raises(ValidationError):
        EasyPasalConfig(
            generator=GeneratorConfig(
                cpu_default={"temperature": 3.5}  # type: ignore[arg-type]
            )
        )


def test_rag_pipeline_from_config() -> None:
    """Verify RAGPipeline.from_config assembles retriever and generator properly."""
    config = EasyPasalConfig()

    mock_retriever = MagicMock()
    mock_gen = MagicMock()

    with (
        patch(
            "easypasal.retrieval.retriever.HybridRetriever.from_paths",
            return_value=mock_retriever,
        ) as mock_from_paths,
        patch(
            "easypasal.generation.pipeline.get_generator_backend",
            return_value=mock_gen,
        ) as mock_get_gen,
    ):
        pipeline = RAGPipeline.from_config(config=config, override_provider="mock")

        assert pipeline.retriever == mock_retriever
        assert pipeline.generator == mock_gen
        assert pipeline.config == config

        mock_from_paths.assert_called_once_with(
            index_dir=config.retrieval.index_dir,
            strategy=config.chunking.strategy,
            model_name=config.embeddings.model_name,
            collection_name=config.retrieval.collection_name,
        )

        mock_get_gen.assert_called_once_with(
            provider="mock",
            repo_id=config.generator.cpu_default.repo_id,
            filename=config.generator.cpu_default.filename,
            n_ctx=config.generator.cpu_default.n_ctx,
            n_threads=config.generator.cpu_default.n_threads,
        )


def test_rag_pipeline_ask_uses_config_defaults() -> None:
    """Verify pipeline.ask() defaults to config retrieval and generation parameters."""
    config = EasyPasalConfig()
    mock_retriever = MagicMock()
    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        doc_id="POJK_22_2023",
        doc_number="POJK 22/2023",
        citation_label="POJK 22/2023, Pasal 1",
        score=0.9,
        text="Definisi konsumen keuangan.",
    )
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="konsumen",
        mode="dense",
        chunks=[chunk],
        total_candidates=1,
        dense_count=1,
        bm25_count=0,
        duration_ms=10.0,
    )
    mock_gen = MagicMock()
    mock_gen.generate.return_value = MagicMock(
        text="Menurut [POJK 22/2023, Pasal 1], konsumen adalah nasabah.",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        duration_s=0.5,
    )

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_gen, config=config)

    output = pipeline.ask("Apa itu konsumen?")

    assert isinstance(output, GenerationOutput)
    # Check that retrieve was invoked with dense mode and top_k=5 from config
    mock_retriever.retrieve.assert_called_once_with(
        query="Apa itu konsumen?",
        top_k=5,
        mode="dense",
        status_filter="berlaku",
    )
    # Check that generator was invoked with max_tokens=160 and temp=0.1 from config
    mock_gen.generate.assert_called_once()
    assert mock_gen.generate.call_args[1]["temperature"] == 0.1
    assert mock_gen.generate.call_args[1]["max_tokens"] == 160
