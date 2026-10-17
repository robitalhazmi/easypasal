"""Centralized configuration schemas and loader for EasyPasal (T4.5).

Provides Pydantic models for the frozen baseline variant and variant configurations,
supporting YAML parsing, serialization, and runtime pipeline assembly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("easypasal.config")

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
BASELINE_CONFIG_PATH = Path("configs/baseline_v0.1.yaml")


class CorpusConfig(BaseModel):
    """Configuration for regulation corpus manifest and storage."""

    total_documents: int = Field(default=21, ge=1)
    manifest_path: str = "data/corpus_manifest.json"
    raw_dir: str = "data/raw"
    parsed_dir: str = "data/parsed"
    themes: list[str] = Field(
        default_factory=lambda: [
            "General Financial Regulation",
            "Capital Markets",
            "IT & Cyber Risk",
            "Consumer Protection",
            "AML / CFT",
        ]
    )


class ChunkingConfig(BaseModel):
    """Configuration for statutory document chunking strategy."""

    strategy: Literal["structure_aware", "fixed_size", "structure_aware_no_header"] = (
        "structure_aware"
    )
    include_breadcrumbs: bool = True
    target_chunk_size: int = Field(default=512, ge=64, le=4096)
    overlap: int = Field(default=64, ge=0, le=512)
    chunks_path: str = "data/chunks/structure_aware_chunks.json"

    @field_validator("overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        target = info.data.get("target_chunk_size", 512)
        if v >= target:
            raise ValueError(
                f"overlap ({v}) must be strictly less than target_chunk_size ({target})"
            )
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for dense vector embedding model and backend."""

    model_name: str = "intfloat/multilingual-e5-large"
    dimension: int = Field(default=1024, ge=64)
    backend: Literal["fastembed", "onnx"] = "fastembed"
    max_length: int = Field(default=512, ge=128, le=8192)
    prefix_query: str = "query: "
    prefix_passage: str = "passage: "
    batch_size: int = Field(default=64, ge=1, le=512)


class RetrievalConfig(BaseModel):
    """Configuration for statutory retrieval pipeline."""

    mode: Literal["dense", "bm25", "hybrid", "hybrid+rerank"] = "dense"
    top_k: int = Field(default=5, ge=1, le=50)
    vector_store: Literal["qdrant", "memory"] = "qdrant"
    index_dir: str = "data/index"
    collection_name: str | None = "easypasal_structure_aware_multilingual_e5_large"
    distance: Literal["Cosine", "Dot", "Euclidean"] = "Cosine"
    score_threshold: float | None = None
    reranker: str | None = None


class GeneratorModelConfig(BaseModel):
    """Configuration for a specific generator LLM backend instance."""

    provider: Literal["llamacpp", "vllm", "mock", "hf", "openai"] = "llamacpp"
    repo_id: str = "bartowski/Llama-3.2-1B-Instruct-GGUF"
    filename: str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    quantization: str = "Q4_K_M"
    n_ctx: int = Field(default=8192, ge=512, le=32768)
    n_threads: int = Field(default=4, ge=1, le=64)
    max_tokens: int = Field(default=160, ge=16, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    stop_tokens: list[str] = Field(
        default_factory=lambda: [
            "<|eot_id|>",
            "<|im_end|>",
            "<|end_of_text|>",
            "<|endoftext|>",
            "</answer>",
            "\n\n\n",
        ]
    )


class GeneratorConfig(BaseModel):
    """Configuration for CPU serving and GPU target generators."""

    cpu_default: GeneratorModelConfig = Field(default_factory=GeneratorModelConfig)
    gpu_target: GeneratorModelConfig | None = Field(
        default_factory=lambda: GeneratorModelConfig(
            provider="vllm",
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            n_ctx=8192,
            max_tokens=160,
            temperature=0.1,
        )
    )


class GuardrailsConfig(BaseModel):
    """Configuration for citation validation, abstention, and filtering."""

    verify_citations: bool = True
    strict_grounding: bool = False
    status_filter: str | None = "berlaku"
    default_language: str = "id"
    enable_router: bool = True
    router_adapter_path: str | None = "experiments/e7_qlora/adapter"


class TrafficSplitConfig(BaseModel):
    """Configuration for deterministic traffic splitting and variant routing."""

    enabled: bool = False
    default_variant: str = "baseline_v0.1"
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "baseline_v0.1": 0.5,
            "gepa_v0.2": 0.5,
        }
    )
    salt: str = "easypasal-traffic-split-v1"

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("Traffic split weights dictionary cannot be empty")
        for k, w in v.items():
            if w < 0:
                raise ValueError(f"Weight for variant '{k}' must be non-negative, got {w}")
        total = sum(v.values())
        if total <= 0:
            raise ValueError("Sum of traffic split weights must be greater than zero")
        return v


class MonitoringConfig(BaseModel):
    """Configuration for structured request logging, PII redaction, and feedback (Plan §9)."""

    enabled: bool = True
    log_file: str | None = "logs/requests.jsonl"
    feedback_file: str | None = "logs/feedback.jsonl"
    redact_pii: bool = True
    salt: str = "easypasal-monitoring-salt-v1"


class EasyPasalConfig(BaseModel):
    """Master configuration container for EasyPasal system."""

    version: str = "0.1"
    variant_id: str = "baseline_v0.1"
    prompt_variant: Literal["gepa", "baseline"] = "baseline"
    description: str = "EasyPasal v0.1 Frozen Production Baseline"
    created_at: str | None = "2026-10-15"
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    traffic_split: TrafficSplitConfig = Field(default_factory=TrafficSplitConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> EasyPasalConfig:
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated EasyPasalConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If YAML syntax is invalid or schema validation fails.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid YAML structure in {p}: expected dictionary")
            return cls.model_validate(data)
        except yaml.YAMLError as err:
            raise ValueError(f"YAML parsing error in {p}: {err}") from err

    def to_yaml(self, path: Path | str | None = None) -> str:
        """Serialize configuration to a YAML string, optionally saving to a file.

        Args:
            path: Optional destination file path.

        Returns:
            YAML string representation.
        """
        data = self.model_dump(mode="json")
        yaml_str = str(yaml.dump(data, sort_keys=False, allow_unicode=True))
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str

    @classmethod
    def load_default(cls, path: Path | str | None = None) -> EasyPasalConfig:
        """Load default active configuration.

        Checks explicitly provided path, then `configs/default.yaml`, then
        `configs/baseline_v0.1.yaml`, falling back to compiled default model.
        """
        if path is not None:
            return cls.from_yaml(path)

        if DEFAULT_CONFIG_PATH.exists():
            logger.info("Loading active configuration from %s", DEFAULT_CONFIG_PATH)
            return cls.from_yaml(DEFAULT_CONFIG_PATH)

        if BASELINE_CONFIG_PATH.exists():
            logger.info("Loading baseline configuration from %s", BASELINE_CONFIG_PATH)
            return cls.from_yaml(BASELINE_CONFIG_PATH)

        logger.info("Configuration files not found on disk. Instantiating compiled defaults.")
        return cls()
