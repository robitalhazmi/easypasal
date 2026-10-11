"""Dense embedding abstraction and FastEmbed backend for EasyPasal.

Provides an extensible EmbeddingBackend protocol and an ONNX-accelerated
FastEmbed backend optimized for fast CPU inference and memory efficiency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.index.embeddings")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol defining the interface for embedding models."""

    @property
    def model_name(self) -> str:
        """The identifier of the embedding model."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a list of document chunk texts into float vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a float vector."""
        ...


class FastEmbedBackend:
    """ONNX-powered CPU embedding backend using FastEmbed."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path | str | None = None,
        max_length: int = 512,
        threads: int | None = None,
    ) -> None:
        self._model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_length = max_length
        self.threads = threads
        self._model: Any = None
        self._dimension: int | None = None

    def _init_model(self) -> None:
        """Lazily initialize the underlying FastEmbed TextEmbedding engine."""
        if self._model is not None:
            return

        from fastembed import TextEmbedding

        logger.info("Initializing FastEmbed model=%r...", self._model_name)
        self._model = TextEmbedding(
            model_name=self._model_name,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            max_length=self.max_length,
            threads=self.threads,
        )

        # Inspect known dimensions
        for m in TextEmbedding.list_supported_models():
            if m["model"] == self._model_name:
                self._dimension = int(m["dim"])
                break

        if self._dimension is None:
            # Fallback probe
            probe = next(self._model.embed(["probe"]))
            self._dimension = len(probe)

        logger.info(
            "FastEmbed model %r ready (dimension=%d).",
            self._model_name,
            self._dimension,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._init_model()
        assert self._dimension is not None
        return self._dimension

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Generate normalized embedding vectors for document chunks."""
        self._init_model()
        assert self._model is not None

        if not texts:
            return []

        embeddings_gen = self._model.embed(texts, batch_size=batch_size)
        return [[float(x) for x in vec] for vec in embeddings_gen]

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding vector for a search query."""
        self._init_model()
        assert self._model is not None

        gen = self._model.embed([query])
        return [float(x) for x in next(gen)]


def get_embedding_backend(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: Path | str | None = None,
) -> EmbeddingBackend:
    """Factory function to acquire an embedding backend."""
    return FastEmbedBackend(model_name=model_name, cache_dir=cache_dir)
