"""Index building pipeline for EasyPasal.

Builds and persists dense Qdrant vector collections and sparse BM25 indices
from generated JSONL chunks. Logs build time, memory footprint, throughput,
and storage statistics for acceptance criteria verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from easypasal.index.bm25 import BM25Index
from easypasal.index.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingBackend,
    get_embedding_backend,
)
from easypasal.index.vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.index.build")


@dataclass
class IndexStats:
    """Summary metrics of an index build execution."""

    strategy: str
    model_name: str
    dimension: int
    num_chunks: int
    elapsed_seconds: float
    chunks_per_second: float
    qdrant_size_mb: float
    bm25_size_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_dir_size_mb(path: Path) -> float:
    """Compute total size of a file or directory in megabytes."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024 * 1024)


def load_chunks_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load chunk dictionaries from a JSONL file."""
    chunks: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                chunks.append(json.loads(line_str))
    return chunks


def build_indices(
    chunks_file: Path | str,
    index_dir: Path | str = "data/index",
    strategy: str = "structure_aware",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    recreate: bool = True,
    embedding_backend: EmbeddingBackend | None = None,
) -> IndexStats:
    """Build and persist both Qdrant vector store and BM25 index from a chunks JSONL file."""
    chunks_path = Path(chunks_file)
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    logger.info("Loading chunks from %s...", chunks_path)
    chunks = load_chunks_jsonl(chunks_path)
    total_chunks = len(chunks)
    logger.info("Loaded %d chunks for strategy=%r.", total_chunks, strategy)

    start_time = time.perf_counter()

    # 1. Initialize Embedding Backend & Compute Dense Embeddings
    backend = embedding_backend or get_embedding_backend(model_name=model_name)
    dimension = backend.dimension
    logger.info(
        "Embedding %d chunks using %r (dimension=%d, batch_size=%d)...",
        total_chunks,
        backend.model_name,
        dimension,
        batch_size,
    )

    texts = [c.get("text", "") for c in chunks]
    vectors = backend.embed_documents(texts, batch_size=batch_size)

    # 2. Build Qdrant Vector Store
    qdrant_dir = index_path / "qdrant"
    collection_name = f"easypasal_{strategy}"
    logger.info(
        "Upserting vectors into embedded Qdrant (path=%s, collection=%r)...",
        qdrant_dir,
        collection_name,
    )

    store = QdrantVectorStore(path=qdrant_dir)
    store.init_collection(collection_name=collection_name, dimension=dimension, recreate=recreate)
    store.upsert_chunks(
        collection_name=collection_name,
        chunks=chunks,
        vectors=vectors,
        batch_size=batch_size,
    )

    # 3. Build & Serialize BM25 Sparse Index
    bm25_dir = index_path / "bm25"
    bm25_index = BM25Index.build_from_chunks(chunks)
    bm25_index.save(output_dir=bm25_dir, prefix=strategy)

    elapsed = time.perf_counter() - start_time
    chunks_per_sec = total_chunks / elapsed if elapsed > 0 else 0.0

    qdrant_mb = get_dir_size_mb(qdrant_dir)
    bm25_mb = get_dir_size_mb(bm25_dir / f"{strategy}_model.pkl") + get_dir_size_mb(
        bm25_dir / f"{strategy}_meta.json"
    )

    stats = IndexStats(
        strategy=strategy,
        model_name=backend.model_name,
        dimension=dimension,
        num_chunks=total_chunks,
        elapsed_seconds=round(elapsed, 2),
        chunks_per_second=round(chunks_per_sec, 1),
        qdrant_size_mb=round(qdrant_mb, 2),
        bm25_size_mb=round(bm25_mb, 2),
    )

    logger.info(
        "Index build complete: %d chunks indexed in %.2fs (%.1f chunks/s) | Qdrant: %.2f MB | BM25: %.2f MB",
        stats.num_chunks,
        stats.elapsed_seconds,
        stats.chunks_per_second,
        stats.qdrant_size_mb,
        stats.bm25_size_mb,
    )

    # 4. Probe Verification Query
    probe_query = "pelindungan konsumen dan kerugian PUJK"
    logger.info("Executing probe verification query: %r", probe_query)

    # Dense Probe
    query_vec = backend.embed_query(probe_query)
    dense_hits = store.search(collection_name, query_vec, top_k=3)
    logger.info(
        "Top Dense Hit: %s (score=%.4f)",
        dense_hits[0].citation_label if dense_hits else "None",
        dense_hits[0].score if dense_hits else 0.0,
    )

    # Sparse Probe
    sparse_hits = bm25_index.search(probe_query, top_k=3)
    logger.info(
        "Top BM25 Hit: %s (score=%.4f)",
        sparse_hits[0].citation_label if sparse_hits else "None",
        sparse_hits[0].score if sparse_hits else 0.0,
    )

    store.close()
    return stats


def main() -> None:
    """CLI runner to build EasyPasal retrieval indices."""
    parser = argparse.ArgumentParser(
        description="Build embedded Qdrant vector store and BM25 sparse index from corpus chunks."
    )
    parser.add_argument(
        "--strategy",
        choices=["structure_aware", "fixed_size", "all"],
        default="structure_aware",
        help="Chunking strategy to index (default: structure_aware)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model identifier (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/chunks"),
        help="Directory containing chunks JSONL files (default: data/chunks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/index"),
        help="Directory to save index data (default: data/index)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding and upsert (default: 64)",
    )

    args = parser.parse_args()

    strategies = ["structure_aware", "fixed_size"] if args.strategy == "all" else [args.strategy]

    for strat in strategies:
        chunks_file = args.input_dir / f"{strat}.jsonl"
        build_indices(
            chunks_file=chunks_file,
            index_dir=args.output_dir,
            strategy=strat,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
