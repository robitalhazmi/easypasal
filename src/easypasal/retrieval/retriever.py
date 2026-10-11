"""Unified hybrid retrieval engine combining dense vector search and sparse BM25 search."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from easypasal.index.bm25 import BM25Index, BM25SearchResult
from easypasal.index.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingBackend,
    get_embedding_backend,
)
from easypasal.index.vector_store import QdrantVectorStore, VectorSearchResult
from easypasal.retrieval.fusion import DEFAULT_K_RRF, reciprocal_rank_fusion
from easypasal.retrieval.models import RetrievalResult, RetrievedChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.retrieval.retriever")


class HybridRetriever:
    """Hybrid retrieval engine with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        bm25_index: BM25Index,
        embedding_backend: EmbeddingBackend,
        collection_name: str = "easypasal_structure_aware",
        k_rrf: int = DEFAULT_K_RRF,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.qdrant_store = qdrant_store
        self.bm25_index = bm25_index
        self.embedding_backend = embedding_backend
        self.collection_name = collection_name
        self.k_rrf = k_rrf
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    @classmethod
    def from_paths(
        cls,
        index_dir: Path | str = "data/index",
        strategy: str = "structure_aware",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        k_rrf: int = DEFAULT_K_RRF,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> HybridRetriever:
        """Initialize HybridRetriever from serialized on-disk index directories."""
        index_path = Path(index_dir)
        qdrant_dir = index_path / "qdrant"
        bm25_dir = index_path / "bm25"

        if not qdrant_dir.exists():
            raise FileNotFoundError(f"Qdrant directory does not exist: {qdrant_dir}")
        if not bm25_dir.exists():
            raise FileNotFoundError(f"BM25 directory does not exist: {bm25_dir}")

        logger.info("Loading Qdrant vector store from %s...", qdrant_dir)
        qdrant_store = QdrantVectorStore(path=qdrant_dir)

        logger.info("Loading BM25 index from %s (strategy=%r)...", bm25_dir, strategy)
        bm25_index = BM25Index.load(input_dir=bm25_dir, prefix=strategy)

        logger.info("Initializing embedding backend %r...", model_name)
        embedding_backend = get_embedding_backend(model_name=model_name)

        collection_name = f"easypasal_{strategy}"
        return cls(
            qdrant_store=qdrant_store,
            bm25_index=bm25_index,
            embedding_backend=embedding_backend,
            collection_name=collection_name,
            k_rrf=k_rrf,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        status_filter: str | None = None,
        candidate_limit: int | None = None,
    ) -> RetrievalResult:
        """Perform document retrieval using hybrid RRF, pure dense, or pure BM25 search.

        Args:
            query: Input natural language query or statutory reference.
            top_k: Number of final chunks to return.
            mode: Retrieval mode ('hybrid', 'dense', or 'bm25').
            status_filter: Optional regulation status filter (e.g., 'berlaku').
            candidate_limit: Pool size of candidates to retrieve per channel before fusion.

        Returns:
            RetrievalResult containing ranked RetrievedChunk items and provenance data.
        """
        start_time = time.perf_counter()
        query_clean = query.strip()
        if not query_clean:
            return RetrievalResult(
                query=query,
                mode=mode,
                chunks=[],
                total_candidates=0,
                dense_count=0,
                bm25_count=0,
                duration_ms=0.0,
            )

        pool_size = candidate_limit or max(top_k * 3, 30)

        dense_candidates: list[VectorSearchResult] = []
        bm25_candidates: list[BM25SearchResult] = []

        # 1. Dense Vector Retrieval
        if mode in ("hybrid", "dense"):
            query_vec = self.embedding_backend.embed_query(query_clean)
            dense_candidates = self.qdrant_store.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                top_k=pool_size,
            )
            if status_filter:
                dense_candidates = [
                    res for res in dense_candidates if res.status.lower() == status_filter.lower()
                ]

        # 2. Sparse BM25 Retrieval
        if mode in ("hybrid", "bm25"):
            bm25_candidates = self.bm25_index.search(
                query=query_clean,
                top_k=pool_size,
            )
            if status_filter:
                bm25_candidates = [
                    res for res in bm25_candidates if res.status.lower() == status_filter.lower()
                ]

        # 3. Mode Resolution & Ranking
        final_chunks: list[RetrievedChunk] = []

        if mode == "hybrid":
            final_chunks = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                bm25_results=bm25_candidates,
                k_rrf=self.k_rrf,
                dense_weight=self.dense_weight,
                bm25_weight=self.bm25_weight,
                top_k=top_k,
            )
        elif mode == "dense":
            for rank, dense_item in enumerate(dense_candidates[:top_k], start=1):
                final_chunks.append(
                    RetrievedChunk.from_payload(
                        chunk_id=dense_item.chunk_id,
                        score=dense_item.score,
                        payload=dense_item.payload,
                        dense_score=dense_item.score,
                        dense_rank=rank,
                    )
                )
        elif mode == "bm25":
            for rank, bm25_item in enumerate(bm25_candidates[:top_k], start=1):
                final_chunks.append(
                    RetrievedChunk.from_payload(
                        chunk_id=bm25_item.chunk_id,
                        score=bm25_item.score,
                        payload=bm25_item.payload,
                        bm25_score=bm25_item.score,
                        bm25_rank=rank,
                    )
                )
        else:
            raise ValueError(
                f"Unsupported retrieval mode: {mode!r}. Must be 'hybrid', 'dense', or 'bm25'."
            )

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        total_unique = len(
            {c.chunk_id for c in dense_candidates} | {c.chunk_id for c in bm25_candidates}
        )

        return RetrievalResult(
            query=query,
            mode=mode,
            chunks=final_chunks,
            total_candidates=total_unique,
            dense_count=len(dense_candidates),
            bm25_count=len(bm25_candidates),
            duration_ms=duration_ms,
        )

    def close(self) -> None:
        """Release underlying vector store and client resources."""
        self.qdrant_store.close()


def main() -> None:
    """CLI entrypoint to test and verify hybrid retrieval interactively."""
    parser = argparse.ArgumentParser(
        description="EasyPasal Hybrid Retrieval (Dense Qdrant + Sparse BM25 + RRF)",
    )
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--index-dir",
        default="data/index",
        help="Base index directory (default: data/index)",
    )
    parser.add_argument(
        "--strategy",
        default="structure_aware",
        help="Chunking strategy (default: structure_aware)",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="Filter regulation status (e.g. berlaku)",
    )

    args = parser.parse_args()

    retriever = HybridRetriever.from_paths(
        index_dir=args.index_dir,
        strategy=args.strategy,
    )

    result = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        mode=args.mode,
        status_filter=args.status,
    )

    print("\n" + "=" * 80)
    print(f"QUERY: {result.query}")
    print(
        f"MODE: {result.mode.upper()} | TOP-{len(result.chunks)} | LATENCY: {result.duration_ms:.2f} ms | CANDIDATES: {result.total_candidates}"
    )
    print("=" * 80)

    if not result.chunks:
        print("No matching chunks found.")
        return

    for idx, chunk in enumerate(result.chunks, start=1):
        score_str = (
            f"RRF: {chunk.score:.6f}" if result.mode == "hybrid" else f"Score: {chunk.score:.4f}"
        )
        provenance = []
        if chunk.dense_rank is not None:
            provenance.append(f"Dense #{chunk.dense_rank} ({chunk.dense_score:.4f})")
        if chunk.bm25_rank is not None:
            provenance.append(f"BM25 #{chunk.bm25_rank} ({chunk.bm25_score:.2f})")
        prov_str = f" [{', '.join(provenance)}]" if provenance else ""

        print(f"\n[{idx}] {chunk.citation_label} ({score_str}){prov_str}")
        print(f"    Status: {chunk.status} | Doc: {chunk.doc_number}")
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        print(f"    Excerpt: {preview}")


if __name__ == "__main__":
    main()
