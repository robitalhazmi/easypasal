"""Unified hybrid retrieval engine combining dense vector search and sparse BM25 search."""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

from easypasal.index.bm25 import BM25Index, BM25SearchResult
from easypasal.index.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingBackend,
    get_embedding_backend,
    get_model_slug,
)
from easypasal.index.vector_store import QdrantVectorStore, VectorSearchResult
from easypasal.retrieval.fusion import DEFAULT_K_RRF, reciprocal_rank_fusion
from easypasal.retrieval.models import RetrievalResult, RetrievedChunk
from easypasal.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    BaseReranker,
    FlashRankReranker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.retrieval.retriever")


def _is_sector_mismatch(query_lower: str, chunk: RetrievedChunk) -> bool:
    """Determine if a retrieved candidate chunk belongs to a disjoint financial sector from the query."""
    is_p2p_query = bool(
        re.search(r"\b(?:fintech\s+p2p|p2p\s+lending|pinjol|lpbti|peer-to-peer)\b", query_lower)
    )
    has_capital_markets_query = bool(
        re.search(
            r"\b(?:pasar\s+modal|saham|efek|bursa|reksa\s+dana|emiten|kliring)\b", query_lower
        )
    )

    # P2P lending query with Capital Market distractor chunk
    if is_p2p_query and not has_capital_markets_query:
        doc_info = f"{chunk.doc_number} {chunk.doc_title} {chunk.citation_label}".lower()
        if any(
            term in doc_info
            for term in ["pasar modal", "lembaga kliring", "perusahaan efek", "bursa efek"]
        ):
            return True

    return False


class HybridRetriever:
    """Hybrid retrieval engine with Reciprocal Rank Fusion (RRF) and optional cross-encoder reranking."""

    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        bm25_index: BM25Index,
        embedding_backend: EmbeddingBackend,
        collection_name: str = "easypasal_structure_aware",
        k_rrf: int = DEFAULT_K_RRF,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        reranker: BaseReranker | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.qdrant_store = qdrant_store
        self.bm25_index = bm25_index
        self.embedding_backend = embedding_backend
        self.collection_name = collection_name
        self.k_rrf = k_rrf
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.reranker = reranker
        self.score_threshold = score_threshold

    @classmethod
    def from_paths(
        cls,
        index_dir: Path | str = "data/index",
        strategy: str = "structure_aware",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str | None = None,
        k_rrf: int = DEFAULT_K_RRF,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        reranker: BaseReranker | None = None,
        rerank: bool = False,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        score_threshold: float | None = None,
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

        if collection_name is None:
            if model_name == DEFAULT_EMBEDDING_MODEL:
                target_collection = f"easypasal_{strategy}"
            else:
                slug = get_model_slug(model_name)
                target_collection = f"easypasal_{strategy}_{slug}"
        else:
            target_collection = collection_name

        target_reranker = reranker
        if target_reranker is None and rerank:
            logger.info("Initializing FlashRank reranker %r...", reranker_model)
            target_reranker = FlashRankReranker(model_name=reranker_model)

        return cls(
            qdrant_store=qdrant_store,
            bm25_index=bm25_index,
            embedding_backend=embedding_backend,
            collection_name=target_collection,
            k_rrf=k_rrf,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            reranker=target_reranker,
            score_threshold=score_threshold,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        status_filter: str | None = None,
        candidate_limit: int | None = None,
        min_score: float | None = None,
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
        if mode in ("hybrid", "dense", "hybrid+rerank", "hybrid_rerank"):
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
        if mode in ("hybrid", "bm25", "hybrid+rerank", "hybrid_rerank"):
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
        elif mode in ("hybrid+rerank", "hybrid_rerank"):
            rerank_pool_limit = max(top_k, 15)
            fused_candidates = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                bm25_results=bm25_candidates,
                k_rrf=self.k_rrf,
                dense_weight=self.dense_weight,
                bm25_weight=self.bm25_weight,
                top_k=rerank_pool_limit,
            )
            if self.reranker is not None:
                final_chunks = self.reranker.rerank(
                    query=query_clean,
                    chunks=fused_candidates,
                    top_k=top_k,
                )
            else:
                logger.warning(
                    "Hybrid+rerank requested but no reranker configured; falling back to Hybrid RRF."
                )
                final_chunks = fused_candidates[:top_k]
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
                f"Unsupported retrieval mode: {mode!r}. Must be 'hybrid', 'dense', 'bm25', or 'hybrid+rerank'."
            )

        # Cross-Sector Entity Alignment: Deprioritize disjoint sector distractors
        if final_chunks:
            q_low = query_clean.lower()
            aligned_chunks = [c for c in final_chunks if not _is_sector_mismatch(q_low, c)]
            if aligned_chunks:
                final_chunks = aligned_chunks

        # Apply score threshold filter if specified
        eff_min_score = min_score if min_score is not None else self.score_threshold
        if eff_min_score is not None:
            final_chunks = [c for c in final_chunks if c.score >= eff_min_score]

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
        choices=["hybrid", "dense", "bm25", "hybrid+rerank", "hybrid_rerank"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (default: MiniLM-L12-v2)",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"FlashRank reranker model name (default: {DEFAULT_RERANKER_MODEL})",
    )
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

    should_rerank = args.mode in ("hybrid+rerank", "hybrid_rerank")
    retriever = HybridRetriever.from_paths(
        index_dir=args.index_dir,
        strategy=args.strategy,
        model_name=args.model_name,
        rerank=should_rerank,
        reranker_model=args.reranker_model,
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
        if result.mode in ("hybrid+rerank", "hybrid_rerank"):
            score_str = f"Rerank Score: {chunk.score:.4f}"
        elif result.mode == "hybrid":
            score_str = f"RRF: {chunk.score:.6f}"
        else:
            score_str = f"Score: {chunk.score:.4f}"

        provenance = []
        if chunk.rerank_rank is not None:
            provenance.append(f"Rerank #{chunk.rerank_rank} ({chunk.score:.4f})")
        if chunk.dense_rank is not None:
            dense_sc = f"{chunk.dense_score:.4f}" if chunk.dense_score is not None else "N/A"
            provenance.append(f"Dense #{chunk.dense_rank} ({dense_sc})")
        if chunk.bm25_rank is not None:
            bm25_sc = f"{chunk.bm25_score:.2f}" if chunk.bm25_score is not None else "N/A"
            provenance.append(f"BM25 #{chunk.bm25_rank} ({bm25_sc})")
        prov_str = f" [{', '.join(provenance)}]" if provenance else ""

        print(f"\n[{idx}] {chunk.citation_label} ({score_str}){prov_str}")
        print(f"    Status: {chunk.status} | Doc: {chunk.doc_number}")
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        print(f"    Excerpt: {preview}")


if __name__ == "__main__":
    main()
