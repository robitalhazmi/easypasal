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
    """Determine if a retrieved candidate chunk fails sectoral or conjunctive entity-concept alignment.

    Applies two general retrieval hygiene rules:
    1. Sectoral Exclusivity: Excludes statutes exclusive to one financial sub-sector (e.g., pure Banking
       or pure Capital Markets) when the query explicitly targets a disjoint sub-sector (e.g., Fintech P2P).
    2. Conjunctive Entity-Concept Gate: When a query targets a specialized sub-sector entity (e.g., P2P/LPBBTI)
       and asks for a quantitative cap/rate or specific operational rule, excludes split-query distractor
       chunks from umbrella statutes that lack conjunctive coverage of both the target entity and concept.
    """
    is_p2p_query = bool(
        re.search(
            r"\b(?:fintech\s+p2p|p2p\s+lending|p2p\b|pinjol|pinjaman\s+daring|lpbti|lpbbti|peer-to-peer)\b",
            query_lower,
        )
    )
    has_banking_query = bool(
        re.search(
            r"\b(?:bank\b|perbankan|bank\s+umum|bpr\b|tabungan|deposito|giro)\b",
            query_lower,
        )
    )
    has_capital_markets_query = bool(
        re.search(
            r"\b(?:pasar\s+modal|saham|efek|bursa|reksa\s+dana|emiten|perusahaan\s+terbuka|rups|kliring)\b",
            query_lower,
        )
    )

    doc_info = f"{chunk.doc_number} {chunk.doc_title} {chunk.citation_label}".lower()
    chunk_body = f"{doc_info} {chunk.text}".lower()

    is_banking_only_doc = any(
        term in doc_info
        for term in [
            "uu nomor 10 tahun 1998",
            "uu nomor 7 tahun 1992",
            "pojk.03",
            "seojk.03",
            "pojk 21 tahun 2023",
            "oleh bank umum",
            "bagi bank umum",
        ]
    )
    is_capital_markets_only_doc = any(
        term in doc_info
        for term in [
            "uu nomor 8 tahun 1995",
            "pojk.04",
            "pasar modal",
            "lembaga kliring",
            "perusahaan efek",
            "bursa efek",
            "perusahaan terbuka",
            "emiten",
        ]
    )

    # 1. Disjoint Sectoral Exclusivity Check
    if is_p2p_query and not has_banking_query and is_banking_only_doc:
        return True
    if is_p2p_query and not has_capital_markets_query and is_capital_markets_only_doc:
        return True
    if has_banking_query and not has_capital_markets_query and is_capital_markets_only_doc:
        return True
    if has_capital_markets_query and not has_banking_query and is_banking_only_doc:
        return True

    # 2. Conjunctive Entity-Concept Gate for Specialized Sub-Sector Queries
    if is_p2p_query and not has_banking_query and not has_capital_markets_query:
        has_p2p_entity_in_chunk = bool(
            re.search(
                r"\b(?:peer-to-peer|p2p|lpbbti|lpbti|pendanaan\s+bersama|pinjaman\s+daring)\b",
                chunk_body,
            )
        )
        is_general_cross_sector_topic = bool(
            re.search(
                r"\b(?:pengaduan|sengketa|laps|data\s+pribadi|kerahasiaan|pencucian\s+uang|apu|ppt|cdd|edd)\b",
                query_lower,
            )
        )
        if not is_general_cross_sector_topic and not has_p2p_entity_in_chunk:
            return True

        is_rate_or_cap_query = bool(
            re.search(
                r"\b(?:batas\s+maksimum|batas\s+maksimal|manfaat\s+ekonomi|bunga\s+pinjaman|suku\s+bunga|besaran\s+bunga)\b",
                query_lower,
            )
        )
        if is_rate_or_cap_query:
            has_rate_or_cap_in_chunk = bool(
                re.search(
                    r"\b(?:batas\s+maksimum|batas\s+maksimal|manfaat\s+ekonomi|suku\s+bunga|bunga\s+pinjaman)\b",
                    chunk_body,
                )
            )
            if not (has_p2p_entity_in_chunk and has_rate_or_cap_in_chunk):
                return True

    return False


def _expand_primary_statutory_norms(
    query: str,
    chunks: list[RetrievedChunk],
    bm25_index: BM25Index | None,
    status_filter: str | None = None,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Expand cross-referenced or neighboring primary obligation articles when an obligation query

    retrieves secondary remedial, conditional exception, or dependent cross-referencing clauses.
    Ensures the standalone root obligation (e.g., Pasal N ayat (1)) is present and ranked first.
    """
    if not chunks or not query:
        return chunks

    from easypasal.generation.prompts import (
        is_enumeration_query,
        score_primary_obligation_alignment,
    )
    from easypasal.retrieval.fusion import detect_normative_intent

    if detect_normative_intent(query) != "obligation":
        return chunks

    is_enum_q = is_enumeration_query(query)

    # 1. Collect target (doc_identifier, target_pasal) pairs from cross-references, root clauses, and sibling provisions
    target_pasals_by_doc: dict[str, set[str]] = {}
    target_docs_for_enum: set[str] = set()
    for c in chunks:
        doc_keys = [k.lower().strip() for k in (c.doc_id, c.doc_number) if k]
        if not doc_keys:
            continue

        if is_enum_q:
            target_docs_for_enum.update(doc_keys)

        chunk_targets: set[str] = set()

        # A. Explicit cross-references: "sebagaimana dimaksud dalam Pasal 35"
        for ref_match in re.finditer(
            r"sebagaimana\s+dimaksud\s+dalam\s+Pasal\s+(\d+[a-z]?)",
            c.text,
            re.IGNORECASE,
        ):
            chunk_targets.add(ref_match.group(1).lower())

        # B. Root clause of current article and adjacent sibling provisions (e.g. Pasal 74 oral -> Pasal 75 written)
        if c.pasal:
            p_str = str(c.pasal).lower()
            chunk_targets.add(p_str)
            if p_str.isdigit():
                p_num = int(p_str)
                if p_num > 1:
                    chunk_targets.add(str(p_num - 1))
                if p_num > 2:
                    chunk_targets.add(str(p_num - 2))
                chunk_targets.add(str(p_num + 1))

        for k in doc_keys:
            if chunk_targets:
                target_pasals_by_doc.setdefault(k, set()).update(chunk_targets)

    expanded_pool = list(chunks)
    existing_ids = {c.chunk_id for c in expanded_pool}

    # 2. Look up candidate primary obligation chunks in BM25 corpus payloads if available
    corpus_payloads = getattr(bm25_index, "corpus_payloads", None) if bm25_index else None
    if (
        corpus_payloads
        and isinstance(corpus_payloads, list)
        and (target_pasals_by_doc or target_docs_for_enum)
    ):
        for payload in corpus_payloads:
            if not isinstance(payload, dict):
                continue
            cid = str(payload.get("chunk_id", ""))
            if not cid or cid in existing_ids:
                continue
            if (
                "penjelasan" in cid.lower()
                or "penjelasan" in str(payload.get("citation_label", "")).lower()
            ):
                continue

            p_status = str(payload.get("status", "berlaku")).lower()
            if status_filter and p_status != status_filter.lower():
                continue

            p_doc_id = str(payload.get("doc_id", "")).lower().strip()
            p_doc_num = str(payload.get("doc_number", "")).lower().strip()
            matched_targets: set[str] | None = None
            for d_key, t_set in target_pasals_by_doc.items():
                if d_key and d_key in (p_doc_id, p_doc_num):
                    matched_targets = t_set
                    break
            is_enum_doc_match = is_enum_q and bool({p_doc_id, p_doc_num} & target_docs_for_enum)
            if not matched_targets and not is_enum_doc_match:
                continue

            p_pasal = str(payload.get("pasal", "") or "").lower()
            p_ayat = payload.get("ayat")
            if matched_targets and p_pasal in matched_targets:
                allowed_ayats = ("1", "2", "3") if is_enum_q else ("1",)
                if p_ayat is not None and str(p_ayat) not in allowed_ayats:
                    continue
                min_align = 2.0
            elif is_enum_doc_match:
                # For enumeration queries on circulars (SEOJK) or sibling governance chapters,
                # require strong enumeration alignment (>= 4.5)
                min_align = 4.5
            else:
                continue

            cand_chunk = RetrievedChunk.from_payload(
                chunk_id=cid,
                score=chunks[0].score,
                payload=payload,
            )
            if score_primary_obligation_alignment(cand_chunk, query) >= min_align:
                expanded_pool.append(cand_chunk)
                existing_ids.add(cid)

    # 3. Promote the highest-scoring primary obligation chunk to Rank 1 and retain high-scoring sibling provisions
    scored_primaries = [
        (score_primary_obligation_alignment(c, query), idx, c)
        for idx, c in enumerate(expanded_pool)
    ]
    best_score, _best_idx, best_chunk = max(scored_primaries, key=lambda item: (item[0], -item[1]))
    if best_score >= 2.0:
        top_score = max(c.score for c in chunks)
        best_chunk.score = round(top_score + 0.01, 6)
        doc_primary_counts: dict[str, int] = {}
        best_doc_key = (best_chunk.doc_id or best_chunk.doc_number or "").lower().strip()
        if best_doc_key:
            doc_primary_counts[best_doc_key] = 1
        other_primaries: list[RetrievedChunk] = []
        for s, _idx, c in sorted(
            scored_primaries, key=lambda item: (item[0], -item[1]), reverse=True
        ):
            if c.chunk_id == best_chunk.chunk_id or s < 2.0:
                continue
            d_key = (c.doc_id or c.doc_number or "").lower().strip()
            if d_key and doc_primary_counts.get(d_key, 0) >= 3:
                continue
            if d_key:
                doc_primary_counts[d_key] = doc_primary_counts.get(d_key, 0) + 1
            other_primaries.append(c)
        remaining = [
            c
            for c in chunks
            if c.chunk_id != best_chunk.chunk_id
            and c.chunk_id not in {op.chunk_id for op in other_primaries}
        ]
        reordered = [best_chunk] + other_primaries + remaining
        limit = max(len(chunks), top_k)
        return reordered[:limit]

    return chunks


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
                query=query_clean,
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
                query=query_clean,
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
            from easypasal.retrieval.fusion import _compute_normative_salience_boost

            scored_dense = []
            for orig_rank, dense_item in enumerate(dense_candidates, start=1):
                n_boost = _compute_normative_salience_boost(query_clean, dense_item.payload or {})
                adj_score = round(dense_item.score * (1.0 + 0.25 * n_boost), 6)
                scored_dense.append((adj_score, -orig_rank, orig_rank, dense_item))
            scored_dense.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for adj_sc, _neg_r, orig_rank, dense_item in scored_dense[:top_k]:
                final_chunks.append(
                    RetrievedChunk.from_payload(
                        chunk_id=dense_item.chunk_id,
                        score=adj_sc,
                        payload=dense_item.payload,
                        dense_score=dense_item.score,
                        dense_rank=orig_rank,
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

        # Cross-Sector Entity & Conjunctive Alignment: Filter out disjoint sector distractors
        if final_chunks:
            q_low = query_clean.lower()
            final_chunks = [c for c in final_chunks if not _is_sector_mismatch(q_low, c)]

        # Structural & Cross-Reference Primary Norm Expansion for Obligation Queries
        if final_chunks:
            final_chunks = _expand_primary_statutory_norms(
                query=query_clean,
                chunks=final_chunks,
                bm25_index=self.bm25_index,
                status_filter=status_filter,
                top_k=top_k,
            )

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
