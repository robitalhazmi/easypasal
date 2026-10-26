"""Reciprocal Rank Fusion (RRF) for combining dense and sparse search rankings."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from easypasal.retrieval.models import RetrievedChunk

if TYPE_CHECKING:
    from easypasal.index.bm25 import BM25SearchResult
    from easypasal.index.vector_store import VectorSearchResult

logger = logging.getLogger("easypasal.retrieval.fusion")

DEFAULT_K_RRF: int = 60


def _compute_normative_salience_boost(query: str, payload: dict[str, Any]) -> float:
    """Compute a modest, scale-invariant RRF boost (0.0 to 0.25) for chunks carrying

    the direct imperative action predicate or target interrogative answer to the query.
    """
    if not query:
        return 0.0

    text = payload.get("text", "")
    if not text:
        return 0.0

    q_low = query.lower()
    t_low = text.lower()
    boost = 0.0

    # 1. Direct Imperative Action Predicate Concordance
    # e.g., Query: "... wajib menempatkan ..." -> Action verb: "menempatkan"
    # Chunk: "... wajib menempatkan ..." -> Exact imperative predicate concordance!
    m_action = re.search(r"\b(?:wajib|harus|dilarang)\s+(\w+)", q_low)
    if m_action:
        action_verb = m_action.group(1)
        if len(action_verb) >= 4 and re.search(
            rf"\b(?:wajib|harus|dilarang)\s+(?:\w+\s+)?{re.escape(action_verb)}\b",
            t_low,
        ):
            boost += 0.15

    # 2. Interrogative Target Alignment
    # Where? ("di mana") -> Check for geographical/jurisdictional placement terms
    if (
        ("di mana" in q_low or "dimana" in q_low)
        and any(
            loc in t_low
            for loc in [
                "di wilayah indonesia",
                "di indonesia",
                "wilayah negara kesatuan",
                "luar wilayah indonesia",
                "pada kantor",
            ]
        )
        or any(
            k in q_low
            for k in [
                "berapa",
                "batas waktu",
                "jangka waktu",
                "paling lambat",
                "paling lama",
                "maksimum",
                "minimum",
            ]
        )
        and (
            re.search(
                r"\b(?:\d+|satu|dua|tiga|lima|tujuh|sepuluh|empat belas|dua puluh|tiga puluh)\s+(?:hari|bulan|tahun|hari kerja|jam)\b",
                t_low,
            )
            or re.search(r"\brp\s*[\d\.]+", t_low)
        )
    ):
        boost += 0.05

    # 3. Concise Statutory Density Bonus (Anti-Length-Penalization)
    if boost > 0 and len(text) < 450:
        boost += 0.03

    return min(boost, 0.25)


def reciprocal_rank_fusion(
    dense_results: list[VectorSearchResult],
    bm25_results: list[BM25SearchResult],
    k_rrf: int = DEFAULT_K_RRF,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    top_k: int = 10,
    query: str | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked candidate lists from dense vector search and sparse BM25 search.

    Uses Reciprocal Rank Fusion (RRF):
        RRF_score(d) = sum_{m in M} ( w_m / (k_rrf + rank_m(d)) )

    Args:
        dense_results: Ranked list of VectorSearchResult from dense Qdrant retrieval.
        bm25_results: Ranked list of BM25SearchResult from sparse BM25 retrieval.
        k_rrf: Smoothing constant to calibrate rank degradation (standard default: 60).
        dense_weight: Multiplier weight for dense vector rankings.
        bm25_weight: Multiplier weight for BM25 sparse rankings.
        top_k: Maximum number of fused results to return.
        query: Optional user query string to apply normative action salience boosting.

    Returns:
        A list of up to `top_k` `RetrievedChunk` instances sorted by descending RRF score.
    """
    if k_rrf <= 0:
        raise ValueError(f"k_rrf must be positive, got {k_rrf}")
    if top_k <= 0:
        return []

    # Map chunk_id to intermediate aggregation state
    candidates: dict[str, dict[str, Any]] = {}

    # 1. Process Dense Vector Results
    for rank, d_res in enumerate(dense_results, start=1):
        cid = d_res.chunk_id
        if cid not in candidates:
            candidates[cid] = {
                "chunk_id": cid,
                "rrf_score": 0.0,
                "dense_score": d_res.score,
                "dense_rank": rank,
                "bm25_score": None,
                "bm25_rank": None,
                "payload": d_res.payload,
            }
        else:
            candidates[cid]["dense_score"] = d_res.score
            candidates[cid]["dense_rank"] = rank

        candidates[cid]["rrf_score"] += dense_weight / (k_rrf + rank)

    # 2. Process BM25 Sparse Results
    for rank, b_res in enumerate(bm25_results, start=1):
        cid = b_res.chunk_id
        if cid not in candidates:
            candidates[cid] = {
                "chunk_id": cid,
                "rrf_score": 0.0,
                "dense_score": None,
                "dense_rank": None,
                "bm25_score": b_res.score,
                "bm25_rank": rank,
                "payload": b_res.payload,
            }
        else:
            candidates[cid]["bm25_score"] = b_res.score
            candidates[cid]["bm25_rank"] = rank
            # Merge payload keys if dense had incomplete fields
            if not candidates[cid]["payload"]:
                candidates[cid]["payload"] = b_res.payload

        candidates[cid]["rrf_score"] += bm25_weight / (k_rrf + rank)

    # 3. Apply Normative Salience Boosting if query is provided
    if query:
        for item in candidates.values():
            n_boost = _compute_normative_salience_boost(query, item.get("payload") or {})
            if n_boost > 0:
                item["rrf_score"] = item["rrf_score"] * (1.0 + n_boost)

    # 4. Sort candidates by RRF score descending with deterministic secondary keys
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda item: (
            item["rrf_score"],
            item["dense_score"] or -1.0,
            item["bm25_score"] or -1.0,
            item["chunk_id"],
        ),
        reverse=True,
    )

    # 5. Convert top-k candidates to RetrievedChunk instances
    fused_chunks: list[RetrievedChunk] = []
    for item in sorted_candidates[:top_k]:
        chunk = RetrievedChunk.from_payload(
            chunk_id=item["chunk_id"],
            score=round(item["rrf_score"], 6),
            payload=item["payload"],
            dense_score=item["dense_score"],
            dense_rank=item["dense_rank"],
            bm25_score=item["bm25_score"],
            bm25_rank=item["bm25_rank"],
        )
        fused_chunks.append(chunk)

    return fused_chunks
