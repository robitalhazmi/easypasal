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


def detect_normative_intent(query: str) -> str:
    """Classify the user's normative intent: 'obligation', 'permission_exemption', 'sanction_enforcement', or 'general'."""
    if not query:
        return "general"
    q_low = query.lower()

    # Sanction / Enforcement / Authority power queries
    # e.g. "kapan ojk dapat meminta", "apa sanksi", "denda pelanggaran", "wewenang ojk"
    if re.search(
        r"\b(?:sanksi|denda|hukuman|pidana|administratif|teguran|peringatan|pembekuan|pencabutan|penalty|sanction|enforcement|punishment|fine)\b",
        q_low,
    ) or re.search(
        r"\b(?:kapan|apakah)\s+(?:ojk|otoritas|bi|bank indonesia)\s+(?:dapat|berwenang|meminta|memerintahkan)\b",
        q_low,
    ):
        return "sanction_enforcement"

    # Permission / Exemption queries
    # e.g. "kapan boleh di luar negeri", "syarat pengecualian", "izin penempatan luar negeri"
    if re.search(
        r"\b(?:dapat|pengecualian|izin|persetujuan|dispensasi|may|permitted|exception|exemption|waiver)\b",
        q_low,
    ) and not re.search(r"\b(?:wajib|harus|kewajiban|keharusan)\b", q_low):
        return "permission_exemption"

    # Obligation / Prohibition queries
    # e.g. "di mana bank umum wajib menempatkan", "apa kewajiban pujk", "harus dipenuhi"
    if re.search(
        r"\b(?:wajib|harus|kewajiban|keharusan|dilarang|larangan|must|mandatory|required|shall|prohibited|obligation)\b",
        q_low,
    ):
        return "obligation"

    return "general"


def _compute_normative_salience_boost(query: str, payload: dict[str, Any]) -> float:
    """Compute a scale-invariant RRF boost/penalty (-0.25 to +0.35) for chunks carrying
    normative alignment with the query's legal intent.
    """
    if not query:
        return 0.0

    text = payload.get("text", "")
    if not text:
        return 0.0

    q_low = query.lower()
    t_low = text.lower()
    boost = 0.0

    intent = detect_normative_intent(query)

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

    # 2. Intent-Based Normative Primacy Boost / Penalty
    # Check if chunk represents authority corrective action / sanction vs entity primary obligation
    has_authority_remedial = bool(
        re.search(
            r"\b(?:ojk|otoritas jasa keuangan|bank indonesia|bi)\s+(?:dapat\s+meminta|berwenang|memerintahkan|mengenakan|menjatuhkan)\b",
            t_low,
        )
        or re.search(r"\b(?:dikenai|pengenaan)\s+sanksi\b", t_low)
    )
    has_entity_obligation = bool(
        re.search(
            r"\b(?:bank|pujk|perusahaan|penyelenggara|ljk|pihak)\s+(?:wajib|harus|dilarang)\b",
            t_low,
        )
        or re.search(
            r"\bwajib\s+(?:menempatkan|memiliki|menerapkan|menyediakan|menyampaikan|memenuhi|menggunakan|menjaga|melakukan)\b",
            t_low,
        )
    )

    if intent == "obligation":
        if has_entity_obligation and not has_authority_remedial:
            boost += 0.15
        elif has_authority_remedial and not (has_entity_obligation and t_low.startswith("bank")):
            # Downweight remedial/supervisory clause when user asked about entity obligation
            boost -= 0.15
    elif intent == "sanction_enforcement":
        if has_authority_remedial:
            boost += 0.20
        elif has_entity_obligation:
            boost -= 0.10
    elif intent == "permission_exemption" and re.search(
        r"\b(?:bank|pujk|perusahaan)\s+dapat\b|\bsepanjang\s+memperoleh\s+izin\b", t_low
    ):
        boost += 0.15

    # 3. Interrogative Target Alignment
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

    # 4. Concise Statutory Density Bonus (Anti-Length-Penalization for concise general obligations)
    if boost > 0 and len(text) < 450:
        boost += 0.03

    return max(-0.25, min(boost, 0.35))


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
