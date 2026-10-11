"""BM25 sparse inverted index module for Indonesian regulatory texts.

Provides Indonesian-aware legal tokenization preserving statutory article numbers,
regulatory acronyms (OJK, PUJK, LJK), and legal compound words (undang-undang).
Integrates with rank-bm25 with disk persistence and metadata retrieval.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.index.bm25")

# Functional Indonesian stopwords that carry negligible legal semantics
INDONESIAN_FUNCTIONAL_STOPWORDS: set[str] = {
    "dan",
    "yang",
    "di",
    "ke",
    "dari",
    "pada",
    "untuk",
    "atau",
    "ini",
    "itu",
    "dengan",
    "adalah",
    "oleh",
    "sebagai",
    "secara",
    "tersebut",
    "serta",
    "atas",
    "dalam",
    "yaitu",
    "bagi",
    "jika",
    "maka",
    "akan",
    "telah",
    "antara",
    "lain",
    "hal",
}


def tokenize_indonesian_legal(text: str, filter_stopwords: bool = True) -> list[str]:
    """Tokenize Indonesian legal text into tokens preserving acronyms and numbers.

    Preserves legal numbers (e.g. '71', '2023'), acronyms ('ojk', 'pujk'), and
    hyphenated legal compounds ('undang-undang', 'tata-tertib').
    """
    raw_tokens = re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)?\b", text.lower())
    if not filter_stopwords:
        return raw_tokens
    return [t for t in raw_tokens if t not in INDONESIAN_FUNCTIONAL_STOPWORDS]


@dataclass
class BM25SearchResult:
    """Represents a search result from BM25 sparse retrieval."""

    chunk_id: str
    score: float
    citation_label: str
    doc_id: str
    doc_number: str
    pasal: str | None
    ayat: str | None
    status: str
    text: str
    payload: dict[str, Any]


class BM25Index:
    """Sparse inverted index using BM25Okapi over regulatory chunks."""

    def __init__(
        self,
        bm25: BM25Okapi | None = None,
        corpus_ids: list[str] | None = None,
        corpus_payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        self.bm25 = bm25
        self.corpus_ids: list[str] = corpus_ids or []
        self.corpus_payloads: list[dict[str, Any]] = corpus_payloads or []

    @classmethod
    def build_from_chunks(cls, chunks: list[dict[str, Any]]) -> BM25Index:
        """Construct a BM25 index from a list of chunk dictionaries."""
        logger.info("Building BM25 index over %d chunks...", len(chunks))
        tokenized_corpus: list[list[str]] = []
        corpus_ids: list[str] = []
        corpus_payloads: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            # Tokenize context-enriched text for maximum term match
            text = chunk.get("text", "")
            tokens = tokenize_indonesian_legal(text)
            tokenized_corpus.append(tokens)
            corpus_ids.append(chunk_id)
            corpus_payloads.append(chunk)

        bm25 = BM25Okapi(tokenized_corpus)
        return cls(bm25=bm25, corpus_ids=corpus_ids, corpus_payloads=corpus_payloads)

    def search(self, query: str, top_k: int = 10) -> list[BM25SearchResult]:
        """Query the BM25 index and return the top-k highest scoring chunks."""
        if not self.bm25 or not self.corpus_ids:
            return []

        query_tokens = tokenize_indonesian_legal(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[
            :top_k
        ]

        results: list[BM25SearchResult] = []
        for idx in top_indices:
            score = float(doc_scores[idx])
            if score <= 0:
                continue
            payload = self.corpus_payloads[idx]
            results.append(
                BM25SearchResult(
                    chunk_id=self.corpus_ids[idx],
                    score=score,
                    citation_label=payload.get("citation_label", ""),
                    doc_id=payload.get("doc_id", ""),
                    doc_number=payload.get("doc_number", ""),
                    pasal=payload.get("pasal"),
                    ayat=payload.get("ayat"),
                    status=payload.get("status", "berlaku"),
                    text=payload.get("text", ""),
                    payload=payload,
                )
            )
        return results

    def save(self, output_dir: Path | str, prefix: str = "bm25") -> Path:
        """Serialize BM25 index model and metadata to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_file = out_path / f"{prefix}_model.pkl"
        meta_file = out_path / f"{prefix}_meta.json"

        with open(model_file, "wb") as f:
            pickle.dump(self.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta_data = {
            "num_chunks": len(self.corpus_ids),
            "corpus_ids": self.corpus_ids,
            "corpus_payloads": self.corpus_payloads,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False)

        logger.info(
            "Saved BM25 index (%d chunks) to %s (%.2f MB)",
            len(self.corpus_ids),
            out_path,
            (model_file.stat().st_size + meta_file.stat().st_size) / (1024 * 1024),
        )
        return out_path

    @classmethod
    def load(cls, input_dir: Path | str, prefix: str = "bm25") -> BM25Index:
        """Load serialized BM25 index from disk."""
        in_path = Path(input_dir)
        model_file = in_path / f"{prefix}_model.pkl"
        meta_file = in_path / f"{prefix}_meta.json"

        if not model_file.exists() or not meta_file.exists():
            raise FileNotFoundError(
                f"BM25 index files not found in {in_path} (looked for {model_file.name} and {meta_file.name})"
            )

        with open(model_file, "rb") as f:
            bm25 = pickle.load(f)

        with open(meta_file, encoding="utf-8") as f:
            meta_data = json.load(f)

        return cls(
            bm25=bm25,
            corpus_ids=meta_data["corpus_ids"],
            corpus_payloads=meta_data["corpus_payloads"],
        )
