"""Qdrant embedded vector store module for regulatory chunks.

Supports local embedded disk storage and in-memory execution with zero
external server daemon dependencies. Generates deterministic UUIDv5 IDs
from chunk_id to maintain consistency across sparse and dense indices.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.index.vector_store")


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUIDv5 identifier from a chunk_id string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


@dataclass
class VectorSearchResult:
    """Represents a search result from Qdrant vector retrieval."""

    chunk_id: str
    point_id: str
    score: float
    citation_label: str
    doc_id: str
    doc_number: str
    pasal: str | None
    ayat: str | None
    status: str
    text: str
    payload: dict[str, Any]


class QdrantVectorStore:
    """Embedded Qdrant vector store wrapper."""

    def __init__(
        self,
        path: Path | str | None = None,
        url: str | None = None,
    ) -> None:
        self.path = Path(path) if path and path != ":memory:" else None
        if path == ":memory:":
            self.client = QdrantClient(location=":memory:")
            logger.info("Initialized in-memory Qdrant instance.")
        elif self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.path))
            logger.info("Initialized embedded Qdrant instance at %s", self.path)
        elif url:
            self.client = QdrantClient(url=url)
            logger.info("Connected to remote Qdrant instance at %s", url)
        else:
            self.client = QdrantClient(location=":memory:")
            logger.info("Initialized default in-memory Qdrant instance.")

    def init_collection(
        self,
        collection_name: str,
        dimension: int,
        recreate: bool = False,
    ) -> None:
        """Create or verify the existence of a collection with cosine distance."""
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name in existing:
            if recreate:
                logger.info("Recreating collection %r...", collection_name)
                self.client.delete_collection(collection_name=collection_name)
            else:
                logger.info("Collection %r already exists.", collection_name)
                return

        logger.info(
            "Creating collection %r (dimension=%d, distance=COSINE)...",
            collection_name,
            dimension,
        )
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """Batch upsert chunks and their dense embeddings into Qdrant."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match vectors count ({len(vectors)})"
            )

        total = len(chunks)
        logger.info(
            "Upserting %d points into collection %r (batch_size=%d)...",
            total,
            collection_name,
            batch_size,
        )

        with tqdm(
            total=total,
            desc=f"Upserting Qdrant ({collection_name})",
            unit="chunks",
            disable=not show_progress,
        ) as pbar:
            for i in range(0, total, batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_vectors = vectors[i : i + batch_size]

                points: list[models.PointStruct] = []
                for chunk, vec in zip(batch_chunks, batch_vectors, strict=True):
                    cid = chunk["chunk_id"]
                    p_uuid = chunk_id_to_uuid(cid)
                    points.append(
                        models.PointStruct(
                            id=p_uuid,
                            vector=vec,
                            payload=chunk,
                        )
                    )

                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                pbar.update(len(batch_chunks))

        logger.info("Successfully upserted %d points into %r.", total, collection_name)
        return total

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[VectorSearchResult]:
        """Query Qdrant collection with a dense query vector."""
        # Use query_points (or search)
        scored_points = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        results: list[VectorSearchResult] = []
        for sp in scored_points:
            payload = sp.payload or {}
            cid = str(payload.get("chunk_id", str(sp.id)))
            results.append(
                VectorSearchResult(
                    chunk_id=cid,
                    point_id=str(sp.id),
                    score=float(sp.score),
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

    def count(self, collection_name: str) -> int:
        """Return the count of indexed points in a collection."""
        res = self.client.count(collection_name=collection_name)
        return res.count

    def close(self) -> None:
        """Close client connection."""
        self.client.close()
