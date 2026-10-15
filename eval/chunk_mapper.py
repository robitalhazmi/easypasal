"""Span-to-article coverage mapper: maps fixed-size chunks to canonical structure-aware chunk IDs.

Enables fair, scientifically rigorous evaluation of fixed-size baseline chunks against
ground-truth gold items whose relevant_chunk_ids are indexed by statutory hierarchy.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval.chunk_mapper")

DEFAULT_MAP_PATH = Path("data/chunks/fixed_to_structure_map.json")


class ChunkSpanMapper:
    """Bi-directional mapper between fixed-size chunks and canonical structure-aware chunks."""

    def __init__(
        self,
        fixed_to_structure: dict[str, list[str]] | None = None,
        structure_to_fixed: dict[str, list[str]] | None = None,
    ) -> None:
        self.fixed_to_structure: dict[str, list[str]] = fixed_to_structure or {}
        self.structure_to_fixed: dict[str, list[str]] = structure_to_fixed or {}

    @classmethod
    def build_from_chunks(
        cls,
        structure_chunks_path: Path | str = "data/chunks/structure_aware.jsonl",
        fixed_chunks_path: Path | str = "data/chunks/fixed_size.jsonl",
    ) -> ChunkSpanMapper:
        """Build mapping by computing text and span overlap between fixed and structure chunks.

        Args:
            structure_chunks_path: Path to structure-aware JSONL chunks.
            fixed_chunks_path: Path to fixed-size JSONL chunks.

        Returns:
            Populated ChunkSpanMapper instance.
        """
        s_path = Path(structure_chunks_path)
        f_path = Path(fixed_chunks_path)

        if not s_path.exists():
            raise FileNotFoundError(f"Structure chunks file not found: {s_path}")
        if not f_path.exists():
            raise FileNotFoundError(f"Fixed chunks file not found: {f_path}")

        logger.info("Building ChunkSpanMapper from %s and %s...", s_path, f_path)

        # 1. Group chunks by doc_id
        struct_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with open(s_path, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    c = json.loads(line_str)
                    struct_by_doc[c["doc_id"]].append(c)

        fixed_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with open(f_path, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    c = json.loads(line_str)
                    fixed_by_doc[c["doc_id"]].append(c)

        f_to_s: dict[str, list[str]] = defaultdict(list)
        s_to_f: dict[str, list[str]] = defaultdict(list)

        # 2. Match overlapping chunks within each document
        for doc_id, s_list in struct_by_doc.items():
            f_list = fixed_by_doc.get(doc_id, [])
            if not f_list:
                continue

            for sc in s_list:
                s_id = sc["chunk_id"]
                s_raw = sc.get("raw_content", "").strip()
                if not s_raw:
                    continue

                # Distinctive substrings: start, middle, end
                sample_start = s_raw[:50] if len(s_raw) >= 50 else s_raw
                sample_mid = (
                    s_raw[len(s_raw) // 2 : len(s_raw) // 2 + 50] if len(s_raw) > 100 else ""
                )
                sample_end = s_raw[-50:] if len(s_raw) >= 50 else ""

                for fc in f_list:
                    f_id = fc["chunk_id"]
                    f_text = fc.get("text", "")

                    # Check substring containment
                    if (
                        (sample_start and sample_start in f_text)
                        or (sample_mid and sample_mid in f_text)
                        or (sample_end and sample_end in f_text)
                        or (len(s_raw) <= 50 and s_raw in f_text)
                    ):
                        if s_id not in f_to_s[f_id]:
                            f_to_s[f_id].append(s_id)
                        if f_id not in s_to_f[s_id]:
                            s_to_f[s_id].append(f_id)

        mapper = cls(fixed_to_structure=dict(f_to_s), structure_to_fixed=dict(s_to_f))
        logger.info(
            "ChunkSpanMapper built: %d fixed chunks mapped to %d structure chunks.",
            len(mapper.fixed_to_structure),
            len(mapper.structure_to_fixed),
        )
        return mapper

    def save(self, output_path: Path | str = DEFAULT_MAP_PATH) -> Path:
        """Persist mapping dictionary to JSON."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "fixed_to_structure": self.fixed_to_structure,
            "structure_to_fixed": self.structure_to_fixed,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved ChunkSpanMapper to %s", target)
        return target

    @classmethod
    def load(cls, input_path: Path | str = DEFAULT_MAP_PATH) -> ChunkSpanMapper:
        """Load mapping dictionary from JSON."""
        target = Path(input_path)
        if not target.exists():
            raise FileNotFoundError(f"Mapper file not found: {target}")
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            fixed_to_structure=data.get("fixed_to_structure", {}),
            structure_to_fixed=data.get("structure_to_fixed", {}),
        )

    @classmethod
    def load_or_build(
        cls,
        cache_path: Path | str = DEFAULT_MAP_PATH,
        structure_chunks_path: Path | str = "data/chunks/structure_aware.jsonl",
        fixed_chunks_path: Path | str = "data/chunks/fixed_size.jsonl",
    ) -> ChunkSpanMapper:
        """Load from cache if available, or build and persist otherwise."""
        path = Path(cache_path)
        if path.exists():
            try:
                return cls.load(path)
            except Exception:
                logger.warning("Failed to load cached mapper from %s, rebuilding...", path)
        mapper = cls.build_from_chunks(
            structure_chunks_path=structure_chunks_path,
            fixed_chunks_path=fixed_chunks_path,
        )
        mapper.save(path)
        return mapper

    def get_structure_ids(self, fixed_id: str) -> list[str]:
        """Get canonical structure-aware chunk IDs covered by a fixed-size chunk ID."""
        return self.fixed_to_structure.get(fixed_id, [])

    def map_retrieved_chunk_ids(
        self,
        retrieved_ids: Sequence[str],
        strategy: str,
    ) -> list[str]:
        """Normalize retrieved chunk IDs for evaluation against gold canonical IDs.

        If strategy is 'fixed_size', maps each retrieved chunk ID to its primary
        canonical structure-aware ID (or multiple IDs while preserving rank).
        If strategy is already structure-aware, returns retrieved_ids unchanged.
        """
        if strategy != "fixed_size":
            return list(retrieved_ids)

        mapped_ranked: list[str] = []
        seen: set[str] = set()

        for fid in retrieved_ids:
            s_ids = self.get_structure_ids(fid)
            if s_ids:
                for sid in s_ids:
                    if sid not in seen:
                        seen.add(sid)
                        mapped_ranked.append(sid)
            else:
                # If unmapped, retain original fid so it counts as non-match
                if fid not in seen:
                    seen.add(fid)
                    mapped_ranked.append(fid)

        return mapped_ranked
