"""Unit tests for ChunkSpanMapper (T4.1 fair fixed-size retrieval evaluation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.chunk_mapper import ChunkSpanMapper


@pytest.fixture
def sample_chunks_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create sample structure-aware and fixed-size JSONL files for testing."""
    struct_file = tmp_path / "structure_aware.jsonl"
    fixed_file = tmp_path / "fixed_size.jsonl"

    struct_data = [
        {
            "chunk_id": "DOC1_pasal_1_ayat_1",
            "doc_id": "DOC1",
            "text": "[DOC1 | BAB I | Pasal 1 ayat (1)]\n(1) Lembaga Jasa Keuangan wajib menjaga stabilitas sistem perbankan nasional.",
            "raw_content": "(1) Lembaga Jasa Keuangan wajib menjaga stabilitas sistem perbankan nasional.",
        },
        {
            "chunk_id": "DOC1_pasal_1_ayat_2",
            "doc_id": "DOC1",
            "text": "[DOC1 | BAB I | Pasal 1 ayat (2)]\n(2) Kewajiban stabilitas dilaksanakan sesuai pedoman pengawasan OJK.",
            "raw_content": "(2) Kewajiban stabilitas dilaksanakan sesuai pedoman pengawasan OJK.",
        },
    ]

    fixed_data = [
        {
            "chunk_id": "DOC1_fix_0001",
            "doc_id": "DOC1",
            "text": "Lembaga Jasa Keuangan wajib menjaga stabilitas sistem perbankan nasional.",
            "raw_content": "Lembaga Jasa Keuangan wajib menjaga stabilitas sistem perbankan nasional.",
        },
        {
            "chunk_id": "DOC1_fix_0002",
            "doc_id": "DOC1",
            "text": "Kewajiban stabilitas dilaksanakan sesuai pedoman pengawasan OJK.",
            "raw_content": "Kewajiban stabilitas dilaksanakan sesuai pedoman pengawasan OJK.",
        },
        {
            "chunk_id": "DOC1_fix_0003",
            "doc_id": "DOC1",
            "text": "Teks tambahan tanpa tumpang tindih dengan ketentuan hukum di atas.",
            "raw_content": "Teks tambahan tanpa tumpang tindih dengan ketentuan hukum di atas.",
        },
    ]

    with open(struct_file, "w", encoding="utf-8") as f:
        for item in struct_data:
            f.write(json.dumps(item) + "\n")

    with open(fixed_file, "w", encoding="utf-8") as f:
        for item in fixed_data:
            f.write(json.dumps(item) + "\n")

    return struct_file, fixed_file


def test_chunk_span_mapper_build(sample_chunks_fixture: tuple[Path, Path]) -> None:
    """Test building ChunkSpanMapper from JSONL files."""
    struct_file, fixed_file = sample_chunks_fixture
    mapper = ChunkSpanMapper.build_from_chunks(
        structure_chunks_path=struct_file,
        fixed_chunks_path=fixed_file,
    )

    assert "DOC1_fix_0001" in mapper.fixed_to_structure
    assert mapper.get_structure_ids("DOC1_fix_0001") == ["DOC1_pasal_1_ayat_1"]

    assert "DOC1_fix_0002" in mapper.fixed_to_structure
    assert mapper.get_structure_ids("DOC1_fix_0002") == ["DOC1_pasal_1_ayat_2"]

    # Unmapped chunk returns empty list
    assert mapper.get_structure_ids("DOC1_fix_0003") == []
    assert mapper.get_structure_ids("NON_EXISTENT") == []


def test_chunk_span_mapper_save_and_load(
    tmp_path: Path, sample_chunks_fixture: tuple[Path, Path]
) -> None:
    """Test serializing and reloading ChunkSpanMapper."""
    struct_file, fixed_file = sample_chunks_fixture
    mapper = ChunkSpanMapper.build_from_chunks(
        structure_chunks_path=struct_file,
        fixed_chunks_path=fixed_file,
    )

    save_path = tmp_path / "map_cache.json"
    saved = mapper.save(save_path)
    assert saved.exists()

    loaded = ChunkSpanMapper.load(save_path)
    assert loaded.fixed_to_structure == mapper.fixed_to_structure
    assert loaded.structure_to_fixed == mapper.structure_to_fixed


def test_chunk_span_mapper_load_or_build(
    tmp_path: Path, sample_chunks_fixture: tuple[Path, Path]
) -> None:
    """Test load_or_build caching logic."""
    struct_file, fixed_file = sample_chunks_fixture
    cache_path = tmp_path / "cached_mapper.json"

    assert not cache_path.exists()
    mapper1 = ChunkSpanMapper.load_or_build(
        cache_path=cache_path,
        structure_chunks_path=struct_file,
        fixed_chunks_path=fixed_file,
    )
    assert cache_path.exists()

    # Second call should load from cache
    mapper2 = ChunkSpanMapper.load_or_build(
        cache_path=cache_path,
        structure_chunks_path=struct_file,
        fixed_chunks_path=fixed_file,
    )
    assert mapper2.fixed_to_structure == mapper1.fixed_to_structure


def test_chunk_span_mapper_missing_file_error(tmp_path: Path) -> None:
    """Test FileNotFoundError when source chunks file is missing."""
    with pytest.raises(FileNotFoundError, match="Structure chunks file not found"):
        ChunkSpanMapper.build_from_chunks(
            structure_chunks_path=tmp_path / "missing_struct.jsonl",
            fixed_chunks_path=tmp_path / "missing_fixed.jsonl",
        )

    struct_file = tmp_path / "exists.jsonl"
    struct_file.write_text("{}\n")
    with pytest.raises(FileNotFoundError, match="Fixed chunks file not found"):
        ChunkSpanMapper.build_from_chunks(
            structure_chunks_path=struct_file,
            fixed_chunks_path=tmp_path / "missing_fixed.jsonl",
        )

    with pytest.raises(FileNotFoundError, match="Mapper file not found"):
        ChunkSpanMapper.load(tmp_path / "non_existent.json")


def test_map_retrieved_chunk_ids(sample_chunks_fixture: tuple[Path, Path]) -> None:
    """Test map_retrieved_chunk_ids with fixed_size and structure_aware strategies."""
    struct_file, fixed_file = sample_chunks_fixture
    mapper = ChunkSpanMapper.build_from_chunks(
        structure_chunks_path=struct_file,
        fixed_chunks_path=fixed_file,
    )

    retrieved = ["DOC1_fix_0001", "DOC1_fix_0002", "DOC1_fix_0003"]

    # When strategy is fixed_size: maps to canonical IDs while preserving rank
    mapped = mapper.map_retrieved_chunk_ids(retrieved, strategy="fixed_size")
    assert mapped == ["DOC1_pasal_1_ayat_1", "DOC1_pasal_1_ayat_2", "DOC1_fix_0003"]

    # When strategy is structure_aware: passes through unchanged
    unchanged = mapper.map_retrieved_chunk_ids(retrieved, strategy="structure_aware")
    assert unchanged == retrieved
