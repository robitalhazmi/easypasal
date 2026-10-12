"""Unit tests for eval.labeling_helper: schema validation, templating, and unverified drafting."""

from __future__ import annotations

import json
from pathlib import Path

import eval as eval_pkg
import pytest
from eval.labeling_helper import (
    GoldItem,
    PassageRecord,
    draft_candidate_questions,
    filter_passages,
    format_draft_display,
    generate_blank_template,
    generate_draft_items_batch,
    get_next_item_id,
    get_theme_for_doc,
    load_passages,
    save_draft_item,
)
from pydantic import ValidationError


def _sample_passage(
    chunk_id: str = "POJK_22_Tahun_2023_pasal_75_ayat_5",
    doc_number: str = "POJK 22 Tahun 2023",
    pasal: str = "75",
    ayat: str = "5",
    text: str = "(5) PUJK wajib menyelesaikan pengaduan paling lambat 10 (sepuluh) hari kerja.",
) -> PassageRecord:
    return PassageRecord(
        chunk_id=chunk_id,
        doc_number=doc_number,
        doc_title="Pelindungan Konsumen dan Masyarakat",
        citation_label=f"{doc_number}, Pasal {pasal} ayat ({ayat})",
        pasal=pasal,
        ayat=ayat,
        text=text,
        status="berlaku",
        theme=get_theme_for_doc(doc_number),
    )


# ==============================================================================
# 1. SCHEMA VALIDATION TESTS
# ==============================================================================


def test_gold_item_schema_valid() -> None:
    """Verify valid GoldItem instance serialization and default unverified status."""
    item = GoldItem(
        id="gold_001",
        category="factual_direct",
        theme="Consumer Protection",
        difficulty="easy",
        language="id",
        question="Berapa batas waktu penanganan pengaduan menurut POJK 22 Tahun 2023?",
        gold_answer="Berdasarkan [POJK 22 Tahun 2023, Pasal 75 ayat (5)], batas waktunya 10 hari kerja.",
        gold_citations=["POJK 22 Tahun 2023, Pasal 75 ayat (5)"],
        relevant_chunk_ids=["POJK_22_Tahun_2023_pasal_75_ayat_5"],
        grounded_facts=["Batas waktu penyelesaian adalah 10 hari kerja."],
        should_abstain=False,
        abstain_reason=None,
        target_regulations=["POJK 22 Tahun 2023"],
    )

    assert item.id == "gold_001"
    assert item.review_status == "unverified"  # Default status must be unverified
    dumped = item.model_dump()
    assert dumped["review_status"] == "unverified"
    assert len(dumped["gold_citations"]) == 1


def test_gold_item_schema_validation_failures() -> None:
    """Verify validation errors for malformed item attributes."""
    valid_kwargs = {
        "id": "gold_001",
        "category": "factual_direct",
        "theme": "Consumer Protection",
        "question": "Pertanyaan valid?",
        "gold_answer": "Jawaban valid.",
    }

    # 1. Invalid ID pattern
    with pytest.raises(ValidationError):
        GoldItem.model_validate({**valid_kwargs, "id": "invalid_id_format"})

    # 2. Invalid category (not in 9 taxonomy keys)
    with pytest.raises(ValidationError):
        GoldItem.model_validate({**valid_kwargs, "category": "non_existent_category"})

    # 3. Question too short (< 5 chars)
    with pytest.raises(ValidationError):
        GoldItem.model_validate({**valid_kwargs, "question": "abc"})

    # 4. Extra forbidden field
    with pytest.raises(ValidationError):
        GoldItem.model_validate({**valid_kwargs, "forbidden_extra_field": True})


# ==============================================================================
# 2. THEME MAPPING & FILTERING TESTS
# ==============================================================================


def test_theme_resolution() -> None:
    """Verify regulatory theme resolution across the 5 domains."""
    assert get_theme_for_doc("POJK 22 Tahun 2023") == "Consumer Protection"
    assert get_theme_for_doc("POJK 11/POJK.03/2022") == "IT & Cyber Risk"
    assert get_theme_for_doc("POJK 8 Tahun 2023") == "AML / CFT"
    assert get_theme_for_doc("UU Nomor 27 Tahun 2022") == "Data Privacy & Secrecy"
    assert get_theme_for_doc("POJK 31/POJK.04/2015") == "Capital Markets"
    assert get_theme_for_doc("Unknown Reg 99/9999") == "General Financial Regulation"


def test_filter_passages() -> None:
    """Verify passage filtering by theme, doc number, pasal, and keywords."""
    p1 = _sample_passage(
        chunk_id="c1", doc_number="POJK 22 Tahun 2023", pasal="75", text="pengaduan konsumen"
    )
    p2 = _sample_passage(
        chunk_id="c2", doc_number="POJK 11/POJK.03/2022", pasal="31", text="insiden siber"
    )
    p3 = _sample_passage(
        chunk_id="c3", doc_number="POJK 8 Tahun 2023", pasal="1", text="pencucian uang"
    )

    passages = [p1, p2, p3]

    # Filter by theme
    assert len(filter_passages(passages, theme="Consumer Protection")) == 1
    assert filter_passages(passages, theme="Consumer Protection")[0].chunk_id == "c1"

    # Filter by pasal
    assert len(filter_passages(passages, pasal="31")) == 1
    assert filter_passages(passages, pasal="31")[0].chunk_id == "c2"

    # Filter by keyword
    assert len(filter_passages(passages, keyword="siber")) == 1
    assert filter_passages(passages, keyword="siber")[0].chunk_id == "c2"

    # Filter by doc_number
    assert len(filter_passages(passages, doc_number="POJK 8")) == 1


# ==============================================================================
# 3. TEMPLATE GENERATION & UNVERIFIED DRAFTING TESTS
# ==============================================================================


def test_generate_blank_template_factual() -> None:
    """Verify blank template pre-populates metadata for factual passage."""
    p = _sample_passage()
    item = generate_blank_template(passage=p, next_id="gold_010", category="factual_direct")

    assert item.id == "gold_010"
    assert item.category == "factual_direct"
    assert item.theme == "Consumer Protection"
    assert item.gold_citations == [p.citation_label]
    assert item.relevant_chunk_ids == [p.chunk_id]
    assert item.should_abstain is False
    assert item.abstain_reason is None
    assert item.review_status == "unverified"  # Explicitly unverified


def test_generate_blank_template_abstention() -> None:
    """Verify template pre-configures standardized refusal for abstention categories."""
    # Indonesian out-of-scope abstention
    item_id = generate_blank_template(
        next_id="gold_099", category="unanswerable_out_of_scope", language="id"
    )
    assert item_id.should_abstain is True
    assert item_id.abstain_reason == "out_of_corpus"
    assert "tidak ditemukan dasar hukum" in item_id.gold_answer
    assert item_id.gold_citations == []
    assert item_id.review_status == "unverified"

    # English adversarial jailbreak abstention
    item_en = generate_blank_template(
        next_id="gold_100", category="adversarial_jailbreak", language="en"
    )
    assert item_en.should_abstain is True
    assert item_en.abstain_reason == "adversarial"
    assert "no legal basis was found" in item_en.gold_answer
    assert item_en.gold_citations == []
    assert item_en.review_status == "unverified"


def test_draft_candidate_questions_patterns() -> None:
    """Verify heuristic drafting generates pattern-anchored questions."""
    # 1. Definitional
    p_def = _sample_passage(
        pasal="1",
        text="Konsumen adalah pihak yang memanfaatkan pelayanan di Lembaga Jasa Keuangan.",
    )
    cands_def = draft_candidate_questions(p_def, category="definitional")
    assert any("definisi" in c.lower() for c in cands_def)

    # 2. Deadlines / Time limits
    p_time = _sample_passage(
        pasal="75",
        text="PUJK menyelesaikan pengaduan paling lambat 10 (sepuluh) hari kerja.",
    )
    cands_time = draft_candidate_questions(p_time, category="factual_direct")
    assert any("batas waktu" in c.lower() or "jangka waktu" in c.lower() for c in cands_time)

    # 3. Obligations (Wajib)
    p_obl = _sample_passage(
        pasal="10",
        text="Bank wajib menerapkan tata kelola teknologi informasi secara andal.",
    )
    cands_obl = draft_candidate_questions(p_obl, category="obligation_and_prohibition")
    assert any("kewajiban" in c.lower() for c in cands_obl)

    # 4. Prohibitions (Dilarang)
    p_proh = _sample_passage(
        pasal="62",
        text="PUJK dilarang menggunakan cara ancaman atau kekerasan dalam penagihan.",
    )
    cands_proh = draft_candidate_questions(p_proh, category="obligation_and_prohibition")
    assert any("dilarang" in c.lower() or "larangan" in c.lower() for c in cands_proh)

    # 5. Sanctions (Sanksi)
    p_sanct = _sample_passage(
        pasal="112",
        text="OJK mengenakan sanksi administratif berupa peringatan tertulis dan denda uang.",
    )
    cands_sanct = draft_candidate_questions(p_sanct, category="sanctions_and_enforcement")
    assert any("sanksi" in c.lower() or "denda" in c.lower() for c in cands_sanct)

    # 6. English slice
    cands_en = draft_candidate_questions(
        p_obl, category="obligation_and_prohibition", language="en"
    )
    assert any("mandatory" in c.lower() or "duties" in c.lower() for c in cands_en)


def test_format_draft_display_contains_unverified_warning() -> None:
    """Verify format_draft_display includes high-visibility UNVERIFIED warnings."""
    p = _sample_passage()
    item = generate_blank_template(passage=p, next_id="gold_005")
    view = format_draft_display(item, p)

    assert "UNVERIFIED DRAFT" in view
    assert "REQUIRES HUMAN REVIEW" in view
    assert "REVIEW STATUS: UNVERIFIED" in view
    assert p.citation_label in view


# ==============================================================================
# 4. SERIALIZATION & BATCH GENERATION TESTS
# ==============================================================================


def test_save_draft_item_and_get_next_id(tmp_path: Path) -> None:
    """Verify appending items to JSONL and sequential ID incrementation."""
    out_file = tmp_path / "test_drafts.jsonl"

    assert get_next_item_id(out_file) == "gold_001"

    p = _sample_passage()
    item1 = generate_blank_template(passage=p, next_id="gold_001")
    save_draft_item(item1, output_path=out_file)

    assert get_next_item_id(out_file) == "gold_002"

    item2 = generate_blank_template(passage=p, next_id="gold_002")
    save_draft_item(item2, output_path=out_file)

    assert get_next_item_id(out_file) == "gold_003"

    with open(out_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 2

    # Verify JSON structure and unverified status in stored file
    parsed = json.loads(lines[0])
    assert parsed["id"] == "gold_001"
    assert parsed["review_status"] == "unverified"


def test_generate_draft_items_batch(tmp_path: Path) -> None:
    """Verify batch draft generation produces scaffold items marked unverified."""
    fake_meta = tmp_path / "mock_meta.json"
    p1 = _sample_passage(chunk_id="c1", doc_number="POJK 22 Tahun 2023", pasal="75")
    p2 = _sample_passage(chunk_id="c2", doc_number="POJK 11/POJK.03/2022", pasal="31")

    mock_data = {
        "num_chunks": 2,
        "corpus_payloads": [
            {
                "chunk_id": p1.chunk_id,
                "doc_number": p1.doc_number,
                "doc_title": p1.doc_title,
                "citation_label": p1.citation_label,
                "pasal": p1.pasal,
                "ayat": p1.ayat,
                "text": p1.text,
                "status": p1.status,
            },
            {
                "chunk_id": p2.chunk_id,
                "doc_number": p2.doc_number,
                "doc_title": p2.doc_title,
                "citation_label": p2.citation_label,
                "pasal": p2.pasal,
                "ayat": p2.ayat,
                "text": p2.text,
                "status": p2.status,
            },
        ],
    }
    with open(fake_meta, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)

    out_file = tmp_path / "batch_drafts.jsonl"
    drafts = generate_draft_items_batch(
        count=2,
        output_path=out_file,
        meta_path=fake_meta,
    )

    assert len(drafts) == 2
    for d in drafts:
        assert d.review_status == "unverified"
        assert "[UNVERIFIED DRAFT]" in d.question

    assert out_file.exists()


def test_load_passages_non_existent_file() -> None:
    """Verify graceful handling when metadata file does not exist."""
    passages = load_passages(Path("non_existent_path.json"))
    assert passages == []


# ==============================================================================
# 5. PACKAGE EXPORTS TEST
# ==============================================================================


def test_eval_package_exports() -> None:
    """Verify dynamic PEP 562 attribute resolution in eval package."""
    assert callable(eval_pkg.load_passages)
    assert callable(eval_pkg.generate_blank_template)
    assert issubclass(eval_pkg.GoldItem, GoldItem)

    with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
        _ = eval_pkg.NonExistent
