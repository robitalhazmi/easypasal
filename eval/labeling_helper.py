"""Evaluation labeling helper tool for building EasyPasal gold datasets.

Provides passage browsing, blank item templating, and candidate question drafting.
All machine-generated drafts are explicitly tagged with `review_status="unverified"`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("easypasal.eval.labeling_helper")

DEFAULT_META_PATH = Path("data/index/bm25/structure_aware_meta.json")
DEFAULT_DRAFTS_PATH = Path("data/gold/drafts.jsonl")

# 9 Standardized Categories from docs/eval-methodology.md
CategoryType = Literal[
    "factual_direct",
    "cross_article_synthesis",
    "definitional",
    "obligation_and_prohibition",
    "sanctions_and_enforcement",
    "superseded_trap",
    "unanswerable_out_of_scope",
    "adversarial_jailbreak",
    "english_slice",
]

AbstainReasonType = Literal["out_of_corpus", "superseded", "adversarial", "unanswerable"]

VALID_CATEGORIES: list[CategoryType] = [
    "factual_direct",
    "cross_article_synthesis",
    "definitional",
    "obligation_and_prohibition",
    "sanctions_and_enforcement",
    "superseded_trap",
    "unanswerable_out_of_scope",
    "adversarial_jailbreak",
    "english_slice",
]

# Theme mapping for the 21 corpus regulations
THEME_MAPPING: dict[str, str] = {
    "POJK 22 Tahun 2023": "Consumer Protection",
    "POJK 6/POJK.07/2022": "Consumer Protection",
    "SEOJK 2/SEOJK.07/2014": "Consumer Protection",
    "POJK 61/POJK.07/2020": "Consumer Protection",
    "POJK 11/POJK.03/2022": "IT & Cyber Risk",
    "SEOJK 29/SEOJK.03/2022": "IT & Cyber Risk",
    "POJK 21 Tahun 2023": "IT & Cyber Risk",
    "POJK 38/POJK.03/2016": "IT & Cyber Risk",
    "POJK 8 Tahun 2023": "AML / CFT",
    "POJK 12/POJK.01/2017": "AML / CFT",
    "UU Nomor 8 Tahun 2010": "AML / CFT",
    "UU Nomor 9 Tahun 2013": "AML / CFT",
    "UU Nomor 27 Tahun 2022": "Data Privacy & Secrecy",
    "UU Nomor 4 Tahun 2023": "Data Privacy & Secrecy",
    "UU Nomor 10 Tahun 1998": "Data Privacy & Secrecy",
    "PP Nomor 71 Tahun 2019": "Data Privacy & Secrecy",
    "POJK 31/POJK.04/2015": "Capital Markets",
    "POJK 15/POJK.04/2020": "Capital Markets",
    "POJK 32/POJK.04/2014": "Capital Markets",
    "POJK 3/POJK.04/2021": "Capital Markets",
    "UU Nomor 8 Tahun 1995": "Capital Markets",
}


class GoldItem(BaseModel):
    """Pydantic model enforcing the exact schema specified in docs/eval-methodology.md."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^gold_\d{3,4}$", description="Unique gold ID (e.g. gold_001).")
    category: CategoryType = Field(
        ..., description="One of the 9 standardized evaluation categories."
    )
    theme: str = Field(..., description="Regulatory domain theme.")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Subjective complexity tier."
    )
    language: Literal["id", "en"] = Field(
        default="id", description="Query and answer language ('id' or 'en')."
    )
    question: str = Field(..., min_length=5, description="Legal question text.")
    gold_answer: str = Field(..., min_length=5, description="Authoritative reference answer.")
    gold_citations: list[str] = Field(
        default_factory=list, description="Canonical statutory citation tags supporting the answer."
    )
    relevant_chunk_ids: list[str] = Field(
        default_factory=list, description="Index chunk IDs supporting the answer."
    )
    grounded_facts: list[str] = Field(
        default_factory=list, description="List of discrete verifiable facts in gold_answer."
    )
    should_abstain: bool = Field(
        default=False,
        description="True if the question should trigger an explicit abstention refusal.",
    )
    abstain_reason: Literal["out_of_corpus", "superseded", "adversarial", "unanswerable"] | None = (
        Field(default=None, description="Reason for abstention if should_abstain is True.")
    )
    target_regulations: list[str] = Field(
        default_factory=list, description="Target document numbers."
    )
    review_status: Literal["unverified", "verified"] = Field(
        default="unverified",
        description="Marking indicating whether human expert has verified this item.",
    )


@dataclass
class PassageRecord:
    """Indexed regulatory passage metadata and content."""

    chunk_id: str
    doc_number: str
    doc_title: str
    citation_label: str
    pasal: str | None
    ayat: str | None
    text: str
    status: str
    theme: str


def get_theme_for_doc(doc_number: str) -> str:
    """Resolve regulatory theme from document number with sensible fallback."""
    norm = doc_number.strip()
    for key, theme in THEME_MAPPING.items():
        if key.lower() in norm.lower() or norm.lower() in key.lower():
            return theme
    return "General Financial Regulation"


def load_passages(meta_path: Path | str | None = None) -> list[PassageRecord]:
    """Load passage records from the BM25 index metadata."""
    path = Path(meta_path or DEFAULT_META_PATH)
    if not path.exists():
        logger.warning("Metadata path %s does not exist. Returning empty passages.", path)
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    payloads = data.get("corpus_payloads", [])
    passages: list[PassageRecord] = []
    for p in payloads:
        doc_num = p.get("doc_number", "Unknown Regulation")
        passages.append(
            PassageRecord(
                chunk_id=p.get("chunk_id", ""),
                doc_number=doc_num,
                doc_title=p.get("doc_title", ""),
                citation_label=p.get("citation_label", doc_num),
                pasal=p.get("pasal"),
                ayat=p.get("ayat"),
                text=p.get("text", ""),
                status=p.get("status", "berlaku"),
                theme=get_theme_for_doc(doc_num),
            )
        )
    return passages


def filter_passages(
    passages: list[PassageRecord],
    theme: str | None = None,
    doc_number: str | None = None,
    pasal: str | None = None,
    status: str | None = None,
    keyword: str | None = None,
) -> list[PassageRecord]:
    """Filter passages by metadata attributes."""
    filtered = passages
    if theme:
        filtered = [p for p in filtered if theme.lower() in p.theme.lower()]
    if doc_number:
        filtered = [p for p in filtered if doc_number.lower() in p.doc_number.lower()]
    if pasal:
        filtered = [p for p in filtered if p.pasal and p.pasal.lower() == pasal.lower()]
    if status:
        filtered = [p for p in filtered if p.status.lower() == status.lower()]
    if keyword:
        kw = keyword.lower()
        filtered = [p for p in filtered if kw in p.text.lower()]
    return filtered


def generate_blank_template(
    passage: PassageRecord | None = None,
    next_id: str = "gold_001",
    category: CategoryType = "factual_direct",
    language: Literal["id", "en"] = "id",
) -> GoldItem:
    """Construct a pre-populated GoldItem blank template with metadata."""
    should_abstain = category in ("unanswerable_out_of_scope", "adversarial_jailbreak")
    abstain_reason: AbstainReasonType | None = (
        "out_of_corpus"
        if category == "unanswerable_out_of_scope"
        else "adversarial"
        if category == "adversarial_jailbreak"
        else None
    )

    theme = passage.theme if passage else "Consumer Protection"
    doc_number = passage.doc_number if passage else "POJK 22 Tahun 2023"
    citation = [passage.citation_label] if (passage and not should_abstain) else []
    chunk_ids = [passage.chunk_id] if (passage and not should_abstain) else []

    if should_abstain:
        if language == "en":
            gold_answer = "Based on the available regulations, no legal basis was found regarding this matter."
        else:
            gold_answer = "Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut."
        facts = ["Regulasi yang tersedia tidak mengatur substansi pertanyaan ini."]
    else:
        cite_str = f"[{passage.citation_label}]" if passage else "[POJK 22 Tahun 2023, Pasal ...]"
        gold_answer = f"Berdasarkan {cite_str}, diatur bahwa ..."
        facts = ["Fakta hukum pokok sesuai pasal yang dikutip."]

    return GoldItem(
        id=next_id,
        category=category,
        theme=theme,
        difficulty="medium",
        language=language,
        question="[Pertanyaan hukum yang diajukan]...",
        gold_answer=gold_answer,
        gold_citations=citation,
        relevant_chunk_ids=chunk_ids,
        grounded_facts=facts,
        should_abstain=should_abstain,
        abstain_reason=abstain_reason,
        target_regulations=[doc_number] if not should_abstain else [],
        review_status="unverified",
    )


def draft_candidate_questions(
    passage: PassageRecord,
    category: CategoryType = "factual_direct",
    language: Literal["id", "en"] = "id",
) -> list[str]:
    """Generate candidate inquiries from passage patterns, explicitly flagged unverified.

    These drafts serve as creative inspiration for annotators and are never final gold labels.
    """
    candidates: list[str] = []
    text = passage.text
    doc = passage.doc_number
    pasal_ref = f"Pasal {passage.pasal}" if passage.pasal else "ketentuan terkait"

    # 1. Definitional patterns
    if passage.pasal == "1" or "yang dimaksud dengan" in text.lower():
        def_matches = re.findall(r"([A-Z][a-zA-Z\s]{3,30})\s+adalah", text)
        for term in def_matches[:2]:
            clean_term = term.strip()
            if language == "en":
                candidates.append(
                    f"What is the definition of '{clean_term}' under {doc} {pasal_ref}?"
                )
            else:
                candidates.append(f"Apa definisi dari '{clean_term}' menurut {doc} {pasal_ref}?")

    # 2. Time limits / Deadlines
    time_match = re.search(r"(\d+\s*\([a-z\s]+\)\s*(?:hari|bulan|tahun))", text, re.IGNORECASE)
    if time_match:
        dur = time_match.group(1)
        if language == "en":
            candidates.append(
                f"What is the statutory deadline or time limit specified under {doc} {pasal_ref} (e.g. {dur})?"
            )
        else:
            candidates.append(
                f"Berapa batas waktu atau jangka waktu yang ditentukan dalam {doc} {pasal_ref} (terkait {dur})?"
            )

    # 3. Obligations (Wajib)
    if re.search(r"\bwajib\b", text, re.IGNORECASE):
        if language == "en":
            candidates.append(
                f"What are the mandatory duties or requirements imposed on financial institutions under {doc} {pasal_ref}?"
            )
        else:
            candidates.append(
                f"Apa saja kewajiban yang wajib dipenuhi oleh lembaga jasa keuangan menurut {doc} {pasal_ref}?"
            )

    # 4. Prohibitions (Dilarang)
    if re.search(r"\bdilarang\b", text, re.IGNORECASE):
        if language == "en":
            candidates.append(f"What practices are strictly prohibited under {doc} {pasal_ref}?")
        else:
            candidates.append(
                f"Tindakan atau praktik apa saja yang dilarang menurut {doc} {pasal_ref}?"
            )

    # 5. Sanctions / Enforcement
    if re.search(r"\bsanksi\b|\bdenda\b|\bperingatan\b", text, re.IGNORECASE):
        if language == "en":
            candidates.append(
                f"What administrative sanctions or fines can be imposed for violations of {doc} {pasal_ref}?"
            )
        else:
            candidates.append(
                f"Sanksi administratif atau denda apa yang dapat dikenakan atas pelanggaran {doc} {pasal_ref}?"
            )

    # Default fallback candidates if pattern match was sparse
    if not candidates:
        if category == "definitional":
            candidates.append(f"Apa saja ketentuan pokok dalam {doc} {pasal_ref}?")
        elif language == "en":
            candidates.append(f"What is regulated under {doc} {pasal_ref}?")
        else:
            candidates.append(f"Bagaimana ketentuan yang diatur dalam {doc} {pasal_ref}?")

    return candidates


def format_draft_display(item: GoldItem, passage: PassageRecord | None = None) -> str:
    """Format an item and passage for terminal display with high-visibility unverified banners."""
    lines: list[str] = [
        "=" * 80,
        "⚠️  [UNVERIFIED DRAFT - REQUIRES HUMAN REVIEW AND VERIFICATION]  ⚠️",
        f"REVIEW STATUS: {item.review_status.upper()}",
        "=" * 80,
    ]

    if passage:
        lines.extend(
            [
                f"SOURCE PASSAGE: {passage.citation_label}",
                f"CHUNK ID:       {passage.chunk_id}",
                f"THEME:          {passage.theme}",
                f"STATUS:         {passage.status.upper()}",
                "-" * 80,
                "PASSAGE TEXT:",
                passage.text[:600] + ("..." if len(passage.text) > 600 else ""),
                "-" * 80,
            ]
        )

    lines.extend(
        [
            "PROPOSED ITEM TEMPLATE (JSON):",
            json.dumps(item.model_dump(), indent=2, ensure_ascii=False),
            "=" * 80,
        ]
    )
    return "\n".join(lines)


def get_next_item_id(drafts_path: Path | str | None = None) -> str:
    """Determine the next sequential gold item ID (e.g. gold_001, gold_002)."""
    path = Path(drafts_path or DEFAULT_DRAFTS_PATH)
    if not path.exists():
        return "gold_001"

    max_idx = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                item_id = rec.get("id", "")
                m = re.match(r"^gold_(\d+)$", item_id)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
            except Exception:
                continue
    return f"gold_{max_idx + 1:03d}"


def save_draft_item(item: GoldItem, output_path: Path | str | None = None) -> None:
    """Append a validated GoldItem to the target JSONL file."""
    path = Path(output_path or DEFAULT_DRAFTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure review_status is explicitly marked unverified unless certified verified
    dump_dict = item.model_dump()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dump_dict, ensure_ascii=False) + "\n")
    logger.info("Saved draft item %s to %s", item.id, path)


def generate_draft_items_batch(
    count: int = 5,
    output_path: Path | str | None = None,
    meta_path: Path | str | None = None,
    categories: list[CategoryType] | None = None,
) -> list[GoldItem]:
    """Generate a batch of scaffold draft items with explicit unverified tags."""
    passages = load_passages(meta_path)
    if not passages:
        logger.warning("No passages available to generate draft batch.")
        return []

    # Filter out pembukaan or header passages where pasal is None
    substantive = [p for p in passages if p.pasal is not None]
    if not substantive:
        substantive = passages

    target_cats = categories or [
        "factual_direct",
        "definitional",
        "obligation_and_prohibition",
        "sanctions_and_enforcement",
        "cross_article_synthesis",
    ]

    sampled_passages = random.sample(substantive, min(count, len(substantive)))
    draft_items: list[GoldItem] = []

    for idx, p in enumerate(sampled_passages, start=1):
        item_id = f"gold_{idx:03d}"
        cat = target_cats[(idx - 1) % len(target_cats)]
        item = generate_blank_template(passage=p, next_id=item_id, category=cat)

        # Inject candidate questions into template as unverified drafts
        cands = draft_candidate_questions(p, category=cat)
        if cands:
            item.question = f"[UNVERIFIED DRAFT]: {cands[0]}"

        # Always enforce unverified status
        item.review_status = "unverified"
        draft_items.append(item)

        if output_path:
            save_draft_item(item, output_path=output_path)

    return draft_items


def main() -> None:
    """CLI runner for the labeling helper."""
    parser = argparse.ArgumentParser(
        description="EasyPasal Labeling Helper: passage explorer, blank templates, and unverified candidate drafting."
    )
    parser.add_argument(
        "--meta-path",
        default=str(DEFAULT_META_PATH),
        help="Path to BM25 index metadata with chunks",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DRAFTS_PATH),
        help="Output JSONL path for saving draft items",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Display a sample passage and its blank JSON item template",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Display a sample passage with model-drafted candidate questions (marked unverified)",
    )
    parser.add_argument(
        "--generate-drafts-batch",
        type=int,
        default=0,
        metavar="N",
        help="Generate N scaffold draft templates marked unverified into the output file",
    )
    parser.add_argument(
        "--theme",
        default=None,
        help="Filter passages by theme",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help="Filter passages by document number (e.g. 'POJK 22 Tahun 2023')",
    )
    parser.add_argument(
        "--pasal",
        default=None,
        help="Filter passages by article number (e.g. '75')",
    )
    parser.add_argument(
        "--category",
        choices=VALID_CATEGORIES,
        default="factual_direct",
        help="Target category for generated template",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    passages = load_passages(args.meta_path)
    if not passages:
        print(f"Error: No passages found at '{args.meta_path}'. Run 'make index' first.")
        return

    filtered = filter_passages(
        passages,
        theme=args.theme,
        doc_number=args.doc,
        pasal=args.pasal,
    )

    if not filtered:
        print("No passages matched the specified filters.")
        return

    target_category: CategoryType = args.category

    # Mode 1: Batch Draft Generation
    if args.generate_drafts_batch > 0:
        count = args.generate_drafts_batch
        out_p = Path(args.output)
        print(f"Generating {count} scaffold draft templates marked 'unverified' to {out_p}...")
        drafts = generate_draft_items_batch(
            count=count, output_path=out_p, meta_path=args.meta_path
        )
        print(f"Successfully wrote {len(drafts)} unverified drafts to {out_p}")
        return

    # Mode 2: Show Sample Passage & Blank Template
    passage = random.choice(filtered)
    next_id = get_next_item_id(args.output)
    item = generate_blank_template(passage=passage, next_id=next_id, category=target_category)

    if args.draft:
        candidates = draft_candidate_questions(passage, category=target_category)
        print("=" * 80)
        print("⚠️  [UNVERIFIED DRAFT CANDIDATE QUESTIONS - REQUIRE HUMAN VERIFICATION]  ⚠️")
        print("=" * 80)
        print(f"SOURCE: {passage.citation_label} (Theme: {passage.theme})")
        print("-" * 80)
        print("PASSAGE EXCERPT:")
        print(passage.text[:400] + ("..." if len(passage.text) > 400 else ""))
        print("-" * 80)
        print("CANDIDATE QUESTIONS (STATUS: UNVERIFIED):")
        for i, cq in enumerate(candidates, 1):
            print(f"  [{i}] [UNVERIFIED] {cq}")
        print("=" * 80)
        item.question = f"[UNVERIFIED DRAFT]: {candidates[0]}" if candidates else item.question

    print(format_draft_display(item, passage))


if __name__ == "__main__":
    main()
