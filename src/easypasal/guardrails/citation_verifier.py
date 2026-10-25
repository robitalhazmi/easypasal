"""Post-hoc citation verification module to validate groundedness against retrieved passages."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from easypasal.generation.citations import check_abstention, extract_citations

if TYPE_CHECKING:
    from easypasal.retrieval.models import RetrievedChunk

logger = logging.getLogger("easypasal.guardrails.citation_verifier")


@dataclass
class ParsedCitation:
    """Deconstructed regulatory citation attributes."""

    raw_text: str
    doc_type: str | None = None
    doc_number: str | None = None
    pasal: str | None = None
    ayat: str | None = None
    angka: str | None = None


@dataclass
class CitationCheck:
    """Result of verifying a single citation against retrieved evidence."""

    citation: str
    is_grounded: bool
    doc_match: bool
    pasal_match: bool
    ayat_match: bool | None
    matched_chunk_id: str | None
    reason: str


@dataclass
class VerificationResult:
    """Comprehensive summary of post-hoc citation grounding verification."""

    is_valid: bool
    total_citations: int
    grounded_count: int
    ungrounded_count: int
    checks: list[CitationCheck] = field(default_factory=list)
    ungrounded_citations: list[str] = field(default_factory=list)
    hallucinated_pasal: list[str] = field(default_factory=list)


def parse_citation_details(citation_text: str) -> ParsedCitation:
    """Parse a citation string into its constituent statutory elements."""
    clean = citation_text.strip().strip("[]")

    # 1. Regulation Type (UU, POJK, PBI, PADG, SEOJK, PP, Perpu)
    doc_type_match = re.search(
        r"\b(Undang-Undang|UU|POJK|PBI|PADG|SEOJK|PP|Perpu)\b",
        clean,
        re.IGNORECASE,
    )
    doc_type = doc_type_match.group(1).upper() if doc_type_match else None
    if doc_type == "UNDANG-UNDANG":
        doc_type = "UU"

    # 2. Regulation Number & Year
    doc_num_match = re.search(
        r"\b(?:Undang-Undang|UU|POJK|PBI|PADG|SEOJK|PP|Perpu)\s+"
        r"(?:Nomor\s+)?([\w\.\/\-]+(?:\s+Tahun\s+\d{4})?)",
        clean,
        re.IGNORECASE,
    )
    doc_number = doc_num_match.group(1).strip() if doc_num_match else None

    # 3. Pasal (Article)
    pasal_match = re.search(r"\bpasal\s+(\d+[a-z]?)", clean, re.IGNORECASE)
    pasal = pasal_match.group(1).lower() if pasal_match else None

    # 4. Ayat (Clause)
    ayat_match = re.search(r"\bayat\s+\((\d+)\)|\bayat\s+(\d+)", clean, re.IGNORECASE)
    ayat = (ayat_match.group(1) or ayat_match.group(2)) if ayat_match else None

    # 5. Angka (Item)
    angka_match = re.search(r"\bangka\s+(\d+)", clean, re.IGNORECASE)
    angka = angka_match.group(1) if angka_match else None

    return ParsedCitation(
        raw_text=citation_text,
        doc_type=doc_type,
        doc_number=doc_number,
        pasal=pasal,
        ayat=ayat,
        angka=angka,
    )


def _normalize_string(text: str) -> str:
    """Normalize text by stripping punctuation and collapsing whitespace."""
    return re.sub(r"[\s\-_,\.\/]+", " ", text.lower()).strip()


def _is_doc_matched(parsed: ParsedCitation, chunk: RetrievedChunk) -> bool:
    """Determine whether the cited regulation matches the retrieved chunk document."""
    if not parsed.doc_number:
        # If no doc number was extracted, check if doc_type matches chunk doc_number
        if parsed.doc_type:
            return parsed.doc_type.lower() in chunk.doc_number.lower()
        return True

    norm_cite_doc = _normalize_string(parsed.doc_number)
    norm_chunk_doc = _normalize_string(chunk.doc_number)
    norm_chunk_id = _normalize_string(chunk.doc_id)

    # Check for direct inclusion in doc_number or doc_id
    if norm_cite_doc in norm_chunk_doc or norm_chunk_doc in norm_cite_doc:
        return True
    if norm_cite_doc in norm_chunk_id:
        return True

    # Check key numerical components (e.g. "22" and "2023")
    cite_digits = set(re.findall(r"\d+", norm_cite_doc))
    chunk_digits = set(re.findall(r"\d+", norm_chunk_doc))
    return bool(
        cite_digits
        and cite_digits.issubset(chunk_digits)
        and parsed.doc_type
        and parsed.doc_type.lower() in norm_chunk_doc
    )


def _is_pasal_matched(parsed: ParsedCitation, chunk: RetrievedChunk) -> bool:
    """Check if the cited article (pasal) exists within the retrieved chunk."""
    if not parsed.pasal:
        return True  # Document-level citation without specific pasal

    target_pasal = parsed.pasal.lower()

    # 1. Match chunk structured metadata
    if chunk.pasal and chunk.pasal.lower() == target_pasal:
        return True

    # 2. Match citation label (e.g. "... Pasal 71 ...")
    if chunk.citation_label:
        chunk_pasal_match = re.search(r"\bpasal\s+(\d+[a-z]?)", chunk.citation_label, re.IGNORECASE)
        if chunk_pasal_match and chunk_pasal_match.group(1).lower() == target_pasal:
            return True

    # 3. Match within chunk text
    text_pasal_match = re.search(
        rf"\bpasal\s+{re.escape(target_pasal)}\b", chunk.text, re.IGNORECASE
    )
    return text_pasal_match is not None


def _is_ayat_matched(parsed: ParsedCitation, chunk: RetrievedChunk) -> bool:
    """Check if the cited clause (ayat) exists within the retrieved chunk."""
    if not parsed.ayat:
        return True

    target_ayat = parsed.ayat

    # 1. Match chunk structured metadata
    if chunk.ayat and str(chunk.ayat) == target_ayat:
        return True

    # 2. Match citation label
    if chunk.citation_label and f"ayat ({target_ayat})" in chunk.citation_label.lower():
        return True

    # 3. Match within chunk text (e.g. "(1) PUJK wajib...")
    ayat_pattern = rf"\(?\b{re.escape(target_ayat)}\b\)?|\bayat\s+\({re.escape(target_ayat)}\)"
    return re.search(ayat_pattern, chunk.text) is not None


def verify_single_citation(
    citation_text: str,
    retrieved_chunks: list[RetrievedChunk],
) -> CitationCheck:
    """Verify a single citation against the retrieved chunks."""
    parsed = parse_citation_details(citation_text)

    # 1. Find chunks matching the document
    matching_doc_chunks = [c for c in retrieved_chunks if _is_doc_matched(parsed, c)]

    if not matching_doc_chunks:
        return CitationCheck(
            citation=citation_text,
            is_grounded=False,
            doc_match=False,
            pasal_match=False,
            ayat_match=False if parsed.ayat else None,
            matched_chunk_id=None,
            reason=f"Regulasi '{parsed.doc_number or parsed.doc_type or citation_text}' tidak ditemukan dalam bukti yang diretrieve.",
        )

    # 2. Check Pasal match among matching document chunks
    matching_pasal_chunks = [c for c in matching_doc_chunks if _is_pasal_matched(parsed, c)]

    if parsed.pasal and not matching_pasal_chunks:
        return CitationCheck(
            citation=citation_text,
            is_grounded=False,
            doc_match=True,
            pasal_match=False,
            ayat_match=False if parsed.ayat else None,
            matched_chunk_id=None,
            reason=f"Pasal {parsed.pasal} tidak ditemukan dalam dokumen '{matching_doc_chunks[0].doc_number}' yang diretrieve.",
        )

    # 3. Check Ayat match if specified
    if parsed.ayat:
        matching_ayat_chunks = [c for c in matching_pasal_chunks if _is_ayat_matched(parsed, c)]
        if not matching_ayat_chunks:
            return CitationCheck(
                citation=citation_text,
                is_grounded=False,
                doc_match=True,
                pasal_match=True,
                ayat_match=False,
                matched_chunk_id=matching_pasal_chunks[0].chunk_id,
                reason=f"Ayat ({parsed.ayat}) tidak ditemukan dalam Pasal {parsed.pasal} yang diretrieve.",
            )
        matched_chunk = matching_ayat_chunks[0]
        ayat_matched = True
    else:
        matched_chunk = matching_pasal_chunks[0]
        ayat_matched = None

    return CitationCheck(
        citation=citation_text,
        is_grounded=True,
        doc_match=True,
        pasal_match=True if parsed.pasal else True,
        ayat_match=ayat_matched,
        matched_chunk_id=matched_chunk.chunk_id,
        reason="Sitasi terverifikasi dan didukung oleh bukti regulasi.",
    )


def verify_citations(
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
    citations: list[str] | None = None,
) -> VerificationResult:
    """Verify that all citations in the generated answer are grounded in the retrieved chunks.

    Args:
        answer: Generated response text from RAG pipeline.
        retrieved_chunks: Candidate passages retrieved for the query.
        citations: Optional pre-extracted citations list. If None, extracted automatically.

    Returns:
        VerificationResult detailing grounding status, pass/fail checks, and hallucinated pasal.
    """
    cites = citations if citations is not None else extract_citations(answer)

    # If no citations were generated:
    if not cites:
        is_abstain = check_abstention(answer)
        return VerificationResult(
            is_valid=is_abstain,
            total_citations=0,
            grounded_count=0,
            ungrounded_count=0 if is_abstain else 1,
            checks=[],
            ungrounded_citations=[]
            if is_abstain
            else ["Tidak ada sitasi regulasi yang dicantumkan"],
            hallucinated_pasal=[],
        )

    checks: list[CitationCheck] = []
    ungrounded: list[str] = []
    hallucinated: list[str] = []

    for cite in cites:
        check = verify_single_citation(cite, retrieved_chunks)
        checks.append(check)
        if not check.is_grounded:
            ungrounded.append(cite)
            if check.doc_match and not check.pasal_match:
                parsed = parse_citation_details(cite)
                hallucinated.append(f"Pasal {parsed.pasal} ({parsed.doc_number or 'Regulasi'})")
            elif not check.doc_match:
                hallucinated.append(cite)

    grounded_count = len(cites) - len(ungrounded)
    is_valid = len(ungrounded) == 0

    return VerificationResult(
        is_valid=is_valid,
        total_citations=len(cites),
        grounded_count=grounded_count,
        ungrounded_count=len(ungrounded),
        checks=checks,
        ungrounded_citations=ungrounded,
        hallucinated_pasal=hallucinated,
    )
