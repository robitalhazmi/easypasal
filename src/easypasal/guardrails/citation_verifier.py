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
    realigned_citations: dict[str, str] = field(default_factory=dict)


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


def _extract_numerical_claims(statement: str, parsed: ParsedCitation) -> list[str]:
    """Extract financial figures, currencies, and percentages asserted in a statement,

    ignoring standard statutory metadata digits (regulation number, year, pasal, ayat).
    """
    candidates = re.findall(
        r"(?:Rp\s*[\d\.\,]+|(?:\b\d+[\.,]?\d*|\b\d+)\s*(?:miliar|triliun|juta|ribu|persen|%)\b)",
        statement,
        re.IGNORECASE,
    )
    standalone_large = re.findall(r"\b\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?\b", statement)
    all_raw = set(candidates + standalone_large)

    filtered: list[str] = []
    meta_digits: set[str] = set()
    if parsed.doc_number:
        meta_digits.update(re.findall(r"\d+", parsed.doc_number))
    if parsed.pasal:
        meta_digits.update(re.findall(r"\d+", parsed.pasal))
    if parsed.ayat:
        meta_digits.update(re.findall(r"\d+", parsed.ayat))

    for item in all_raw:
        item_digits = set(re.findall(r"\d+", item))
        if item_digits and item_digits.issubset(meta_digits):
            continue
        filtered.append(item.strip())

    return filtered


def _find_surrounding_statement(answer: str, citation: str) -> str:
    """Find the specific sentence or bullet point in answer containing the citation."""
    sentences = re.split(r"(?<=[.!?\n])\s+", answer)
    for s in sentences:
        if citation in s:
            return s
    return answer


def verify_single_citation(
    citation_text: str,
    retrieved_chunks: list[RetrievedChunk],
    surrounding_statement: str | None = None,
    question: str | None = None,
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

    # 4. Verify Numerical Claim Grounding if numerical figures are asserted
    if surrounding_statement and matched_chunk:
        numerical_claims = _extract_numerical_claims(surrounding_statement, parsed)
        for claim in numerical_claims:
            claim_digits = re.findall(r"\d+", claim)
            if not claim_digits:
                continue
            primary_digit = claim_digits[0]
            chunk_digits = set(re.findall(r"\d+", matched_chunk.text))
            if primary_digit not in chunk_digits and not any(
                primary_digit in d for d in chunk_digits
            ):
                return CitationCheck(
                    citation=citation_text,
                    is_grounded=False,
                    doc_match=True,
                    pasal_match=True,
                    ayat_match=ayat_matched,
                    matched_chunk_id=matched_chunk.chunk_id,
                    reason=f"Klaim angka/nominal '{claim}' pada sitasi {citation_text} tidak didukung oleh teks regulasi yang diretrieve.",
                )

    # 5. Verify Query-Aware Sector & Concept Alignment if question is provided
    if question and matched_chunk:
        from easypasal.retrieval.retriever import _is_sector_mismatch

        if _is_sector_mismatch(question.lower(), matched_chunk):
            return CitationCheck(
                citation=citation_text,
                is_grounded=False,
                doc_match=True,
                pasal_match=True,
                ayat_match=ayat_matched,
                matched_chunk_id=matched_chunk.chunk_id,
                reason=f"Ketidaksesuaian sektor/konteks: sitasi {citation_text} tidak mengatur subjek yang ditanyakan.",
            )

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
    question: str | None = None,
) -> VerificationResult:
    """Verify that all citations in the generated answer are grounded in the retrieved chunks.

    Args:
        answer: Generated response text from RAG pipeline.
        retrieved_chunks: Candidate passages retrieved for the query.
        citations: Optional pre-extracted citations list. If None, extracted automatically.
        question: Optional user question to validate sector and concept alignment.

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
        statement = _find_surrounding_statement(answer, cite)
        check = verify_single_citation(
            cite,
            retrieved_chunks,
            surrounding_statement=statement,
            question=question,
        )
        checks.append(check)
        if not check.is_grounded:
            ungrounded.append(cite)
            if check.doc_match and not check.pasal_match:
                parsed = parse_citation_details(cite)
                hallucinated.append(f"Pasal {parsed.pasal} ({parsed.doc_number or 'Regulasi'})")
            elif (
                not check.doc_match
                or "Klaim angka" in check.reason
                or "Ketidaksesuaian sektor" in check.reason
            ):
                hallucinated.append(f"{cite} [{check.reason}]")
            else:
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


def detect_and_realign_inverted_citations(
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
    question: str,
) -> tuple[str, list[str], dict[str, str]]:
    """Detect and automatically realign inverted statutory citations where a supervisory/remedial
    clause (e.g. OJK dapat meminta) was mistakenly cited instead of the entity's primary general
    obligation clause (e.g. Bank/PUJK wajib), or vice versa.

    Returns:
        tuple of (realigned_answer_text, realigned_citations_list, mapping_of_replacements)
    """
    if not answer or not retrieved_chunks or not question:
        cites = extract_citations(answer)
        return answer, cites, {}

    q_low = question.lower()
    is_obligation_query = bool(
        re.search(
            r"\b(?:wajib|harus|kewajiban|keharusan|dilarang|larangan|must|mandatory|required|shall|prohibited|obligation)\b",
            q_low,
        )
        and not re.search(
            r"\b(?:kapan|apakah)\s+(?:ojk|otoritas|bi|bank indonesia)\s+(?:dapat|berwenang|meminta|memerintahkan)\b",
            q_low,
        )
    )

    if not is_obligation_query:
        cites = extract_citations(answer)
        return answer, cites, {}

    current_cites = extract_citations(answer)
    if not current_cites:
        return answer, [], {}

    realigned_text = answer
    replacements: dict[str, str] = {}

    from easypasal.generation.prompts import _detect_statutory_role

    for cite in current_cites:
        parsed = parse_citation_details(cite)
        matching_chunks = [
            c
            for c in retrieved_chunks
            if _is_doc_matched(parsed, c) and _is_pasal_matched(parsed, c)
        ]
        if not matching_chunks:
            continue

        matched_chunk = matching_chunks[0]
        chunk_role = _detect_statutory_role(matched_chunk.text)

        # If cited chunk is an authority remedial / sanction power clause:
        if chunk_role == "[Peran: Wewenang Korektif Otoritas / Sanksi]":
            # Search among all retrieved chunks in the SAME regulation for a primary entity obligation chunk
            candidate_primaries = [
                c
                for c in retrieved_chunks
                if _is_doc_matched(parsed, c)
                and _detect_statutory_role(c.text) == "[Peran: Kewajiban Pokok Entitas]"
                and (not c.status or c.status == "berlaku")
            ]
            if candidate_primaries:
                primary_chunk = candidate_primaries[0]
                new_label = primary_chunk.citation_label
                if new_label and new_label != cite:
                    clean_cite = cite.strip("[]")
                    clean_new = new_label.strip("[]")
                    realigned_text = re.sub(
                        rf"\[\s*{re.escape(clean_cite)}\s*\]",
                        f"[{clean_new}]",
                        realigned_text,
                    )
                    realigned_text = re.sub(
                        rf"\b{re.escape(clean_cite)}\b",
                        clean_new,
                        realigned_text,
                    )
                    replacements[cite] = f"[{clean_new}]"
                    logger.warning(
                        "Inverted legal reasoning detected for query %r: realigned remedial citation %s -> %s",
                        question[:50],
                        cite,
                        new_label,
                    )

    new_citations = extract_citations(realigned_text)
    return realigned_text, new_citations, replacements
