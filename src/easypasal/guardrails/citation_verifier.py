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

        # 6. Verify Normative Role & Modality Entailment for Obligation Queries
        from easypasal.generation.prompts import _detect_statutory_role
        from easypasal.retrieval.fusion import detect_normative_intent

        if detect_normative_intent(question) == "obligation":
            chunk_role = _detect_statutory_role(matched_chunk.text)
            has_primary_in_pool = any(
                _is_doc_matched(parsed, c)
                and _detect_statutory_role(c.text) == "[Peran: Kewajiban Pokok Entitas]"
                for c in retrieved_chunks
            )
            if chunk_role == "[Peran: Wewenang Korektif Otoritas / Sanksi]" and has_primary_in_pool:
                return CitationCheck(
                    citation=citation_text,
                    is_grounded=False,
                    doc_match=True,
                    pasal_match=True,
                    ayat_match=ayat_matched,
                    matched_chunk_id=matched_chunk.chunk_id,
                    reason=f"Ketidaksesuaian hierarki norma: sitasi {citation_text} mengatur wewenang korektif/sanksi otoritas, bukan kewajiban pokok entitas.",
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


def realign_answer_subject(answer: str, question: str) -> str:
    """Ensure the grammatical subject of the opening sentence matches the query's subject entity.

    Prevents regulatory cross-entity contamination (e.g. answering a "Bank Umum" question
    with "PUJK wajib..." when the statutory rule applies to Bank).
    Also harmonizes sentence-wide co-references (e.g. 'diterima PUJK' -> 'diterima Lembaga Jasa Keuangan')
    while strictly protecting official regulation citations inside bracketed tags.
    """
    if not answer or not question:
        return answer

    from easypasal.generation.prompts import (
        _KNOWN_INSTITUTIONAL_SUBJECTS,
        extract_query_subject,
    )

    q_subject = extract_query_subject(question)
    if not q_subject:
        return answer

    clean_ans = answer.strip()
    segments = re.split(r"(\[[^\]]+\])", clean_ans)
    prep_words = r"(?:sejak|diterima|oleh|kepada|pada|di|terhadap|bagi|untuk|antara|dengan|by|to|for|at|in|with|from|received\s+by)"
    action_words = r"(?:wajib|harus|dilarang|dapat|memiliki|menyediakan|menempatkan|melakukan|must|shall|should|may|is\s+required\s+to|are\s+required\s+to)"

    modified = False

    for entity in _KNOWN_INSTITUTIONAL_SUBJECTS:
        if entity.lower() == q_subject.lower():
            continue
        # If q_subject is "Bank Umum" and entity is "Bank", they are legally equivalent, don't replace
        if q_subject.lower() in ("bank umum", "bank") and entity.lower() in (
            "bank umum",
            "bank",
        ):
            continue

        e_pattern = re.escape(entity)

        # Process only prose segments (even indices), leaving bracketed citations (odd indices) intact
        for i in range(0, len(segments), 2):
            seg = segments[i]

            # 1. Leading entity or list-item entity duty (e.g. '^PUJK wajib...', '^\n- PUJK wajib...')
            lead_pat = rf"(^|[\r\n]+[-*•\d.]*\s*)({e_pattern})(\s+{action_words}\b)"
            new_seg = re.sub(lead_pat, rf"\g<1>{q_subject}\g<3>", seg, flags=re.IGNORECASE)

            # 2. Prepositional / relational co-reference: e.g. 'diterima PUJK', 'oleh PUJK'
            prep_pat = rf"(\b{prep_words}\s+)({e_pattern})\b"
            new_seg = re.sub(prep_pat, rf"\g<1>{q_subject}", new_seg, flags=re.IGNORECASE)

            if new_seg != seg:
                segments[i] = new_seg
                modified = True

    if modified:
        realigned = "".join(segments)
        logger.warning(
            "Subject consistency guardrail: harmonized entity co-reference -> %r for query %r",
            q_subject,
            question[:50],
        )
        return realigned

    return answer


def _attach_bracket_citation_to_opening(text: str, citation_label: str) -> str:
    """Ensure the opening sentence of an answer carries the primary bracketed citation."""
    clean_label = citation_label.strip("[]").strip()
    if not text or not clean_label:
        return text

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text

    first_para = paragraphs[0]
    if f"[{clean_label}]" in first_para or re.search(r"\[[^\]]+\]", first_para):
        return text

    # Strip trailing redundant " menurut regulasi OJK" if immediately before terminal punctuation
    first_para = re.sub(
        r"\s+menurut\s+regulasi\s+(?:OJK|Otoritas\s+Jasa\s+Keuangan|BI|Bank\s+Indonesia)(?=[.!?]?$)",
        "",
        first_para,
        flags=re.IGNORECASE,
    )

    if first_para.endswith((".", "!", "?")):
        first_para = f"{first_para[:-1].rstrip()} [{clean_label}]{first_para[-1]}"
    else:
        first_para = f"{first_para} [{clean_label}]."

    paragraphs[0] = first_para
    return "\n\n".join(paragraphs)


def _strip_remedial_dump_paragraphs(text: str, inverted_pasal: str | None = None) -> str:
    """Strip trailing paragraphs that dump supervisory/remedial clauses after a valid BLUF opening."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        return text

    first_para = paragraphs[0]
    if not re.search(
        r"\b(?:wajib|harus|dilarang|must|shall|required|prohibited)\b",
        first_para,
        re.IGNORECASE,
    ):
        return text

    kept: list[str] = [first_para]
    dropping_remedial_list = False

    for para in paragraphs[1:]:
        p_norm = re.sub(r"\s+", " ", para.lower()).strip()

        # Check if this paragraph is a continuation item (a., b., c., d.) of a dropped remedial block
        if dropping_remedial_list and re.match(r"^(?:[a-z][\.\)]|\(\d+\)|\d+[\.\)])\s+", p_norm):
            continue

        is_remedial_dump = bool(
            re.search(
                r"\b(?:ojk|otoritas\s+jasa\s+keuangan|bank\s+indonesia)\s*(?:\([a-z]+\)\s*)?"
                r"(?:dapat\s+meminta|memiliki\s+kebijakan\s+untuk\s+meminta|berwenang\s+meminta)\b",
                p_norm,
            )
            or (
                inverted_pasal
                and re.search(rf"\bpasal\s+{re.escape(inverted_pasal.lower())}\b", p_norm)
                and re.search(
                    r"\b(?:dapat\s+meminta|penilaian\s+ojk|penilaian\s+otoritas)\b", p_norm
                )
            )
        )
        if is_remedial_dump:
            dropping_remedial_list = True
            continue

        dropping_remedial_list = False
        kept.append(para)

    return "\n\n".join(kept)


def reformat_narrative_citations_to_brackets(text: str) -> str:
    """Transform conversational narrative citation phrasing into canonical square-bracketed citations.

    Converts phrases like ', seperti yang ditetapkan dalam POJK 22 Tahun 2023, Pasal 71 ayat (1)'
    or ', as stipulated in UU Nomor 4 Tahun 2023, Pasal 245 ayat (1)' into '[...]'.
    """
    if not text:
        return text

    # Indonesian narrative citations
    pattern_id = re.compile(
        r",?\s*(?:seperti\s+yang\s+(?:ditetapkan|diatur|tercantum)\s+dalam|"
        r"sebagaimana\s+(?:ditetapkan|diatur|tercantum|dimaksud)\s+dalam|"
        r"sesuai\s+(?:dengan\s+)?(?:ketentuan\s+)?(?:dalam\s+)?|"
        r"berdasarkan\s+ketentuan\s+)\s*"
        r"([\w\s\./]+?,\s*Pasal\s+\d+[a-z]?(?:\s+(?:ayat\s+\(\d+[a-z]?\)|angka\s+\d+))?)",
        re.IGNORECASE,
    )

    def _repl_id(m: re.Match[str]) -> str:
        cit = m.group(1).strip()
        return f" [{cit}]"

    text = pattern_id.sub(_repl_id, text)

    # English narrative citations
    pattern_en = re.compile(
        r",?\s*(?:as\s+(?:stipulated|stated|specified|regulated|set\s+out)\s+in|"
        r"according\s+to|pursuant\s+to)\s*"
        r"([\w\s\./]+?,\s*Pasal\s+\d+[a-z]?(?:\s+(?:ayat\s+\(\d+[a-z]?\)|angka\s+\d+))?)",
        re.IGNORECASE,
    )

    def _repl_en(m: re.Match[str]) -> str:
        cit = m.group(1).strip()
        return f" [{cit}]"

    text = pattern_en.sub(_repl_en, text)
    return text


def simplify_dangling_cross_references(text: str) -> str:
    """Replace dangling internal statutory cross-references in generated prose with self-contained descriptions.

    Converts phrases like 'dokumen sebagaimana dimaksud dalam Pasal 71 ayat (2) dan/atau ayat (3)'
    into 'dokumen yang dipersyaratkan' so the answer remains self-contained and does not leak
    unretrieved cross-referenced article numbers.
    """
    if not text:
        return text

    # 1. "dokumen sebagaimana dimaksud dalam Pasal X [ayat (Y) dan/atau ayat (Z)]" -> "dokumen yang dipersyaratkan"
    text = re.sub(
        r"\bdokumen\s+sebagaimana\s+dimaksud\s+(?:dalam|pada)\s+Pasal\s+\d+[a-z]?"
        r"(?:\s+ayat\s+\(\d+[a-z]?\))?"
        r"(?:\s+(?:dan/atau|serta|dan|atau)\s+ayat\s+\(\d+[a-z]?\))*",
        "dokumen yang dipersyaratkan",
        text,
        flags=re.IGNORECASE,
    )

    # 2. "persyaratan sebagaimana dimaksud dalam Pasal X..." -> "persyaratan yang ditetapkan"
    text = re.sub(
        r"\bpersyaratan\s+sebagaimana\s+dimaksud\s+(?:dalam|pada)\s+Pasal\s+\d+[a-z]?"
        r"(?:\s+ayat\s+\(\d+[a-z]?\))?"
        r"(?:\s+(?:dan/atau|serta|dan|atau)\s+ayat\s+\(\d+[a-z]?\))*",
        "persyaratan yang ditetapkan",
        text,
        flags=re.IGNORECASE,
    )

    return text


_SENTENCE_STOPWORDS: set[str] = {
    "yang",
    "dalam",
    "untuk",
    "pada",
    "oleh",
    "dari",
    "atau",
    "serta",
    "dengan",
    "sejak",
    "paling",
    "lama",
    "hari",
    "kerja",
    "wajib",
    "harus",
    "melakukan",
    "lembaga",
    "jasa",
    "keuangan",
    "pelaku",
    "usaha",
    "pujk",
    "bank",
    "umum",
    "konsumen",
    "pasal",
    "ayat",
    "angka",
    "huruf",
    "tahun",
    "nomor",
    "pojk",
    "sebagaimana",
    "dimaksud",
    "tersebut",
    "apabila",
    "jika",
    "juga",
    "selain",
}


def _score_sentence_chunk_match(sentence: str, chunk: RetrievedChunk) -> float:
    """Compute fine-grained lexical, numerical, and modality alignment between a single sentence and a chunk."""
    from easypasal.generation.prompts import _legal_stem

    # Strip bracketed citations and statutory designators before comparing numbers
    s_clean = re.sub(r"\[[^\]]+\]", " ", sentence.lower())
    s_no_ref = re.sub(
        r"\b(?:pasal|ayat|angka|tahun|nomor|pojk|uu|pbi|padg|seojk)\s*\(?\d+[a-z]?\)?",
        " ",
        s_clean,
    )
    c_body = re.sub(r"^\[[^\]]+\]\s*", "", chunk.text.strip()).lower()
    c_no_ref = re.sub(
        r"\b(?:pasal|ayat|angka|tahun|nomor|pojk|uu|pbi|padg|seojk)\s*\(?\d+[a-z]?\)?",
        " ",
        c_body,
    )

    s_nums = set(re.findall(r"\b\d+\b", s_no_ref))
    c_nums = set(re.findall(r"\b\d+\b", c_no_ref))

    score = 0.0
    if s_nums:
        if s_nums <= c_nums:
            score += 3.5
        elif s_nums & c_nums:
            score += 1.5
        else:
            score -= 2.5

    # Modality concordance (oral vs written)
    s_lisan = bool(re.search(r"\blisan\b", s_clean))
    c_lisan = bool(re.search(r"\blisan\b", c_body))
    s_tertulis = bool(re.search(r"\btertulis\b", s_clean))
    c_tertulis = bool(re.search(r"\btertulis\b", c_body))

    if s_lisan:
        if c_lisan:
            score += 2.5
        elif c_tertulis:
            score -= 2.0
    if s_tertulis:
        if c_tertulis:
            score += 2.5
        elif c_lisan:
            score -= 2.0

    s_tokens = {
        _legal_stem(t) for t in re.findall(r"[a-z]{4,}", s_no_ref) if t not in _SENTENCE_STOPWORDS
    }
    c_tokens = {
        _legal_stem(t) for t in re.findall(r"[a-z]{4,}", c_no_ref) if t not in _SENTENCE_STOPWORDS
    }
    s_tokens = {t for t in s_tokens if len(t) >= 3}
    c_tokens = {t for t in c_tokens if len(t) >= 3}

    if s_tokens and c_tokens:
        overlap = len(s_tokens & c_tokens)
        score += (overlap / len(s_tokens)) * 3.0 + min(overlap * 0.25, 1.5)

    return round(score, 4)


def decouple_confabulated_causality(
    text: str,
    retrieved_chunks: list[RetrievedChunk] | None = None,
) -> str:
    """Decouple confabulated causal/conditional links between unrelated statutory clauses.

    Removes false or self-contradictory conditions such as:
    - 'Dalam hal tidak terdapat kesepakatan terhadap hasil penanganan pengaduan..., Lembaga Jasa Keuangan wajib menangani pengaduan secara tertulis...'
    - 'Jika dokumen pengaduan belum lengkap, Lembaga Jasa Keuangan juga wajib melakukan tindak lanjut dan menyelesaikan Pengaduan secara tertulis paling lama 10 (sepuluh) hari kerja sejak dokumen ... diterima secara lengkap'
    and restores them as independent additive statements ('Selain itu, ...').
    """
    if not text:
        return text

    # 1. Explicit disagreement condition falsely attached to written complaint duty
    pattern = re.compile(
        r"\bDalam\s+hal\s+tidak\s+terdapat\s+kesepakatan\s+terhadap\s+hasil\s+penanganan\s+pengaduan\s+yang\s+dilakukan\s+oleh\s+[^,]+,\s*",
        re.IGNORECASE,
    )
    if pattern.search(text):
        text = pattern.sub("Selain itu, ", text)

    # 2. Self-contradictory "incomplete document" condition attached to a clause requiring complete documents / written resolution
    contradictory_pattern = re.compile(
        r"(?:^|(?<=[.!?]\s))((?:Jika|Apabila|Dalam\s+hal|Bila)\s+[^,.!?]{3,100}?\b(?:belum|tidak)\s+lengkap[^,.!?]*,\s+)"
        r"([^.!?]*?\b(?:secara\s+lengkap|telah\s+lengkap|melengkapi\s+dokumen|menyelesaikan\s+pengaduan\s+secara\s+tertulis)\b)",
        re.IGNORECASE,
    )
    if contradictory_pattern.search(text):
        text = contradictory_pattern.sub(r"Selain itu, \2", text)

    # 3. General chunk-verified conditional bridge sanitizer:
    # If a non-opening sentence begins with a conditional clause ("Jika/Apabila/Dalam hal X, <Subject> wajib..."),
    # check if the best-matching retrieved chunk for the main obligation is actually unconditional.
    if retrieved_chunks:
        cond_sent_pattern = re.compile(
            r"(?<=[.!?]\s)((?:Jika|Apabila|Dalam\s+hal|Bila)\s+([^,.!?]{5,120}),\s+)([^.!?]+\b(?:wajib|harus|dilarang|must|shall)\b[^.!?]*[.!?])",
            re.IGNORECASE,
        )

        def _verify_cond_bridge(m: re.Match[str]) -> str:
            main_clause = m.group(3)
            scored = [(_score_sentence_chunk_match(main_clause, c), c) for c in retrieved_chunks]
            if not scored:
                return m.group(0)
            best_score, best_c = max(scored, key=lambda item: item[0])
            if best_score < 4.5:
                return m.group(0)
            c_body = re.sub(r"^\[[^\]]+\]\s*", "", best_c.text.strip()).lower()
            pre_wajib = re.split(r"\b(?:wajib|harus|dilarang)\b", c_body, maxsplit=1)[0]
            has_statutory_condition = bool(
                re.search(r"\b(?:dalam\s+hal|jika|apabila|bila|sepanjang)\b", pre_wajib)
            )
            if not has_statutory_condition:
                return f"Selain itu, {main_clause}"
            return m.group(0)

        text = cond_sent_pattern.sub(_verify_cond_bridge, text)

    return text


def align_sentence_level_citations(
    text: str,
    retrieved_chunks: list[RetrievedChunk],
    question: str,
) -> tuple[str, dict[str, str]]:
    """Verify and align citations at the individual sentence level for multi-sentence normative answers.

    Ensures that:
    1. Every distinct normative sentence in a multi-sentence answer carries its own bracketed citation
       when strongly grounded in a retrieved primary obligation chunk (preventing single-citation collapse).
    2. If a sentence carries a sibling citation whose chunk does not match the sentence's numerical/modality
       claims, it is realigned to the chunk that specifically grounds that sentence.
    """
    if not text or not retrieved_chunks:
        return text, {}

    from easypasal.generation.prompts import (
        _detect_statutory_role,
        score_primary_obligation_alignment,
    )

    primary_candidates = [
        c
        for c in retrieved_chunks
        if _detect_statutory_role(c.text) == "[Peran: Kewajiban Pokok Entitas]"
        and (not c.status or c.status == "berlaku")
        and c.citation_label
        and score_primary_obligation_alignment(c, question) > 0.0
    ]
    if not primary_candidates:
        return text, {}

    paragraphs = text.split("\n\n")
    normative_re = re.compile(
        r"\b(?:wajib|harus|dilarang|must|shall|required|prohibited)\b",
        re.IGNORECASE,
    )

    # Count total normative sentences across the answer
    all_sentences: list[str] = []
    for p in paragraphs:
        if p.strip():
            all_sentences.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", p.strip()))
    normative_count = sum(
        1 for s in all_sentences if normative_re.search(s) and len(s.split()) >= 5
    )

    replacements: dict[str, str] = {}
    new_paragraphs: list[str] = []
    sent_counter = 0

    for para in paragraphs:
        if not para.strip():
            new_paragraphs.append(para)
            continue

        # Do not split bullet/lettered enumeration items
        if re.match(r"^(?:[a-z][\.\)]|\(\d+\)|\d+[\.\)]|[-*•])\s+", para.strip(), re.IGNORECASE):
            new_paragraphs.append(para)
            continue

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", para.strip())
        updated_sentences: list[str] = []

        for sent in sentences:
            sent_counter += 1
            if not normative_re.search(sent) or len(sent.split()) < 5:
                updated_sentences.append(sent)
                continue

            # Score all candidate primary chunks against this specific sentence
            scored_matches = [
                (
                    _score_sentence_chunk_match(sent, c)
                    + 0.05 * min(score_primary_obligation_alignment(c, question), 6.0),
                    _score_sentence_chunk_match(sent, c),
                    c,
                )
                for c in primary_candidates
            ]
            _combined, best_raw_score, best_c = max(scored_matches, key=lambda item: item[0])
            best_label = best_c.citation_label.strip("[]").strip()

            bracket_matches = list(re.finditer(r"\[([^\]]+)\]", sent))
            if not bracket_matches:
                # Attach missing per-sentence citation if part of a multi-sentence answer and strongly matched
                if normative_count >= 2 and best_raw_score >= 4.5 and best_label:
                    if sent.endswith((".", "!", "?")):
                        sent = f"{sent[:-1].rstrip()} [{best_label}]{sent[-1]}"
                    else:
                        sent = f"{sent} [{best_label}]."
                    replacements[f"<uncited_sentence_{sent_counter}>"] = f"[{best_label}]"
                updated_sentences.append(sent)
                continue

            # Sentence already has a bracketed citation; verify whether it matches this sentence or a sibling
            existing_cite = bracket_matches[-1].group(1).strip()
            parsed_existing = parse_citation_details(existing_cite)
            matched_existing = [
                c
                for c in retrieved_chunks
                if _is_doc_matched(parsed_existing, c)
                and _is_pasal_matched(parsed_existing, c)
                and _is_ayat_matched(parsed_existing, c)
            ]
            existing_raw_score = (
                _score_sentence_chunk_match(sent, matched_existing[0]) if matched_existing else -5.0
            )

            if (
                best_label.lower() != existing_cite.lower()
                and best_raw_score >= 4.5
                and best_raw_score > existing_raw_score + 2.5
            ):
                sent = re.sub(
                    rf"\[\s*{re.escape(existing_cite)}\s*\]",
                    f"[{best_label}]",
                    sent,
                )
                replacements[existing_cite] = f"[{best_label}]"

            updated_sentences.append(sent)

        new_paragraphs.append(" ".join(updated_sentences))

    return "\n\n".join(new_paragraphs), replacements


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

    if check_abstention(answer):
        return answer, [], {}

    # 0. Canonical normalization: simplify internal cross-references, reformat narrative citations,
    # and decouple confabulated causality
    answer = simplify_dangling_cross_references(answer)
    answer = reformat_narrative_citations_to_brackets(answer)
    answer = decouple_confabulated_causality(answer, retrieved_chunks=retrieved_chunks)

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

    from easypasal.generation.prompts import (
        _detect_statutory_role,
        score_primary_obligation_alignment,
    )

    # 0b. Sentence-level claim-to-chunk citation alignment for multi-sentence answers
    answer, sent_replacements = align_sentence_level_citations(answer, retrieved_chunks, question)

    current_cites = extract_citations(answer)
    realigned_text = answer
    replacements: dict[str, str] = dict(sent_replacements)

    # If no citations were extracted at all, attach the top primary obligation chunk if strongly aligned
    if not current_cites:
        primaries = sorted(
            [
                c
                for c in retrieved_chunks
                if _detect_statutory_role(c.text) == "[Peran: Kewajiban Pokok Entitas]"
                and (not c.status or c.status == "berlaku")
                and c.citation_label
            ],
            key=lambda ch: score_primary_obligation_alignment(ch, question),
            reverse=True,
        )
        if primaries and score_primary_obligation_alignment(primaries[0], question) >= 2.5:
            best_label = primaries[0].citation_label
            realigned_text = _attach_bracket_citation_to_opening(realigned_text, best_label)
            replacements["<uncited>"] = f"[{best_label}]"
            return realigned_text, extract_citations(realigned_text), replacements
        return answer, [], {}

    for cite in current_cites:
        parsed = parse_citation_details(cite)

        # 0. Check if surrounding statement describes consumer dispute remedies while mistakenly citing ayat 1
        stmt = _find_surrounding_statement(realigned_text, cite)
        if (
            stmt
            and parsed.ayat == "1"
            and re.search(
                r"\b(?:konsumen\s+dapat|mengajukan\s+sengketa|menyampaikan\s+pengaduan\s+kepada\s+otoritas)\b",
                stmt,
                re.IGNORECASE,
            )
        ):
            remedy_ayat_chunks = [
                c
                for c in retrieved_chunks
                if _is_doc_matched(parsed, c)
                and _is_pasal_matched(parsed, c)
                and str(c.ayat) == "2"
            ]
            if remedy_ayat_chunks and remedy_ayat_chunks[0].citation_label:
                target_label = remedy_ayat_chunks[0].citation_label
                realigned_text = re.sub(
                    rf"\[\s*{re.escape(cite.strip('[]'))}\s*\]",
                    f"[{target_label.strip('[]')}]",
                    realigned_text,
                )
                replacements[cite] = f"[{target_label.strip('[]')}]"
                continue

        # 1. Search among all retrieved chunks in the SAME regulation for primary entity obligation chunks,
        # ranked by primary obligation alignment score
        candidate_primaries = sorted(
            [
                c
                for c in retrieved_chunks
                if _is_doc_matched(parsed, c)
                and _detect_statutory_role(c.text) == "[Peran: Kewajiban Pokok Entitas]"
                and (not c.status or c.status == "berlaku")
            ],
            key=lambda ch: score_primary_obligation_alignment(ch, question),
            reverse=True,
        )
        if not candidate_primaries:
            continue

        primary_chunk = candidate_primaries[0]
        primary_label = primary_chunk.citation_label
        if not primary_label or primary_label == cite:
            continue

        primary_parsed = parse_citation_details(primary_label)

        # Check if the cited clause is different from the primary clause (or missing pasal/ayat)
        is_different_clause = bool(
            not parsed.pasal
            or (
                parsed.pasal
                and primary_parsed.pasal
                and parsed.pasal.lower() != primary_parsed.pasal.lower()
            )
            or (parsed.ayat and primary_parsed.ayat and parsed.ayat != primary_parsed.ayat)
        )

        matching_chunks = [
            c
            for c in retrieved_chunks
            if _is_doc_matched(parsed, c)
            and _is_pasal_matched(parsed, c)
            and _is_ayat_matched(parsed, c)
        ]

        should_realign = False
        if not parsed.pasal:
            should_realign = True
        elif matching_chunks:
            matched_c = matching_chunks[0]
            chunk_role = _detect_statutory_role(matched_c.text)
            if chunk_role in (
                "[Peran: Wewenang Korektif Otoritas / Sanksi]",
                "[Peran: Opsi Pengecualian / Izin Bersyarat]",
                "[Peran: Penjelasan / Keterangan Tambahan]",
            ):
                should_realign = True
            elif (
                is_different_clause
                and score_primary_obligation_alignment(primary_chunk, question)
                > score_primary_obligation_alignment(matched_c, question) + 0.5
            ):
                # Do not overwrite a valid sibling primary obligation citation if it specifically grounds its sentence
                if (
                    chunk_role == "[Peran: Kewajiban Pokok Entitas]"
                    and stmt
                    and (
                        _score_sentence_chunk_match(stmt, matched_c)
                        >= _score_sentence_chunk_match(stmt, primary_chunk)
                        or _score_sentence_chunk_match(stmt, matched_c) >= 4.0
                    )
                ):
                    should_realign = False
                else:
                    should_realign = True
        elif is_different_clause:
            should_realign = True

        if should_realign:
            clean_cite = cite.strip("[]")
            clean_new = primary_label.strip("[]")

            # Strip trailing remedial dump paragraphs before replacing inline text
            realigned_text = _strip_remedial_dump_paragraphs(
                realigned_text, inverted_pasal=parsed.pasal
            )

            # Replace bracketed citation
            realigned_text = re.sub(
                rf"\[\s*{re.escape(clean_cite)}\s*\]",
                f"[{clean_new}]",
                realigned_text,
            )
            # Replace inverted inline prose ("Pasal X ayat (Y) dari POJK ...")
            if parsed.pasal:
                inv_pattern = (
                    rf"\bPasal\s+{re.escape(parsed.pasal)}"
                    rf"(?:\s+ayat\s+\(\d+\))?(?:\s+angka\s+\d+)?"
                    rf"\s+(?:dari\s+|dalam\s+|pada\s+)?"
                    rf"(?:Undang-Undang|UU|POJK|PBI|PADG|SEOJK|PP|Perpu)\s+"
                    rf"(?:Nomor\s+)?[\w\.\/\-]+(?:\s+Tahun\s+\d{{4}})?"
                )
                realigned_text = re.sub(
                    inv_pattern,
                    f"[{clean_new}]",
                    realigned_text,
                    flags=re.IGNORECASE,
                )

            realigned_text = re.sub(
                rf"\b{re.escape(clean_cite)}\b",
                clean_new,
                realigned_text,
            )

            # Ensure opening BLUF sentence has a bracketed citation
            realigned_text = _attach_bracket_citation_to_opening(realigned_text, clean_new)

            replacements[cite] = f"[{clean_new}]"
            logger.warning(
                "Inverted legal reasoning detected for query %r: realigned non-primary citation %s -> %s",
                question[:50],
                cite,
                primary_label,
            )

    new_citations = extract_citations(realigned_text)
    return realigned_text, new_citations, replacements
