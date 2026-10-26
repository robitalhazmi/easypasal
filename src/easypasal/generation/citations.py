"""Citation extraction and abstention detection utilities for regulatory answers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.retrieval.models import RetrievedChunk

# Regex matching statutory mentions in Indonesian legal format
_STATUTORY_REGEX = re.compile(
    r"\b(?:Undang-Undang|UU|POJK|PBI|PADG|SEOJK|PP|Perpu)\s+"
    r"(?:Nomor\s+)?[\w\.\/\-]+(?:\s+Tahun\s+\d{4})?"
    r"(?:,?\s+Pasal\s+\d+[a-z]?(?:\s+ayat\s+\(\d+\))?(?:\s+angka\s+\d+)?)?",
    re.IGNORECASE,
)

# Regex matching bracket citations, e.g. [POJK 22 Tahun 2023, Pasal 71 ayat (1)]
_BRACKET_CITATION_REGEX = re.compile(
    r"\[\s*((?:(?:Undang-Undang|UU|POJK|PBI|PADG|SEOJK|PP|Perpu|Penjelasan)\s+[\w\.\/\-]+|Pasal\s+\d+)[^\]\n]{0,100})\s*\]",
    re.IGNORECASE,
)

_ABSTAIN_PHRASES_ID = [
    "tidak ditemukan dalam regulasi",
    "tidak ditemukan dasar hukum",
    "tidak terdapat dasar hukum",
    "tidak terdapat informasi",
    "tidak terdapat dalam regulasi",
    "tidak ada informasi mengenai",
    "tidak ada dasar hukum",
    "tidak menyebutkan informasi",
    "tidak diatur dalam regulasi",
    "di luar cakupan regulasi",
    "tidak memuat ketentuan mengenai",
    "tidak memuat informasi",
    "belum dikemukakan dalam dokumen",
    "tidak dikemukakan dalam dokumen",
    "tidak disebutkan dalam dokumen",
    "tidak disebutkan dalam regulasi",
    "belum diatur dalam dokumen",
    "belum diatur dalam regulasi",
    "tidak tercantum dalam dokumen",
    "tidak tercantum dalam regulasi",
    "dokumen yang ada tidak memuat",
    "regulasi yang ada tidak memuat",
    "tidak dijelaskan dalam dokumen",
    "tidak dijelaskan dalam regulasi",
    "tidak ada ketentuan mengenai",
    "belum ada ketentuan mengenai",
    "tidak diatur dalam korpus",
    "tidak ditemukan dalam korpus",
    "tidak diatur dalam peraturan",
    "tidak ditemukan dalam peraturan",
]

_ABSTAIN_PHRASES_EN = [
    "information not found in",
    "no legal basis was found",
    "not found in the available regulations",
    "not found in the provided regulations",
    "no information found",
    "outside the scope of",
    "does not contain information",
    "not regulated in the regulations",
    "not stated in the provided documents",
    "not stated in the documents",
    "not mentioned in the documents",
    "not specified in the available documents",
    "not found in the documents",
    "has not been provided in the documents",
    "no further details are provided",
    "not mentioned in the regulations",
    "is not specified in the documents",
    "not contained in the documents",
]


def extract_citations(text: str) -> list[str]:
    """Extract structured regulatory citation labels from generated answer text.

    Scans for explicit bracket citations (e.g. `[POJK 22 Tahun 2023, Pasal 71]`)
    and inline statutory designations (e.g. `UU Nomor 4 Tahun 2023, Pasal 243`).
    """
    citations: list[str] = []
    seen: set[str] = set()

    # 1. Bracket Citations
    for match in _BRACKET_CITATION_REGEX.finditer(text):
        raw = match.group(1).strip()
        # Clean leading/trailing artifacts
        clean = re.sub(r"^(?:DOKUMEN\s+\d+:\s*)?", "", raw).strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            citations.append(clean)

    # 2. Inline Statutory Pattern Matches
    for match in _STATUTORY_REGEX.finditer(text):
        raw = match.group(0).strip()
        r_low = raw.lower()
        if raw and r_low not in seen and not any(r_low in c.lower() for c in citations):
            seen.add(r_low)
            citations.append(raw)

    return citations


def check_abstention(text: str, language: str = "id") -> bool:
    """Determine whether the response text constitutes an explicit abstention refusal.

    Returns True if the text indicates that the required information or legal basis
    is not found within the available corpus passages.
    """
    clean_lower = text.strip().lower()

    # If the response cites official regulations, it is overwhelmingly likely a substantive answer.
    # Only flag as abstention if the opening sentence / primary thesis is an explicit refusal.
    citations = extract_citations(text)
    if citations:
        opening_segment = re.split(r"(?<=[.!?\n])\s+", clean_lower)[0]
        canonical_refusals = [
            "tidak ditemukan dasar hukum",
            "tidak terdapat dasar hukum",
            "tidak ada dasar hukum",
            "tidak ditemukan dalam regulasi",
            "tidak diatur dalam regulasi",
            "tidak terdapat informasi",
            "no legal basis was found",
            "not found in the available regulations",
            "not found in the provided regulations",
        ]
        return any(phrase in opening_segment for phrase in canonical_refusals)

    if language == "en":
        return any(phrase in clean_lower for phrase in _ABSTAIN_PHRASES_EN)

    return any(phrase in clean_lower for phrase in _ABSTAIN_PHRASES_ID)


def verify_citation_coverage(
    citations: list[str],
    retrieved_chunks: list[RetrievedChunk],
) -> tuple[list[str], list[str]]:
    """Compare extracted citations against retrieved passage metadata.

    A citation is grounded if its core statutory identifier (e.g., regulation type,
    number, and year) corresponds to at least one retrieved chunk.
    """
    grounded: list[str] = []
    ungrounded: list[str] = []

    known_targets: list[str] = []
    for c in retrieved_chunks:
        doc_clean = re.sub(r"[,_]", " ", c.doc_number.lower()).strip()
        known_targets.append(doc_clean)
        if c.citation_label:
            known_targets.append(re.sub(r"[,_]", " ", c.citation_label.lower()).strip())

    for cite in citations:
        cite_clean = re.sub(r"[,_]", " ", cite.lower()).strip()
        # Isolate statutory document prefix (strip "pasal ...")
        cite_doc = re.split(r"\bpasal\b", cite_clean)[0].strip()

        matched = False
        for target in known_targets:
            target_doc = re.split(r"\bpasal\b", target)[0].strip()
            if cite_doc and (cite_doc in target or target_doc in cite_clean):
                matched = True
                break

        if matched:
            grounded.append(cite)
        else:
            ungrounded.append(cite)

    return grounded, ungrounded
