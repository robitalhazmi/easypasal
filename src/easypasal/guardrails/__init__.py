"""Guardrails module: injection detection, citation verification, and question routing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.guardrails.citation_verifier import (
        CitationCheck,
        ParsedCitation,
        VerificationResult,
        parse_citation_details,
        verify_citations,
        verify_single_citation,
    )

__all__ = [
    "CitationCheck",
    "ParsedCitation",
    "VerificationResult",
    "parse_citation_details",
    "verify_citations",
    "verify_single_citation",
]

_VERIFIER_SYMBOLS = {
    "CitationCheck",
    "ParsedCitation",
    "VerificationResult",
    "parse_citation_details",
    "verify_citations",
    "verify_single_citation",
}


def __getattr__(name: str) -> object:
    if name in _VERIFIER_SYMBOLS:
        import easypasal.guardrails.citation_verifier as _verifier

        return getattr(_verifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
