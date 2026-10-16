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
    from easypasal.guardrails.question_router import (
        QuestionRouter,
        RouteDecision,
        route_question,
    )

__all__ = [
    "CitationCheck",
    "ParsedCitation",
    "VerificationResult",
    "parse_citation_details",
    "verify_citations",
    "verify_single_citation",
    "QuestionRouter",
    "RouteDecision",
    "route_question",
]

_VERIFIER_SYMBOLS = {
    "CitationCheck",
    "ParsedCitation",
    "VerificationResult",
    "parse_citation_details",
    "verify_citations",
    "verify_single_citation",
}

_ROUTER_SYMBOLS = {
    "QuestionRouter",
    "RouteDecision",
    "route_question",
}


def __getattr__(name: str) -> object:
    if name in _VERIFIER_SYMBOLS:
        import easypasal.guardrails.citation_verifier as _verifier

        return getattr(_verifier, name)
    if name in _ROUTER_SYMBOLS:
        import easypasal.guardrails.question_router as _router

        return getattr(_router, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
