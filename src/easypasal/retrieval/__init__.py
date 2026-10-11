"""Retrieval module: hybrid dense/sparse search, RRF fusion, and candidate ranking."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.retrieval.fusion import (
        DEFAULT_K_RRF,
        reciprocal_rank_fusion,
    )
    from easypasal.retrieval.models import (
        RetrievalResult,
        RetrievedChunk,
    )
    from easypasal.retrieval.retriever import HybridRetriever

__all__ = [
    "DEFAULT_K_RRF",
    "HybridRetriever",
    "RetrievalResult",
    "RetrievedChunk",
    "reciprocal_rank_fusion",
]

_FUSION_SYMBOLS = {
    "DEFAULT_K_RRF",
    "reciprocal_rank_fusion",
}

_MODEL_SYMBOLS = {
    "RetrievalResult",
    "RetrievedChunk",
}

_RETRIEVER_SYMBOLS = {
    "HybridRetriever",
}


def __getattr__(name: str) -> object:
    if name in _FUSION_SYMBOLS:
        import easypasal.retrieval.fusion as _fusion

        return getattr(_fusion, name)
    if name in _MODEL_SYMBOLS:
        import easypasal.retrieval.models as _models

        return getattr(_models, name)
    if name in _RETRIEVER_SYMBOLS:
        import easypasal.retrieval.retriever as _retriever

        return getattr(_retriever, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
