"""API module: FastAPI routing, request/response schemas, and structured logging."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.api.app import create_app
    from easypasal.api.middleware import RequestIDMiddleware
    from easypasal.api.schemas import (
        AskRequest,
        AskResponse,
        CitationCheckResponse,
        CitationVerificationResponse,
        ErrorResponse,
        FeedbackPayload,
        FeedbackResponse,
        HealthResponse,
        RetrievedChunkResponse,
    )
    from easypasal.api.traffic import (
        VARIANT_BASELINE_V0_1,
        VARIANT_GEPA_V0_2,
        compute_deterministic_variant,
        map_variant_to_prompt,
        normalize_variant,
        resolve_variant,
    )

__all__ = [
    "AskRequest",
    "AskResponse",
    "CitationCheckResponse",
    "CitationVerificationResponse",
    "ErrorResponse",
    "FeedbackPayload",
    "FeedbackResponse",
    "HealthResponse",
    "RequestIDMiddleware",
    "RetrievedChunkResponse",
    "VARIANT_BASELINE_V0_1",
    "VARIANT_GEPA_V0_2",
    "compute_deterministic_variant",
    "create_app",
    "map_variant_to_prompt",
    "normalize_variant",
    "resolve_variant",
]

_APP_SYMBOLS = {"create_app"}
_MIDDLEWARE_SYMBOLS = {"RequestIDMiddleware"}
_SCHEMA_SYMBOLS = {
    "AskRequest",
    "AskResponse",
    "CitationCheckResponse",
    "CitationVerificationResponse",
    "ErrorResponse",
    "FeedbackPayload",
    "FeedbackResponse",
    "HealthResponse",
    "RetrievedChunkResponse",
}
_TRAFFIC_SYMBOLS = {
    "VARIANT_BASELINE_V0_1",
    "VARIANT_GEPA_V0_2",
    "compute_deterministic_variant",
    "map_variant_to_prompt",
    "normalize_variant",
    "resolve_variant",
}


def __getattr__(name: str) -> object:
    if name in _APP_SYMBOLS:
        import easypasal.api.app as _app

        return getattr(_app, name)
    if name in _MIDDLEWARE_SYMBOLS:
        import easypasal.api.middleware as _mw

        return getattr(_mw, name)
    if name in _SCHEMA_SYMBOLS:
        import easypasal.api.schemas as _schemas

        return getattr(_schemas, name)
    if name in _TRAFFIC_SYMBOLS:
        import easypasal.api.traffic as _traffic

        return getattr(_traffic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
