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
        HealthResponse,
        RetrievedChunkResponse,
    )

__all__ = [
    "AskRequest",
    "AskResponse",
    "CitationCheckResponse",
    "CitationVerificationResponse",
    "ErrorResponse",
    "HealthResponse",
    "RequestIDMiddleware",
    "RetrievedChunkResponse",
    "create_app",
]

_APP_SYMBOLS = {"create_app"}
_MIDDLEWARE_SYMBOLS = {"RequestIDMiddleware"}
_SCHEMA_SYMBOLS = {
    "AskRequest",
    "AskResponse",
    "CitationCheckResponse",
    "CitationVerificationResponse",
    "ErrorResponse",
    "HealthResponse",
    "RetrievedChunkResponse",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
