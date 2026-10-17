"""Authentication dependency for VPS log collector endpoints (Plan §9 / Task T7.2).

Enforces Bearer token or X-Collector-Token header authentication using
constant-time comparison (hmac.compare_digest) to prevent timing attacks.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from easypasal.collector.config import CollectorConfig


def get_collector_config(request: Request) -> CollectorConfig:
    """Retrieve collector configuration from application state."""
    config: CollectorConfig | None = getattr(request.app.state, "collector_config", None)
    if config is None:
        config = CollectorConfig()
    return config


def verify_collector_token(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_collector_token: Annotated[str | None, Header(alias="X-Collector-Token")] = None,
    config: Annotated[CollectorConfig, Depends(get_collector_config)] = None,  # type: ignore[assignment]
) -> str:
    """Validate Bearer token or X-Collector-Token header against configured secret.

    Raises:
        HTTPException: 401 Unauthorized if token is missing or invalid.
    """
    effective_config = config if config is not None else CollectorConfig()
    provided_token: str | None = None

    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided_token = parts[1]
        elif len(parts) == 1:
            provided_token = parts[0]

    if not provided_token and x_collector_token:
        provided_token = x_collector_token.strip()

    if not provided_token or not hmac.compare_digest(provided_token, effective_config.auth_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing collector authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided_token
