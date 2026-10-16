"""API middleware: Request-ID tracking and structured logging."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("easypasal.api.access")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns or propagates a unique Request-ID for every HTTP interaction."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        # Extract client-supplied request ID or generate a fresh UUID
        request_id = request.headers.get("X-Request-ID")
        if not request_id or not request_id.strip():
            request_id = str(uuid.uuid4())
        else:
            request_id = request_id.strip()

        # Attach to request state for downstream handlers
        request.state.request_id = request_id

        # Process the request
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Inject X-Request-ID into outgoing response headers
        response.headers["X-Request-ID"] = request_id

        # Determine client address and active variant
        client_host = request.client.host if request.client else "unknown"
        variant = (
            getattr(request.state, "variant", None)
            or response.headers.get("X-EasyPasal-Variant")
            or "none"
        )

        # Emit structured log
        logger.info(
            "%s %s %s -> %d (%.2fms) [req_id=%s variant=%s client=%s]",
            request.method,
            request.url.path,
            f"?{request.url.query}" if request.url.query else "",
            response.status_code,
            duration_ms,
            request_id,
            variant,
            client_host,
            extra={
                "request_id": request_id,
                "variant": variant,
                "client_ip": client_host,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response
