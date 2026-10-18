# syntax=docker/dockerfile:1
# Multi-stage lightweight Dockerfile for EasyPasal Log Collector microservice (Plan §9 / Task T7.2).
# Target memory footprint: <= 90 MB RSS RAM on a 2 GB VPS.

FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create isolated venv and install only collector runtime dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
        "fastapi>=0.115.0" \
        "uvicorn>=0.32.0" \
        "pydantic>=2.10.0" \
        "httpx>=0.27.0" \
        "pyyaml>=6.0.0" && \
    /opt/venv/bin/pip install --no-cache-dir --no-deps -e .

FROM python:3.12-slim AS runner

WORKDIR /app

# Security: non-root user and persistent SQLite directory
RUN groupadd -g 10001 collector && \
    useradd -u 10001 -g collector -s /bin/sh -m collector && \
    mkdir -p /data && \
    chown -R collector:collector /data /app

COPY --from=builder --chown=collector:collector /opt/venv /opt/venv
COPY --from=builder --chown=collector:collector /app /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EASYPASAL_COLLECTOR_HOST=0.0.0.0 \
    EASYPASAL_COLLECTOR_PORT=8080 \
    EASYPASAL_COLLECTOR_DB_PATH=/data/collector.db

USER collector

EXPOSE 8080

VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)" || exit 1

ENTRYPOINT ["python", "-m", "easypasal.collector.app"]
