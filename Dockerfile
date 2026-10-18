# syntax=docker/dockerfile:1
# Multi-stage production Dockerfile for EasyPasal application services (Plan §8 & §10 / Task T7.7).
# Powers both the FastAPI REST API service (port 8000) and the Gradio UI demo (port 7860).
# Packages prebuilt Qdrant dense vector store and BM25 index for zero-overhead cold starts.

FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install application and dependencies into isolated /opt/venv
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Security: non-root user with UID 1000
RUN useradd -m -u 1000 user && \
    mkdir -p /app/data/index /app/configs /app/logs /home/user/.cache && \
    chown -R user:user /app /home/user

COPY --from=builder --chown=user:user /opt/venv /opt/venv
COPY --from=builder --chown=user:user /app/src /app/src
COPY --chown=user:user configs/ /app/configs/
COPY --chown=user:user data/corpus_manifest.csv /app/data/corpus_manifest.csv
COPY --chown=user:user data/index/ /app/data/index/
COPY --chown=user:user pyproject.toml /app/pyproject.toml
COPY --chown=user:user README.md /app/README.md

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HOME=/home/user

USER user

EXPOSE 8000 7860

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:7860/ || exit 1

CMD ["python", "-m", "easypasal.api.app", "--host", "0.0.0.0", "--port", "8000"]
