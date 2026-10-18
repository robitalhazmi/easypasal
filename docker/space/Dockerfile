# syntax=docker/dockerfile:1
# Production Dockerfile for EasyPasal Hugging Face Space (Plan §8.3 / Task T7.6).
# Configured for Hugging Face Spaces free CPU Basic Tier (2 vCPUs, 16 GB RAM) on port 7860.
# Packages prebuilt Qdrant vector store and BM25 index directly into the container image.

FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install application and runtime dependencies into /opt/venv
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces expects default user with UID 1000
RUN useradd -m -u 1000 user && \
    mkdir -p /app/data/index /app/configs /home/user/.cache && \
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
    PORT=7860 \
    HOME=/home/user

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

ENTRYPOINT ["python", "-m", "easypasal.ui.app", "--host", "0.0.0.0", "--port", "7860"]
