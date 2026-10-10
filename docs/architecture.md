# EasyPasal — Architecture & System Design

## Overview
EasyPasal implements a modular, citation-grounded RAG architecture with separate environments for offline preparation, live CPU demo serving, VPS-based monitoring, and GPU experiment execution.

## System Topology
*(Detailed topology and diagrams from implementation plan §3 will be maintained here as the system components are implemented.)*

## Key Hardware & Hosting Constraints (Verified in T0.6)

### 1. Live Demo: Hugging Face Space (Free CPU Basic Tier)
- **Compute & Memory:** 2 vCPUs, 16 GB RAM.
- **Storage:** 50 GB non-persistent / ephemeral disk space.
- **Sleep & Inactivity Behavior:** Automatically sleeps after **48 hours** of inactivity. Cold start latency upon wake-up is $\approx 10\text{–}30\text{ seconds}$.
- **Architectural Fit:**
  - Quantized 4-bit GGUF model ($\approx 1.0\text{–}2.0\text{ GB}$) + embedding model ($\approx 1.1\text{ GB}$) + embedded Qdrant vector store easily fit in 16 GB RAM with $\approx 12\text{ GB}$ headroom.
  - Ephemeral disk constraint requires packaging the pre-built index inside the Docker image or pulling on container startup.
  - Telemetry and query logs are dispatched asynchronously over HTTP to the VPS log sink to preserve statelessness.

### 2. Experiment Execution: Free GPU Environments (Colab & Kaggle)
- **Google Colab (Free Tier):**
  - Dynamic allocation (NVIDIA T4 with 15–16 GB VRAM).
  - Max session length of 12 hours; idle timeout after $\approx 90\text{ minutes}$.
- **Kaggle (Free Tier):**
  - Predictable quota of **30 GPU hours per week** (resets weekly, NVIDIA T4 / 2x T4).
  - Max interactive session limit of 12 hours.
- **Experiment Protocol:** All GPU workloads (E5 GEPA, E6 vLLM benchmark, E7 QLoRA, and Phase 3 judged evaluations) are packaged as reproducible, standalone notebooks/scripts with pinned random seeds and automatic artifact export (saving metrics JSON and LoRA adapters directly to the repo).

### 3. Monitoring & Operations: VPS Suitability & Memory Budget (1 vCPU / 2 GB RAM)
To respect the strict 2048 MB memory ceiling of the personal VPS:

| Component | Stack | Estimated Peak RAM |
|---|---|---|
| **Log Collector Service** | FastAPI + Uvicorn + SQLite (with auth token) | $60\text{–}90\text{ MB}$ |
| **Monitoring & Drift Analysis** | Streamlit/Gradio tab + Evidently batch report | $150\text{–}300\text{ MB}$ |
| **Documentation Server** | Caddy / Nginx serving MkDocs Material (HTTPS) | $20\text{–}30\text{ MB}$ |
| **Host System & Docker Daemon** | Ubuntu 24.04 LTS + Linux kernel + systemd | $350\text{–}500\text{ MB}$ |
| **Total Memory Footprint** | | **$\approx 600\text{–}920\text{ MB}$ ($<45\%$ of VPS limit)** |

*Architectural Justification:* Heavy enterprise stacks (Langfuse, Grafana, Prometheus, PostgreSQL) require 2.5–4.5 GB of RAM and would trigger out-of-memory (OOM) fatal crashes on a 2 GB VPS. The custom FastAPI + SQLite collector and periodic Evidently report pipeline provides transparent, production-grade observability within a $<1\text{ GB}$ memory envelope. No LLM or embedding inference runs on the VPS.


## Architectural Patterns: Decision Models vs. Generative LLMs

To minimize p95 latency and compute costs in production RAG workflows:
- **Decision Models (Single-Pass Prefill):** Classification tasks (prompt injection detection, regulation scope verification, abstention detection, and NLI citation entailment checks) use decision models (such as **Kev** / LoRA pointer heads on causal LLM backbones). Because they perform a single forward pass without autoregressive token decoding, they return calibrated probabilities in $10\text{–}30\text{ ms}$.
- **Generative Models (Autoregressive):** Reserved exclusively for synthesizing final citation-grounded answers when the query passes guardrails and sufficient regulatory evidence is retrieved.
