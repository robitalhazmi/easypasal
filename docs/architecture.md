# EasyPasal — Architecture & System Design

## Overview
EasyPasal implements a modular, citation-grounded RAG architecture with separate environments for offline preparation, live CPU demo serving, VPS-based monitoring, and GPU experiment execution.

## System Topology
*(Detailed topology and diagrams from implementation plan §3 will be maintained here as the system components are implemented.)*

## Key Hardware & Hosting Constraints
- **Local Laptop:** CPU-only, 16 GB RAM (ingestion, indexing, smoke tests).
- **Hugging Face Space (Demo):** Free CPU tier.
- **VPS (Monitoring & Collector):** 1 vCPU, 2 GB RAM (FastAPI + SQLite collector, Evidently drift reports).
- **Colab / Kaggle (GPU):** Free GPU environments for vLLM serving benchmarks, DSPy/GEPA optimization, and QLoRA fine-tuning.

## Architectural Patterns: Decision Models vs. Generative LLMs

To minimize p95 latency and compute costs in production RAG workflows:
- **Decision Models (Single-Pass Prefill):** Classification tasks (prompt injection detection, regulation scope verification, abstention detection, and NLI citation entailment checks) use decision models (such as **Kev** / LoRA pointer heads on causal LLM backbones). Because they perform a single forward pass without autoregressive token decoding, they return calibrated probabilities in $10\text{–}30\text{ ms}$.
- **Generative Models (Autoregressive):** Reserved exclusively for synthesizing final citation-grounded answers when the query passes guardrails and sufficient regulatory evidence is retrieved.
