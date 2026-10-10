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
