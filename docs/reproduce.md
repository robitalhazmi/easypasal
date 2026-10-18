# EasyPasal — Complete Reproduction Guide

Welcome to the **EasyPasal** reproduction tutorial. This document provides step-by-step instructions to reproduce every baseline, evaluation score, experimental milestone (E1–E7), counterfactual simulation, and operational deployment drill reported in this repository.

> **Honest-Claims & Independence Notice (Plan §13)**
> EasyPasal is an **independent engineering and research project** developed by Robit Hazmi. It is **not** an official publication of, and has no formal affiliation with, Bank Indonesia (BI), the Financial Services Authority (OJK), or the Indonesia Stock Exchange (IDX).
> All reported A/B metrics reflect **offline counterfactual replay simulations** and paired evaluations against a 322-item gold benchmark. Production telemetry reflects **developer-conducted and synthetic query sessions** under realistic constraints, rather than live organic commercial traffic. All reported GPU serving throughputs were physically benchmarked on free Google Colab NVIDIA Tesla T4 GPU instances.

---

## Table of Contents
1. [System Requirements & Hardware Targets](#1-system-requirements--hardware-targets)
2. [Quickstart: Zero to Running in Under 2 Minutes](#2-quickstart-zero-to-running-in-under-2-minutes)
3. [Environment Setup with uv](#3-environment-setup-with-uv)
4. [Corpus Ingestion & Index Building](#4-corpus-ingestion--index-building)
5. [Gold Dataset & Judge Calibration Reproduction](#5-gold-dataset--judge-calibration-reproduction)
6. [CI Smoke Evaluation Gate](#6-ci-smoke-evaluation-gate)
7. [Reproducing Core Experiments (E1 to E4)](#7-reproducing-core-experiments-e1-to-e4)
8. [Reproducing Advanced Milestones (E5, E6, E7)](#8-reproducing-advanced-milestones-e5-e6-e7)
9. [Offline Counterfactual Replay A/B Simulation](#9-offline-counterfactual-replay-ab-simulation)
10. [Full Microservice Stack via Docker Compose](#10-full-microservice-stack-via-docker-compose)
11. [Operational Runbook & Telemetry Drills](#11-operational-runbook--telemetry-drills)
12. [Troubleshooting & FAQs](#12-troubleshooting--faqs)

---

## 1. System Requirements & Hardware Targets

| Environment | Minimum Specs | Recommended Specs | Role in EasyPasal |
|---|---|---|---|
| **Local CPU (Standard)** | 4 CPU cores, 8 GB RAM, 10 GB disk | 8 CPU cores, 16 GB RAM, 20 GB disk | Corpus parsing, embedding indexing, CI smoke tests, CPU generator serving (Llama-3.2-1B-Instruct), and Docker Compose. |
| **Google Colab / Kaggle (Free GPU)** | 1x NVIDIA Tesla T4 (15–16 GB VRAM), 12 GB RAM | 1x NVIDIA A10G / V100 GPU (Colab Pro optional) | E5 DSPy GEPA optimization, E6 vLLM serving benchmark, and E7 QLoRA multi-task router fine-tuning. |
| **Lightweight VPS** | 1 vCPU, 2 GB RAM, 10 GB disk | 2 vCPU, 4 GB RAM, 20 GB disk | Standalone Log Collector (FastAPI + SQLite WAL) and Monitoring Dashboard (<90 MB RSS footprint). |

### Software Prerequisites
- **Operating System:** Linux (Ubuntu 22.04+ recommended), macOS (Apple Silicon or Intel), or Windows with WSL2.
- **Python:** Python 3.11 or Python 3.12.
- **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (fast Python package and project manager).
- **Containerization (Optional):** Docker Engine 24.0+ and Docker Compose v2.20+.
- **Build Utilities:** GNU `make`, `git`, `curl`.

---

## 2. Quickstart: Zero to Running in Under 2 Minutes

If you want to spin up the complete EasyPasal system immediately without re-ingesting raw PDFs from government portals, use our automated prebuilt index path and Docker Compose:

```bash
# 1. Clone repository
git clone https://github.com/robitalhazmi/easypasal.git
cd easypasal

# 2. Build and launch all microservices (API, UI Demo, Log Collector)
make docker-up
```

Once launched:
- **Gradio Interactive UI:** [`http://localhost:7860`](http://localhost:7860)
- **FastAPI Documentation:** [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **Monitoring & Telemetry Dashboard:** [`http://localhost:8080/dashboard`](http://localhost:8080/dashboard)

To view real-time multi-service logs:
```bash
make docker-logs
```

To stop and remove containers cleanly:
```bash
make docker-down
```

---

## 3. Environment Setup with uv

For full developer workflows, testing, and benchmark execution, set up a local virtual environment:

### 3.1 Install `uv`
If `uv` is not already installed on your system:
```bash
# On Linux and macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip:
pip install uv
```

### 3.2 Initialize Repository & Pre-commit Hooks
Run the automated setup target:
```bash
make setup
```
This single command executes:
1. `uv sync --all-groups`: resolves and installs all runtime, development, evaluation, and test dependencies into `.venv`.
2. `uv run pre-commit install`: registers Git hooks for ruff formatting and code quality checks.

### 3.3 Verify Code Quality & Test Suite
Verify that your local environment passes all 384 tests and strict static analysis:
```bash
make check
```
*(Runs `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest tests/` with 86% coverage).*

---

## 4. Corpus Ingestion & Index Building

EasyPasal indexes 21 verified Indonesian financial statutory instruments spanning Banking, P2P Fintech, Capital Markets, Consumer Protection, and Data Privacy (`data/corpus_manifest.csv`).

### Option A: Fast Path — Download Prebuilt Indexes (< 1 minute)
If you wish to skip document scraping and embedding computation:
```bash
make download-index
```
This fetches pre-computed Qdrant vector embeddings and BM25 sparse indexes directly from the release bundle into `data/index/`.

### Option B: Full Path — Ingest & Index from Scratch (~5–8 minutes)
To reproduce the full raw ingestion and indexing pipeline:

```bash
# Step 1: Download raw statutory PDFs from official JDIH portals
make ingest

# Step 2: Parse raw PDFs into clean structural JSON documents
make parse

# Step 3: Segment documents into semantic Pasal/Ayat chunks (and fixed-size baselines)
make chunk

# Step 4: Compute Multilingual-E5-Large embeddings and build Qdrant + BM25 stores
make index
```

Verification: check that `data/index/qdrant/` contains the collection files and `data/index/bm25/` contains the serialized sparse index.

---

## 5. Gold Dataset & Judge Calibration Reproduction

EasyPasal's evaluation centerpiece is a 322-item gold benchmark (`data/gold/gold_dataset.jsonl`) divided into:
- `train.jsonl` ($N=216$): Used for prompt tuning and router fine-tuning.
- `dev.jsonl` ($N=40$): Used for intermediate experimental variant selection.
- `test.jsonl` ($N=66$): Strictly held-out split for final release verification.

### 5.1 Validate Gold Dataset Schema
Ensure all 322 items conform strictly to the required fields, citation schemas, and taxonomy categories:
```bash
make validate-gold
```

### 5.2 Reproduce Human Judge Calibration (PABAK $\kappa = 0.9429$)
EasyPasal uses an open-weight LLM judge calibrated against 35 human expert labels (`eval/judge_calibration/calibration_data_35.json`). To recompute judge agreement statistics:
```bash
make calibrate
```
Expected Output:
```text
Evaluated pairs: 35
Overall Judge Agreement: 97.1% (34 / 35 items)
Cohen's Kappa (PABAK) : 0.9429 (Near-perfect agreement)
Mean Absolute Error    : 0.0149
```

### 5.3 Verify Human Annotation Self-Agreement
To inspect the self-consistency of the human gold labels across a 20% spot-check sample:
```bash
make agreement
```
Expected Output: `Self-Agreement: 100.0%` with zero intra-annotator contradictions.

---

## 6. CI Smoke Evaluation Gate

The CI smoke evaluation is an automated regression gate executed on every commit. It evaluates 30 fixed gold items (25 answerable, 5 unanswerable) on deterministic retrieval and safety metrics without requiring a GPU.

Run the CI smoke gate locally:
```bash
make smoke-eval
```

Expected Output:
```text
================================================================================================
                    🚦 EasyPasal CI Smoke Evaluation & Regression Gate (T3.6)
================================================================================================
Evaluated Items: 30 total (25 answerable, 5 unanswerable)
Gate Verdict   : ✅ PASSED (All thresholds satisfied)
------------------------------------------------------------------------------------------------
Gate Check                   | Observed     | Target/Floor   | Status     | Detail
------------------------------------------------------------------------------------------------
Recall@5 Regression Gate     | 0.2600       | 0.2100         | ✅ PASS     | Candidate 0.2600 vs threshold 0.2100
Recall@5 Absolute Floor      | 0.2600       | 0.2000         | ✅ PASS     | Candidate 0.2600 vs floor 0.2000
MRR Regression Gate          | 0.1533       | 0.1033         | ✅ PASS     | Candidate 0.1533 vs threshold 0.1033
MRR Absolute Floor           | 0.1533       | 0.1000         | ✅ PASS     | Candidate 0.1533 vs floor 0.1000
Abstention Accuracy Gate     | 1.0000       | 0.9000         | ✅ PASS     | Observed 1.0000 vs floor 0.9000
p95 Latency Ceiling Gate     | ~828 ms      | 1500.0000      | ✅ PASS     | Well below ceiling
================================================================================================
```

---

## 7. Reproducing Core Experiments (E1 to E4)

All core architecture decisions are documented in [`docs/decision-log.md`](decision-log.md). You can reproduce the exact quantitative results across `dev` ($N=40$) and `test` ($N=66$) splits:

### 7.1 Experiment E1: Chunking Strategy Comparison
Compares Fixed-Size (512 tokens), Structure-Aware (Pasal/Ayat), and Structure-Aware + Context Hierarchy:
```bash
make exp-e1
```
- **Winner:** Structure-Aware + Context Hierarchy (Recall@5 = 0.5050 on `dev`, +12.5% gain over fixed-size).

### 7.2 Experiment E2: Multilingual Embedding Models
Compares `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, and `intfloat/multilingual-e5-large`:
```bash
make exp-e2
```
- **Winner:** `intfloat/multilingual-e5-large` (Recall@5 = 0.5250 on `dev`, +10.0% gain over MiniLM on Indonesian legal phrasing).

### 7.3 Experiment E3: Retrieval Pipeline Architecture
Compares Pure Dense, Pure BM25, Hybrid RRF ($k=60$), and Hybrid + BAAI Cross-Encoder Reranker:
```bash
make exp-e3
```
- **Winner:** Pure Dense (Recall@5 = 0.5250, p95 latency = 82 ms; Hybrid RRF tied quality but added +65 ms, and Reranker added +680 ms without statutory gain).

### 7.4 Experiment E4: Generator Selection on CPU
Compares `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, and `Qwen/Qwen2.5-3B-Instruct` under 4-bit quantization:
```bash
make exp-e4
```
- **Winner:** `Llama-3.2-1B-Instruct` for CPU baseline (100% Faithfulness, 14.8 tok/s, 1.2 GB RAM); `Qwen2.5-3B-Instruct` selected for GPU serving.

---

## 8. Reproducing Advanced Milestones (E5, E6, E7)

### 8.1 Experiment E5: DSPy + GEPA Prompt Optimization
E5 optimized the system prompt using DSPy and Genetic Pareto Prompt Optimization (GEPA) against a multi-objective reward (Faithfulness, Grounding Fidelity, Abstention Precision).

1. **Local Pipeline & Evaluation Harness:**
   ```bash
   make exp-e5
   ```
2. **Google Colab Free GPU Notebook:**
   Open [`notebooks/e5_gepa_prompt_optimization.ipynb`](../notebooks/e5_gepa_prompt_optimization.ipynb) in Colab. Run all cells to observe the Pareto frontier evolution across 10 generations.
   - **Result:** Grounding Fidelity increased from 0.6897 to 0.7170 (+4.0%) on the held-out test split with zero prompt overfitting.

### 8.2 Experiment E6: vLLM GPU Serving Benchmark
E6 benchmarked high-throughput serving of `Qwen/Qwen2.5-3B-Instruct` on an NVIDIA Tesla T4 GPU (16 GB GDDR6) under concurrent workloads ($c \in \{1, 4, 16, 32\}$).

1. **Local Async Harness & Schema Verification:**
   ```bash
   make exp-e6
   # or
   make bench
   ```
2. **Colab Tesla T4 GPU Benchmark:**
   Open [`notebooks/e6_vllm_benchmark.ipynb`](../notebooks/e6_vllm_benchmark.ipynb) on a free Colab GPU runtime and execute the benchmark.
   - **Observed Metrics on NVIDIA T4:**
     - $c=1$: 31.75 tok/s, TTFT 207.8 ms, p95 6.30s.
     - $c=32$: **187.79 tok/s**, p95 11.38s (**78.2x speedup** over CPU 1B, **391.2x speedup** over CPU 3B).

### 8.3 Experiment E7: QLoRA Multi-Task Router Fine-Tuning
E7 trained a parameter-efficient adapter (PEFT QLoRA 4-bit, rank $r=16$) on `Qwen/Qwen2.5-1.5B-Instruct` to act as an upfront question router (9 categories: 5 legal themes, out-of-scope, legal advice, prompt injection, conversational).

1. **Local Dataset Synthesis, Training Harness, & Evaluation:**
   ```bash
   make exp-e7
   ```
2. **Colab GPU Training Notebook:**
   Open [`notebooks/e7_qlora_training.ipynb`](../notebooks/e7_qlora_training.ipynb) to inspect the training loop, loss curves, and Hugging Face PEFT adapter card.
   - **Observed Metrics:** **1.0000 Theme Macro-F1**, 100% prompt injection precision, 21.56 ms inference latency.

---

## 9. Offline Counterfactual Replay A/B Simulation

To rigorously evaluate the baseline system (`baseline_v0.1`) against the GEPA-optimized system (`gepa_v0.2`) without risking live compliance errors, we replayed gold questions through both pipelines under identical retrieval conditions.

Execute the simulation on the development split:
```bash
make replay-ab
```

Or execute on the held-out test set ($N=66$):
```bash
uv run python -m eval.replay_ab_simulation --dataset data/gold/test.jsonl --output results/t6_1_replay_ab_simulation_test.json
```

### Statistical Comparison with Bootstrap Confidence Intervals
To compute paired bootstrap confidence intervals ($B=2{,}000$ iterations) and permutation significance tests ($P=5{,}000$ permutations):
```bash
make compare
```

Key Findings on Test Set ($N=66$):
- **Faithfulness:** 1.0000 vs 1.0000 (100% adherence, zero hallucinations).
- **Grounding Fidelity:** 0.7170 vs 0.6897 (**+4.0% relative gain**, $p < 0.05$).
- **Abstention Accuracy:** 0.8182 vs 0.8182 (exact parity on unanswerable/adversarial queries).

---

## 10. Full Microservice Stack via Docker Compose

EasyPasal provides an operational production-grade deployment orchestrated by Docker Compose:
- **`rag-api`** (FastAPI service, port `8000`)
- **`gradio-ui`** (Gradio demo interface, port `7860`)
- **`log-collector`** (Lightweight SQLite WAL logging server, port `8080`)

### 10.1 Build & Start Containers
```bash
# Build multi-stage Docker images
make docker-build

# Start stack in detached mode
make docker-up
```

### 10.2 Test Container Health & Endpoints
Verify all three services are healthy:
```bash
# Check Docker container statuses
docker compose ps

# Check API health
curl -s http://localhost:8000/health | jq .

# Check Log Collector health
curl -s http://localhost:8080/health | jq .
```

### 10.3 Send a Test Query via cURL
Test the REST API directly:
```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Berapa modal disetor minimum untuk mendirikan bank umum menurut POJK?",
    "variant": "B",
    "language": "id"
  }' | jq .
```

---

## 11. Operational Runbook & Telemetry Drills

### 11.1 Rehearse Zero-Downtime Rollback Drill (Task T7.8)
Test the hot-swap configuration rollback mechanism under active query load:
```bash
make rehearse-rollback
```
What this drill executes:
1. Baselines active system under `gepa_v0.2`.
2. Simulates an operational incident (abrupt latency spike or bad prompt).
3. Hot-swaps configuration to `baseline_v0.1` via atomic symlink update.
4. Verifies zero service restart, zero downtime, and instant recovery.
- **Observed Result:** Hot-swap completed in **50.44 ms** with 0 dropped requests. Audit log saved to `reports/rollback_rehearsal.json`.

### 11.2 Run Real Usage Query Session (Task T7.9)
Populate production telemetry logs with 35 realistic statutory questions across 8 domains:
```bash
make real-usage
```
This script:
- Sends 35 realistic Indonesian legal queries through `RAGPipeline` (20 to Variant B, 15 to Variant A).
- Performs automated regex PII redaction (NIK, phone numbers, email addresses).
- Simulates realistic user satisfaction feedback (+1 and -1 ratings).
- Writes structured telemetry to `logs/requests.jsonl`, `logs/feedback.jsonl`, and SQLite `collector.db`.
- Updates the telemetry summary in `reports/real_usage_session.json`.

### 11.3 Generate Telemetry Dashboard Infographic
To render and inspect the monitoring dashboard without opening a browser:
```bash
uv run python scripts/generate_dashboard_preview.py
```
Outputs an updated high-resolution screenshot to [`docs/images/dashboard_monitoring.png`](images/dashboard_monitoring.png).

---

## 12. Troubleshooting & FAQs

### Q1: `portalocker.exceptions.AlreadyLocked: Resource temporarily unavailable`
- **Cause:** Qdrant's embedded storage locks `data/index/qdrant/.lock` exclusively per process. If an API process, Gradio UI, or previous script crashed without closing its connection, the lock remains held.
- **Fix:**
  ```bash
  # Ensure no orphan Python processes are running
  pkill -f "easypasal"
  # Safely clear the lock file if no process is active
  rm -f data/index/qdrant/.lock
  ```

### Q2: Hugging Face model download timeouts
- **Cause:** Slow or unstable internet connection during the initial download of `intfloat/multilingual-e5-large` (1.3 GB) or generator models.
- **Fix:** Set the Hugging Face timeout environment variable before running:
  ```bash
  export HF_HUB_ENABLE_HF_TRANSFER=0
  export REQUESTS_TIMEOUT=120
  ```

### Q3: Low-memory environments (< 8 GB RAM)
- **Recommendation:**
  1. Use the prebuilt indexes: `make download-index`.
  2. For the generation model, the CPU default `Llama-3.2-1B-Instruct` consumes only ~1.2 GB RAM under 4-bit quantization. If memory remains constrained, run the pure retrieval pipeline or use the lightweight mock generator:
     ```bash
     export EASYPASAL_MOCK_GENERATOR=1
     ```

### Q4: Colab GPU out-of-memory during vLLM benchmark
- **Fix:** In `notebooks/e6_vllm_benchmark.ipynb`, ensure `gpu_memory_utilization=0.85` and `max_model_len=2048` are set so the KV-cache allocation fits comfortably inside the 16 GB VRAM budget of the Tesla T4.

---

### Need Help or Found an Issue?
Please consult the [Operational Runbook](runbook.md) or open an issue on the [GitHub Repository](https://github.com/robitalhazmi/easypasal/issues).
