# EasyPasal

[![CI](https://github.com/robitalhazmi/easypasal/actions/workflows/ci.yml/badge.svg)](https://github.com/robitalhazmi/easypasal/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **Evaluated, cited answers on Indonesian financial regulation.**

---

> [!IMPORTANT]
> **Independence Notice:** EasyPasal is an independent portfolio project. It is not affiliated with, endorsed by, or representing the Financial Services Authority (OJK) or Bank Indonesia (BI), and it does not provide legal advice.

---

## Overview

EasyPasal is an evaluation-first, citation-grounded RAG assistant designed to answer Indonesian financial regulation questions (Bahasa Indonesia + English) from official public OJK and Bank Indonesia documents. Built like a production LLM system, it emphasizes:

- **Grounded or Silent:** Answers must cite exact regulation, article (*pasal*), and clause (*ayat*), or explicitly abstain.
- **Evaluation Gate in CI:** Every change is benchmarked against an author-labeled gold evaluation set; no variant is promoted without passing deterministic metric thresholds and paired statistical tests.
- **Efficient & Free Stack:** Runs locally on CPU (4-bit GGUF via `llama.cpp`), lightweight VPS monitoring (SQLite + Evidently), and free Colab/Kaggle GPU pipelines for vLLM benchmarks, prompt optimization (DSPy/GEPA), and QLoRA fine-tuning.

---

## Results & Evaluation (Centerpiece)

### Statutory Question-Answering Evaluation (Held-Out Test Set, $N = 66$)

| Pipeline Variant | Recall@5 | Citation F1 (Art) | Grounding Fidelity | Abstention Acc. | p95 Latency (CPU) | Status / Decision |
|---|---|---|---|---|---|---|
| **Baseline (v0.1 / M1)** | 0.3679 | 0.2677 | 0.5808 | 0.8030 | 96.4 s | Frozen Baseline (`configs/baseline_v0.1.yaml`) |
| **+ GEPA Prompt (E5)** | 0.3679 | 0.2601 | **0.6212** (+4.0%) | **0.8182** | 96.4 s | Prompts merged (`src/easypasal/generation/prompts.py`) |
| **+ vLLM Serving (E6)** | 0.3679 | **0.3939** (+47.1%) | **0.6212** | **0.8182** | **8.82 s** ($c=32$) | Production GPU Serving Target (T4 16GB) |
| **+ QLoRA (E7)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | Planned (Phase 6) |

*Note: All results evaluated on author-labeled gold Indonesian regulatory datasets with 95% bootstrap confidence intervals and paired permutation tests. No metrics are fabricated.*

### Inference & Serving Benchmarks: vLLM GPU vs llama.cpp CPU (Experiment E6)

Benchmarked on **NVIDIA Tesla T4 (16 GB GDDR6)** vs **Intel/AMD quad-core CPU** across concurrency levels $c \in \{1, 4, 16, 32\}$ using 1,500-token statutory contexts:

| Concurrency ($c$) | vLLM GPU Tok/s (3B) | vLLM p95 Latency | vLLM p95 TTFT | CPU Tok/s (1B) | CPU p95 Latency (1B) | Throughput Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$c = 1$** (Single Client) | 38.40 tok/s | 2.57 s | 49.0 ms | 2.87 tok/s | 91.90 s | **13.4x** |
| **$c = 4$** (Team Load) | 148.20 tok/s | 3.08 s | 80.2 ms | 2.92 tok/s | 365.20 s | **50.8x** |
| **$c = 16$** (Department Load) | 365.80 tok/s | 5.28 s | 171.7 ms | 2.65 tok/s | 1,450.40 s (24 min) | **138.0x** |
| **$c = 32$** (Stress Test) | **482.50 tok/s** | **8.82 s** | **319.0 ms** | 2.40 tok/s | 2,900.00 s (48 min) | **201.0x** |

![vLLM GPU vs llama.cpp CPU Serving Benchmark](docs/assets/e6_benchmark_comparison.png)

- **Reproducibility Notebook:** [`notebooks/e6_vllm_benchmark.ipynb`](notebooks/e6_vllm_benchmark.ipynb)
- **Serving Benchmark Harness:** `make bench` / `make exp-e6` (`experiments/e6_vllm/run_benchmark.py`)

---

## Quickstart

### Prerequisites
- Linux / macOS with Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- (Optional) Docker & docker-compose

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/robitalhazmi/easypasal.git
cd easypasal

# Set up virtual environment, dependencies, and pre-commit hooks
make setup

# Run linting and test suite
make lint
make test
```

### Running the REST API Service

Start the local FastAPI service on port 8000:
```bash
make serve
# Or specify a custom provider:
# uv run python -m easypasal.api.app --host 0.0.0.0 --port 8000 --provider mock
```

Verify service readiness:
```bash
curl -s http://localhost:8000/health
```

Ask a regulatory question with automatic citation verification and request tracking:
```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: req-example-001" \
  -d '{"question": "Berapa batas waktu penanganan pengaduan konsumen menurut POJK 22 Tahun 2023?"}'
```

Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.


### Key Makefile Targets

| Target | Description |
|---|---|
| `make setup` | Create virtual environment, sync dependencies with `uv`, install pre-commit |
| `make lint` | Run code quality checks (`ruff`, formatting, `mypy` typing) |
| `make test` | Run pytest suite with coverage |
| `make check` | Run all lint and test checks |
| `make ingest` | Fetch and parse corpus from manifest (Phase 1) |
| `make index` | Build dense (Qdrant) and sparse (BM25) indexes (Phase 1) |
| `make serve` | Run local FastAPI + Gradio interface (Phase 1) |
| `make eval` | Run evaluation harness on gold set (Phase 3) |
| `make bench` | Run local CPU latency benchmark |

---

## Repository Structure

```
easypasal/
├── README.md                    # Project pitch, results table, quickstart
├── LICENSE                      # Apache-2.0
├── pyproject.toml / uv.lock     # Tooling and dependency declarations
├── Makefile                     # Standard developer commands
├── docker-compose.yml           # Local multi-service demo deployment
├── .github/workflows/ci.yml     # Automated lint, test, and smoke eval gate
├── configs/                     # System and experiment variant configurations
├── data/
│   ├── corpus_manifest.csv      # In-force regulation metadata and sources
│   ├── README.md                # Data provenance and licensing guide
│   └── gold/                    # Labeled gold evaluation splits (train/dev/test)
├── src/easypasal/
│   ├── ingest/                  # PDF download, parsing, and structure-aware chunking
│   ├── index/                   # Embeddings, Qdrant vector store, and BM25 index
│   ├── retrieval/               # Hybrid retrieval and Reciprocal Rank Fusion
│   ├── generation/              # Citation-grounded prompts, LLM client, parsing
│   ├── guardrails/              # Scope, prompt injection checks, routing
│   ├── api/                     # FastAPI service and structured request logging
│   └── ui/                      # Gradio interface
├── eval/
│   ├── metrics/                 # Deterministic and LLM-judged metrics
│   ├── judge_calibration/       # Human-vs-judge calibration data
│   ├── reports/                 # Evaluation metric plots and tables
│   └── run_eval.py              # Offline evaluation runner
├── experiments/                 # Tracked experiments (E1 chunking ... E5 GEPA, E7 QLoRA)
├── benchmarks/vllm/             # vLLM serving benchmark scripts & results
├── monitoring/                  # Lightweight VPS log collector & drift reports
├── docs/                        # Architecture, decision log, stakeholder memo, runbook
└── tests/                       # Unit and integration test suite
```

---

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
