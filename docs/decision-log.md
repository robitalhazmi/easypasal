# EasyPasal — Decision Log

This document records the hypotheses, experimental setups, quantitative evaluation results, and architectural decisions made across project development.

## Summary of Decisions

| Experiment ID | Focus Area | Date | Decision | Key Rationale / Trade-off |
|---|---|---|---|---|
| **E0** | Project Scaffolding & Naming | 2026-10-10 | Approved | `EasyPasal` name verified available across PyPI/GitHub; scaffolded with uv |
| **E1** | Chunking Strategy | *TBD* | *TBD* | *TBD* |
| **E2** | Embedding Models | *TBD* | *TBD* | *TBD* |
| **E3** | Retrieval Pipeline | *TBD* | *TBD* | *TBD* |
| **E4** | Generator Model (CPU) | *TBD* | *TBD* | *TBD* |
| **E5** | Prompt Optimization (GEPA) | *TBD* | *TBD* | *TBD* |
| **E6** | vLLM Serving Benchmark | *TBD* | *TBD* | *TBD* |
| **E7** | QLoRA Fine-tuning | *TBD* | *TBD* | *TBD* |

---

## Detailed Decision Logs

### Decision: Project Naming ("EasyPasal") Verification (T0.2)
- **Date:** 2026-10-10
- **Status:** Approved (Retain name `EasyPasal`)
- **Investigation Findings:**
  - **PyPI:** Queried `pypi.org/project/easypasal` — package namespace is unregistered and available.
  - **GitHub:** Queried `github.com` — no conflicting software, AI, or compliance repositories under `easypasal` or `easy-pasal`.
  - **Web Search & Trademarks:** In Indonesian, *"Pasal"* means legal article/clause. The only web presence found is an unrelated local retail shop in Nepal (`easypasal.com`, where *"pasal"* means shop in Nepali), which has no domain overlap with an open-source Indonesian regulatory compliance assistant.
- **Decision:** Keep the name **`EasyPasal`** across repository, package namespace (`src/easypasal`), and documentation.

---

### Decision: Generator & Embedding Model Shortlist (T0.5)
- **Date:** 2026-10-10
- **Status:** Shortlist Approved for Experiments E2 & E4

#### 1. Embedding Model Candidates (Experiment E2)
| Model | Params | Dim | Context | License | Notes / Suitability |
|---|---|---|---|---|---|
| `BAAI/bge-m3` | ~568M | 1024 | 8,192 tokens | MIT | SOTA multilingual retrieval, multi-vector & dense support, long context for legal articles |
| `intfloat/multilingual-e5-base` | ~278M | 768 | 512 tokens | MIT | High CPU throughput, compact memory footprint, requires `query:`/`passage:` prefix |
| `intfloat/multilingual-e5-small` | ~118M | 384 | 512 tokens | MIT | Ultra-lightweight baseline for lowest-latency constraints |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | ~278M | 768 | 512 tokens | Apache-2.0 | Standard sentence-transformer comparison baseline |

#### 2. Generator Model Candidates for CPU / Live Demo (Experiment E4)
All local generators run quantized via 4-bit GGUF (`Q4_K_M`) using `llama.cpp` on the 16 GB laptop CPU and free CPU Hugging Face Space:

| Model | Params | Q4_K_M GGUF Size | RAM Footprint | License | Notes / Suitability |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~1.0 GB | ~1.8 GB | Apache-2.0 | High tokens/sec on CPU, strong Indonesian instruction following |
| `Qwen/Qwen2.5-3B-Instruct` | 3.1B | ~2.0 GB | ~3.5 GB | Qwen License | Excellent balance of reasoning depth and CPU latency |
| `Qwen/Qwen2.5-7B-Instruct` | 7.6B | ~4.7 GB | ~7.5 GB | Apache-2.0 | Highest quality open small model, tested on laptop (fits in 16 GB RAM) |
| `meta-llama/Llama-3.2-1B-Instruct` | 1.2B | ~0.8 GB | ~1.5 GB | Llama 3.2 Community | Ultra-fast inference, concise outputs |
| `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | ~2.0 GB | ~3.5 GB | Llama 3.2 Community | Strong reasoning and summarization capability |
| `google/gemma-2-2b-it` | 2.6B | ~1.6 GB | ~3.0 GB | Gemma License | Good multilingual performance, tested as lightweight baseline |
| `aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct` | 8.0B | ~4.9 GB | ~8.0 GB | Llama 3 Community | Southeast Asian / Indonesian domain-specialized comparison |

#### 3. LLM-as-Judge Candidates (Colab / Kaggle Free GPU - Phase 3 & 5)
- Primary Judge: `Qwen/Qwen2.5-7B-Instruct` / `Qwen2.5-14B-Instruct` (Apache-2.0)
- Secondary Calibration Judge: `meta-llama/Llama-3.1-8B-Instruct` (Llama 3.1 Community)
