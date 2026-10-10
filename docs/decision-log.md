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
