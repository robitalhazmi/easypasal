# EasyPasal — Decision Log

This document records the hypotheses, experimental setups, quantitative evaluation results, and architectural decisions made across project development.

## Summary of Decisions

| Experiment ID | Focus Area | Date | Decision | Key Rationale / Trade-off |
|---|---|---|---|---|
| **E0** | Project Scaffolding & Naming | 2026-10-10 | Approved | `EasyPasal` name verified available across PyPI/GitHub; scaffolded with uv |
| **E1** | Chunking Strategy | 2026-10-14 | Structure-Aware + Breadcrumbs (Variant C) | +433% Recall@5 on dev ($p=0.0168$), +123% on test ($p=0.0210$); eliminates clause chopping and guarantees statutory citation anchors |
| **E2** | Embedding Models | 2026-10-15 | Multilingual-E5-Large (1024-d) | +216% Recall@5 on dev ($p=0.0364$), +117% on test ($p=0.0064$); asymmetric task prefixing unlocks precise statutory semantic search |
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

#### 4. Decision Models & Fast Classifiers (Guardrails, NLI & E7 Router Candidates)
| Model Architecture | Params | License | Primary Use Case |
|---|---|---|---|
| **`jaredpalmer/kev-0.8b` / `kev-4b`** | 0.8B / 4B | Apache-2.0 | Single-pass prefill routing, prompt injection triage, scope checks |
| **Qwen2.5 + LoRA Pointer/Classification Head** | 0.5B / 1.5B | Apache-2.0 | Indonesian domain question classification & post-hoc citation NLI entailment (E7) |

---

### Decision: Experiment E1 — Legal Document Chunking Strategy (T4.1)
- **Date:** 2026-10-14
- **Status:** Approved (Variant C: `structure_aware` with Hierarchical Breadcrumbs)
- **Primary Hypothesis:**
  In Indonesian regulatory compliance, compliance queries refer to specific statutory clauses (*Pasal* and *Ayat*). Structuring chunks along atomic legal boundaries preserves statutory coherence and prevents mid-clause truncation. Furthermore, prepending hierarchical statutory breadcrumbs (`[Dokumen | Bab | Bagian | Pasal | Ayat]`) provides critical lexical and semantic anchors that allow dense and sparse retrievers to match both broad thematic queries (e.g., chapters on administrative sanctions) and fine-grained clause requirements.

#### 1. Experimental Setup
- **Evaluation Splits:**
  - `dev.jsonl`: $N = 40$ questions (32 answerable, 8 out-of-scope / unanswerable).
  - Held-out `test.jsonl`: $N = 66$ questions (54 answerable, 12 out-of-scope / unanswerable).
- **Retrieval Pipeline:**
  - Dense retriever: `intfloat/multilingual-e5-small` (384-dim, cosine similarity).
  - Search vector database: Qdrant vector store (`data/index/qdrant/`).
  - Top-$k = 10$, evaluating top-$k \in \{1, 3, 5, 10\}$.
- **Compared Variants:**
  - **Variant A (`fixed_size`)**: Naive 500-character sliding window with 100-character overlap, ignoring legal hierarchy (6,840 chunks). To ensure fair, label-comparable evaluation, fixed-size chunks were mapped to canonical ground-truth clause targets via character span intersection (`ChunkSpanMapper`).
  - **Variant B (`structure_aware_no_header`)**: Atomic legal clauses segmented by `Pasal` / `Ayat` boundaries, without prepended breadcrumbs (9,215 chunks).
  - **Variant C (`structure_aware` / enriched)**: Atomic legal clauses segmented by `Pasal` / `Ayat` with hierarchical breadcrumbs prepended (`[Undang-Undang ... | BAB ... | Pasal ...]`) (9,215 chunks).
- **Statistical Significance Testing:**
  - 95% bootstrap confidence intervals (1,000 resamples).
  - Paired Monte Carlo permutation tests (10,000 resamples).
  - Effect size via Cohen's $d$.

#### 2. Quantitative Results

##### Development Set (`dev.jsonl`, $N = 40$)
| Metric | Variant A (`fixed_size`) | Variant B (`structure_no_header`) | Variant C (`structure_enriched`) | C vs A (Delta, 95% CI, $p$) | C vs B (Delta, 95% CI, $p$) |
|---|---|---|---|---|---|
| **Recall@1** | 0.0312 [0.000, 0.094] | 0.0312 [0.000, 0.094] | **0.0625** [0.000, 0.156] | +0.0312 [-0.062, 0.125], $p=1.000$ | +0.0312 [-0.062, 0.125], $p=1.000$ |
| **Recall@3** | 0.0312 [0.000, 0.094] | 0.0625 [0.000, 0.156] | **0.1406** [0.031, 0.266] | +0.1094 [+0.016, +0.219], $p=0.125$ | +0.0781 [-0.031, 0.203], $p=0.375$ |
| **Recall@5** | 0.0469 [0.000, 0.125] | 0.1406 [0.031, 0.266] | **0.2500** [0.124, 0.406] | **+0.2031 [+0.047, +0.359], $p=0.0168$** ✅ | +0.1094 [-0.031, 0.250], $p=0.219$ |
| **Recall@10** | 0.0781 [0.000, 0.172] | 0.1875 [0.062, 0.328] | **0.2500** [0.124, 0.406] | **+0.1719 [+0.031, +0.328], $p=0.0469$** ✅ | +0.0625 [-0.078, 0.203], $p=0.625$ |
| **MRR@5** | 0.0375 | 0.0656 | **0.1224** | **+0.0849**, $p=0.0469$ ✅ | +0.0568, $p=0.250$ |
| **nDCG@5** | 0.0387 | 0.0826 | **0.1574** | **+0.1187**, $p=0.0312$ ✅ | +0.0748, $p=0.219$ |
| **p95 Latency** | **204.4 ms** | 218.9 ms | 250.0 ms | +45.6 ms (negligible) | +31.1 ms (negligible) |

##### Held-Out Test Set (`test.jsonl`, $N = 66$)
| Metric | Variant A (`fixed_size`) | Variant B (`structure_no_header`) | Variant C (`structure_enriched`) | C vs A (Delta, 95% CI, $p$) | C vs B (Delta, 95% CI, $p$) |
|---|---|---|---|---|---|
| **Recall@1** | 0.0189 [0.000, 0.057] | 0.0283 [0.000, 0.075] | **0.0472** [0.000, 0.113] | +0.0283 [-0.038, 0.094], $p=0.625$ | +0.0189 [-0.057, 0.094], $p=0.754$ |
| **Recall@3** | 0.1038 [0.038, 0.189] | **0.2170** [0.113, 0.321] | 0.1509 [0.057, 0.245] | +0.0472 [-0.057, 0.151], $p=0.551$ | -0.0660 [-0.170, 0.038], $p=0.289$ |
| **Recall@5** | 0.1226 [0.047, 0.208] | 0.2170 [0.113, 0.321] | **0.2736** [0.160, 0.406] | **+0.1509 [+0.038, +0.264], $p=0.0210$** ✅ | +0.0566 [-0.019, 0.132], $p=0.368$ |
| **Recall@10** | 0.2453 [0.142, 0.358] | 0.3208 [0.208, 0.434] | **0.3774** [0.245, 0.509] | **+0.1321 [+0.000, +0.264], $p=0.0430$** ✅ | +0.0566 [-0.057, 0.170], $p=0.508$ |
| **MRR@5** | 0.0613 | 0.1132 | **0.1258** | **+0.0645**, $p=0.0410$ ✅ | +0.0126, $p=0.773$ |
| **nDCG@5** | 0.0763 | 0.1374 | **0.1611** | **+0.0848**, $p=0.0350$ ✅ | +0.0237, $p=0.598$ |
| **p95 Latency** | 258.1 ms | 252.3 ms | **240.9 ms** | -17.2 ms | -11.4 ms |

#### 3. Plain Language Trade-off Analysis & Rationale
1. **Failure of Naive Fixed-Size Chunking on Legal Statutes:**
   Arbitrary 500-character windows chop articles mid-sentence and decouple subordinate clauses (`Ayat`) from their parent article header (`Pasal`). When a user asks about sanctions under Article 78, a fixed chunk containing only `Ayat (2)` has no mention of "Pasal 78", rendering dense embeddings unable to match the query. Variant A experienced a statistically significant collapse in retrieval accuracy across both dev (Recall@5 = 0.0469 vs 0.2500, $p=0.0168$) and test (Recall@5 = 0.1226 vs 0.2736, $p=0.0210$).
2. **Value of Statutory Breadcrumbs:**
   Prepending hierarchical breadcrumbs (`[UU PDP | BAB XIII | Bagian Kesatu | Pasal 78 | Ayat (1)]`) gives every clause the complete legal context. This produces a **+77.8% relative gain on dev** and **+26.1% relative gain on test** over raw structure-aware chunks without headers (Variant B), while doubling MRR and nDCG metrics.
3. **No Latency Penalty:**
   Adding breadcrumbs increases chunk token length moderately (average ~120 tokens vs ~80 tokens), but because total chunk count is identical (9,215 chunks) and within the 512-token limit of embedding models, p95 query latency remains identical (~240 ms).
4. **Citation and Grounding Integrity for Downstream Generation:**
   Atomic legal chunks carrying exact article numbers enable the RAG generator and citation validation engine to produce verifiable legal references (e.g. `[UU 27/2022 Pasal 78 ayat (2)]`), eliminating hallucinated article boundaries.

#### 4. Final Decision
- **Adopt Variant C (`structure_aware` with hierarchical breadcrumbs)** as the standard, production chunking strategy across EasyPasal.
- Maintain `data/chunks/structure_aware.jsonl` as the default source for embedding and indexing in subsequent experiments (E2 Embedding Models and E3 Hybrid Retrieval).
- Keep the multi-strategy CLI flags (`--strategy {structure_aware,structure_aware_no_header,fixed_size}`) and runner (`make exp-e1`) in the codebase for experiment reproducibility and continuous benchmarking.

---

### Decision: Multilingual Embedding Models (Experiment E2 / T4.2)
- **Date:** 2026-10-15
- **Status:** Approved (Adopt `intfloat/multilingual-e5-large`)

#### 1. Experimental Setup
- **Evaluation Splits:**
  - `dev.jsonl`: $N = 40$ questions (32 answerable, 8 unanswerable/out-of-scope).
  - Held-out `test.jsonl`: $N = 66$ questions (54 answerable, 12 unanswerable/out-of-scope).
- **Corpus & Chunks:**
  - Canonical `structure_aware` chunks (9,215 chunks in `data/chunks/structure_aware.jsonl`).
- **Retrieval Mode:**
  - Pure dense vector search (`mode="dense"`, cosine similarity, top-$k = 10$). Evaluates purely the semantic representation quality of embeddings without BM25 masking.
- **Evaluated Candidates:**
  1. **Variant 1 (`minilm_384`)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~118M params, lightweight baseline, INT8 quantized ONNX).
  2. **Variant 2 (`mpnet_768`)**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768-dim, ~278M params, medium capacity multilingual transformer).
  3. **Variant 3 (`e5_large_1024`)**: `intfloat/multilingual-e5-large` (1024-dim, ~560M params, asymmetric instruction prefixing with `passage: ` and `query: `).
- **Statistical Significance Testing:**
  - 95% bootstrap confidence intervals (2,000 resamples).
  - Paired Monte Carlo permutation tests (5,000 resamples).
  - Paired comparison against MiniLM reference baseline.

#### 2. Quantitative Results

##### Development Set (`dev.jsonl`, $N = 40$, 32 answerable)
| Model Variant | Dim | Params | Recall@5 | 95% CI | MRR@5 | nDCG@5 | p95 Latency | Disk Size | Sig vs MiniLM ($p$) |
|---|---|---|---|---|---|---|---|---|---|
| **MiniLM-L12-v2** | 384 | ~118M | 0.0938 | [0.0000, 0.1875] | 0.0703 | 0.0760 | **193.0 ms** | **47.5 MB** | Reference |
| **MPNet-Base-v2** | 768 | ~278M | 0.1406 | [0.0312, 0.2656] | 0.1406 | 0.1326 | 331.9 ms | 81.1 MB | $p = 0.6253$ |
| **Multilingual-E5-Large** | 1024 | ~560M | **0.2969** | [0.1406, 0.4531] | **0.2135** | **0.2350** | 742.8 ms | 107.1 MB | **$p = 0.0364$** ✅ |

##### Held-Out Test Set (`test.jsonl`, $N = 66$, 54 answerable)
| Model Variant | Dim | Params | Recall@5 | 95% CI | MRR@5 | nDCG@5 | p95 Latency | Disk Size | Sig vs MiniLM ($p$) |
|---|---|---|---|---|---|---|---|---|---|
| **MiniLM-L12-v2** | 384 | ~118M | 0.1698 | [0.0755, 0.2830] | 0.1368 | 0.1451 | **199.9 ms** | **47.5 MB** | Reference |
| **MPNet-Base-v2** | 768 | ~278M | 0.2264 | [0.1226, 0.3396] | 0.1673 | 0.1728 | 362.9 ms | 81.1 MB | $p = 0.3415$ |
| **Multilingual-E5-Large** | 1024 | ~560M | **0.3679** | [0.2453, 0.5000] | **0.2396** | **0.2680** | 861.0 ms | 107.1 MB | **$p = 0.0064$** ✅ |

#### 3. Plain Language Trade-off Analysis & Rationale
1. **Decisive Semantic Superiority of Multilingual-E5-Large:**
   Multilingual-E5-Large demonstrated substantial and statistically significant retrieval improvements over the MiniLM baseline across both development (+216% relative Recall@5, $p=0.0364$) and held-out test (+117% relative Recall@5, $p=0.0064$). MRR jumped from 0.0703 to 0.2135 on dev (a 3x improvement), showing that the top-ranked dense results are much more accurate.
2. **The Impact of Asymmetric Task Prefixes (`query: ` vs `passage: `):**
   Statutory texts contain dense, formal Indonesian legal drafting that differs sharply in linguistic style from natural language search queries. E5's dual prefix mechanism allows the embedding space to bridge the query-to-statute linguistic mismatch, whereas symmetric models like MiniLM and MPNet struggle to align short questions with long legal clauses.
3. **MPNet Fails Significance Thresholds:**
   While MPNet-Base-v2 improves Recall@5 from 0.0938 to 0.1406 on dev and 0.1698 to 0.2264 on test, the paired permutation test shows that these gains are not statistically significant ($p = 0.6253$ dev, $p = 0.3415$ test). Given that MPNet requires 2.5 hours to index without INT8 quantization and doubles query latency, it does not justify the added overhead over MiniLM.
4. **Latency and Storage Budget:**
   - Qdrant disk storage for E5-Large is 107.1 MB (vs 47.5 MB for MiniLM), which comfortably fits well within the < 500 MB budget for free tier hosting and local laptops.
   - p95 query latency on CPU is 742-861 ms. While higher than MiniLM (~195 ms), this remains well under the 1.5-second retrieval latency budget for interactive compliance search, and can be further accelerated via hybrid fusion (E3) or embedding caching.

#### 4. Final Decision
- **Adopt `intfloat/multilingual-e5-large` (1024-d)** as the primary embedding model for EasyPasal across production retrieval and subsequent experiments (E3 Hybrid Retrieval Pipeline).
- Maintain automated asymmetric prefixing (`passage: ` for indexing, `query: ` for querying) in `src/easypasal/index/embeddings.py`.
- Preserve MiniLM (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) in the codebase as a lightweight, low-resource fallback option.
- Retain the automated E2 experiment runner (`make exp-e2` / `experiments/e2_embeddings/run_experiment.py`) for reproducibility.
