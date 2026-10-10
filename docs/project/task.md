# EasyPasal — Task List

> Source of truth for progress. Companion to `implementation_plan.md` (the *what and why*) and `walkthrough.md` (the *how to work and present it*).
> Grouped by **phase** (not by week). Target finish **2026-10-24**; application deadline **2026-10-31**.

**Legend**
- `[ ]` not started · `[/]` in progress · `[x]` done · `[~]` descoped (note why)
- **P0** must ship · **P1** strongly desired · **P2** stretch
- 👤 needs you (judgment, labeling, accounts) · 🤖 good to delegate to the Antigravity agent · 🤝 pair on it
- **AC** = acceptance criteria (how we know it's done)

**Rule for the agent:** before starting a task, read `implementation_plan.md`. After finishing, tick the box, add a one-line note with the commit hash, and never fabricate results — leave `TBD` instead.

---

## Phase 0 — Setup and verification (P0)

- [x] **T0.1** 👤 Create public GitHub repo `easypasal` (Apache-2.0, README stub, `.gitignore`) (commit `d6993c8`)
  - AC: repo exists, license file present, cloned locally in Antigravity
- [x] **T0.2** 👤 Check the name "EasyPasal" on GitHub, PyPI, and a web search; record result in `docs/decision-log.md`
  - AC: decision noted (keep or rename)
- [x] **T0.3** 🤖 Scaffold project: `pyproject.toml`, uv, ruff, pytest, pre-commit, `Makefile`, folder structure from plan §11 (commit `8efcf33`)
  - AC: `make setup && make test` passes on a clean clone
- [x] **T0.4** 🤖 Add GitHub Actions skeleton (lint + tests) (commit `6f9ce82`)
  - AC: first CI run is green
- [ ] **T0.5** 🤝 Verify current models/versions/licenses (generator shortlist incl. Indonesian-tuned, embeddings); record in `docs/decision-log.md`
  - AC: shortlist of 2–4 generators and 2–3 embedding models with license + size notes
- [ ] **T0.6** 🤝 Verify free Hugging Face Space limits, Colab/Kaggle GPU quotas, and VPS suitability for the log collector
  - AC: constraints recorded in `docs/architecture.md`
- [ ] **T0.7** 👤 Choose regulations for the corpus (15–30 docs, four themes) and confirm current status/amendments
  - AC: `data/corpus_manifest.csv` filled (title, number/year, issuer, URL, retrieval date, status, language)

## Phase 1 — Ingestion and baseline RAG (P0)

- [ ] **T1.1** 🤖 Download script that fetches corpus files from manifest URLs, stores hashes
  - AC: `make ingest` reproduces the raw corpus locally (raw PDFs not committed if licensing/size unclear)
- [ ] **T1.2** 🤖 PDF parsing with PyMuPDF; handle headers/footers, hyphenation, and numbering
  - AC: text extraction spot-checked on 5 documents
- [ ] **T1.3** 🤖 Structure-aware chunker (Bab → Bagian → Pasal → Ayat) with metadata; plus fixed-size baseline chunker
  - AC: unit tests on sample articles; each chunk carries doc_id, pasal, ayat, status, source URL
- [ ] **T1.4** 🤖 Embedding + Qdrant index + BM25 index build (`make index`)
  - AC: index builds on laptop CPU; index size and build time logged
- [ ] **T1.5** 🤖 Hybrid retrieval with RRF fusion
  - AC: retrieval returns top-k with scores and metadata
- [ ] **T1.6** 🤝 Generation module: local small model via llama.cpp/Ollama, citation-required prompt, abstain behavior, language matching
  - AC: answers 10 hand-picked questions with citations; abstains on 3 out-of-corpus questions
- [ ] **T1.7** 🤖 Post-hoc citation verification (cited pasal must exist in retrieved set)
  - AC: unit tests for pass/fail cases
- [ ] **T1.8** 🤖 FastAPI service (`/ask`, `/health`) and request-ID logging
  - AC: `curl` example in README works

## Phase 2 — Gold evaluation set (P0)

- [ ] **T2.1** 👤 Write labeling guide: categories, answer style, citation format (`docs/eval-methodology.md`)
  - AC: guide committed
- [ ] **T2.2** 🤖 Build labeling helper (script that shows a source passage and a blank item template; optional model-drafted candidate questions)
  - AC: helper runs locally; drafts are clearly marked "unverified"
- [ ] **T2.3** 👤 Label ≥100 items (target 150–200) across all categories incl. unanswerable, superseded, adversarial, and an English slice
  - AC: JSONL validated against schema; every gold citation checked against the source text
- [ ] **T2.4** 🤖 Create train/dev/test splits (stratified by category, fixed seed) and a schema validator
  - AC: `make validate-gold` passes; split sizes reported
- [ ] **T2.5** 👤 Re-label a 20% sample after a few days; compute self-agreement
  - AC: agreement rate recorded in `docs/eval-methodology.md`

## Phase 3 — Evaluation harness and CI gate (P0)

- [ ] **T3.1** 🤖 Deterministic metrics: Recall@k, MRR, nDCG, citation precision/recall, abstention accuracy
  - AC: unit-tested on toy examples with known answers
- [ ] **T3.2** 🤖 Eval runner (`make eval`) producing a results table + per-category breakdown + MLflow run
  - AC: one command runs the full retrieval eval on `dev`
- [ ] **T3.3** 🤝 LLM-judged metrics (faithfulness, correctness) using RAGAS/DeepEval-style prompts with a stronger open-weight judge on free GPU
  - AC: judged eval runs on Colab/Kaggle; outputs saved
- [ ] **T3.4** 👤 Judge calibration: hand-label 30–40 outputs, compare with judge, report agreement
  - AC: calibration section in `docs/eval-methodology.md`
- [ ] **T3.5** 🤖 Bootstrap confidence intervals + paired comparison utility
  - AC: variant comparison output includes CIs
- [ ] **T3.6** 🤖 CI smoke eval (≈30 fixed items, retrieval + deterministic metrics, no GPU) with stored baseline and thresholds
  - AC: a deliberately broken retrieval change makes CI fail (demonstrate and keep as a test)

## Phase 4 — Core experiments E1–E4 (P0)

- [ ] **T4.1** 🤝 E1 chunking strategy → decision logged
  - AC: results on `dev`, final check on `test`, trade-off notes in plain language
- [ ] **T4.2** 🤝 E2 embedding models → decision logged
- [ ] **T4.3** 🤝 E3 retrieval pipeline (dense / BM25 / hybrid / +rerank) → decision logged
- [ ] **T4.4** 🤝 E4 generator model comparison (quality vs latency/memory on CPU) → decision logged
- [ ] **T4.5** 🤖 Freeze the **baseline variant** in `configs/` and tag the release `v0.1`
  - AC: tag exists; CI baseline updated

## Phase 5 — Prompt optimization and inference (P1)

- [ ] **T5.1** 🤝 E5: DSPy + GEPA run optimizing the answer prompt on `train`, tune on `dev`, report on `test`
  - AC: before/after table with CIs; check for overfitting; optimized prompt committed
- [ ] **T5.2** 🤝 E6: vLLM benchmark notebook on free GPU (throughput, TTFT, p95 latency at concurrency 1/4/16/32) vs llama.cpp CPU baseline
  - AC: results table + plot in README; notebook reproducible; GPU type and settings recorded
- [ ] **T5.3** 🤖 Variant switch in the API (config flag + deterministic hashing) and logging of variant
  - AC: two variants can serve side by side

## Phase 6 — Replay A/B and fine-tuning (P1/P2)

- [ ] **T6.1** 🤝 Replay-based A/B simulation: replay logged/dev queries through variants A and B, same judge, paired CIs (P1)
  - AC: decision documented; clearly labeled as offline replay, not a user test
- [ ] **T6.2** 👤 Choose E7 task (recommended: question router/classifier for theme + answerability + injection detection) (P2)
  - AC: choice recorded in decision log
- [ ] **T6.3** 🤝 E7: QLoRA fine-tune on free GPU with PEFT + TRL (P2)
  - AC: adapter card, training config, before/after metrics, regression check on other categories
- [ ] **T6.4** 🤖 Integrate router/adapter into guardrails only if it improves `dev` and `test` (P2)
  - AC: gated by eval results; otherwise documented as a negative result

## Phase 7 — Monitoring and deployment (P0)

- [ ] **T7.1** 🤖 Structured request logging in the app (fields from plan §9) with basic redaction
  - AC: logs validated against a schema
- [ ] **T7.2** 🤖 Log collector (FastAPI + SQLite) for the VPS, Dockerized, with auth token
  - AC: runs within the VPS memory budget; verified with load test
- [ ] **T7.3** 🤖 Dashboard: latency p50/p95, volume, abstain rate, retrieval-score distribution, feedback
  - AC: reachable on VPS; screenshots saved for README
- [ ] **T7.4** 🤝 Drift/quality report with Evidently (P2 if time is short)
  - AC: report generated from logs vs gold-set distribution
- [ ] **T7.5** 🤖 Simple threshold alerts (abstain rate, p95 latency) writing flagged events
  - AC: test alert triggers in a drill
- [ ] **T7.6** 👤 Deploy the demo to a free Hugging Face Space (prebuilt index in image)
  - AC: public URL works from a phone; cold-start time recorded
- [ ] **T7.7** 🤖 `docker-compose.yml` for full local run (fallback demo)
  - AC: `docker compose up` works on a clean laptop
- [ ] **T7.8** 🤖 Runbook with rollback and index rebuild steps (`docs/runbook.md`)
  - AC: rollback rehearsed once
- [ ] **T7.9** 👤 Generate some real usage: ask 30+ varied questions yourself (and optionally invite a few friends) so monitoring shows genuine logs
  - AC: dashboard shows non-trivial logged traffic; **do not** inflate or fake it

## Phase 8 — Documentation and communication (P0)

- [ ] **T8.1** 🤝 README: pitch, independence notice, results table, architecture diagram, live demo link, quickstart
  - AC: a stranger can understand the project in 2 minutes
- [ ] **T8.2** 👤 Stakeholder memo (1–2 pages) for Risk/Compliance: what shipped, what didn't, trade-offs, risks, future work (incl. credit-risk extension)
  - AC: reads without ML jargon; every number matches the decision log
- [ ] **T8.3** 🤝 `docs/decision-log.md` finalized (E1–E7, with decisions and reasoning)
- [ ] **T8.4** 🤝 `docs/reproduce.md` — "how to reproduce this" tutorial
  - AC: a friend can reproduce core results following only this guide
- [ ] **T8.5** 👤 Record a 3–5 minute demo video (fallback + portfolio)
- [ ] **T8.6** 🤖 Add badges (CI, license) and repo topics; final lint pass
- [ ] **T8.7** 👤 Honest-claims audit: compare README/memo/resume wording against plan §13
  - AC: no claim of real production users or real A/B tests unless true

## Phase 9 — Application packaging (P0)

- [ ] **T9.1** 👤 Update resume for the OCBC Data Scientist role: lead with ML/LLM systems, evaluation, fraud detection, thesis results; add EasyPasal
- [ ] **T9.2** 👤 Update LinkedIn headline/About/Featured with the repo and demo link
- [ ] **T9.3** 👤 Prepare a 20-minute walkthrough using `walkthrough.md` Part C and practice once aloud
- [ ] **T9.4** 👤 Final check of every link (repo, demo, docs, video) from a logged-out browser
- [ ] **T9.5** 👤 Submit the application (target **2026-10-28**; hard deadline **2026-10-31**)

---

## Cut-line reminder
If time runs short, descope in this order and record it as future work: **E7 QLoRA → drift report → replay A/B → vLLM benchmark → GEPA.** Never cut: gold set, eval harness, CI gate, live demo, monitoring, stakeholder memo.

## Progress log
| Date | Update |
|---|---|
| 2026-10-10 | Plan, task list, and walkthrough created. |
| 2026-10-10 | T0.1: Added Apache-2.0 LICENSE and README stub (commit `d6993c8`). |
| 2026-10-10 | T0.3: Scaffolded project directory structure, `pyproject.toml` with uv, ruff, mypy, pytest, pre-commit, `Makefile`, and initial smoke test (commit `8efcf33`). |
| 2026-10-10 | T0.4: Added GitHub Actions CI workflow for lint and test (commit `6f9ce82`). |
| 2026-10-10 | T0.2: Verified name "EasyPasal" on GitHub, PyPI, and web; decision recorded in `docs/decision-log.md`. |
