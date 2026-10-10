# EasyPasal — Walkthrough

> A living guide in five parts. **Part A** is how to build the project with Antigravity + GitHub. **Part B** is the tour a reviewer will take. **Part C** is a 20-minute demo script. **Part D** is interview preparation mapped to the job posting. **Part E** is the honest-claims guide.
>
> Anything marked **TBD** must be replaced with real results. Never invent numbers.

---

## Part A — How to work on this project (Antigravity IDE + GitHub, on your laptop)

### A1. One-time setup
1. Create the public repo `easypasal` on GitHub (Apache-2.0) and clone it locally.
2. Open the folder in Antigravity IDE.
3. Copy these three files into `docs/project/` (or the repo root): `implementation_plan.md`, `task.md`, `walkthrough.md`.
4. Commit them as the first commit: `docs: add plan, tasks, walkthrough`.
5. Make sure your laptop has: Python 3.11+, `uv`, Docker, Git, and (for local LLM tests) Ollama or llama.cpp. Your laptop is CPU-only with 16 GB RAM, so use small 4-bit models locally.

### A2. Daily working loop
1. **Pull** the latest changes: `git pull`.
2. Open `task.md`; pick the next unchecked task in the current phase.
3. Tell the agent which task you're working on, and point it at the plan. Suggested prompt:
   > "Read `implementation_plan.md` and `task.md`. Work on task **T1.3**. Follow the repo structure in plan §11, write tests first or alongside, run lint and tests, and summarize what changed. Do not invent results; leave TBD where data is missing."
4. **Review the diff yourself** before accepting. You are the owner; you must be able to explain every file in an interview.
5. Run `make lint test` (and `make eval` when relevant).
6. Tick the task in `task.md`, add a one-line note, and commit using clear messages (e.g., `feat(ingest): structure-aware chunker (T1.3)`).
7. **Push** to GitHub and confirm CI is green.

### A3. Branching (simple and professional)
- `main` is always green and deployable.
- One short-lived branch per task group (`feat/ingest`, `exp/e1-chunking`) and open a pull request, even if you're the only reviewer. The CI gate then visibly runs on each PR — which is itself evidence for the posting's "evaluation decides what ships" line.
- Tag releases: `v0.1` (baseline), `v0.2` (after experiments), `v1.0` (final).

### A4. Where the GPU work happens
Laptop and VPS can't do GPU work. For **E3 judged evals, E5 (GEPA), E6 (vLLM), E7 (QLoRA)**:
1. Keep notebooks/scripts in the repo with pinned configs.
2. Open them in Colab or Kaggle (free GPU), run, and save outputs (metrics JSON, adapters, plots) back to the repo or to a Hugging Face Hub repo.
3. Record GPU type, library versions, seed, and runtime in the result file. Free sessions are time-limited, so keep runs small and resumable.

### A5. Working rules
- **Secrets:** never commit tokens (Hugging Face, VPS, log-collector token). Use environment variables and `.env` (git-ignored).
- **Test set discipline:** `test` split is used only for final decisions. If you peek and tune on it, say so in the decision log.
- **Log everything that matters:** every experiment gets a row in `docs/decision-log.md` with hypothesis → config → result → decision → trade-off.
- **Small commits:** it keeps the Git history readable for reviewers.

### A6. Handy commands (to exist after Phase 0–1)
```bash
make setup      # create env, install deps, pre-commit
make ingest     # download + parse corpus from the manifest
make index      # build Qdrant + BM25 indexes
make serve      # run API + Gradio locally
make eval       # run evaluation on dev split, write MLflow run + report
make test       # unit tests
make lint       # ruff + type checks
make bench      # CPU latency benchmark (GPU/vLLM runs are notebooks)
```

---

## Part B — Project tour (what a reviewer will see)

> Fill in each block with real results as phases complete. Until then, leave **TBD**.

### B1. The problem in plain language
Bank staff and product teams must repeatedly check what Indonesian regulations require. Searching long PDFs is slow, and generic chatbots can sound confident while citing nothing. EasyPasal answers in Bahasa Indonesia or English, cites the exact regulation, article (pasal) and clause (ayat), and says "no legal basis found" when the documents don't support an answer.

### B2. The system
- Architecture diagram: see `implementation_plan.md` §3 (copy the final version into `docs/architecture.md`).
- Corpus: **TBD** documents across four themes (consumer protection, IT risk, AML/CFT, personal data).
- Pipeline: structure-aware chunking → hybrid retrieval (dense + BM25) → cite-or-abstain generation → citation verification.

### B3. Evaluation results (the centerpiece)

| Variant | Recall@5 | Citation recall | Faithfulness | Abstention acc. | p95 latency | Decision |
|---|---|---|---|---|---|---|
| Baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| Best of E1–E4 | TBD | TBD | TBD | TBD | TBD | TBD |
| + GEPA prompt | TBD | TBD | TBD | TBD | TBD | TBD |
| + QLoRA router (if done) | TBD | TBD | TBD | TBD | TBD | TBD |

- Gold set: **TBD** items (train/dev/test = TBD/TBD/TBD); self-agreement **TBD**; judge-vs-human agreement **TBD**.
- CI gate: describe the smoke eval, thresholds, and a screenshot of a PR that was blocked by it (**TBD**).

### B4. Inference results (vLLM)
| Setup | Hardware | Concurrency | Throughput (tok/s) | TTFT | p95 latency |
|---|---|---|---|---|---|
| llama.cpp 4-bit | Laptop CPU | 1 | TBD | TBD | TBD |
| vLLM | Free GPU (type TBD) | 1/4/16/32 | TBD | TBD | TBD |

One-sentence takeaway for a non-technical reader: **TBD** (e.g., "On a free GPU, vLLM served X times as many requests per second as the CPU setup at similar quality").

### B5. Operations
- Live demo: **TBD URL** · Monitoring dashboard: **TBD URL** · Docs: **TBD URL**
- What is monitored, thresholds, and one example of an alert drill: **TBD**.
- Rollback rehearsal: **TBD**.

### B6. What the stakeholder memo says
Summarize in three bullets for a Risk/Compliance reader: what shipped, what the system will refuse to do, and the top three limitations. (Final text lives in `docs/stakeholder-memo.md`.)

---

## Part C — 20-minute demo script

| Time | What you show | What you say (key message) |
|---|---|---|
| 0:00–1:30 | README top section | "I built an evaluation-first compliance assistant to show how I take an LLM workflow from design to monitoring." State the independence notice. |
| 1:30–4:00 | Live demo: 3 questions | (1) answerable Indonesian question with citations, (2) English question, (3) out-of-scope question showing abstention. "Grounded or silent." |
| 4:00–7:00 | Architecture diagram | Walk the flow; explain why a thin custom pipeline, why hybrid retrieval, and why the app and monitoring are split. |
| 7:00–11:00 | Gold set + eval harness | Show categories, splits, and the `make eval` output. Explain judge calibration and why deterministic metrics come first. |
| 11:00–14:00 | Decision log | Pick one experiment where the result surprised you or a "fancier" option lost. This is your best story for "evals that influenced decisions." |
| 14:00–16:00 | CI gate | Show a PR blocked by the eval gate. "No variant ships without passing the gate." |
| 16:00–18:00 | vLLM + GEPA results | Present as "benchmarked / optimized, with results and caveats," not as mastery. |
| 18:00–19:30 | Monitoring dashboard | Latency, abstain rate, drift, feedback; describe the rollback runbook. |
| 19:30–20:00 | Limits + next steps | Credit-risk extension, larger models, real user study. Invite questions. |

**Backup plan:** if the live demo is slow or down, switch to the recorded video and run the local `docker compose` demo.

---

## Part D — Interview preparation, mapped to the posting

### D1. Likely questions and where your evidence is

| Question | Where to point | Notes |
|---|---|---|
| "Tell me about an LLM/RAG system you took to production." | EasyPasal + earlier course RAG chatbot | Be precise: EasyPasal is a production-style system with monitoring and a live demo; it had **limited, self-generated traffic**. Earlier RAG work was a course demo. |
| "How do you decide whether something is good enough to ship?" | Gold set, CI gate, decision log | Explain thresholds, paired comparisons, confidence intervals, and the held-out test split. |
| "Describe an eval that changed a decision." | Decision log (pick the best experiment) | Hypothesis → result → what you didn't ship and why. |
| "How would you A/B test an LLM change in a bank?" | Plan §6.4 | Offline paired eval first, shadow/replay, then limited traffic with guardrails; discuss power, risk review, and rollback. Be explicit about what you did vs. what you would do. |
| "How did you reduce hallucinations?" | Cite-or-abstain prompt, citation verification, abstention metric, adversarial set | Show numbers. |
| "Why vLLM? What did you measure?" | Benchmark table | Throughput vs latency trade-off, batching, memory; mention limits of a free GPU. |
| "What is GEPA and what did it do?" | Experiment E5 | Reflective prompt evolution using eval feedback; show before/after and overfitting checks. |
| "Why QLoRA, and did it help?" | Experiment E7 | Memory-efficient fine-tuning; report honestly, including negative results. |
| "How would this change for credit risk or regulatory compliance in a real bank?" | Stakeholder memo | Data privacy, model risk management, audit trails, human-in-the-loop, regulator expectations. |
| "Explain trade-offs to a non-technical stakeholder." | Memo + one-slide summary | Practice a 60-second version with no jargon. |
| "What did you personally build in your fraud system?" | Resume (SG-EDTS) | Prepare precise ownership: data, model, pipeline, reporting. Note it was an alert system without a monitoring/retraining loop, and say what you'd add now (this project is that lesson applied). |
| "Link your thesis to this role." | Thesis | Same discipline: careful baselines, ablations, quantified gains; prompts as adaptive components. Keep the GEPA comparison modest. |

### D2. One-minute story (draft)
"My background is in applied ML — fraud detection in production at SG-EDTS and vision-language research for my master's thesis. For this role I wanted to show the part that matters most when LLMs meet banking: evaluation. EasyPasal answers questions on Indonesian financial regulation with citations, and every change goes through a gold-set evaluation and a CI gate before it ships. I benchmarked vLLM, optimized prompts with GEPA, and monitor the live system. I can walk you through the decisions the evals changed." *(Edit to match your real results.)*

### D3. Questions to ask them
- How are LLM changes currently evaluated and approved with Risk and Compliance?
- What does the path from experiment to production look like for AI at Banking Industry?
- Which banking use cases are the first priority for the team?

---

## Part E — Honest-claims guide

**Safe to say (when true)**
- "Production-style RAG system with offline evals, CI gate, monitoring, and a live demo."
- "Benchmarked vLLM against a CPU baseline on a free GPU."
- "Optimized prompts with GEPA and evaluated on a held-out split."
- "Hand-built a gold set of N items and calibrated the LLM judge against human labels."

**Avoid**
- Real customer/production-scale claims, real A/B tests on users, or "fine-tuned for banking" if E7 is small.
- Any metric you cannot regenerate with a documented command.
- Implying affiliation with OJK or Bank Indonesia.

**Always include**
- The independence notice and "not legal advice" disclaimer in the README and UI.
- A "Limitations" section: small corpus, small models, single-annotator gold set, limited traffic.

---

## Final pre-submission checklist
- [ ] README, memo, decision log, and reproduce guide published
- [ ] All links work when logged out
- [ ] Demo video recorded
- [ ] Resume and LinkedIn updated and consistent with the repo
- [ ] Honest-claims audit completed
- [ ] Application submitted (target 2026-10-28, deadline 2026-10-31)
