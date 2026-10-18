# EasyPasal — Operations & Disaster Recovery Runbook

This runbook serves as the single source of truth for operations, on-call engineers, and release managers maintaining the EasyPasal regulatory question-answering system. It details standard operating procedures, incident triage, operational rollback playbooks, disaster recovery index rebuilds, and the automated rollback rehearsal drill (Task T7.8).

---

## 1. System Topology & Operational Architecture

EasyPasal is structured as a resilient, multi-service architecture designed to operate either across distributed infrastructure (VPS + Hugging Face Spaces) or fully containerized locally via Docker Compose.

```
                                  ┌──────────────────────────────────────────────┐
                                  │           Client / Browser / App             │
                                  └───────┬──────────────┬──────────────┬────────┘
                                          │ :7860        │ :8000        │ :8080
                                          ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ EasyPasal Infrastructure & Docker Network (`easypasal_default`)                         │
│                                                                                         │
│  ┌───────────────────────┐   health   ┌───────────────────────┐   health   ┌──────────┐ │
│  │     `easypasal-ui`    ├───────────►│    `easypasal-app`    ├───────────►│`collector`│ │
│  │ (Gradio Demo :7860)   │            │(FastAPI REST API :8000│            │  (:8080) │ │
│  └───────────┬───────────┘            └───────────┬───────────┘            └─────┬────┘ │
│              │                                    │                              │      │
│              │ Telemetry                          │ Log & Feedback Forwarding    │      │
│              └────────────────────────────────────┴─────────────────────────────►│      │
│                                                                                  ▼      │
│                                                                            [collector.db│
│                                                                             SQLite WAL] │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Microservices & Port Allocations
| Service | Container / Process | Default Port | Internal Endpoints | Primary Function | Resource Limits / Budget |
|---|---|:---:|---|---|---|
| **Collector & Telemetry** | `easypasal-collector` | `8080` | `/health`, `/dashboard`, `/reports/drift`, `/v1/logs`, `/v1/alerts` | SQLite WAL ingestion, Evidently drift analysis, KPI dashboard | Memory RSS $\le 90\text{ MB}$, disk $\le 1\text{ GB}$ |
| **REST API Backend** | `easypasal-app` | `8000` | `/health`, `/ask`, `/feedback`, `/docs` | Statutory retrieval (Qdrant + BM25), guardrail routing, answer generation | 2 vCPU, 4–8 GB RAM |
| **Interactive Web UI** | `easypasal-ui` | `7860` | `/` | Gradio two-column workspace, bilingual toggle, citation inspector | 2 vCPU, 2–4 GB RAM |
| **Standalone Demo** | Hugging Face Space | `7860` | `/` (Public URL) | Zero-friction public evaluation demo with baked prebuilt index | 2 vCPU, 16 GB RAM (Free tier) |

### 1.2 Configuration Precedence Hierarchy
When resolving operational configurations, EasyPasal applies strict precedence:
1. **Explicit Request Payload:** `variant` field in `POST /ask`.
2. **HTTP Request Header:** `X-EasyPasal-Variant` or `X-Variant`.
3. **Deterministic Traffic Splitting:** Hash-bucket routing when `traffic_split.enabled = true`.
4. **Active Configuration File:** Path specified via `EASYPASAL_ACTIVE_CONFIG` (e.g. `configs/gepa_v0.2.yaml`).
5. **Environment Variable:** `EASYPASAL_VARIANT` (e.g. `baseline_v0.1` or `gepa_v0.2`).
6. **Frozen Baseline Fallback:** `configs/baseline_v0.1.yaml`.

---

## 2. Standard Operating Procedures (SOPs)

### 2.1 Local Development Environment
```bash
# 1. Environment initialization and dependency synchronization
make setup

# 2. Download prebuilt Qdrant & BM25 indices (takes <20s from Hugging Face Space)
make download-index

# 3. Launch REST API backend (:8000)
make serve

# 4. Launch Gradio UI demo (:7860) in a separate terminal
make ui
```

### 2.2 Multi-Service Docker Compose Stack (Local Fallback Demo)
```bash
# Build all service containers with multi-stage non-root images
make docker-build

# Start stack in background with inter-service health validation
make docker-up

# Tail real-time aggregated logs across all three containers
make docker-logs

# Stop containers and cleanly tear down networks
make docker-down
```

### 2.3 System Health Verification
Verify readiness and indexed statutory vector counts via curl:
```bash
curl -s http://localhost:8000/health | jq .
```
Expected output format:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "development",
  "index_ready": true,
  "vector_count": 9215,
  "bm25_doc_count": 9215,
  "backend_provider": "MockGeneratorBackend",
  "active_variant": "gepa_v0.2",
  "traffic_split_enabled": false
}
```

---

## 3. Telemetry Monitoring, Threshold Alerting & Incident Playbooks (Plan §9 / T7.5)

### 3.1 Monitored Boundaries & Alert Thresholds
Operational telemetry evaluated by the collector engine:

| Metric | Warning Ceiling | Critical Ceiling | Root Cause Rationale | Primary Remediation |
|---|---|---|---|---|
| **Abstention Rate** (`abstain_rate`) | $> 35.0\%$ | $\ge 50.0\%$ | Retrieval starvation, corrupted vector store, or adversarial prompt injection surges. | Inspect `/v1/logs/requests?abstained=true`; execute variant rollback or index rebuild. |
| **p95 Latency** (`p95_latency_ms`) | $> 5{,}000\text{ ms}$ | $\ge 10{,}000\text{ ms}$ | Host CPU exhaustion, GPU PagedAttention queue backlog, or excessive token generation. | Check collector host memory `/health` ($\le 90\text{ MB}$ budget); scale worker threads. |
| **Low-Evidence Rate** (`low_evidence_rate`) | $> 30.0\%$ | $> 50.0\%$ | Queries falling below top-1 score $<0.60$; indicates user queries outside regulatory domain. | Review Evidently drift report (`/reports/drift`); expand statutory corpus manifest if needed. |
| **Sample Size Gate** | $N \ge 10$ requests | — | Prevents spurious false alarms during cold starts. | Allow system traffic to accumulate before triggering automated alerts. |

### 3.2 Automated Alert Drill Procedure
To verify that alerting pipelines, database event persistence, and webhook dispatches operate as intended without waiting for real production incidents:

```bash
# 1. Trigger an automated synthetic alert drill via CLI
uv run python -m easypasal.monitoring.alerts --action drill --metric abstain_rate

# 2. Or trigger via the authenticated Collector REST API
curl -X POST "http://localhost:8080/v1/alerts/drill?metric=abstain_rate" \
     -H "Authorization: Bearer easypasal-vps-secret-token-v1"

# 3. Query flagged alert events from SQLite
uv run python -m easypasal.monitoring.alerts --action list

# 4. View active alerts in the dashboard
curl -s "http://localhost:8080/v1/alerts/active?lookback_minutes=60" \
     -H "Authorization: Bearer easypasal-vps-secret-token-v1" | jq .
```

### 3.3 Incident Triage & Resolution Playbooks

#### Playbook A: Abstention Rate Surge (`abstain_rate > 35%`)
```
[Alert: Abstain Rate Spike]
          │
          ▼
Query Collector Logs: GET /v1/logs/requests?abstained=true&limit=20
          │
          ├─────────────────────────────────────────┐
          ▼                                         ▼
   routed == true?                           routed == false?
   (Safety Guardrail Filter)                 (Post-Retrieval Abstention)
          │                                         │
          ▼                                         ▼
   theme == "Out of Scope" or Injection?     Check retrieval_score_top1:
          │                                         │
     [Expected]                                     ├────────────────────────┐
   Legitimate rejection;                            ▼                        ▼
   no action needed.                       score < 0.60?            score >= 0.75?
                                                    │                        │
                                                    ▼                        ▼
                                             Retrieval Starvation     Model Hallucination
                                             Execute Playbook 4.4     Execute Playbook 4.1
                                             (Index Rebuild)          (Variant Rollback)
```

#### Playbook B: Latency Degradation (`p95_latency_ms > 5000ms`)
1. **Host Memory Check:** Verify collector memory RSS via `curl http://localhost:8080/health`. If RSS $> 90\text{ MB}$, execute WAL checkpoint: `sqlite3 data/sqlite/collector.db 'PRAGMA wal_checkpoint(TRUNCATE);'`.
2. **Worker Pool Check:** For CPU generation, ensure `OMP_NUM_THREADS` matches physical cores.
3. **LLM Generator Mode:** Verify `EASYPASAL_LLM_PROVIDER`. If set to physical `llamacpp` on a resource-constrained node, fall back to `mock` or reduce `n_ctx` from 2048 to 1024.

---

## 4. Operational Rollback Procedures

### 4.1 Variant Rollback Procedure (`gepa_v0.2` $\rightarrow$ `baseline_v0.1`)
**Trigger Criteria:** Prompt regression detected, formatting deviations in citations, Grounding Fidelity drop, or an unexplained abstention spike on candidate variant `gepa_v0.2`.

#### Step-by-Step Rollback Execution:
1. **Option A: Hot-Swap via Environment Variable (Zero Downtime in Docker Compose):**
   ```bash
   # Update environment variable in docker-compose or .env
   export EASYPASAL_VARIANT=baseline_v0.1
   export EASYPASAL_ACTIVE_CONFIG=configs/baseline_v0.1.yaml

   # Hot-restart the application container without tearing down network or collector
   docker compose up -d --no-deps app
   ```

2. **Option B: Emergency Traffic Routing Override:**
   If running deterministic traffic splitting (`traffic_split.enabled = true`), immediately route 100% of traffic to Variant A (`baseline_v0.1`):
   ```bash
   # Set variant B probability to 0.0
   export EASYPASAL_ROUTING_PROBABILITY_B=0.0
   ```

3. **Post-Rollback Verification:**
   ```bash
   # 1. Verify health reflects baseline_v0.1
   curl -s http://localhost:8000/health | jq .active_variant
   # Expected: "baseline_v0.1"

   # 2. Verify /ask response header and grounding
   curl -s -i -X POST http://localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"question": "Berapa modal disetor minimum pendirian BPR?", "language": "id"}' \
        | grep -i "x-easypasal-variant"
   # Expected: X-EasyPasal-Variant: baseline_v0.1
   ```

### 4.2 Docker Compose Stack Rollback
**Trigger Criteria:** Application container failure, fatal dependency collision, or unrecoverable error during rolling upgrade.

```bash
# 1. Roll back to previous image tag or stable release commit
git checkout v0.1

# 2. Recreate containers while preserving persistent volumes (collector_data, model_cache)
docker compose up -d --build

# 3. Verify inter-service health status
docker compose ps
```
*Note: SQLite WAL database in `collector_data` volume is completely preserved during container recreation.*

### 4.3 Hugging Face Space Demo Rollback (Plan §8.3 / T7.6)
**Trigger Criteria:** Space container crashes on boot or runtime error on the public Space (`robitalhazmi/easypasal`).

```bash
# 1. Identify previous stable commit
git log --oneline -n 5

# 2. Revert space deployment
cd dist/hf_space
git revert HEAD --no-edit
git push space main

# 3. Verify cold-start and HTTP 200 OK
uv run python scripts/deploy_hf_space.py \
    --space-id robitalhazmi/easypasal \
    --measure-cold-start \
    --timeout 120
```

### 4.4 Corrupted Index Recovery
**Trigger Criteria:** `AlreadyLocked` errors, corrupted Qdrant collection files, or missing vector directories.

```bash
# Instant zero-friction recovery from official Hugging Face Space repository:
make download-index
# Or force clean overwrite:
uv run python scripts/download_prebuilt_index.py --force
```

---

## 5. Disaster Recovery: Statutory Index Rebuild Procedures

If the local or remote index storage becomes unrecoverable or when new statutory instruments are added to `data/corpus_manifest.csv`, execute a clean index rebuild following the **Atomic Rebuild Protocol**.

### 5.1 Corpus Manifest Validation
Verify that all 21 canonical statutory instruments exist in `data/corpus_manifest.csv` and checksums match:
```bash
wc -l data/corpus_manifest.csv
# Expected: 22 lines (header + 21 statutory instruments)
```

### 5.2 Atomic Staging Rebuild Protocol
To prevent `portalocker.exceptions.AlreadyLocked` or runtime query interruptions during vector indexing, never overwrite `data/index/` directly while the application is reading from it. Always build into `data/index_staging/`:

```bash
# Step 1: Clean and prepare staging directory
rm -rf data/index_staging
mkdir -p data/index_staging

# Step 2: Build dense Qdrant collection & sparse BM25 index into staging
uv run python -m easypasal.index.build \
    --strategy structure_aware \
    --index-dir data/index_staging \
    --model-name intfloat/multilingual-e5-large \
    --batch-size 64

# Step 3: Validate staging index integrity
test -d data/index_staging/qdrant && test -d data/index_staging/bm25 && echo "Staging index verified."

# Step 4: Perform atomic directory swap
TIMESTAMP=$(date +%Y%m%d%H%M%S)
mv data/index "data/index_backup_${TIMESTAMP}"
mv data/index_staging data/index

# Step 5: Run CI smoke evaluation gate to ensure zero regression against gold baseline
make smoke-eval
```

---

## 6. Automated Rollback Rehearsal Drill (Task T7.8 / Acceptance Criteria)

### 6.1 Purpose & Frequency
In compliance with **Plan §9 / Task T7.8**, operational rollback procedures must be formally rehearsed and audited to guarantee zero manual guessing during live production incidents. Rehearsals must be executed:
- At least once prior to production releases (rehearsal completed and audited).
- After major prompt or model configuration upgrades (e.g. promoting `gepa_v0.2`).
- As part of monthly disaster recovery drills.

### 6.2 Executing the Rehearsal Harness
The automated rollback drill harness executes all phases end-to-end, validating status codes, response headers, and index availability:

```bash
# Execute via Makefile target
make rehearse-rollback

# Or invoke directly via Python CLI
uv run python scripts/rehearse_rollback.py \
    --from-variant gepa_v0.2 \
    --to-variant baseline_v0.1 \
    --provider mock \
    --output reports/rollback_rehearsal.json
```

### 6.3 5-Phase Rehearsal Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Pre-Rollback Health & Verification                            │
│ Query /health and /ask; assert active_variant == "gepa_v0.2"           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Simulated Incident Trigger                                    │
│ Synthesize critical regression alert (INC-SIM-20261018-01)             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Rollback Execution                                            │
│ Hot-swap active config to baseline_v0.1; measure swap latency (<20ms)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Post-Rollback Health & Verification                           │
│ Query /health and /ask; assert active_variant == "baseline_v0.1"       │
│ Assert header X-EasyPasal-Variant == "baseline_v0.1"                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Index Disaster Recovery & Manifest Verification               │
│ Check Qdrant vectors, BM25 postings, and corpus manifest integrity     │
└────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Rehearsal Audit Certificate
The automated drill generates a machine-verifiable audit certificate persisted to [`reports/rollback_rehearsal.json`](file:///home/robit/Documents/Project/easypasal/reports/rollback_rehearsal.json):

```json
{
  "timestamp": "2026-10-18T03:15:00.000000+00:00",
  "mode": "in-process",
  "method": "hot-swap",
  "from_variant": "gepa_v0.2",
  "to_variant": "baseline_v0.1",
  "provider": "mock",
  "phases": {
    "phase1_pre_rollback": {
      "status": "PASS",
      "health_status": "ok",
      "active_variant": "gepa_v0.2",
      "response_variant": "gepa_v0.2",
      "header_variant": "gepa_v0.2",
      "answer_length": 232,
      "citations_count": 1
    },
    "phase2_incident_trigger": {
      "incident_id": "INC-SIM-20261018-01",
      "type": "SIMULATED_REGRESSION_ALERT",
      "description": "Evidently detected prompt grounding drift and p95 latency degraded in gepa_v0.2.",
      "trigger_metric": "abstain_rate_spike",
      "severity": "CRITICAL",
      "remediation": "Initiate immediate rollback to baseline_v0.1."
    },
    "phase3_rollback_execution": {
      "status": "PASS",
      "target_variant": "baseline_v0.1",
      "rollback_duration_ms": 14.09,
      "method": "hot-swap"
    },
    "phase4_post_rollback": {
      "status": "PASS",
      "health_status": "ok",
      "active_variant": "baseline_v0.1",
      "response_variant": "baseline_v0.1",
      "header_variant": "baseline_v0.1",
      "answer_length": 232,
      "citations_count": 1
    },
    "phase5_index_recovery": {
      "status": "PASS",
      "index_dir": "data/index",
      "index_ready": true,
      "qdrant_present": true,
      "bm25_present": true,
      "remote_snapshot_repo": "robitalhazmi/easypasal",
      "rebuild_command": "make index",
      "download_command": "make download-index"
    }
  },
  "status": "SUCCESS",
  "total_rehearsal_duration_ms": 13770.99
}
```

---

## 7. Routine Maintenance & Database Housekeeping

### 7.1 SQLite WAL Checkpointing
To keep the SQLite database file compact and prevent memory bloat on low-resource VPS hosts:
```bash
# Checkpoint and truncate the write-ahead log file
sqlite3 data/sqlite/collector.db "PRAGMA wal_checkpoint(TRUNCATE);"
```

### 7.2 Log File Pruning
To prevent disk exhaustion over long-running deployments, rotate or compress raw JSONL logs:
```bash
# Rotate logs older than 30 days
gzip -c logs/requests.jsonl > "logs/requests_$(date +%Y%m%d).jsonl.gz"
> logs/requests.jsonl
```

### 7.3 Cache Cleanliness
```bash
# Clean temporary build and python test artifacts
make clean
```
