# EasyPasal — Operations Runbook

## 1. Local Deployment
```bash
make setup
make ingest
make index
make serve
```

## 2. Variant Rollback Procedure
1. Identify the previous stable variant configuration tag or commit hash.
2. Update `configs/` or `EASYPASAL_VARIANT` environment variable.
3. Restart application service.
4. Verify `/health` endpoint and run smoke test suite.

## 3. Index Rebuilding & Drift Recovery
1. Validate integrity of `data/corpus_manifest.csv`.
2. Execute `make ingest` to re-fetch raw PDFs if needed.
3. Execute `make index` to regenerate Qdrant vectors and BM25 postings.

## 4. Telemetry Monitoring, Threshold Alerting & Drill Procedures (Plan §9 / Task T7.5)

### 4.1 Monitored Thresholds & Boundaries
Operational telemetry is continuously evaluated against configured quality and performance ceilings:

| Metric | Warning Threshold | Critical Threshold | Rationale & Remediation |
|---|---|---|---|
| **Abstention Rate** (`abstain_rate`) | $> 35.0\%$ | $\ge 50.0\%$ | Spikes signal retrieval starvation, vector index failure, or adversarial prompt injection bursts. Inspect `/v1/logs/requests` for ungrounded queries. |
| **p95 Latency** (`p95_latency_ms`) | $> 5{,}000\text{ ms}$ | $\ge 10{,}000\text{ ms}$ | High latency degrades user experience. Check CPU saturation, worker pool exhaustion, or large prompt token counts. |
| **Low-Evidence Rate** (`low_evidence_rate`) | $> 30.0\%$ | $> 50.0\%$ | Measures fraction of queries with top-1 retrieval score $< 0.60$. High rates indicate user questions outside current statutory corpus scope. |
| **Minimum Evaluation Sample** | $N \ge 10$ requests | — | Prevents spurious cold-start alerts before statistical significance is achieved. |

### 4.2 Automated Alert Drill Procedure
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
# Open http://localhost:8080/dashboard or inspect the active alerts endpoint:
curl "http://localhost:8080/v1/alerts/active?lookback_minutes=60" \
     -H "Authorization: Bearer easypasal-vps-secret-token-v1"
```

### 4.3 Incident Triage & Resolution Runbook

#### A. Abstention Rate Spike (`abstain_rate > 35%`)
1. **Check Log Classification:** Query `GET /v1/logs/requests?abstained=true&limit=20`.
2. **Determine Root Cause:**
   - If `routed=true` and `theme="Out of Scope"`: User queries are genuine out-of-domain questions; normal behavior.
   - If `routed=false` and `retrieval_score_top1 < 0.60`: Vector store or embeddings failing to retrieve matching passages. Verify Qdrant collection status.
   - If `retrieval_score_top1 >= 0.75` but generator abstained: Prompt hallucination refusal or strict citation verification rejection. Check model generation logs.

#### B. Latency Spike (`p95_latency_ms > 5000ms`)
1. Check collector host memory via `/health` (`budget_mb = 90.0`).
2. Verify CPU core utilization.
3. For GPU vLLM deployments, check GPU VRAM allocation and PagedAttention queue backlog.

## 5. Hugging Face Space Demo Deployment (Plan §8.3 / Task T7.6)

### 5.1 Architecture & Resource Footprint
The interactive web demo is containerized as a standalone Hugging Face Docker Space matching the free CPU Basic tier:
- **vCPU:** 2 vCPUs
- **System Memory:** 16 GB RAM
- **Container Port:** 7860 (standard Hugging Face Space port)
- **Base Image:** `python:3.12-slim`
- **Prebuilt Index:** Pre-baked Qdrant dense vector store and BM25 sparse index (`data/index/`) bundled directly into the container image to eliminate cold-start indexing overhead.

### 5.2 Deployment Procedures

#### Option A: Automated CLI Deployment via Token
```bash
# Set your Hugging Face Access Token (write permission)
export HF_TOKEN="hf_..."

# Deploy to target space (creates space if non-existent)
uv run python scripts/deploy_hf_space.py --space-id <hf-username>/easypasal
```

#### Option B: Git Remote Deployment
```bash
# 1. Build and stage the Space bundle
make space-bundle

# 2. Add HF Space as a git remote
git remote add space https://huggingface.co/spaces/<hf-username>/easypasal

# 3. Push staged assets to HF Space
cd dist/hf_space
git init
git add .
git commit -m "deploy: update EasyPasal Space demo"
git push --force space main
```

### 5.3 Cold-Start Latency Measurement
Under the free Hugging Face Space tier, inactive spaces enter a dormant sleep state after 48 hours of inactivity.
Measure the end-to-end wake-up and cold-start latency using the deployment measurement tool:

```bash
# Measure cold start until HTTP 200 OK
uv run python scripts/deploy_hf_space.py \
    --space-id <hf-username>/easypasal \
    --measure-cold-start \
    --timeout 300
```
- **Observed Cold Start Target:** $\le 45\text{ seconds}$ from dormant state to responsive Gradio interface, achieved by eliminating on-the-fly index downloads or runtime model downloads.
