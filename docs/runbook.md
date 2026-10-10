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
