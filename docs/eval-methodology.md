# EasyPasal — Evaluation Methodology

## 1. Principles
Evaluation is the core gatekeeper of EasyPasal. No variant is promoted to release without quantified validation on the held-out test split.

## 2. Metric Taxonomy
- **Retrieval Metrics:** Recall@k, MRR, nDCG@k against labeled gold *Pasal* and *Ayat*.
- **Citation Metrics:** Precision and Recall of generated citation tags against source ground truth.
- **Generation Quality:** Faithfulness and factual correctness (evaluated via calibrated LLM judge and verified deterministically).
- **Abstention Accuracy:** Correctly returning abstention on unanswerable or out-of-scope inquiries.
- **Operational Metrics:** Latency (p50/p95), TTFT, throughput (tokens/sec), memory footprint.

## 3. Judge Calibration
*(Calibration statistics, Cohen's kappa, and human agreement rates will be documented in Phase 3 / T3.4.)*
