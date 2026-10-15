"""Offline evaluation runner: benchmarks retrieval against gold datasets, prints tables, and logs to MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from eval.labeling_helper import GoldItem
from eval.metrics.retrieval import RetrievalMetrics, compute_retrieval_metrics
from eval.split_eval import load_gold_items

# Ensure MLflow allows file store if used and silences agent hints
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"
logger = logging.getLogger("eval.run_eval")


@dataclass
class ItemEvalResult:
    """Individual item retrieval evaluation result."""

    id: str
    category: str
    theme: str
    question: str
    should_abstain: bool
    gold_chunks: list[str]
    retrieved_chunks: list[str]
    metrics: RetrievalMetrics
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "theme": self.theme,
            "question": self.question,
            "should_abstain": self.should_abstain,
            "gold_chunks": self.gold_chunks,
            "retrieved_chunks": self.retrieved_chunks,
            "metrics": self.metrics.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class CategoryEvalSummary:
    """Aggregated retrieval metrics for a single evaluation category."""

    category: str
    total_items: int
    answerable_items: int
    recall_at_k: dict[int, float]
    hit_at_k: dict[int, float]
    mrr_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert category summary to serializable dictionary."""
        return {
            "category": self.category,
            "total_items": self.total_items,
            "answerable_items": self.answerable_items,
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "hit_at_k": {str(k): round(v, 4) for k, v in self.hit_at_k.items()},
            "mrr_at_k": {str(k): round(v, 4) for k, v in self.mrr_at_k.items()},
            "ndcg_at_k": {str(k): round(v, 4) for k, v in self.ndcg_at_k.items()},
        }


@dataclass
class EvalReport:
    """Comprehensive offline evaluation report."""

    dataset_path: str
    total_items: int
    answerable_items: int
    unanswerable_items: int
    top_k: int
    k_values: list[int]
    mean_recall_at_k: dict[int, float]
    mean_hit_at_k: dict[int, float]
    mean_precision_at_k: dict[int, float]
    mean_mrr_at_k: dict[int, float]
    mean_ndcg_at_k: dict[int, float]
    mean_latency_ms: float
    p95_latency_ms: float
    categories: dict[str, CategoryEvalSummary]
    item_results: list[ItemEvalResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def gate_recall_pass(self) -> bool:
        """Recall@5 >= 80% acceptance gate."""
        return self.mean_recall_at_k.get(5, 0.0) >= 0.80

    @property
    def gate_mrr_pass(self) -> bool:
        """MRR (at cut-off or top-k) >= 0.65 acceptance gate."""
        mrr_val = self.mean_mrr_at_k.get(self.top_k) or self.mean_mrr_at_k.get(5, 0.0)
        return mrr_val >= 0.65

    @property
    def gate_overall_pass(self) -> bool:
        """Whether all deterministic acceptance gates are met."""
        return self.gate_recall_pass and self.gate_mrr_pass

    def to_dict(self) -> dict[str, Any]:
        """Convert complete evaluation report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "dataset_path": self.dataset_path,
            "total_items": self.total_items,
            "answerable_items": self.answerable_items,
            "unanswerable_items": self.unanswerable_items,
            "top_k": self.top_k,
            "k_values": self.k_values,
            "gates": {
                "recall_at_5": {
                    "observed": round(self.mean_recall_at_k.get(5, 0.0), 4),
                    "threshold": 0.80,
                    "passed": self.gate_recall_pass,
                },
                "mrr": {
                    "observed": round(
                        self.mean_mrr_at_k.get(self.top_k) or self.mean_mrr_at_k.get(5, 0.0), 4
                    ),
                    "threshold": 0.65,
                    "passed": self.gate_mrr_pass,
                },
                "overall_passed": self.gate_overall_pass,
            },
            "overall_metrics": {
                "recall_at_k": {str(k): round(v, 4) for k, v in self.mean_recall_at_k.items()},
                "hit_at_k": {str(k): round(v, 4) for k, v in self.mean_hit_at_k.items()},
                "precision_at_k": {
                    str(k): round(v, 4) for k, v in self.mean_precision_at_k.items()
                },
                "mrr_at_k": {str(k): round(v, 4) for k, v in self.mean_mrr_at_k.items()},
                "ndcg_at_k": {str(k): round(v, 4) for k, v in self.mean_ndcg_at_k.items()},
                "latency_mean_ms": round(self.mean_latency_ms, 2),
                "latency_p95_ms": round(self.p95_latency_ms, 2),
            },
            "categories": {cat: sum_obj.to_dict() for cat, sum_obj in self.categories.items()},
            "item_results": [it.to_dict() for it in self.item_results],
        }

    def save_json(self, output_path: Path | str) -> Path:
        """Save report to JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return target


def evaluate_retrieval(
    retriever: Any,
    items: Sequence[GoldItem],
    top_k: int = 10,
    k_values: Sequence[int] = (1, 3, 5, 10),
    mode: str = "hybrid",
    dataset_path: str = "data/gold/dev.jsonl",
    strategy: str = "structure_aware",
    chunk_mapper: Any | None = None,
    show_progress: bool = True,
) -> EvalReport:
    """Run retrieval evaluation over gold items and compute aggregate metrics.

    Args:
        retriever: HybridRetriever or compatible duck-typed retriever.
        items: Sequence of GoldItem objects.
        top_k: Top-k cut-off for retriever.
        k_values: Depths to compute metrics for.
        mode: Retrieval mode ('hybrid', 'dense', or 'bm25').
        dataset_path: Identifier path for the dataset being evaluated.
        strategy: Chunking strategy being evaluated.
        chunk_mapper: Optional ChunkSpanMapper to normalize non-canonical chunk IDs.
        show_progress: Whether to show a tqdm progress bar during evaluation.

    Returns:
        Populated EvalReport dataclass.
    """
    sorted_k = sorted(k_values)
    item_results: list[ItemEvalResult] = []
    latencies: list[float] = []

    cat_items: dict[str, list[ItemEvalResult]] = defaultdict(list)

    iterable = (
        tqdm(items, desc=f"Evaluating ({mode} | {strategy})", unit="query")
        if show_progress
        else items
    )
    for item in iterable:
        start_t = time.perf_counter()
        retrieval_res = retriever.retrieve(query=item.question, top_k=top_k, mode=mode)
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(latency_ms)

        retrieved_ids = [c.chunk_id for c in retrieval_res.chunks]
        gold_chunks = item.relevant_chunk_ids

        # If a chunk mapper is provided (e.g. for fixed-size evaluation), map IDs to canonical targets
        if chunk_mapper is not None:
            eval_retrieved_ids = chunk_mapper.map_retrieved_chunk_ids(
                retrieved_ids, strategy=strategy
            )
        else:
            eval_retrieved_ids = retrieved_ids

        metrics = compute_retrieval_metrics(
            retrieved=eval_retrieved_ids,
            relevant=gold_chunks,
            k_values=sorted_k,
        )

        item_result = ItemEvalResult(
            id=item.id,
            category=item.category,
            theme=item.theme,
            question=item.question,
            should_abstain=item.should_abstain,
            gold_chunks=gold_chunks,
            retrieved_chunks=retrieved_ids,
            metrics=metrics,
            latency_ms=latency_ms,
        )
        item_results.append(item_result)
        cat_items[item.category].append(item_result)

    # Retrieval metrics are evaluated over answerable queries (ground-truth chunks > 0)
    answerable_results = [it for it in item_results if len(it.gold_chunks) > 0]
    total_ans = len(answerable_results)
    total_unans = len(item_results) - total_ans

    mean_recalls: dict[int, float] = {}
    mean_hits: dict[int, float] = {}
    mean_precisions: dict[int, float] = {}
    mean_mrrs: dict[int, float] = {}
    mean_ndcgs: dict[int, float] = {}

    for k in sorted_k:
        if total_ans > 0:
            mean_recalls[k] = (
                sum(it.metrics.recall_at_k[k] for it in answerable_results) / total_ans
            )
            mean_hits[k] = sum(it.metrics.hit_at_k[k] for it in answerable_results) / total_ans
            mean_precisions[k] = (
                sum(it.metrics.precision_at_k[k] for it in answerable_results) / total_ans
            )
            mean_mrrs[k] = sum(it.metrics.mrr_at_k[k] for it in answerable_results) / total_ans
            mean_ndcgs[k] = sum(it.metrics.ndcg_at_k[k] for it in answerable_results) / total_ans
        else:
            mean_recalls[k] = 0.0
            mean_hits[k] = 0.0
            mean_precisions[k] = 0.0
            mean_mrrs[k] = 0.0
            mean_ndcgs[k] = 0.0

    # Per-category summary
    categories_summary: dict[str, CategoryEvalSummary] = {}
    for cat, c_results in sorted(cat_items.items()):
        c_ans = [it for it in c_results if len(it.gold_chunks) > 0]
        n_c_ans = len(c_ans)

        c_rec: dict[int, float] = {}
        c_hit: dict[int, float] = {}
        c_mrr: dict[int, float] = {}
        c_ndcg: dict[int, float] = {}

        for k in sorted_k:
            if n_c_ans > 0:
                c_rec[k] = sum(it.metrics.recall_at_k[k] for it in c_ans) / n_c_ans
                c_hit[k] = sum(it.metrics.hit_at_k[k] for it in c_ans) / n_c_ans
                c_mrr[k] = sum(it.metrics.mrr_at_k[k] for it in c_ans) / n_c_ans
                c_ndcg[k] = sum(it.metrics.ndcg_at_k[k] for it in c_ans) / n_c_ans
            else:
                c_rec[k] = 0.0
                c_hit[k] = 0.0
                c_mrr[k] = 0.0
                c_ndcg[k] = 0.0

        categories_summary[cat] = CategoryEvalSummary(
            category=cat,
            total_items=len(c_results),
            answerable_items=n_c_ans,
            recall_at_k=c_rec,
            hit_at_k=c_hit,
            mrr_at_k=c_mrr,
            ndcg_at_k=c_ndcg,
        )

    # Latency statistics
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p95_idx = int(len(sorted_lat) * 0.95)
    p95_lat = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0.0

    return EvalReport(
        dataset_path=dataset_path,
        total_items=len(items),
        answerable_items=total_ans,
        unanswerable_items=total_unans,
        top_k=top_k,
        k_values=list(sorted_k),
        mean_recall_at_k=mean_recalls,
        mean_hit_at_k=mean_hits,
        mean_precision_at_k=mean_precisions,
        mean_mrr_at_k=mean_mrrs,
        mean_ndcg_at_k=mean_ndcgs,
        mean_latency_ms=mean_lat,
        p95_latency_ms=p95_lat,
        categories=categories_summary,
        item_results=item_results,
    )


def print_eval_report(report: EvalReport) -> None:
    """Render clean, structured terminal tables for evaluation metrics."""
    print("=" * 80)
    print("📊 EasyPasal Retrieval Evaluation Report (T3.2)")
    print("=" * 80)
    print(f"Dataset Split      : {report.dataset_path}")
    print(
        f"Total Inquiries    : {report.total_items} items "
        f"({report.answerable_items} answerable, {report.unanswerable_items} out-of-scope/adversarial)"
    )
    print(
        f"Retrieval Latency  : Mean: {report.mean_latency_ms:.1f}ms | p95: {report.p95_latency_ms:.1f}ms"
    )
    print()

    # Overall Summary Table
    print(f"{'Retrieval Metric':<24} | {'Observed':<10} | {'Target Gate':<14} | {'Status':<8}")
    print("-" * 80)

    def badge(passed: bool) -> str:
        return "✅ PASS" if passed else "❌ FAIL"

    rec_5 = report.mean_recall_at_k.get(5, 0.0)
    mrr_top = report.mean_mrr_at_k.get(report.top_k) or report.mean_mrr_at_k.get(5, 0.0)

    print(f"{'Recall@1':<24} | {report.mean_recall_at_k.get(1, 0.0):>8.1%}  | {'-':<14} | {'-':<8}")
    print(f"{'Recall@3':<24} | {report.mean_recall_at_k.get(3, 0.0):>8.1%}  | {'-':<14} | {'-':<8}")
    print(
        f"{'Recall@5':<24} | {rec_5:>8.1%}  | {'>= 80.0%':<14} | {badge(report.gate_recall_pass)}"
    )
    print(
        f"{'Recall@10':<24} | {report.mean_recall_at_k.get(10, 0.0):>8.1%}  | {'-':<14} | {'-':<8}"
    )
    print(f"{'-' * 24} | {'-' * 10} | {'-' * 14} | {'-' * 8}")
    print(f"{'Hit@1':<24} | {report.mean_hit_at_k.get(1, 0.0):>8.1%}  | {'-':<14} | {'-':<8}")
    print(f"{'Hit@3':<24} | {report.mean_hit_at_k.get(3, 0.0):>8.1%}  | {'-':<14} | {'-':<8}")
    print(f"{'Hit@5':<24} | {report.mean_hit_at_k.get(5, 0.0):>8.1%}  | {'-':<14} | {'-':<8}")
    print(f"{'Hit@10':<24} | {report.mean_hit_at_k.get(10, 0.0):>8.1%}  | {'-':<14} | {'-':<8}")
    print(f"{'-' * 24} | {'-' * 10} | {'-' * 14} | {'-' * 8}")
    print(f"{'MRR':<24} | {mrr_top:>8.3f}  | {'>= 0.650':<14} | {badge(report.gate_mrr_pass)}")
    print(f"{'nDCG@5':<24} | {report.mean_ndcg_at_k.get(5, 0.0):>8.3f}  | {'-':<14} | {'-':<8}")
    print(f"{'nDCG@10':<24} | {report.mean_ndcg_at_k.get(10, 0.0):>8.3f}  | {'-':<14} | {'-':<8}")
    print("-" * 80)
    print()

    # Per-Category Breakdown Table
    print("📁 Per-Category Retrieval Breakdown:")
    print(
        f"{'Category':<28} | {'Items':<5} | {'Ans':<4} | {'Rec@5':<8} | {'Hit@5':<8} | {'MRR':<7} | {'nDCG@5':<7}"
    )
    print("-" * 80)
    for cat, c in sorted(report.categories.items()):
        rec5_str = f"{c.recall_at_k.get(5, 0.0):>6.1%}" if c.answerable_items > 0 else "  N/A"
        hit5_str = f"{c.hit_at_k.get(5, 0.0):>6.1%}" if c.answerable_items > 0 else "  N/A"
        mrr_str = (
            f"{c.mrr_at_k.get(report.top_k, 0.0):>6.3f}" if c.answerable_items > 0 else "  N/A"
        )
        ndcg_str = f"{c.ndcg_at_k.get(5, 0.0):>6.3f}" if c.answerable_items > 0 else "  N/A"

        print(
            f"{cat:<28} | {c.total_items:>5} | {c.answerable_items:>4} | {rec5_str:<8} | {hit5_str:<8} | {mrr_str:<7} | {ndcg_str:<7}"
        )
    print("-" * 80)
    print()

    if report.gate_overall_pass:
        print("🎉 ALL RETRIEVAL ACCEPTANCE GATES MET! System meets deployment thresholds.")
    else:
        print("⚠️ SOME RETRIEVAL METRICS BELOW THRESHOLD. See details above.")
    print("=" * 80)


def log_to_mlflow(
    report: EvalReport,
    experiment_name: str = "easypasal-retrieval-eval",
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
) -> str | None:
    """Log evaluation report metrics, parameters, and artifact to MLflow.

    Args:
        report: EvalReport object to record.
        experiment_name: MLflow experiment name.
        run_name: Optional human-readable run name.
        params: Optional extra parameters (e.g. strategy, mode, k_rrf).
        tracking_uri: Optional MLflow tracking URI (defaults to file:./mlruns).

    Returns:
        MLflow run_id if logging succeeded, else None.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow is not installed. Skipping MLflow run logging.")
        return None

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    elif not mlflow.get_tracking_uri():
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

    mlflow.set_experiment(experiment_name)

    now_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    computed_run_name = run_name or f"eval_{report.top_k}k_{now_str}"

    with mlflow.start_run(run_name=computed_run_name) as run:
        # 1. Log Parameters
        all_params: dict[str, Any] = {
            "dataset_path": report.dataset_path,
            "total_items": report.total_items,
            "answerable_items": report.answerable_items,
            "unanswerable_items": report.unanswerable_items,
            "top_k": report.top_k,
        }
        if params:
            all_params.update(params)
        mlflow.log_params(all_params)

        # 2. Log Overall Metrics
        metrics_dict: dict[str, float] = {
            "latency_mean_ms": report.mean_latency_ms,
            "latency_p95_ms": report.p95_latency_ms,
            "gate_recall_pass": 1.0 if report.gate_recall_pass else 0.0,
            "gate_mrr_pass": 1.0 if report.gate_mrr_pass else 0.0,
            "gate_overall_pass": 1.0 if report.gate_overall_pass else 0.0,
        }
        for k in report.k_values:
            metrics_dict[f"recall_at_{k}"] = report.mean_recall_at_k.get(k, 0.0)
            metrics_dict[f"hit_at_{k}"] = report.mean_hit_at_k.get(k, 0.0)
            metrics_dict[f"precision_at_{k}"] = report.mean_precision_at_k.get(k, 0.0)
            metrics_dict[f"mrr_at_{k}"] = report.mean_mrr_at_k.get(k, 0.0)
            metrics_dict[f"ndcg_at_{k}"] = report.mean_ndcg_at_k.get(k, 0.0)

        # 3. Log Per-Category Metrics
        for cat, c in report.categories.items():
            safe_cat = cat.replace("/", "_").replace(" ", "_")
            metrics_dict[f"cat_{safe_cat}_recall_at_5"] = c.recall_at_k.get(5, 0.0)
            metrics_dict[f"cat_{safe_cat}_hit_at_5"] = c.hit_at_k.get(5, 0.0)
            metrics_dict[f"cat_{safe_cat}_mrr"] = c.mrr_at_k.get(report.top_k, 0.0)
            metrics_dict[f"cat_{safe_cat}_ndcg_at_5"] = c.ndcg_at_k.get(5, 0.0)

        mlflow.log_metrics(metrics_dict)

        # 4. Log Detailed Report Artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp_f:
            json.dump(report.to_dict(), tmp_f, indent=2, ensure_ascii=False)
            tmp_path = tmp_f.name

        try:
            mlflow.log_artifact(tmp_path, artifact_path="reports")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        logger.info("Successfully logged evaluation run to MLflow (run_id=%s)", run.info.run_id)
        return str(run.info.run_id)


def main() -> None:
    """CLI entrypoint for offline retrieval evaluation."""
    parser = argparse.ArgumentParser(description="EasyPasal Offline Retrieval Evaluation Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/gold/dev.jsonl",
        help="Path to evaluation dataset JSONL (default: data/gold/dev.jsonl)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Path to index directory (default: data/index)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="structure_aware",
        choices=["structure_aware", "structure_aware_no_header", "fixed_size"],
        help="Chunking strategy to evaluate (default: structure_aware)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "dense", "bm25", "hybrid+rerank", "hybrid_rerank"],
        help="Retrieval search mode (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated k values for metrics (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="easypasal-retrieval-eval",
        help="MLflow experiment name (default: easypasal-retrieval-eval)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom MLflow run name",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name (default: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full evaluation JSON report",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow run tracking",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar during evaluation",
    )

    args = parser.parse_args()

    # 1. Parse k values
    k_vals = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    # 2. Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {dataset_path}")
    items = load_gold_items(dataset_path)

    # 3. Initialize retriever
    from easypasal.retrieval.retriever import HybridRetriever

    should_rerank = args.mode in ("hybrid+rerank", "hybrid_rerank")
    retriever = HybridRetriever.from_paths(
        index_dir=args.index_dir,
        strategy=args.strategy,
        model_name=args.model_name,
        rerank=should_rerank,
    )

    # 4. Optional chunk mapper for non-canonical chunking (e.g. fixed_size)
    mapper = None
    if args.strategy == "fixed_size":
        from eval.chunk_mapper import ChunkSpanMapper

        mapper = ChunkSpanMapper.load_or_build()

    # 5. Run evaluation
    report = evaluate_retrieval(
        retriever=retriever,
        items=items,
        top_k=args.top_k,
        k_values=k_vals,
        mode=args.mode,
        dataset_path=str(dataset_path),
        strategy=args.strategy,
        chunk_mapper=mapper,
        show_progress=not args.no_progress,
    )

    # 5. Print terminal report
    print_eval_report(report)

    # 6. Save JSON report if requested
    if args.output_json:
        saved_path = report.save_json(args.output_json)
        print(f"📄 Detailed report saved to: {saved_path}")

    # 7. Log to MLflow
    if not args.no_mlflow:
        run_id = log_to_mlflow(
            report=report,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            params={
                "strategy": args.strategy,
                "mode": args.mode,
                "top_k": args.top_k,
                "model_name": args.model_name,
            },
        )
        if run_id:
            print(f"📈 MLflow Run logged: ID={run_id} (experiment: {args.experiment_name})")


if __name__ == "__main__":
    main()
