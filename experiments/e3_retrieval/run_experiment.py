"""Experiment E3: Retrieval Pipeline Architecture (T4.3).

Benchmarks four retrieval configurations on canonical structure_aware chunks and Multilingual-E5-Large embeddings:
  - Variant 1 (dense): Pure Dense Semantic Search (intfloat/multilingual-e5-large, 1024-d)
  - Variant 2 (bm25): Pure BM25 Sparse Search (Indonesian legal tokenization)
  - Variant 3 (hybrid): Hybrid RRF Fusion (Dense + BM25, k_rrf=60)
  - Variant 4 (hybrid_rerank): Hybrid RRF (pool N=15) + Multilingual Cross-Encoder (ms-marco-MultiBERT-L-12)

Evaluates on dev split, computes 95% bootstrap CIs and paired permutation tests,
validates on held-out test split, and logs operational trade-offs (Recall@k, MRR, nDCG, p95 latency).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

from eval.compare_variants import compare_retrieval_runs
from eval.metrics.bootstrap import bootstrap_ci, paired_bootstrap_comparison
from eval.run_eval import EvalReport, evaluate_retrieval
from eval.split_eval import load_gold_items

from easypasal.retrieval.reranker import DEFAULT_RERANKER_MODEL, FlashRankReranker
from easypasal.retrieval.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments.e3_retrieval")

CANONICAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


class PipelineVariant(TypedDict):
    id: str
    name: str
    mode: str
    rerank: bool
    description: str


VARIANTS: list[PipelineVariant] = [
    {
        "id": "dense",
        "name": "Pure Dense (E5-Large)",
        "mode": "dense",
        "rerank": False,
        "description": "Pure dense vector retrieval using 1024-dim Multilingual-E5-Large with asymmetric prefixes",
    },
    {
        "id": "bm25",
        "name": "Pure BM25 (Indonesian Tokenizer)",
        "mode": "bm25",
        "rerank": False,
        "description": "Sparse inverted index lexical matching with Indonesian legal tokenization",
    },
    {
        "id": "hybrid",
        "name": "Hybrid RRF (Dense + BM25)",
        "mode": "hybrid",
        "rerank": False,
        "description": "Reciprocal Rank Fusion (k_rrf=60) combining Multilingual-E5-Large and BM25",
    },
    {
        "id": "hybrid_rerank",
        "name": "Hybrid + Reranker (MiniLM-L12)",
        "mode": "hybrid+rerank",
        "rerank": True,
        "description": "Hybrid RRF candidate pool (N=15) refined by 12-layer quantized cross-encoder reranker",
    },
]


def run_variant_eval(
    variant: PipelineVariant,
    dataset_path: Path | str,
    index_dir: Path | str = "data/index",
    strategy: str = "structure_aware",
    model_name: str = CANONICAL_EMBEDDING_MODEL,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    top_k: int = 10,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    show_progress: bool = True,
) -> EvalReport:
    """Run retrieval evaluation for a single retrieval pipeline variant."""
    d_path = Path(dataset_path)
    items = load_gold_items(d_path)

    reranker = FlashRankReranker(model_name=reranker_model) if variant["rerank"] else None

    retriever = HybridRetriever.from_paths(
        index_dir=index_dir,
        strategy=strategy,
        model_name=model_name,
        reranker=reranker,
    )

    try:
        report = evaluate_retrieval(
            retriever=retriever,
            items=items,
            top_k=top_k,
            k_values=k_values,
            mode=variant["mode"],
            dataset_path=str(d_path),
            strategy=strategy,
            show_progress=show_progress,
        )
    finally:
        retriever.close()

    return report


def format_e3_results_table(
    results: dict[str, EvalReport],
    baseline_key: str = "dense",
) -> str:
    """Format structured comparison table for Experiment E3."""
    baseline_report = results.get(baseline_key)
    lines: list[str] = [
        "=" * 125,
        "                     📊 EasyPasal Experiment E3: Retrieval Pipeline Architecture Results                  ",
        "=" * 125,
        f"{'Pipeline Variant':<32} | {'Recall@5':<10} | {'95% CI':<19} | {'MRR':<8} | {'nDCG@5':<8} | {'p95 (ms)':<9} | {'Mean (ms)':<10} | {'Sig vs Dense':<14}",
        "-" * 125,
    ]

    for v in VARIANTS:
        vid = v["id"]
        report = results.get(vid)
        if not report:
            continue

        r5 = report.mean_recall_at_k.get(5, 0.0)
        mrr = report.mean_mrr_at_k.get(5, 0.0)
        ndcg5 = report.mean_ndcg_at_k.get(5, 0.0)
        p95 = report.p95_latency_ms
        mean_lat = report.mean_latency_ms

        ans_results = [it for it in report.item_results if len(it.gold_chunks) > 0]
        r5_vals = [it.metrics.recall_at_k.get(5, 0.0) for it in ans_results]
        ci = bootstrap_ci(r5_vals, n_resamples=2000, confidence=0.95, seed=42)
        ci_str = f"[{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]"

        if vid == baseline_key or baseline_report is None:
            sig_badge = "Reference"
        else:
            base_ans = [it for it in baseline_report.item_results if len(it.gold_chunks) > 0]
            base_r5 = [it.metrics.recall_at_k.get(5, 0.0) for it in base_ans]
            paired = paired_bootstrap_comparison(
                baseline=base_r5,
                candidate=r5_vals,
                metric_name="recall_at_5",
                n_resamples=2000,
                n_permutations=5000,
                confidence=0.95,
                seed=42,
            )
            sig_badge = f"p={paired.p_value:.4f}"

        lines.append(
            f"{v['name']:<32} | {r5:<10.4f} | {ci_str:<19} | {mrr:<8.4f} | {ndcg5:<8.4f} | {p95:<9.1f} | {mean_lat:<10.1f} | {sig_badge:<14}"
        )

    lines.append("=" * 125)
    return "\n".join(lines)


def run_experiment_e3(
    split: str = "both",
    output_dir: Path | str = "data/eval_outputs",
    index_dir: Path | str = "data/index",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end Experiment E3 across dev and held-out test splits."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits_to_run = ["dev"] if split == "dev" else ["test"] if split == "test" else ["dev", "test"]
    all_summary: dict[str, Any] = {}

    for s in splits_to_run:
        dataset_file = f"data/gold/{s}.jsonl"
        logger.info("==================================================")
        logger.info("Executing Experiment E3 on split: %s (%s)", s, dataset_file)
        logger.info("==================================================")

        split_results: dict[str, EvalReport] = {}

        for v in VARIANTS:
            vid = v["id"]
            logger.info(
                "Evaluating variant: %s (mode=%s, rerank=%s)...", v["name"], v["mode"], v["rerank"]
            )
            rep = run_variant_eval(
                variant=v,
                dataset_path=dataset_file,
                index_dir=index_dir,
                strategy="structure_aware",
                model_name=CANONICAL_EMBEDDING_MODEL,
                show_progress=show_progress,
            )
            split_results[vid] = rep

            out_file = out_path / f"e3_{vid}_{s}.json"
            rep.save_json(out_file)
            logger.info("Saved report for %s to %s", vid, out_file)

        # Print comparison table
        table_str = format_e3_results_table(split_results, baseline_key="dense")
        print(f"\n--- Split: {s.upper()} ---")
        print(table_str)

        # Statistical paired comparisons against Dense baseline
        comparisons: dict[str, Any] = {}
        for cand_vid in ["bm25", "hybrid", "hybrid_rerank"]:
            if cand_vid in split_results and "dense" in split_results:
                comp = compare_retrieval_runs(
                    baseline_data=split_results["dense"].to_dict(),
                    candidate_data=split_results[cand_vid].to_dict(),
                    baseline_path="dense",
                    candidate_path=cand_vid,
                )
                comp_file = out_path / f"e3_comp_{cand_vid}_vs_dense_{s}.json"
                comp_file.write_text(json.dumps(comp.to_dict(), indent=2), encoding="utf-8")
                comparisons[f"{cand_vid}_vs_dense"] = comp.to_dict()

        # Also compare hybrid_rerank vs hybrid directly
        if "hybrid_rerank" in split_results and "hybrid" in split_results:
            comp_rerank = compare_retrieval_runs(
                baseline_data=split_results["hybrid"].to_dict(),
                candidate_data=split_results["hybrid_rerank"].to_dict(),
                baseline_path="hybrid",
                candidate_path="hybrid_rerank",
            )
            comp_rerank_file = out_path / f"e3_comp_hybrid_rerank_vs_hybrid_{s}.json"
            comp_rerank_file.write_text(
                json.dumps(comp_rerank.to_dict(), indent=2), encoding="utf-8"
            )
            comparisons["hybrid_rerank_vs_hybrid"] = comp_rerank.to_dict()

        all_summary[s] = {
            "variants": {
                vid: rep.to_dict()["overall_metrics"] for vid, rep in split_results.items()
            },
            "comparisons": comparisons,
        }

    summary_file = out_path / "e3_retrieval_summary.json"
    summary_file.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    logger.info("Experiment E3 complete! Summary written to %s", summary_file)
    return all_summary


def main() -> None:
    """CLI entrypoint for running Experiment E3."""
    parser = argparse.ArgumentParser(
        description="EasyPasal Experiment E3: Retrieval Pipeline Architecture (Dense / BM25 / Hybrid / Rerank)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="both",
        help="Dataset split to evaluate (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval_outputs",
        help="Directory to save evaluation reports and summaries",
    )
    parser.add_argument(
        "--index-dir",
        default="data/index",
        help="Base index directory containing qdrant and bm25 indices",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar during evaluation",
    )

    args = parser.parse_args()

    run_experiment_e3(
        split=args.split,
        output_dir=args.output_dir,
        index_dir=args.index_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
