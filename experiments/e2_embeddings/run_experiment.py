"""Experiment E2: Multilingual Embedding Models (T4.2).

Benchmarks three multilingual dense embedding candidates on canonical structure_aware chunks:
  - Variant 1 (minilm_384): sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim, ~118M params)
  - Variant 2 (mpnet_768): sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768-dim, ~278M params)
  - Variant 3 (e5_large_1024): intfloat/multilingual-e5-large (1024-dim, ~560M params, asymmetric prefixes)

Evaluates dense retrieval on dev split, computes 95% bootstrap CIs and paired permutation tests,
validates top candidates on held-out test split, and logs operational trade-offs (latency, index size).
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

from easypasal.index.embeddings import DEFAULT_EMBEDDING_MODEL, get_model_slug
from easypasal.retrieval.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments.e2_embeddings")


class EmbeddingVariant(TypedDict):
    id: str
    name: str
    model_name: str
    dimension: int
    params: str
    description: str


VARIANTS: list[EmbeddingVariant] = [
    {
        "id": "minilm_384",
        "name": "MiniLM-L12-v2 (384-d)",
        "model_name": DEFAULT_EMBEDDING_MODEL,
        "dimension": 384,
        "params": "~118M",
        "description": "Lightweight 384-dim multilingual sentence transformer baseline (fast, low memory)",
    },
    {
        "id": "mpnet_768",
        "name": "MPNet-Base-v2 (768-d)",
        "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dimension": 768,
        "params": "~278M",
        "description": "Medium capacity 768-dim multilingual transformer combining masked and permuted LM",
    },
    {
        "id": "e5_large_1024",
        "name": "Multilingual-E5-Large (1024-d)",
        "model_name": "intfloat/multilingual-e5-large",
        "dimension": 1024,
        "params": "~560M",
        "description": "High capacity 1024-dim multilingual model with asymmetric instruction prefixes",
    },
]


def get_collection_size_mb(index_dir: Path | str, collection_name: str) -> float:
    """Measure total disk usage of a Qdrant collection in Megabytes."""
    col_path = Path(index_dir) / "qdrant" / "collection" / collection_name
    if not col_path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in col_path.rglob("*") if f.is_file())
    return round(total_bytes / (1024.0 * 1024.0), 2)


def run_variant_eval(
    model_name: str,
    dataset_path: Path | str,
    index_dir: Path | str = "data/index",
    strategy: str = "structure_aware",
    top_k: int = 10,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    mode: str = "dense",
    show_progress: bool = True,
) -> EvalReport:
    """Run retrieval evaluation for a single embedding model variant."""
    d_path = Path(dataset_path)
    items = load_gold_items(d_path)

    retriever = HybridRetriever.from_paths(
        index_dir=index_dir,
        strategy=strategy,
        model_name=model_name,
    )

    report = evaluate_retrieval(
        retriever=retriever,
        items=items,
        top_k=top_k,
        k_values=k_values,
        mode=mode,
        dataset_path=str(d_path),
        strategy=strategy,
        show_progress=show_progress,
    )
    return report


def format_e2_results_table(
    results: dict[str, EvalReport],
    index_dir: Path | str = "data/index",
    strategy: str = "structure_aware",
    baseline_key: str = "minilm_384",
) -> str:
    """Format structured comparison table for Experiment E2."""
    baseline_report = results.get(baseline_key)
    lines: list[str] = [
        "=" * 125,
        "                     📊 EasyPasal Experiment E2: Multilingual Embedding Models Results                    ",
        "=" * 125,
        f"{'Model Variant':<28} | {'Dim':<5} | {'Recall@5':<10} | {'95% CI':<19} | {'MRR':<8} | {'nDCG@5':<8} | {'p95 (ms)':<9} | {'Disk (MB)':<9} | {'Sig vs MiniLM':<14}",
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

        model_name = v["model_name"]
        if model_name == DEFAULT_EMBEDDING_MODEL:
            target_col = f"easypasal_{strategy}"
        else:
            slug = get_model_slug(model_name)
            target_col = f"easypasal_{strategy}_{slug}"

        disk_mb = get_collection_size_mb(index_dir, target_col)

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
            f"{v['name']:<28} | {v['dimension']:<5} | {r5:<10.4f} | {ci_str:<19} | {mrr:<8.4f} | {ndcg5:<8.4f} | {p95:<9.1f} | {disk_mb:<9.1f} | {sig_badge:<14}"
        )

    lines.append("=" * 125)
    return "\n".join(lines)


def run_experiment_e2(
    split: str = "both",
    mode: str = "dense",
    output_dir: Path | str = "data/eval_outputs",
    index_dir: Path | str = "data/index",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end Experiment E2 across dev and held-out test splits."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits_to_run = ["dev"] if split == "dev" else ["test"] if split == "test" else ["dev", "test"]
    all_summary: dict[str, Any] = {}

    for s in splits_to_run:
        dataset_file = f"data/gold/{s}.jsonl"
        logger.info("==================================================")
        logger.info("Executing Experiment E2 on split: %s (%s, mode=%s)", s, dataset_file, mode)
        logger.info("==================================================")

        split_results: dict[str, EvalReport] = {}

        for v in VARIANTS:
            vid = v["id"]
            model_name = v["model_name"]
            logger.info("Evaluating %s (%s)...", v["name"], model_name)
            rep = run_variant_eval(
                model_name=model_name,
                dataset_path=dataset_file,
                index_dir=index_dir,
                strategy="structure_aware",
                mode=mode,
                show_progress=show_progress,
            )
            split_results[vid] = rep

            out_file = out_path / f"e2_{vid}_{s}_{mode}.json"
            rep.save_json(out_file)
            logger.info("Saved report for %s to %s", vid, out_file)

        # Print comparison table
        table_str = format_e2_results_table(split_results, index_dir=index_dir)
        print(f"\n--- Split: {s.upper()} (Mode: {mode}) ---")
        print(table_str)

        # Statistical paired comparisons against MiniLM baseline
        comparisons: dict[str, Any] = {}
        for cand_vid in ["mpnet_768", "e5_large_1024"]:
            if cand_vid in split_results and "minilm_384" in split_results:
                comp = compare_retrieval_runs(
                    baseline_data=split_results["minilm_384"].to_dict(),
                    candidate_data=split_results[cand_vid].to_dict(),
                    baseline_path="minilm_384",
                    candidate_path=cand_vid,
                )
                comp_file = out_path / f"e2_comp_{cand_vid}_vs_minilm_{s}_{mode}.json"
                comp_file.write_text(json.dumps(comp.to_dict(), indent=2), encoding="utf-8")
                comparisons[f"{cand_vid}_vs_minilm"] = comp.to_dict()

        all_summary[s] = {
            "mode": mode,
            "variants": {
                vid: rep.to_dict()["overall_metrics"] for vid, rep in split_results.items()
            },
            "comparisons": comparisons,
        }

    summary_file = out_path / f"e2_embeddings_summary_{mode}.json"
    summary_file.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    logger.info("Experiment E2 complete! Summary written to %s", summary_file)
    return all_summary


def main() -> None:
    """CLI runner for Experiment E2."""
    parser = argparse.ArgumentParser(
        description="Run Experiment E2: Multilingual Embedding Models."
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="both",
        help="Dataset split to evaluate (default: both)",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid"],
        default="dense",
        help="Retrieval search mode (default: dense)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval_outputs"),
        help="Directory to save evaluation reports (default: data/eval_outputs)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Directory containing retrieval indices (default: data/index)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during evaluation",
    )

    args = parser.parse_args()
    run_experiment_e2(
        split=args.split,
        mode=args.mode,
        output_dir=args.output_dir,
        index_dir=args.index_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
