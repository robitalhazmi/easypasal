"""Experiment E1: Legal Chunking Strategy (T4.1).

Benchmarks three chunking paradigms for Indonesian financial regulations:
  - Variant A: fixed_size (500 char sliding window, 100 char overlap)
  - Variant B: structure_aware_no_header (atomic Pasal/Ayat without breadcrumbs)
  - Variant C: structure_aware (atomic Pasal/Ayat with hierarchical breadcrumbs)

Evaluates retrieval on dev split, computes 95% bootstrap CIs and paired permutation tests,
validates top candidates on held-out test split, and logs trade-offs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from eval.chunk_mapper import ChunkSpanMapper
from eval.compare_variants import compare_retrieval_runs
from eval.metrics.bootstrap import bootstrap_ci, paired_bootstrap_comparison
from eval.run_eval import EvalReport, evaluate_retrieval
from eval.split_eval import load_gold_items

from easypasal.retrieval.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments.e1_chunking")

VARIANTS = [
    {
        "id": "variant_a_fixed_size",
        "name": "Variant A (Fixed-Size 500/100)",
        "strategy": "fixed_size",
        "description": "500-char sliding window, 100-char overlap, naive slicing without statutory hierarchy",
    },
    {
        "id": "variant_b_structure_no_header",
        "name": "Variant B (Structure-Aware No Header)",
        "strategy": "structure_aware_no_header",
        "description": "Atomic legal clauses (Pasal/Ayat) without document or chapter breadcrumb headers",
    },
    {
        "id": "variant_c_structure_enriched",
        "name": "Variant C (Structure-Aware + Context Header)",
        "strategy": "structure_aware",
        "description": "Atomic legal clauses with prepended statutory breadcrumbs [Doc | Bab | Bagian | Pasal | Ayat]",
    },
]


def run_variant_eval(
    strategy: str,
    dataset_path: Path | str,
    index_dir: Path | str = "data/index",
    top_k: int = 10,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    mode: str = "hybrid",
    mapper: ChunkSpanMapper | None = None,
) -> EvalReport:
    """Run retrieval evaluation for a single chunking variant."""
    d_path = Path(dataset_path)
    items = load_gold_items(d_path)

    retriever = HybridRetriever.from_paths(
        index_dir=index_dir,
        strategy=strategy,
    )

    chunk_mapper = mapper if strategy == "fixed_size" else None

    report = evaluate_retrieval(
        retriever=retriever,
        items=items,
        top_k=top_k,
        k_values=k_values,
        mode=mode,
        dataset_path=str(d_path),
        strategy=strategy,
        chunk_mapper=chunk_mapper,
    )
    return report


def format_e1_results_table(
    results: dict[str, EvalReport],
    baseline_key: str = "variant_c_structure_enriched",
) -> str:
    """Format structured comparison table for Experiment E1."""
    baseline_report = results.get(baseline_key)
    lines: list[str] = [
        "=" * 115,
        "                      📊 EasyPasal Experiment E1: Chunking Strategy Results                       ",
        "=" * 115,
        f"{'Variant':<42} | {'Recall@5':<10} | {'95% CI':<19} | {'MRR':<8} | {'nDCG@5':<8} | {'p95 (ms)':<9} | {'Sig vs Var C':<12}",
        "-" * 115,
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
                baseline=r5_vals,
                candidate=base_r5,
                metric_name="recall_at_5",
                n_resamples=2000,
                n_permutations=5000,
                confidence=0.95,
                seed=42,
            )
            sig_badge = f"p={paired.p_value:.4f}"

        lines.append(
            f"{v['name']:<42} | {r5:<10.4f} | {ci_str:<19} | {mrr:<8.4f} | {ndcg5:<8.4f} | {p95:<9.1f} | {sig_badge:<12}"
        )

    lines.append("=" * 115)
    return "\n".join(lines)


def run_experiment_e1(
    split: str = "both",
    output_dir: Path | str = "data/eval_outputs",
    index_dir: Path | str = "data/index",
) -> dict[str, Any]:
    """Execute end-to-end Experiment E1 across dev and held-out test splits."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mapper = ChunkSpanMapper.load_or_build()

    splits_to_run = ["dev"] if split == "dev" else ["test"] if split == "test" else ["dev", "test"]
    all_summary: dict[str, Any] = {}

    for s in splits_to_run:
        dataset_file = f"data/gold/{s}.jsonl"
        logger.info("==================================================")
        logger.info("Executing Experiment E1 on split: %s (%s)", s, dataset_file)
        logger.info("==================================================")

        split_results: dict[str, EvalReport] = {}

        for v in VARIANTS:
            vid = v["id"]
            strategy = v["strategy"]
            logger.info("Evaluating %s (strategy=%r)...", v["name"], strategy)
            rep = run_variant_eval(
                strategy=strategy,
                dataset_path=dataset_file,
                index_dir=index_dir,
                mapper=mapper,
            )
            split_results[vid] = rep

            out_file = out_path / f"e1_{vid}_{s}.json"
            rep.save_json(out_file)
            logger.info("Saved report for %s to %s", vid, out_file)

        # Print comparison table
        table_str = format_e1_results_table(split_results)
        print(f"\n--- Split: {s.upper()} ---")
        print(table_str)

        # Statistical paired comparison between Variant C and Variant A
        comp_c_vs_a = compare_retrieval_runs(
            baseline_data=split_results["variant_a_fixed_size"].to_dict(),
            candidate_data=split_results["variant_c_structure_enriched"].to_dict(),
            baseline_path="variant_a_fixed_size",
            candidate_path="variant_c_structure_enriched",
        )
        comp_file_ca = out_path / f"e1_comp_variant_c_vs_a_{s}.json"
        comp_file_ca.write_text(json.dumps(comp_c_vs_a.to_dict(), indent=2), encoding="utf-8")

        # Statistical paired comparison between Variant C and Variant B
        comp_c_vs_b = compare_retrieval_runs(
            baseline_data=split_results["variant_b_structure_no_header"].to_dict(),
            candidate_data=split_results["variant_c_structure_enriched"].to_dict(),
            baseline_path="variant_b_structure_no_header",
            candidate_path="variant_c_structure_enriched",
        )
        comp_file_cb = out_path / f"e1_comp_variant_c_vs_b_{s}.json"
        comp_file_cb.write_text(json.dumps(comp_c_vs_b.to_dict(), indent=2), encoding="utf-8")

        all_summary[s] = {
            "variants": {
                vid: rep.to_dict()["overall_metrics"] for vid, rep in split_results.items()
            },
            "comp_c_vs_a": comp_c_vs_a.to_dict(),
            "comp_c_vs_b": comp_c_vs_b.to_dict(),
        }

    summary_file = out_path / "e1_chunking_summary.json"
    summary_file.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    logger.info("Experiment E1 complete! Summary written to %s", summary_file)
    return all_summary


def main() -> None:
    """CLI runner for Experiment E1."""
    parser = argparse.ArgumentParser(description="Run Experiment E1: Legal Chunking Strategy.")
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="both",
        help="Dataset split to evaluate (default: both)",
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

    args = parser.parse_args()
    run_experiment_e1(
        split=args.split,
        output_dir=args.output_dir,
        index_dir=args.index_dir,
    )


if __name__ == "__main__":
    main()
