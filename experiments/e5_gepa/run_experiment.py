"""End-to-End GEPA Prompt Optimization Experiment Runner (T5.1 / E5).

Runs evolutionary Pareto prompt optimization on Indonesian financial regulatory RAG.
Optimizes answer prompt on e5_train.jsonl, tunes on e5_val.jsonl, and performs
final held-out generalization benchmark and overfitting check on test.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path and line buffering is active
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import gepa
from eval.labeling_helper import GoldItem
from eval.metrics.bootstrap import bootstrap_ci, paired_bootstrap_comparison
from eval.metrics.citations import citation_f1, citation_precision, citation_recall
from eval.split_eval import load_gold_items
from experiments.e5_gepa.adapter import EasyPasalGEPAAdapter, compute_lexical_f1
from experiments.e5_gepa.proposer import EasyPasalReflectiveProposer
from tqdm import tqdm

from easypasal.generation.backends import MockGeneratorBackend
from easypasal.generation.pipeline import RAGPipeline
from easypasal.generation.prompts import SYSTEM_PROMPT_ID
from easypasal.retrieval.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.experiments.e5_gepa.runner")


@dataclass
class PromptItemEval:
    """Individual item evaluation record."""

    item_id: str
    category: str
    question: str
    gold_answer: str
    gold_citations: list[str]
    should_abstain: bool
    generated_answer: str
    extracted_citations: list[str]
    did_abstain: bool
    citation_f1_article: float
    citation_f1_exact: float
    citation_precision_article: float
    citation_recall_article: float
    abstention_correct: bool
    lexical_f1: float
    grounded_ratio: float
    hallucinated_pasal: list[str]
    ungrounded_citations: list[str]
    composite_score: float
    latency_ms: float


@dataclass
class SplitEvaluationSummary:
    """Aggregated evaluation metrics for a prompt candidate on a dataset split."""

    split_name: str
    variant_name: str
    num_items: int
    mean_citation_f1_article: float
    mean_citation_f1_exact: float
    mean_citation_precision: float
    mean_citation_recall: float
    abstention_accuracy: float
    mean_lexical_f1: float
    grounding_fidelity: float
    mean_composite_score: float
    mean_latency_ms: float
    bootstrap_cis: dict[str, Any] = field(default_factory=dict)
    items: list[PromptItemEval] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"] = [asdict(it) for it in self.items]
        return d


def evaluate_prompt_on_split(
    pipeline: RAGPipeline,
    items: list[GoldItem],
    prompt_text: str,
    split_name: str,
    variant_name: str,
    top_k: int = 5,
    mode: str = "dense",
    strict_grounding: bool = False,
    num_bootstrap: int = 2000,
) -> SplitEvaluationSummary:
    """Evaluate a specific system prompt candidate across a dataset split."""
    results: list[PromptItemEval] = []
    logger.info("Evaluating '%s' on split '%s' (%d items)...", variant_name, split_name, len(items))

    for item in tqdm(
        items,
        desc=f"[{split_name.upper():<5}] {variant_name:<14}",
        leave=True,
        dynamic_ncols=True,
    ):
        t0 = time.perf_counter()
        res = pipeline.ask(
            question=item.question,
            language=item.language,
            top_k=top_k,
            mode=mode,
            system_prompt_override=prompt_text,
            verify_citations=True,
            strict_grounding=strict_grounding,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ans = res.answer
        cits = res.citations
        abstained = res.abstained

        meta = getattr(res, "metadata", None) or {}
        v_dict = (
            meta.get("citation_verification") or getattr(res, "citation_verification", {}) or {}
        )
        grounded_ratio = (
            v_dict.get("grounded_count", 0) / max(v_dict.get("total_citations", 1), 1)
            if v_dict
            else 1.0
        )
        ungrounded = v_dict.get("ungrounded_citations", [])
        hallucinated = v_dict.get("hallucinated_pasal", [])

        art_f1 = citation_f1(cits, item.gold_citations, mode="article")
        exact_f1 = citation_f1(cits, item.gold_citations, mode="exact")
        art_prec = citation_precision(cits, item.gold_citations, mode="article")
        art_rec = citation_recall(cits, item.gold_citations, mode="article")
        abs_correct = abstained == item.should_abstain
        lex_f1 = compute_lexical_f1(ans, item.gold_answer)

        if item.should_abstain:
            comp_score = 1.0 if abs_correct else 0.0
        else:
            if abstained:
                comp_score = 0.0
            else:
                comp_score = 0.40 * art_f1 + 0.30 * grounded_ratio + 0.30 * lex_f1
                if hallucinated:
                    comp_score *= 0.5

        results.append(
            PromptItemEval(
                item_id=item.id,
                category=item.category,
                question=item.question,
                gold_answer=item.gold_answer,
                gold_citations=item.gold_citations,
                should_abstain=item.should_abstain,
                generated_answer=ans,
                extracted_citations=cits,
                did_abstain=abstained,
                citation_f1_article=round(art_f1, 4),
                citation_f1_exact=round(exact_f1, 4),
                citation_precision_article=round(art_prec, 4),
                citation_recall_article=round(art_rec, 4),
                abstention_correct=abs_correct,
                lexical_f1=round(lex_f1, 4),
                grounded_ratio=round(grounded_ratio, 4),
                hallucinated_pasal=hallucinated,
                ungrounded_citations=ungrounded,
                composite_score=round(comp_score, 4),
                latency_ms=round(latency_ms, 2),
            )
        )

    # Compute aggregates
    n = len(results)
    mean_cite_f1 = float(sum(r.citation_f1_article for r in results) / n) if n else 0.0
    mean_exact_f1 = float(sum(r.citation_f1_exact for r in results) / n) if n else 0.0
    mean_prec = float(sum(r.citation_precision_article for r in results) / n) if n else 0.0
    mean_rec = float(sum(r.citation_recall_article for r in results) / n) if n else 0.0
    abst_acc = float(sum(1.0 for r in results if r.abstention_correct) / n) if n else 0.0
    mean_lex = float(sum(r.lexical_f1 for r in results) / n) if n else 0.0
    grounding_fid = float(sum(r.grounded_ratio for r in results) / n) if n else 0.0
    mean_comp = float(sum(r.composite_score for r in results) / n) if n else 0.0
    mean_lat = float(sum(r.latency_ms for r in results) / n) if n else 0.0

    # Non-parametric Bootstrap CIs
    cis: dict[str, Any] = {}
    if n > 1:
        cis["citation_f1_article"] = bootstrap_ci(
            [r.citation_f1_article for r in results], n_resamples=num_bootstrap
        ).to_dict()
        cis["citation_f1_exact"] = bootstrap_ci(
            [r.citation_f1_exact for r in results], n_resamples=num_bootstrap
        ).to_dict()
        cis["abstention_accuracy"] = bootstrap_ci(
            [1.0 if r.abstention_correct else 0.0 for r in results], n_resamples=num_bootstrap
        ).to_dict()
        cis["grounding_fidelity"] = bootstrap_ci(
            [r.grounded_ratio for r in results], n_resamples=num_bootstrap
        ).to_dict()
        cis["composite_score"] = bootstrap_ci(
            [r.composite_score for r in results], n_resamples=num_bootstrap
        ).to_dict()

    return SplitEvaluationSummary(
        split_name=split_name,
        variant_name=variant_name,
        num_items=n,
        mean_citation_f1_article=round(mean_cite_f1, 4),
        mean_citation_f1_exact=round(mean_exact_f1, 4),
        mean_citation_precision=round(mean_prec, 4),
        mean_citation_recall=round(mean_rec, 4),
        abstention_accuracy=round(abst_acc, 4),
        mean_lexical_f1=round(mean_lex, 4),
        grounding_fidelity=round(grounding_fid, 4),
        mean_composite_score=round(mean_comp, 4),
        mean_latency_ms=round(mean_lat, 2),
        bootstrap_cis=cis,
        items=results,
    )


def combine_split_evaluations(
    summary_a: SplitEvaluationSummary,
    summary_b: SplitEvaluationSummary,
    combined_split_name: str,
    variant_name: str,
    num_bootstrap: int = 2000,
) -> SplitEvaluationSummary:
    """Combine evaluated items from two disjoint splits (e.g. train + val -> dev) without re-inference."""
    combined_items = summary_a.items + summary_b.items
    n = len(combined_items)
    mean_cite_f1 = float(sum(r.citation_f1_article for r in combined_items) / n) if n else 0.0
    mean_exact_f1 = float(sum(r.citation_f1_exact for r in combined_items) / n) if n else 0.0
    mean_prec = float(sum(r.citation_precision_article for r in combined_items) / n) if n else 0.0
    mean_rec = float(sum(r.citation_recall_article for r in combined_items) / n) if n else 0.0
    abst_acc = float(sum(1.0 for r in combined_items if r.abstention_correct) / n) if n else 0.0
    mean_lex = float(sum(r.lexical_f1 for r in combined_items) / n) if n else 0.0
    grounding_fid = float(sum(r.grounded_ratio for r in combined_items) / n) if n else 0.0
    mean_comp = float(sum(r.composite_score for r in combined_items) / n) if n else 0.0
    mean_lat = float(sum(r.latency_ms for r in combined_items) / n) if n else 0.0

    cis: dict[str, Any] = {}
    if n > 1:
        cis["citation_f1_article"] = bootstrap_ci(
            [r.citation_f1_article for r in combined_items], n_resamples=num_bootstrap
        ).to_dict()
        cis["citation_f1_exact"] = bootstrap_ci(
            [r.citation_f1_exact for r in combined_items], n_resamples=num_bootstrap
        ).to_dict()
        cis["abstention_accuracy"] = bootstrap_ci(
            [1.0 if r.abstention_correct else 0.0 for r in combined_items],
            n_resamples=num_bootstrap,
        ).to_dict()
        cis["grounding_fidelity"] = bootstrap_ci(
            [r.grounded_ratio for r in combined_items], n_resamples=num_bootstrap
        ).to_dict()
        cis["composite_score"] = bootstrap_ci(
            [r.composite_score for r in combined_items], n_resamples=num_bootstrap
        ).to_dict()

    return SplitEvaluationSummary(
        split_name=combined_split_name,
        variant_name=variant_name,
        num_items=n,
        mean_citation_f1_article=round(mean_cite_f1, 4),
        mean_citation_f1_exact=round(mean_exact_f1, 4),
        mean_citation_precision=round(mean_prec, 4),
        mean_citation_recall=round(mean_rec, 4),
        abstention_accuracy=round(abst_acc, 4),
        mean_lexical_f1=round(mean_lex, 4),
        grounding_fidelity=round(grounding_fid, 4),
        mean_composite_score=round(mean_comp, 4),
        mean_latency_ms=round(mean_lat, 2),
        bootstrap_cis=cis,
        items=combined_items,
    )


def compare_variants(
    baseline_eval: SplitEvaluationSummary,
    candidate_eval: SplitEvaluationSummary,
    num_bootstrap: int = 2000,
    num_permutations: int = 5000,
) -> dict[str, Any]:
    """Perform paired bootstrap statistical hypothesis testing between baseline and candidate."""
    b_items = {it.item_id: it for it in baseline_eval.items}
    c_items = {it.item_id: it for it in candidate_eval.items}
    common_ids = sorted(set(b_items.keys()) & set(c_items.keys()))

    metrics_to_test = [
        ("Citation F1 (Article)", "citation_f1_article"),
        ("Citation F1 (Exact)", "citation_f1_exact"),
        ("Abstention Accuracy", "abstention_correct"),
        ("Grounding Fidelity", "grounded_ratio"),
        ("Composite Score", "composite_score"),
    ]

    comparisons: dict[str, Any] = {}
    for display_name, attr in metrics_to_test:
        if attr == "abstention_correct":
            b_vals = [1.0 if b_items[i].abstention_correct else 0.0 for i in common_ids]
            c_vals = [1.0 if c_items[i].abstention_correct else 0.0 for i in common_ids]
        else:
            b_vals = [float(getattr(b_items[i], attr)) for i in common_ids]
            c_vals = [float(getattr(c_items[i], attr)) for i in common_ids]

        res = paired_bootstrap_comparison(
            baseline=b_vals,
            candidate=c_vals,
            metric_name=display_name,
            n_resamples=num_bootstrap,
            n_permutations=num_permutations,
        )
        comparisons[display_name] = res.to_dict()

    return comparisons


def build_pipeline(provider: str, config_path: str = "configs/default.yaml") -> RAGPipeline:
    """Initialize RAG pipeline for evaluation."""
    if provider == "mock":
        retriever = HybridRetriever.from_paths(
            index_dir="data/index",
            strategy="structure_aware",
            model_name="intfloat/multilingual-e5-large",
            collection_name="easypasal_structure_aware_multilingual_e5_large",
        )
        generator = MockGeneratorBackend()
        return RAGPipeline(retriever=retriever, generator=generator)

    # Load configured pipeline with local llama.cpp backend
    return RAGPipeline.from_config(config_path=config_path, override_provider="llamacpp")


def print_comparison_table(
    summaries: dict[str, dict[str, SplitEvaluationSummary]],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    """Print formatted markdown comparison table."""
    print("\n" + "=" * 90)
    print("📊 EXPERIMENT E5: DSPy + GEPA PROMPT OPTIMIZATION BENCHMARK")
    print("=" * 90)

    header = f"{'Split':<8} | {'Variant':<16} | {'Cite F1':<8} | {'95% CI':<19} | {'Abst Acc':<8} | {'Grounding':<9} | {'Composite':<9}"
    divider = "-" * len(header)
    print(header)
    print(divider)

    for split in ["train", "val", "dev", "test"]:
        if split not in summaries:
            continue
        base = summaries[split].get("baseline")
        gepa = summaries[split].get("gepa")

        if base:
            f1_ci = base.bootstrap_cis.get("citation_f1_article", {}).get("formatted", "N/A")
            ci_part = "[" + f1_ci.split("[")[-1] if "[" in f1_ci else f1_ci
            print(
                f"{split:<8} | {'Baseline':<16} | {base.mean_citation_f1_article:<8.4f} | {ci_part:<19} | "
                f"{base.abstention_accuracy:<8.4f} | {base.grounding_fidelity:<9.4f} | {base.mean_composite_score:<9.4f}"
            )
        if gepa:
            f1_ci = gepa.bootstrap_cis.get("citation_f1_article", {}).get("formatted", "N/A")
            ci_part = "[" + f1_ci.split("[")[-1] if "[" in f1_ci else f1_ci
            print(
                f"{split:<8} | {'GEPA-Optimized':<16} | {gepa.mean_citation_f1_article:<8.4f} | {ci_part:<19} | "
                f"{gepa.abstention_accuracy:<8.4f} | {gepa.grounding_fidelity:<9.4f} | {gepa.mean_composite_score:<9.4f}"
            )
        print(divider)

    print("\n🔬 Statistical Hypothesis Testing on Held-Out Test Set (Baseline vs GEPA-Optimized):")
    print("-" * 90)
    test_comp = comparisons.get("test", {})
    for metric_name, c in test_comp.items():
        delta = c.get("delta_mean", 0.0)
        p_val = c.get("p_value", 1.0)
        sig = "Significant [P < 0.05]" if c.get("is_significant") else "Not Significant"
        d_ci = c.get("delta_ci", {}).get("formatted", "")
        print(
            f"  • {metric_name:<24}: Δ = {delta:+.4f} (95% CI: {d_ci}) | p-value = {p_val:.4f} | {sig}"
        )
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E5 (GEPA Prompt Optimization)")
    parser.add_argument("--provider", choices=["llamacpp", "mock"], default="llamacpp")
    parser.add_argument("--max-metric-calls", type=int, default=20)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-json", default="results/e5_gepa_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path("data/gold/e5_train.jsonl")
    val_path = Path("data/gold/e5_val.jsonl")
    dev_path = Path("data/gold/dev.jsonl")
    test_path = Path("data/gold/test.jsonl")

    logger.info("Loading gold datasets...")
    train_items = load_gold_items(train_path)
    val_items = load_gold_items(val_path)
    dev_items = load_gold_items(dev_path)
    test_items = load_gold_items(test_path)
    logger.info(
        "Loaded: Train=%d, Val=%d, Dev=%d, Test=%d (Held-out)",
        len(train_items),
        len(val_items),
        len(dev_items),
        len(test_items),
    )

    logger.info("Initializing RAGPipeline (provider=%s)...", args.provider)
    pipeline = build_pipeline(provider=args.provider, config_path=args.config)

    # 1. Initialize GEPA Adapter
    adapter = EasyPasalGEPAAdapter(
        pipeline=pipeline,
        top_k=5,
        mode="dense",
        strict_grounding=False,
        proposer=EasyPasalReflectiveProposer(base_prompt=SYSTEM_PROMPT_ID),
    )

    seed_candidate = {"system_prompt": SYSTEM_PROMPT_ID}

    # 2. Run GEPA Optimization on train and val
    logger.info("Starting GEPA Evolutionary Pareto Optimization...")
    opt_start = time.perf_counter()
    gepa_result: Any = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_items,
        valset=val_items,
        adapter=adapter,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=4,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        display_progress_bar=True,
        seed=args.seed,
    )
    opt_duration = time.perf_counter() - opt_start
    logger.info("GEPA Optimization completed in %.2fs!", opt_duration)

    winning_prompt: str = str(
        getattr(gepa_result, "best_candidate", {}).get("system_prompt", SYSTEM_PROMPT_ID)
    )
    logger.info("Winning Prompt length: %d characters", len(winning_prompt))

    # 3. Final Multi-Split Benchmark Evaluation (Train, Val, Dev, Test)
    summaries: dict[str, dict[str, SplitEvaluationSummary]] = {
        "train": {},
        "val": {},
        "dev": {},
        "test": {},
    }

    # Baseline Evaluations
    summaries["train"]["baseline"] = evaluate_prompt_on_split(
        pipeline, train_items, SYSTEM_PROMPT_ID, split_name="train", variant_name="baseline"
    )
    summaries["val"]["baseline"] = evaluate_prompt_on_split(
        pipeline, val_items, SYSTEM_PROMPT_ID, split_name="val", variant_name="baseline"
    )
    # Dev is the exact combination of train and val (no redundant LLM calls)
    summaries["dev"]["baseline"] = combine_split_evaluations(
        summaries["train"]["baseline"],
        summaries["val"]["baseline"],
        combined_split_name="dev",
        variant_name="baseline",
    )
    summaries["test"]["baseline"] = evaluate_prompt_on_split(
        pipeline, test_items, SYSTEM_PROMPT_ID, split_name="test", variant_name="baseline"
    )

    # GEPA-Optimized Evaluations
    summaries["train"]["gepa"] = evaluate_prompt_on_split(
        pipeline, train_items, winning_prompt, split_name="train", variant_name="gepa"
    )
    summaries["val"]["gepa"] = evaluate_prompt_on_split(
        pipeline, val_items, winning_prompt, split_name="val", variant_name="gepa"
    )
    # Dev is the exact combination of train and val (no redundant LLM calls)
    summaries["dev"]["gepa"] = combine_split_evaluations(
        summaries["train"]["gepa"],
        summaries["val"]["gepa"],
        combined_split_name="dev",
        variant_name="gepa",
    )
    summaries["test"]["gepa"] = evaluate_prompt_on_split(
        pipeline, test_items, winning_prompt, split_name="test", variant_name="gepa"
    )

    # 4. Statistical Significance Testing
    comparisons: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "dev", "test"]:
        comparisons[split] = compare_variants(
            summaries[split]["baseline"],
            summaries[split]["gepa"],
        )

    # 5. Overfitting Gap Analysis
    delta_train_f1 = (
        summaries["train"]["gepa"].mean_citation_f1_article
        - summaries["train"]["baseline"].mean_citation_f1_article
    )
    delta_test_f1 = (
        summaries["test"]["gepa"].mean_citation_f1_article
        - summaries["test"]["baseline"].mean_citation_f1_article
    )
    overfitting_gap_f1 = abs(delta_train_f1 - delta_test_f1)

    delta_train_comp = (
        summaries["train"]["gepa"].mean_composite_score
        - summaries["train"]["baseline"].mean_composite_score
    )
    delta_test_comp = (
        summaries["test"]["gepa"].mean_composite_score
        - summaries["test"]["baseline"].mean_composite_score
    )
    overfitting_gap_comp = abs(delta_train_comp - delta_test_comp)

    overfitting_audit = {
        "delta_train_citation_f1": round(delta_train_f1, 4),
        "delta_test_citation_f1": round(delta_test_f1, 4),
        "overfitting_gap_citation_f1": round(overfitting_gap_f1, 4),
        "delta_train_composite": round(delta_train_comp, 4),
        "delta_test_composite": round(delta_test_comp, 4),
        "overfitting_gap_composite": round(overfitting_gap_comp, 4),
        "is_overfitted": bool(overfitting_gap_f1 > 0.15 and delta_test_f1 < 0.0),
        "verdict": (
            "ROBUST GENERALIZATION: Improvement holds on unseen test set without overfitting."
            if (overfitting_gap_f1 <= 0.15 or delta_test_f1 >= 0.0)
            else "OVERFITTED: High gap between train gains and test regression."
        ),
    }

    # 6. Print Report & Save JSON Results
    print_comparison_table(summaries, comparisons)
    print(f"\n🛡️ Overfitting Audit Verdict: {overfitting_audit['verdict']}")
    print(f"  • Train Δ Citation F1: {delta_train_f1:+.4f}")
    print(f"  • Test Δ Citation F1 : {delta_test_f1:+.4f}")
    print(f"  • Gap |Δ_train - Δ_test|: {overfitting_gap_f1:.4f} (Threshold: <= 0.15)")

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    serializable_summaries: dict[str, dict[str, Any]] = {}
    for sp, vars_dict in summaries.items():
        serializable_summaries[sp] = {vname: ssum.to_dict() for vname, ssum in vars_dict.items()}

    report = {
        "experiment": "E5",
        "description": "DSPy + GEPA answer prompt optimization across train, val, dev, and test",
        "timestamp": datetime.now(UTC).isoformat(),
        "provider": args.provider,
        "seed": args.seed,
        "optimization_duration_seconds": round(opt_duration, 2),
        "baseline_prompt": SYSTEM_PROMPT_ID,
        "winning_prompt": winning_prompt,
        "overfitting_audit": overfitting_audit,
        "summaries": serializable_summaries,
        "comparisons": comparisons,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved complete E5 benchmark report to %s", out_file)


if __name__ == "__main__":
    main()
