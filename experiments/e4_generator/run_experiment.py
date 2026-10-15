"""Experiment E4: Generator Model Comparison on CPU (T4.4).

Benchmarks three candidate local generator models (Qwen2.5-1.5B, Llama-3.2-1B, Qwen2.5-3B)
running 4-bit quantized GGUF (Q4_K_M) via in-process llama.cpp on CPU:
  - Candidate 1 (llama_3_2_1b): Llama-3.2-1B-Instruct (1.23B, compact footprint, fast prefill)
  - Candidate 2 (qwen_2_5_1_5b): Qwen2.5-1.5B-Instruct (1.54B, strong Indonesian instruction following) [Reference]
  - Candidate 3 (qwen_2_5_3b): Qwen2.5-3B-Instruct (3.09B, deeper reasoning capacity and statutory nuance)

Evaluates on development split (dev.jsonl, N=40), validates on held-out test split (test.jsonl, N=66),
computes 95% bootstrap CIs and paired Monte Carlo permutation tests, and evaluates whether the 3B model
"earns its latency and memory overhead" on CPU within a strict interactive latency envelope.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from eval.labeling_helper import GoldItem
from eval.metrics.abstention import abstention_accuracy
from eval.metrics.bootstrap import bootstrap_ci, paired_bootstrap_comparison
from eval.metrics.citations import citation_f1, citation_precision, citation_recall
from eval.split_eval import load_gold_items
from tqdm import tqdm

from easypasal.generation.backends import (
    GeneratorBackend,
    LlamaCppBackend,
    MockGeneratorBackend,
    get_process_peak_memory_mb,
)
from easypasal.generation.citations import (
    verify_citation_coverage,
)
from easypasal.generation.models import GenerationOutput
from easypasal.generation.pipeline import RAGPipeline
from easypasal.retrieval.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments.e4_generator")

CANONICAL_RETRIEVAL_MODEL = "intfloat/multilingual-e5-large"
CANONICAL_STRATEGY = "structure_aware"
DEFAULT_OUTPUT_DIR = Path("data/eval_outputs")


class GeneratorCandidate(TypedDict):
    id: str
    name: str
    repo_id: str
    filename: str
    params: str
    quant: str
    description: str


CANDIDATES: list[GeneratorCandidate] = [
    {
        "id": "llama_3_2_1b",
        "name": "Llama-3.2-1B-Instruct",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "params": "1.23B",
        "quant": "Q4_K_M",
        "description": "Ultra-compact 1.2B SLM with fast CPU prefill and low RAM footprint",
    },
    {
        "id": "qwen_2_5_1_5b",
        "name": "Qwen2.5-1.5B-Instruct",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "params": "1.54B",
        "quant": "Q4_K_M",
        "description": "Balanced 1.5B multilingual SLM with strong Indonesian compliance reasoning (Reference)",
    },
    {
        "id": "qwen_2_5_3b",
        "name": "Qwen2.5-3B-Instruct",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "params": "3.09B",
        "quant": "Q4_K_M",
        "description": "Medium 3.1B SLM with enhanced statutory reasoning depth and clause synthesis",
    },
]


def compute_lexical_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 overlap between predicted answer and reference answer."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    prec = sum(1 for t in pred_tokens if t in common) / len(pred_tokens)
    rec = sum(1 for t in ref_tokens if t in common) / len(ref_tokens)
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


@dataclass
class ItemGenerationRecord:
    item_id: str
    question: str
    category: str
    theme: str
    should_abstain: bool
    gold_citations: list[str]
    predicted_answer: str
    predicted_citations: list[str]
    abstained: bool
    is_abstention_correct: bool
    citation_precision_exact: float
    citation_recall_exact: float
    citation_f1_exact: float
    citation_precision_article: float
    citation_recall_article: float
    citation_f1_article: float
    grounded_citation_ratio: float
    lexical_token_f1: float
    tokens_generated: int
    tokens_per_second: float
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "category": self.category,
            "theme": self.theme,
            "should_abstain": self.should_abstain,
            "gold_citations": self.gold_citations,
            "predicted_answer": self.predicted_answer,
            "predicted_citations": self.predicted_citations,
            "abstained": self.abstained,
            "is_abstention_correct": self.is_abstention_correct,
            "citation_precision_exact": round(self.citation_precision_exact, 4),
            "citation_recall_exact": round(self.citation_recall_exact, 4),
            "citation_f1_exact": round(self.citation_f1_exact, 4),
            "citation_precision_article": round(self.citation_precision_article, 4),
            "citation_recall_article": round(self.citation_recall_article, 4),
            "citation_f1_article": round(self.citation_f1_article, 4),
            "grounded_citation_ratio": round(self.grounded_citation_ratio, 4),
            "lexical_token_f1": round(self.lexical_token_f1, 4),
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class GeneratorEvalReport:
    candidate_id: str
    candidate_name: str
    params: str
    quant: str
    dataset_path: str
    total_items: int
    answerable_items: int
    unanswerable_items: int
    mean_citation_f1_exact: float
    mean_citation_f1_article: float
    mean_citation_precision: float
    mean_citation_recall: float
    abstention_accuracy: float
    mean_grounded_citation_ratio: float
    mean_lexical_token_f1: float
    mean_tokens_per_second: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    peak_memory_mb: float
    model_size_mb: float
    records: list[ItemGenerationRecord] = field(default_factory=list)
    bootstrap_cis: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "params": self.params,
            "quant": self.quant,
            "dataset_path": self.dataset_path,
            "total_items": self.total_items,
            "answerable_items": self.answerable_items,
            "unanswerable_items": self.unanswerable_items,
            "mean_citation_f1_exact": round(self.mean_citation_f1_exact, 4),
            "mean_citation_f1_article": round(self.mean_citation_f1_article, 4),
            "mean_citation_precision": round(self.mean_citation_precision, 4),
            "mean_citation_recall": round(self.mean_citation_recall, 4),
            "abstention_accuracy": round(self.abstention_accuracy, 4),
            "mean_grounded_citation_ratio": round(self.mean_grounded_citation_ratio, 4),
            "mean_lexical_token_f1": round(self.mean_lexical_token_f1, 4),
            "mean_tokens_per_second": round(self.mean_tokens_per_second, 2),
            "latency_mean_ms": round(self.latency_mean_ms, 2),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "model_size_mb": round(self.model_size_mb, 2),
            "bootstrap_cis": self.bootstrap_cis,
            "timestamp": self.timestamp,
            "records": [r.to_dict() for r in self.records],
        }


def evaluate_generator_candidate(
    candidate: GeneratorCandidate,
    dataset_path: Path | str,
    retriever: HybridRetriever,
    dry_run: bool = False,
    show_progress: bool = True,
    n_ctx: int = 8192,
    top_k: int = 5,
    max_tokens: int = 160,
) -> GeneratorEvalReport:
    """Evaluate a single generator candidate across all gold items."""
    d_path = Path(dataset_path)
    items: list[GoldItem] = load_gold_items(d_path)

    logger.info("Evaluating candidate generator '%s' on %s...", candidate["name"], d_path.name)

    # 1. Initialize generator backend
    model_size_mb = 0.0
    if dry_run:
        logger.info("Running in dry-run mode (using MockGeneratorBackend)...")
        generator: GeneratorBackend = MockGeneratorBackend(model_name=candidate["id"])
    else:
        logger.info(
            "Loading GGUF generator from %s / %s...", candidate["repo_id"], candidate["filename"]
        )
        generator = LlamaCppBackend(
            repo_id=candidate["repo_id"],
            filename=candidate["filename"],
            n_ctx=n_ctx,
        )
        if isinstance(generator, LlamaCppBackend) and generator.model_path.exists():
            model_size_mb = round(generator.model_path.stat().st_size / (1024 * 1024), 2)

    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    records: list[ItemGenerationRecord] = []
    pbar = tqdm(
        items,
        desc=f"Generating ({candidate['id']} | {d_path.stem})",
        disable=not show_progress,
        unit="query",
    )

    for item in pbar:
        # Run end-to-end RAG pipeline
        out: GenerationOutput = pipeline.ask(
            question=item.question,
            mode="dense",
            top_k=top_k,
            max_tokens=max_tokens,
            verify_citations=True,
            strict_grounding=False,
        )

        # 1. Citations evaluation
        p_exact = citation_precision(out.citations, item.gold_citations, mode="exact")
        r_exact = citation_recall(out.citations, item.gold_citations, mode="exact")
        f1_exact = citation_f1(out.citations, item.gold_citations, mode="exact")

        p_article = citation_precision(out.citations, item.gold_citations, mode="article")
        r_article = citation_recall(out.citations, item.gold_citations, mode="article")
        f1_article = citation_f1(out.citations, item.gold_citations, mode="article")

        # 2. Grounding verification against retrieved chunks
        grounded, ungrounded = verify_citation_coverage(out.citations, out.retrieved_chunks)
        total_cites = len(grounded) + len(ungrounded)
        grounded_ratio = len(grounded) / total_cites if total_cites > 0 else 1.0

        # 3. Abstention correctness
        is_abstain_correct = out.abstained == item.should_abstain

        # 4. Lexical answer overlap
        lex_f1 = compute_lexical_token_f1(out.answer, item.gold_answer)

        record = ItemGenerationRecord(
            item_id=item.id,
            question=item.question,
            category=item.category,
            theme=item.theme,
            should_abstain=item.should_abstain,
            gold_citations=item.gold_citations,
            predicted_answer=out.answer,
            predicted_citations=out.citations,
            abstained=out.abstained,
            is_abstention_correct=is_abstain_correct,
            citation_precision_exact=p_exact,
            citation_recall_exact=r_exact,
            citation_f1_exact=f1_exact,
            citation_precision_article=p_article,
            citation_recall_article=r_article,
            citation_f1_article=f1_article,
            grounded_citation_ratio=grounded_ratio,
            lexical_token_f1=lex_f1,
            tokens_generated=out.completion_tokens,
            tokens_per_second=out.tokens_per_second,
            latency_ms=out.latency_ms,
        )
        records.append(record)

    peak_ram = get_process_peak_memory_mb()

    # Aggregate metrics
    answerable = [r for r in records if not r.should_abstain]
    mean_f1_exact = float(np.mean([r.citation_f1_exact for r in records]))
    mean_f1_article = float(np.mean([r.citation_f1_article for r in records]))
    mean_prec = float(np.mean([r.citation_precision_article for r in records]))
    mean_rec = float(np.mean([r.citation_recall_article for r in records]))

    abst_preds = [r.abstained for r in records]
    abst_gold = [r.should_abstain for r in records]
    abst_acc = abstention_accuracy(abst_preds, abst_gold)

    mean_grounded = float(np.mean([r.grounded_citation_ratio for r in records]))
    mean_lex_f1 = float(np.mean([r.lexical_token_f1 for r in answerable])) if answerable else 0.0

    tok_s_vals = [r.tokens_per_second for r in records if r.tokens_per_second > 0]
    mean_tok_s = float(np.mean(tok_s_vals)) if tok_s_vals else 0.0

    latencies = [r.latency_ms for r in records]
    lat_mean = float(np.mean(latencies))
    lat_p50 = float(np.percentile(latencies, 50))
    lat_p95 = float(np.percentile(latencies, 95))

    # Bootstrap CIs for key indicators
    cis: dict[str, Any] = {}
    cis["citation_f1_article"] = bootstrap_ci([r.citation_f1_article for r in records]).to_dict()
    cis["citation_f1_exact"] = bootstrap_ci([r.citation_f1_exact for r in records]).to_dict()
    cis["abstention_accuracy"] = bootstrap_ci(
        [float(r.is_abstention_correct) for r in records]
    ).to_dict()
    cis["tokens_per_second"] = bootstrap_ci(tok_s_vals).to_dict() if tok_s_vals else {}
    cis["latency_ms"] = bootstrap_ci(latencies).to_dict()

    # Clean up generator backend and reclaim memory
    if hasattr(generator, "close"):
        generator.close()
    del generator
    import gc

    gc.collect()

    return GeneratorEvalReport(
        candidate_id=candidate["id"],
        candidate_name=candidate["name"],
        params=candidate["params"],
        quant=candidate["quant"],
        dataset_path=str(d_path),
        total_items=len(records),
        answerable_items=len(answerable),
        unanswerable_items=len(records) - len(answerable),
        mean_citation_f1_exact=mean_f1_exact,
        mean_citation_f1_article=mean_f1_article,
        mean_citation_precision=mean_prec,
        mean_citation_recall=mean_rec,
        abstention_accuracy=abst_acc,
        mean_grounded_citation_ratio=mean_grounded,
        mean_lexical_token_f1=mean_lex_f1,
        mean_tokens_per_second=mean_tok_s,
        latency_mean_ms=lat_mean,
        latency_p50_ms=lat_p50,
        latency_p95_ms=lat_p95,
        peak_memory_mb=peak_ram,
        model_size_mb=model_size_mb,
        records=records,
        bootstrap_cis=cis,
    )


def compare_generator_candidates(
    baseline_report: GeneratorEvalReport,
    candidate_report: GeneratorEvalReport,
    n_resamples: int = 2000,
    n_permutations: int = 5000,
) -> dict[str, Any]:
    """Run paired permutation tests and bootstrap comparison between baseline and candidate generator."""
    b_map = {r.item_id: r for r in baseline_report.records}
    c_map = {r.item_id: r for r in candidate_report.records}
    common_ids = [r.item_id for r in baseline_report.records if r.item_id in c_map]

    metrics: dict[str, Any] = {}

    target_metrics = [
        ("Citation F1 (Article)", lambda r: r.citation_f1_article),
        ("Citation F1 (Exact)", lambda r: r.citation_f1_exact),
        ("Abstention Acc", lambda r: float(r.is_abstention_correct)),
        ("Lexical F1", lambda r: r.lexical_token_f1),
        ("Tokens/s", lambda r: r.tokens_per_second),
        ("Latency (ms)", lambda r: r.latency_ms),
    ]

    for m_name, extractor in target_metrics:
        b_vals = [extractor(b_map[i]) for i in common_ids]
        c_vals = [extractor(c_map[i]) for i in common_ids]
        comp = paired_bootstrap_comparison(
            baseline=b_vals,
            candidate=c_vals,
            metric_name=m_name,
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            higher_is_better=(m_name != "Latency (ms)"),
        )
        metrics[m_name] = comp.to_dict()

    return {
        "baseline_id": baseline_report.candidate_id,
        "candidate_id": candidate_report.candidate_id,
        "common_items": len(common_ids),
        "metrics": metrics,
    }


def print_summary_table(
    reports: dict[str, GeneratorEvalReport],
    comparisons: dict[str, Any],
    split_name: str,
    reference_id: str = "qwen_2_5_1_5b",
) -> None:
    """Print clean formatted benchmark table to stdout."""
    header = (
        f"{'Generator Candidate':<26} | {'Params':<7} | {'Cite F1':<8} | {'95% CI':<19} | "
        f"{'Abst Acc':<8} | {'Tok/s':<7} | {'p95 Lat (s)':<11} | {'Peak RAM':<9} | {'Sig vs Ref':<12}"
    )
    divider = "-" * len(header)

    print("\n" + "=" * len(header))
    title = f"📊 EasyPasal Experiment E4: Generator Model Comparison ({split_name.upper()} SPLIT)"
    print(f"{title:^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(divider)

    for c in CANDIDATES:
        cid = c["id"]
        rep = reports.get(cid)
        if not rep:
            continue

        f1 = rep.mean_citation_f1_article
        ci_str = rep.bootstrap_cis.get("citation_f1_article", {}).get("formatted", "N/A")
        ci_box = "[" + ci_str.split("[")[-1] if "[" in ci_str and "]" in ci_str else ci_str
        abst = rep.abstention_accuracy
        tok_s = rep.mean_tokens_per_second
        p95_s = rep.latency_p95_ms / 1000.0
        ram_mb = f"{rep.peak_memory_mb:.0f} MB"

        if cid == reference_id:
            sig_str = "Reference"
        else:
            comp_key = f"{cid}_vs_{reference_id}"
            comp = comparisons.get(comp_key, {})
            m = comp.get("metrics", {}).get("Citation F1 (Article)", {})
            p_val = m.get("p_value")
            sig_str = f"p={p_val:.4f}" if p_val is not None else "N/A"

        row = (
            f"{rep.candidate_name:<26} | {rep.params:<7} | {f1:<8.4f} | {ci_box:<19} | "
            f"{abst:<8.4f} | {tok_s:<7.1f} | {p95_s:<11.2f} | {ram_mb:<9} | {sig_str:<12}"
        )
        print(row)

    print("=" * len(header))


def run_experiment(
    split: str = "both",
    candidate_ids: list[str] | None = None,
    dry_run: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    show_progress: bool = True,
    n_resamples: int = 2000,
    n_permutations: int = 5000,
    n_ctx: int = 8192,
    max_tokens: int = 160,
) -> dict[str, Any]:
    """Execute end-to-end Experiment E4 benchmark across requested splits."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_to_run = ["dev", "test"] if split == "both" else [split]
    eval_candidates = (
        [c for c in CANDIDATES if c["id"] in candidate_ids] if candidate_ids else CANDIDATES
    )

    # Initialize winning E3 retrieval pipeline: Pure Dense with Multilingual-E5-Large
    logger.info(
        "Initializing canonical E3 retriever (mode=dense, model=%s)...", CANONICAL_RETRIEVAL_MODEL
    )
    retriever = HybridRetriever.from_paths(
        index_dir="data/index",
        strategy=CANONICAL_STRATEGY,
        model_name=CANONICAL_RETRIEVAL_MODEL,
    )

    all_results: dict[str, Any] = {}

    try:
        for s in splits_to_run:
            dataset_path = Path(f"data/gold/{s}.jsonl")
            if not dataset_path.exists():
                raise FileNotFoundError(f"Split dataset {dataset_path} not found.")

            reports: dict[str, GeneratorEvalReport] = {}
            for candidate in eval_candidates:
                rep = evaluate_generator_candidate(
                    candidate=candidate,
                    dataset_path=dataset_path,
                    retriever=retriever,
                    dry_run=dry_run,
                    show_progress=show_progress,
                    n_ctx=n_ctx,
                    max_tokens=max_tokens,
                )
                reports[candidate["id"]] = rep

                # Save individual report
                out_path = out_dir / f"e4_{candidate['id']}_{s}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(rep.to_dict(), f, indent=2)
                logger.info("Saved report for %s to %s", candidate["id"], out_path)

            # Paired comparisons against reference baseline (qwen_2_5_1_5b)
            comparisons: dict[str, Any] = {}
            ref_rep = reports.get("qwen_2_5_1_5b")
            if ref_rep:
                for candidate in eval_candidates:
                    cid = candidate["id"]
                    if cid == "qwen_2_5_1_5b":
                        continue
                    comp = compare_generator_candidates(
                        baseline_report=ref_rep,
                        candidate_report=reports[cid],
                        n_resamples=n_resamples,
                        n_permutations=n_permutations,
                    )
                    comparisons[f"{cid}_vs_qwen_2_5_1_5b"] = comp
                    comp_path = out_dir / f"e4_comp_{cid}_vs_qwen_2_5_1_5b_{s}.json"
                    with open(comp_path, "w", encoding="utf-8") as f:
                        json.dump(comp, f, indent=2)

            print_summary_table(reports, comparisons, split_name=s)

            all_results[s] = {
                "candidates": {k: v.to_dict() for k, v in reports.items()},
                "comparisons": comparisons,
            }

        # Save unified master summary
        summary_path = out_dir / "e4_generator_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Experiment E4 complete! Master summary saved to %s", summary_path)

    finally:
        retriever.close()

    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment E4: Generator Model Comparison on CPU (T4.4)."
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="both",
        help="Evaluation split to benchmark (default: both).",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=[c["id"] for c in CANDIDATES],
        default=None,
        help="Subset of candidate IDs to evaluate (default: all candidates).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast offline smoke evaluation using MockGeneratorBackend without downloading weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save evaluation reports (default: data/eval_outputs).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Bootstrap resample iterations (default: 2000).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help="Permutation test resamples (default: 5000).",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Context window size in tokens for llama.cpp (default: 8192).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=160,
        help="Maximum generation tokens per query (default: 160).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        split=args.split,
        candidate_ids=args.candidates,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        n_resamples=args.n_resamples,
        n_permutations=args.n_permutations,
        n_ctx=args.n_ctx,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
