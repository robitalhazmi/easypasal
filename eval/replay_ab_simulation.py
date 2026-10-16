"""Offline Replay-Based A/B Simulation Runner (Task T6.1).

Replays historical and gold evaluation queries through Variant A (baseline_v0.1)
and Variant B (gepa_v0.2), evaluates both variant responses under the exact same
judge backend, and computes paired bootstrap confidence intervals (B=2,000)
and Monte Carlo permutation tests (P=5,000).

HONEST-CLAIMS AUDIT (§13 / Plan §6.4):
This evaluation is strictly an offline counterfactual replay simulation using
gold/logged queries and an automated LLM judge. It is NOT an online A/B test
with live human end-users.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from easypasal.config import EasyPasalConfig
from easypasal.generation.backends import (
    MockGeneratorBackend,
)
from easypasal.generation.pipeline import RAGPipeline
from easypasal.retrieval.models import RetrievedChunk
from eval.metrics.bootstrap import (
    PairedComparisonResult,
    paired_bootstrap_comparison,
)
from eval.metrics.citations import (
    citation_f1,
    citation_precision,
    citation_recall,
)
from eval.metrics.judge import (
    BaseJudgeBackend,
    MockJudgeBackend,
    OpenAICompatibleJudgeBackend,
    TransformersJudgeBackend,
    evaluate_correctness,
    evaluate_faithfulness,
)
from eval.split_eval import load_gold_items

logger = logging.getLogger("eval.replay_ab_simulation")

DISCLAIMER_TEXT: str = (
    "OFFLINE REPLAY SIMULATION NOTICE: This evaluation is an offline counterfactual "
    "replay simulation using gold/logged queries and an automated LLM judge. "
    "It is NOT an online A/B test with live human end-users (Plan §6.4 & §13)."
)

DEFAULT_VARIANT_A: str = "baseline_v0.1"
DEFAULT_VARIANT_B: str = "gepa_v0.2"
DEFAULT_DATASET: str = "data/gold/dev.jsonl"
DEFAULT_OUTPUT: str = "results/t6_1_replay_ab_simulation_dev.json"


@dataclass
class ReplayedItemPair:
    """Side-by-side counterfactual outputs and paired evaluations for a single query."""

    item_id: str
    category: str
    theme: str
    question: str
    gold_answer: str
    gold_citations: list[str]
    should_abstain: bool

    # Variant A (Baseline)
    variant_a_answer: str
    variant_a_citations: list[str]
    variant_a_abstained: bool
    variant_a_grounded_ratio: float
    variant_a_latency_ms: float

    # Variant B (Candidate / GEPA)
    variant_b_answer: str
    variant_b_citations: list[str]
    variant_b_abstained: bool
    variant_b_grounded_ratio: float
    variant_b_latency_ms: float

    # Judge Metrics
    faithfulness_a: float
    faithfulness_b: float
    correctness_a: float
    correctness_b: float

    # Citation & Grounding Metrics
    citation_f1_article_a: float
    citation_f1_article_b: float
    citation_f1_exact_a: float
    citation_f1_exact_b: float
    citation_precision_a: float
    citation_precision_b: float
    citation_recall_a: float
    citation_recall_b: float

    # Abstention Compliance
    abstention_correct_a: bool
    abstention_correct_b: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert item pair to dictionary."""
        return asdict(self)


@dataclass
class ReplaySimulationSummary:
    """Comprehensive summary of offline replay-based A/B simulation."""

    timestamp: str
    dataset_path: str
    variant_a_id: str
    variant_b_id: str
    judge_backend: str
    disclaimer: str
    total_queries: int
    answerable_queries: int
    unanswerable_queries: int
    metrics: dict[str, PairedComparisonResult]
    category_metrics: dict[str, dict[str, PairedComparisonResult]] = field(default_factory=dict)
    items: list[ReplayedItemPair] = field(default_factory=list)

    @property
    def total_significant_gains(self) -> int:
        """Count of metrics with statistically significant gain."""
        return sum(1 for m in self.metrics.values() if m.status == "SIG_GAIN")

    @property
    def total_regressions(self) -> int:
        """Count of metrics with statistically significant regression."""
        return sum(1 for m in self.metrics.values() if m.status == "REGRESSION")

    @property
    def total_neutral(self) -> int:
        """Count of metrics with no statistically significant change."""
        return sum(1 for m in self.metrics.values() if m.status == "NEUTRAL")

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete simulation summary to dictionary."""
        return {
            "timestamp": self.timestamp,
            "simulation_type": "offline_replay_ab",
            "disclaimer": self.disclaimer,
            "dataset_path": self.dataset_path,
            "variant_a_id": self.variant_a_id,
            "variant_b_id": self.variant_b_id,
            "judge_backend": self.judge_backend,
            "total_queries": self.total_queries,
            "answerable_queries": self.answerable_queries,
            "unanswerable_queries": self.unanswerable_queries,
            "summary": {
                "significant_gains": self.total_significant_gains,
                "regressions": self.total_regressions,
                "neutral": self.total_neutral,
            },
            "metrics": {name: res.to_dict() for name, res in self.metrics.items()},
            "category_metrics": {
                cat: {name: res.to_dict() for name, res in cat_res.items()}
                for cat, cat_res in self.category_metrics.items()
            },
            "items": [item.to_dict() for item in self.items],
        }

    def save_json(self, output_path: Path | str) -> Path:
        """Save report to JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return target


def get_judge_backend(
    backend_type: str = "mock",
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "dummy",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    timeout: float = 60.0,
) -> BaseJudgeBackend:
    """Factory for LLM judge backend instances."""
    b_type = backend_type.lower().strip()
    if b_type == "mock":
        return MockJudgeBackend()
    if b_type in ("openai", "vllm", "ollama"):
        return OpenAICompatibleJudgeBackend(
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout,
        )
    if b_type == "transformers":
        return TransformersJudgeBackend(model_name=model_name)
    raise ValueError(f"Unknown judge backend type: {backend_type!r}")


def _extract_grounded_ratio(output_metadata: dict[str, Any] | None) -> float:
    """Helper to extract grounded citation ratio from verification metadata."""
    if not output_metadata:
        return 1.0
    v_dict = output_metadata.get("citation_verification") or {}
    total = v_dict.get("total_citations", 0)
    if total == 0:
        return 1.0
    grounded = v_dict.get("grounded_count", 0)
    return float(grounded) / float(total)


def run_replay_simulation(
    dataset_path: Path | str = DEFAULT_DATASET,
    variant_a_id: str = DEFAULT_VARIANT_A,
    variant_b_id: str = DEFAULT_VARIANT_B,
    judge_backend: BaseJudgeBackend | None = None,
    pipeline: RAGPipeline | None = None,
    config_a: EasyPasalConfig | None = None,
    config_b: EasyPasalConfig | None = None,
    replay_logs_path: Path | str | None = None,
    n_resamples: int = 2000,
    n_permutations: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
    top_k: int = 5,
    mode: str = "dense",
    show_progress: bool = True,
) -> ReplaySimulationSummary:
    """Execute counterfactual replay-based A/B simulation on a query dataset.

    Args:
        dataset_path: Path to evaluation queries (JSONL).
        variant_a_id: Baseline variant identifier (e.g. 'baseline_v0.1').
        variant_b_id: Candidate variant identifier (e.g. 'gepa_v0.2').
        judge_backend: Initialized BaseJudgeBackend (defaults to MockJudgeBackend).
        pipeline: Optional pre-configured RAGPipeline.
        config_a: Optional EasyPasalConfig for Variant A.
        config_b: Optional EasyPasalConfig for Variant B.
        n_resamples: Bootstrap iterations for confidence intervals.
        n_permutations: Monte Carlo permutation test resamples.
        confidence: Desired CI confidence level (default 0.95).
        seed: Random seed for reproducibility.
        top_k: Number of retrieved passages.
        mode: Retrieval strategy ('dense', 'hybrid', etc.).
        show_progress: Whether to render a tqdm progress bar.

    Returns:
        ReplaySimulationSummary containing paired metric evaluations and item pairs.
    """
    items = load_gold_items(dataset_path)
    if not items:
        raise ValueError(f"No valid gold items found in {dataset_path}")

    eff_variant_a = config_a.variant_id if config_a is not None else variant_a_id
    eff_variant_b = config_b.variant_id if config_b is not None else variant_b_id

    judge = judge_backend or MockJudgeBackend()
    judge_name = judge.__class__.__name__

    # Initialize pipeline if not supplied
    if pipeline is None:
        try:
            cfg = config_a or EasyPasalConfig.load_default()
            pipeline = RAGPipeline.from_config(cfg, override_provider="mock")
        except Exception:
            # Fallback to pure mock pipeline
            gen = MockGeneratorBackend()
            pipeline = RAGPipeline(retriever=None, generator=gen)  # type: ignore[arg-type]

    replayed_pairs: list[ReplayedItemPair] = []
    logged_a: dict[str, Any] = {}
    logged_b: dict[str, Any] = {}
    if replay_logs_path is not None and Path(replay_logs_path).exists():
        try:
            with open(replay_logs_path, encoding="utf-8") as f:
                logs_data = json.load(f)
            summaries = logs_data.get("summaries", {})
            ds_str = str(dataset_path).lower()
            split_key = "test" if "test" in ds_str else ("train" if "train" in ds_str else "dev")
            if split_key in summaries:
                logged_a = {
                    it["item_id"]: it
                    for it in summaries[split_key].get("baseline", {}).get("items", [])
                }
                logged_b = {
                    it["item_id"]: it
                    for it in summaries[split_key].get("gepa", {}).get("items", [])
                }
                logger.info(
                    "Replaying historical responses from %s [split=%s]: %d base / %d gepa items",
                    replay_logs_path,
                    split_key,
                    len(logged_a),
                    len(logged_b),
                )
        except Exception as err:
            logger.warning(
                "Failed to load historical replay logs from %s: %s", replay_logs_path, err
            )

    logger.info(
        "Starting Replay A/B Simulation: %d queries | Variant A: %s vs Variant B: %s | Judge: %s",
        len(items),
        eff_variant_a,
        eff_variant_b,
        judge_name,
    )

    iterator = tqdm(items, desc="Replaying A/B Queries", disable=not show_progress)
    for item in iterator:
        # 1. Retrieve Candidate Passages (Once per query to ensure identical counterfactual context)
        chunks: list[RetrievedChunk] = []
        if getattr(pipeline, "retriever", None) is not None:
            retrieval_res = pipeline.retriever.retrieve(
                query=item.question,
                top_k=top_k,
                mode=mode,
                status_filter="berlaku",
            )
            chunks = retrieval_res.chunks

        # Construct context passage block for judge verification
        if chunks:
            context_text = "\n\n---\n\n".join(c.text for c in chunks if c.text)
        elif item.should_abstain:
            context_text = "Tidak ada ketentuan dalam korpus regulasi keuangan yang relevan untuk pertanyaan ini."
        else:
            context_text = f"Konteks regulasi untuk {', '.join(item.gold_citations)}"

        # 2 & 3. Obtain Responses for Variant A and Variant B (from replay logs or pipeline)
        if item.id in logged_a and item.id in logged_b:
            rec_a = logged_a[item.id]
            rec_b = logged_b[item.id]
            ans_a = rec_a["generated_answer"]
            cits_a = rec_a.get("extracted_citations", [])
            abstained_a = rec_a.get("did_abstain", False)
            grounded_a = float(rec_a.get("grounded_ratio", 1.0))
            lat_a = float(rec_a.get("latency_ms", 0.0))

            ans_b = rec_b["generated_answer"]
            cits_b = rec_b.get("extracted_citations", [])
            abstained_b = rec_b.get("did_abstain", False)
            grounded_b = float(rec_b.get("grounded_ratio", 1.0))
            lat_b = float(rec_b.get("latency_ms", 0.0))
        else:
            t0_a = time.perf_counter()
            out_a = pipeline.ask(
                question=item.question,
                language=item.language,
                top_k=top_k,
                mode=mode,
                prompt_variant="baseline",
                variant=eff_variant_a,
                retrieved_chunks=chunks,
                verify_citations=True,
                strict_grounding=False,
            )
            lat_a = (time.perf_counter() - t0_a) * 1000.0
            ans_a = out_a.answer
            cits_a = out_a.citations
            abstained_a = out_a.abstained
            grounded_a = _extract_grounded_ratio(out_a.metadata)

            t0_b = time.perf_counter()
            out_b = pipeline.ask(
                question=item.question,
                language=item.language,
                top_k=top_k,
                mode=mode,
                prompt_variant="gepa",
                variant=eff_variant_b,
                retrieved_chunks=chunks,
                verify_citations=True,
                strict_grounding=False,
            )
            lat_b = (time.perf_counter() - t0_b) * 1000.0
            ans_b = out_b.answer
            cits_b = out_b.citations
            abstained_b = out_b.abstained
            grounded_b = _extract_grounded_ratio(out_b.metadata)

        # 4. Evaluate both variants with the exact same judge backend
        faith_res_a = evaluate_faithfulness(
            backend=judge,
            question=item.question,
            context=context_text,
            answer=ans_a,
        )
        corr_res_a = evaluate_correctness(
            backend=judge,
            question=item.question,
            gold_answer=item.gold_answer,
            answer=ans_a,
            should_abstain=item.should_abstain,
        )

        faith_res_b = evaluate_faithfulness(
            backend=judge,
            question=item.question,
            context=context_text,
            answer=ans_b,
        )
        corr_res_b = evaluate_correctness(
            backend=judge,
            question=item.question,
            gold_answer=item.gold_answer,
            answer=ans_b,
            should_abstain=item.should_abstain,
        )

        # 5. Deterministic Citation and Grounding Metrics
        f1_art_a = citation_f1(cits_a, item.gold_citations, mode="article")
        f1_art_b = citation_f1(cits_b, item.gold_citations, mode="article")
        f1_ex_a = citation_f1(cits_a, item.gold_citations, mode="exact")
        f1_ex_b = citation_f1(cits_b, item.gold_citations, mode="exact")

        prec_a = citation_precision(cits_a, item.gold_citations, mode="article")
        prec_b = citation_precision(cits_b, item.gold_citations, mode="article")
        rec_a = citation_recall(cits_a, item.gold_citations, mode="article")
        rec_b = citation_recall(cits_b, item.gold_citations, mode="article")

        abstain_ok_a = abstained_a == item.should_abstain
        abstain_ok_b = abstained_b == item.should_abstain

        pair = ReplayedItemPair(
            item_id=item.id,
            category=item.category,
            theme=item.theme,
            question=item.question,
            gold_answer=item.gold_answer,
            gold_citations=item.gold_citations,
            should_abstain=item.should_abstain,
            variant_a_answer=ans_a,
            variant_a_citations=cits_a,
            variant_a_abstained=abstained_a,
            variant_a_grounded_ratio=grounded_a,
            variant_a_latency_ms=lat_a,
            variant_b_answer=ans_b,
            variant_b_citations=cits_b,
            variant_b_abstained=abstained_b,
            variant_b_grounded_ratio=grounded_b,
            variant_b_latency_ms=lat_b,
            faithfulness_a=faith_res_a.score,
            faithfulness_b=faith_res_b.score,
            correctness_a=corr_res_a.score,
            correctness_b=corr_res_b.score,
            citation_f1_article_a=f1_art_a,
            citation_f1_article_b=f1_art_b,
            citation_f1_exact_a=f1_ex_a,
            citation_f1_exact_b=f1_ex_b,
            citation_precision_a=prec_a,
            citation_precision_b=prec_b,
            citation_recall_a=rec_a,
            citation_recall_b=rec_b,
            abstention_correct_a=abstain_ok_a,
            abstention_correct_b=abstain_ok_b,
        )
        replayed_pairs.append(pair)

    # 6. Aggregate Paired Metrics and Compute Statistical Significance
    total_q = len(replayed_pairs)
    ans_pairs = [p for p in replayed_pairs if not p.should_abstain]
    unans_pairs = [p for p in replayed_pairs if p.should_abstain]

    eval_pairs = ans_pairs if ans_pairs else replayed_pairs
    current_seed = seed
    metrics_res: dict[str, PairedComparisonResult] = {}

    # Metric: Faithfulness (Judge)
    metrics_res["Faithfulness"] = paired_bootstrap_comparison(
        baseline=[p.faithfulness_a for p in eval_pairs],
        candidate=[p.faithfulness_b for p in eval_pairs],
        metric_name="Faithfulness",
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=current_seed,
        higher_is_better=True,
    )
    current_seed += 10

    # Metric: Answer Correctness (Judge)
    metrics_res["Correctness"] = paired_bootstrap_comparison(
        baseline=[p.correctness_a for p in eval_pairs],
        candidate=[p.correctness_b for p in eval_pairs],
        metric_name="Correctness",
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=current_seed,
        higher_is_better=True,
    )
    current_seed += 10

    # Metric: Grounding Fidelity
    metrics_res["Grounding Fidelity"] = paired_bootstrap_comparison(
        baseline=[p.variant_a_grounded_ratio for p in eval_pairs],
        candidate=[p.variant_b_grounded_ratio for p in eval_pairs],
        metric_name="Grounding Fidelity",
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=current_seed,
        higher_is_better=True,
    )
    current_seed += 10

    # Metric: Citation F1 (Article)
    metrics_res["Citation F1 (Article)"] = paired_bootstrap_comparison(
        baseline=[p.citation_f1_article_a for p in eval_pairs],
        candidate=[p.citation_f1_article_b for p in eval_pairs],
        metric_name="Citation F1 (Article)",
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=current_seed,
        higher_is_better=True,
    )
    current_seed += 10

    # Metric: Citation F1 (Exact)
    metrics_res["Citation F1 (Exact)"] = paired_bootstrap_comparison(
        baseline=[p.citation_f1_exact_a for p in eval_pairs],
        candidate=[p.citation_f1_exact_b for p in eval_pairs],
        metric_name="Citation F1 (Exact)",
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=current_seed,
        higher_is_better=True,
    )
    current_seed += 10

    # Metric: Citation Precision
    metrics_res["Citation Precision"] = paired_bootstrap_comparison(
        baseline=[p.citation_precision_a for p in eval_pairs],
        candidate=[p.citation_precision_b for p in eval_pairs],
        metric_name="Citation Precision",
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=current_seed,
        higher_is_better=True,
    )
    current_seed += 10

    # Metric: Citation Recall
    metrics_res["Citation Recall"] = paired_bootstrap_comparison(
        baseline=[p.citation_recall_a for p in eval_pairs],
        candidate=[p.citation_recall_b for p in eval_pairs],
        metric_name="Citation Recall",
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=current_seed,
        higher_is_better=True,
    )
    current_seed += 10

    # Metric: Abstention Accuracy (Evaluated over all queries)
    metrics_res["Abstention Accuracy"] = paired_bootstrap_comparison(
        baseline=[1.0 if p.abstention_correct_a else 0.0 for p in replayed_pairs],
        candidate=[1.0 if p.abstention_correct_b else 0.0 for p in replayed_pairs],
        metric_name="Abstention Accuracy",
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=current_seed,
        higher_is_better=True,
    )
    current_seed += 10

    # Metric: Latency (Evaluated over all queries, lower is better)
    metrics_res["Latency (mean ms)"] = paired_bootstrap_comparison(
        baseline=[p.variant_a_latency_ms for p in replayed_pairs],
        candidate=[p.variant_b_latency_ms for p in replayed_pairs],
        metric_name="Latency (mean ms)",
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=current_seed,
        higher_is_better=False,
    )
    current_seed += 10

    # Category Breakdowns for Key Metrics (Faithfulness, Correctness, Grounding)
    cat_to_pairs: dict[str, list[ReplayedItemPair]] = defaultdict(list)
    for p in eval_pairs:
        cat_to_pairs[p.category].append(p)

    category_metrics: dict[str, dict[str, PairedComparisonResult]] = defaultdict(dict)
    for cat, c_pairs in cat_to_pairs.items():
        if len(c_pairs) >= 2:
            category_metrics[cat]["Faithfulness"] = paired_bootstrap_comparison(
                baseline=[p.faithfulness_a for p in c_pairs],
                candidate=[p.faithfulness_b for p in c_pairs],
                metric_name=f"{cat} Faithfulness",
                n_resamples=n_resamples,
                n_permutations=n_permutations,
                confidence=confidence,
                seed=current_seed,
                higher_is_better=True,
            )
            current_seed += 10
            category_metrics[cat]["Correctness"] = paired_bootstrap_comparison(
                baseline=[p.correctness_a for p in c_pairs],
                candidate=[p.correctness_b for p in c_pairs],
                metric_name=f"{cat} Correctness",
                n_resamples=n_resamples,
                n_permutations=n_permutations,
                confidence=confidence,
                seed=current_seed,
                higher_is_better=True,
            )
            current_seed += 10

    return ReplaySimulationSummary(
        timestamp=datetime.now(UTC).isoformat(),
        dataset_path=str(dataset_path),
        variant_a_id=eff_variant_a,
        variant_b_id=eff_variant_b,
        judge_backend=judge_name,
        disclaimer=DISCLAIMER_TEXT,
        total_queries=total_q,
        answerable_queries=len(ans_pairs),
        unanswerable_queries=len(unans_pairs),
        metrics=metrics_res,
        category_metrics=dict(category_metrics),
        items=replayed_pairs,
    )


def format_replay_table(summary: ReplaySimulationSummary) -> str:
    """Format simulation summary into a clean ASCII comparison table."""
    lines: list[str] = []
    w = 120
    lines.append("=" * w)
    lines.append("📊 EasyPasal Counterfactual Replay-Based A/B Simulation (T6.1)".center(w))
    lines.append("=" * w)
    lines.append(f"⚠️  {summary.disclaimer}")
    lines.append("-" * w)
    lines.append(f"Dataset:            {summary.dataset_path}")
    lines.append(f"Variant A (Base):   {summary.variant_a_id} (baseline prompt)")
    lines.append(f"Variant B (Cand):   {summary.variant_b_id} (GEPA-optimized prompt)")
    lines.append(f"Judge Backend:      {summary.judge_backend}")
    lines.append(
        f"Queries Replayed:   {summary.total_queries} total "
        f"({summary.answerable_queries} answerable, {summary.unanswerable_queries} unanswerable)"
    )
    lines.append(
        "Statistical Gate:   95% Bootstrap CI (B=2,000 resamples), Paired Permutations (P=5,000)"
    )
    lines.append("-" * w)

    # Table Header
    header = (
        f"{'Metric':<24} | {'Variant A [95% CI]':<26} | {'Variant B [95% CI]':<26} | "
        f"{'Paired Delta [95% CI]':<26} | {'p-val':<7} | {'Status'}"
    )
    lines.append(header)
    lines.append("-" * w)

    for name, res in summary.metrics.items():
        base_str = f"{res.baseline_mean:.4f} [{res.baseline_ci.lower_bound:.4f}, {res.baseline_ci.upper_bound:.4f}]"
        cand_str = f"{res.candidate_mean:.4f} [{res.candidate_ci.lower_bound:.4f}, {res.candidate_ci.upper_bound:.4f}]"
        delta_str = f"{res.delta_mean:+.4f} [{res.delta_ci.lower_bound:+.4f}, {res.delta_ci.upper_bound:+.4f}]"
        pval_str = f"{res.p_value:.4f}" if res.p_value >= 0.0001 else "<0.0001"

        row = (
            f"{name:<24} | {base_str:<26} | {cand_str:<26} | "
            f"{delta_str:<26} | {pval_str:<7} | {res.status_badge}"
        )
        lines.append(row)

    lines.append("-" * w)
    lines.append(
        f"Summary: {summary.total_significant_gains} significant gain(s), "
        f"{summary.total_regressions} regression(s), "
        f"{summary.total_neutral} neutral"
    )

    if summary.category_metrics:
        lines.append("-" * w)
        lines.append("📁 Per-Category Faithfulness & Correctness Breakdown:")
        cat_header = f"{'Category':<28} | {'Metric':<14} | {'Paired Delta [95% CI]':<26} | {'p-val':<7} | {'Status'}"
        lines.append(cat_header)
        lines.append("-" * w)
        for cat, cat_res in sorted(summary.category_metrics.items()):
            for m_key, res in cat_res.items():
                delta_str = f"{res.delta_mean:+.4f} [{res.delta_ci.lower_bound:+.4f}, {res.delta_ci.upper_bound:+.4f}]"
                pval_str = f"{res.p_value:.4f}" if res.p_value >= 0.0001 else "<0.0001"
                cat_row = f"{cat:<28} | {m_key:<14} | {delta_str:<26} | {pval_str:<7} | {res.status_badge}"
                lines.append(cat_row)
        lines.append("-" * w)

    lines.append("=" * w)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for replay_ab_simulation."""
    parser = argparse.ArgumentParser(
        description="EasyPasal Counterfactual Replay-Based A/B Simulation (T6.1)",
        epilog=(
            "Example:\n"
            "  uv run python -m eval.replay_ab_simulation \\\n"
            "    --dataset data/gold/dev.jsonl \\\n"
            "    --output results/t6_1_replay_ab_simulation_dev.json\n\n"
            "Makefile shortcut:\n"
            "  make replay-ab"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(DEFAULT_DATASET),
        help=f"Path to evaluation queries JSONL (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--variant-a",
        default=DEFAULT_VARIANT_A,
        help=f"Variant A identifier / config (default: {DEFAULT_VARIANT_A})",
    )
    parser.add_argument(
        "--variant-b",
        default=DEFAULT_VARIANT_B,
        help=f"Variant B identifier / config (default: {DEFAULT_VARIANT_B})",
    )
    parser.add_argument(
        "--judge-backend",
        default="mock",
        choices=["mock", "openai", "vllm", "ollama", "transformers"],
        help="Backend to use for LLM judging (default: mock)",
    )
    parser.add_argument(
        "--judge-api-base",
        default="http://localhost:8000/v1",
        help="Base URL for OpenAI-compatible judge server (vLLM/Ollama)",
    )
    parser.add_argument(
        "--judge-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name for judge server",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output path for JSON simulation results (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Bootstrap resamples for confidence intervals (default: 2000)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help="Permutations for significance testing (default: 5000)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence interval level (default: 0.95)",
    )
    parser.add_argument(
        "--replay-logs",
        type=Path,
        default=Path("results/e5_gepa_results.json"),
        help="Optional path to historical generation results to replay (default: results/e5_gepa_results.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        judge = get_judge_backend(
            backend_type=args.judge_backend,
            api_base=args.judge_api_base,
            model_name=args.judge_model,
        )

        summary = run_replay_simulation(
            dataset_path=args.dataset,
            variant_a_id=args.variant_a,
            variant_b_id=args.variant_b,
            judge_backend=judge,
            replay_logs_path=args.replay_logs,
            n_resamples=args.n_resamples,
            n_permutations=args.n_permutations,
            confidence=args.confidence,
            seed=args.seed,
        )

        table_str = format_replay_table(summary)
        print(table_str)

        summary.save_json(args.output)
        logger.info("Saved simulation report to %s", args.output)

    except Exception as exc:
        logger.error("Replay simulation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
