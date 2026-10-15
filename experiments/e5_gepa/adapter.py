"""EasyPasal GEPA Adapter for Reflective Prompt Optimization (T5.1).

Implements the GEPAAdapter protocol to interface EasyPasal's RAG pipeline with
the Genetic-Pareto reflective prompt evolution engine.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from eval.labeling_helper import GoldItem
from eval.metrics.abstention import abstention_accuracy
from eval.metrics.citations import citation_f1
from experiments.e5_gepa.proposer import EasyPasalReflectiveProposer
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from tqdm import tqdm

logger = logging.getLogger("easypasal.experiments.e5_gepa.adapter")


def compute_lexical_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 overlap between predicted and gold strings."""
    toks_pred = re.findall(r"\w+", pred.lower())
    toks_gold = re.findall(r"\w+", gold.lower())
    if not toks_pred and not toks_gold:
        return 1.0
    if not toks_pred or not toks_gold:
        return 0.0
    c_pred = Counter(toks_pred)
    c_gold = Counter(toks_gold)
    common = sum((c_pred & c_gold).values())
    if common == 0:
        return 0.0
    p = common / len(toks_pred)
    r = common / len(toks_gold)
    return (2.0 * p * r) / (p + r)


@dataclass
class GEPATrajectory:
    """Rich diagnostic trajectory capturing execution state for reflection."""

    item_id: str
    category: str
    question: str
    gold_answer: str
    gold_citations: list[str]
    should_abstain: bool
    generated_answer: str
    extracted_citations: list[str]
    did_abstain: bool
    citation_f1: float
    lexical_f1: float
    grounded_ratio: float
    ungrounded_citations: list[str]
    hallucinated_pasal: list[str]
    diagnostic_feedback: str


@dataclass
class GEPARolloutOutput:
    """Summary of program outputs for a single evaluation item."""

    item_id: str
    answer: str
    citations: list[str]
    abstained: bool
    score: float
    metrics: dict[str, float] = field(default_factory=dict)


class EasyPasalGEPAAdapter(GEPAAdapter[GoldItem, GEPATrajectory, GEPARolloutOutput]):
    """GEPA Adapter for optimizing EasyPasal legal RAG system prompts."""

    def __init__(
        self,
        pipeline: Any,
        top_k: int = 5,
        mode: str = "dense",
        strict_grounding: bool = False,
        proposer: Any | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.top_k = top_k
        self.mode = mode
        self.strict_grounding = strict_grounding
        self.proposer = proposer or EasyPasalReflectiveProposer()

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose refined prompt texts based on reflective error traces."""
        return self.proposer.propose(
            candidate=candidate,
            reflective_dataset=reflective_dataset,
            components_to_update=components_to_update,
        )

    def evaluate(
        self,
        batch: list[GoldItem],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[GEPATrajectory, GEPARolloutOutput]:
        """Execute candidate prompt on batch of items and produce multi-objective scores and traces."""
        candidate_prompt = candidate.get("system_prompt", "")
        outputs: list[GEPARolloutOutput] = []
        scores: list[float] = []
        trajectories: list[GEPATrajectory] = []

        desc = "GEPA Train Rollout" if capture_traces else "GEPA Val Eval"
        iterator = (
            tqdm(batch, desc=desc, leave=False, dynamic_ncols=True) if len(batch) > 1 else batch
        )
        for item in iterator:
            try:
                # 1. Run pipeline inference with candidate prompt override
                res = self.pipeline.ask(
                    question=item.question,
                    language=item.language,
                    top_k=self.top_k,
                    mode=self.mode,
                    system_prompt_override=candidate_prompt if candidate_prompt else None,
                    verify_citations=True,
                    strict_grounding=self.strict_grounding,
                )

                ans_text = res.answer
                cits = res.citations
                abstained = res.abstained

                # 2. Extract verification details
                meta = getattr(res, "metadata", None) or {}
                v_dict = (
                    meta.get("citation_verification")
                    or getattr(res, "citation_verification", {})
                    or {}
                )
                grounded_ratio = (
                    v_dict.get("grounded_count", 0) / max(v_dict.get("total_citations", 1), 1)
                    if v_dict
                    else 1.0
                )
                ungrounded = v_dict.get("ungrounded_citations", [])
                hallucinated = v_dict.get("hallucinated_pasal", [])

                # 3. Compute deterministic compliance metrics
                art_f1 = citation_f1(cits, item.gold_citations, mode="article")
                exact_f1 = citation_f1(cits, item.gold_citations, mode="exact")

                abs_acc = abstention_accuracy([abstained], [item.should_abstain])
                lex_f1 = compute_lexical_f1(ans_text, item.gold_answer)

                # 4. Multi-objective composite scoring:
                #    If should_abstain: 100% weight on correct abstention.
                #    If answerable: balance citation precision/recall (40%), grounding (30%), lexical (30%).
                if item.should_abstain:
                    composite_score = 1.0 if (abstained and abs_acc == 1.0) else 0.0
                else:
                    if abstained:
                        # Improper abstention on answerable query
                        composite_score = 0.0
                    else:
                        composite_score = 0.40 * art_f1 + 0.30 * grounded_ratio + 0.30 * lex_f1
                        # Penalize hallucinated pasal
                        if hallucinated:
                            composite_score *= 0.5

                scores.append(round(composite_score, 4))
                outputs.append(
                    GEPARolloutOutput(
                        item_id=item.id,
                        answer=ans_text,
                        citations=cits,
                        abstained=abstained,
                        score=round(composite_score, 4),
                        metrics={
                            "citation_f1_article": art_f1,
                            "citation_f1_exact": exact_f1,
                            "abstention_accuracy": abs_acc,
                            "lexical_f1": lex_f1,
                            "grounded_ratio": grounded_ratio,
                        },
                    )
                )

                # 5. Diagnostic Feedback Construction
                diagnostics: list[str] = []
                if item.should_abstain and not abstained:
                    diagnostics.append(
                        f"FAILED ABSTENTION: Query is out-of-scope/adversarial ({item.category}), "
                        "but model generated an answer instead of using the standard abstention formula."
                    )
                elif not item.should_abstain and abstained:
                    diagnostics.append(
                        "FALSE ABSTENTION: Query has legal basis in context, but model improperly refused to answer."
                    )
                else:
                    if art_f1 < 0.5:
                        diagnostics.append(
                            f"MISSED CITATIONS: Expected {item.gold_citations}, but model cited {cits}."
                        )
                    if ungrounded:
                        diagnostics.append(
                            f"UNGROUNDED CITATIONS: Cited articles {ungrounded} not found in retrieved passages."
                        )
                    if hallucinated:
                        diagnostics.append(
                            f"HALLUCINATION: Fabricated pasal designations: {hallucinated}."
                        )

                diag_str = (
                    " | ".join(diagnostics)
                    if diagnostics
                    else "PERFECT: Response correctly cited and grounded."
                )

                if capture_traces:
                    trajectories.append(
                        GEPATrajectory(
                            item_id=item.id,
                            category=item.category,
                            question=item.question,
                            gold_answer=item.gold_answer,
                            gold_citations=item.gold_citations,
                            should_abstain=item.should_abstain,
                            generated_answer=ans_text,
                            extracted_citations=cits,
                            did_abstain=abstained,
                            citation_f1=art_f1,
                            lexical_f1=lex_f1,
                            grounded_ratio=grounded_ratio,
                            ungrounded_citations=ungrounded,
                            hallucinated_pasal=hallucinated,
                            diagnostic_feedback=diag_str,
                        )
                    )

            except Exception as exc:
                logger.warning("Evaluation failure on %s: %s", item.id, exc)
                scores.append(0.0)
                outputs.append(
                    GEPARolloutOutput(
                        item_id=item.id,
                        answer="",
                        citations=[],
                        abstained=False,
                        score=0.0,
                    )
                )
                if capture_traces:
                    trajectories.append(
                        GEPATrajectory(
                            item_id=item.id,
                            category=item.category,
                            question=item.question,
                            gold_answer=item.gold_answer,
                            gold_citations=item.gold_citations,
                            should_abstain=item.should_abstain,
                            generated_answer="",
                            extracted_citations=[],
                            did_abstain=False,
                            citation_f1=0.0,
                            lexical_f1=0.0,
                            grounded_ratio=0.0,
                            ungrounded_citations=[],
                            hallucinated_pasal=[],
                            diagnostic_feedback=f"ERROR: Exception during inference: {exc}",
                        )
                    )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[GEPATrajectory, GEPARolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Extract high-signal diagnostic error traces to drive prompt reflection and mutation."""
        _ = candidate
        records: list[dict[str, Any]] = []

        if eval_batch.trajectories:
            for traj in eval_batch.trajectories:
                # Only include suboptimal cases in the reflective dataset
                if "PERFECT" not in traj.diagnostic_feedback:
                    records.append(
                        {
                            "Inputs": {
                                "item_id": traj.item_id,
                                "category": traj.category,
                                "question": traj.question,
                            },
                            "Generated Outputs": {
                                "answer": traj.generated_answer[:200] + "..."
                                if len(traj.generated_answer) > 200
                                else traj.generated_answer,
                                "citations": traj.extracted_citations,
                                "abstained": traj.did_abstain,
                            },
                            "Feedback": traj.diagnostic_feedback,
                        }
                    )

        result: dict[str, Sequence[Mapping[str, Any]]] = {}
        for comp in components_to_update:
            result[comp] = records[:8]  # Limit to top 8 high-signal reflections
        return result
