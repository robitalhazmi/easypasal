"""Evaluation suite: metrics, judge calibration, offline runners, and labeling helpers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.compute_self_agreement import (
        AgreementReport,
        evaluate_agreement,
    )
    from eval.labeling_helper import (
        GoldItem,
        PassageRecord,
        draft_candidate_questions,
        filter_passages,
        generate_blank_template,
        generate_draft_items_batch,
        load_passages,
        save_draft_item,
    )
    from eval.metrics import (
        AbstentionMetrics,
        CitationMetrics,
        RetrievalMetrics,
        abstention_accuracy,
        abstention_f1,
        abstention_precision,
        abstention_recall,
        citation_f1,
        citation_precision,
        citation_recall,
        compute_abstention_metrics,
        compute_citation_metrics,
        compute_retrieval_metrics,
        hit_at_k,
        ndcg_at_k,
        recall_at_k,
        reciprocal_rank,
    )
    from eval.run_eval import (
        CategoryEvalSummary,
        EvalReport,
        ItemEvalResult,
        evaluate_retrieval,
        log_to_mlflow,
    )
    from eval.split_eval import (
        split_dataset,
        stratified_split,
    )
    from eval.validate_gold import (
        ValidationReport,
        validate_gold_dataset,
    )

__all__ = [
    "AbstentionMetrics",
    "AgreementReport",
    "CategoryEvalSummary",
    "CitationMetrics",
    "EvalReport",
    "GoldItem",
    "ItemEvalResult",
    "PassageRecord",
    "RetrievalMetrics",
    "ValidationReport",
    "abstention_accuracy",
    "abstention_f1",
    "abstention_precision",
    "abstention_recall",
    "citation_f1",
    "citation_precision",
    "citation_recall",
    "compute_abstention_metrics",
    "compute_citation_metrics",
    "compute_retrieval_metrics",
    "draft_candidate_questions",
    "evaluate_agreement",
    "evaluate_retrieval",
    "filter_passages",
    "generate_blank_template",
    "generate_draft_items_batch",
    "hit_at_k",
    "load_passages",
    "log_to_mlflow",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "save_draft_item",
    "split_dataset",
    "stratified_split",
    "validate_gold_dataset",
]

_HELPER_SYMBOLS = {
    "GoldItem",
    "PassageRecord",
    "draft_candidate_questions",
    "filter_passages",
    "generate_blank_template",
    "generate_draft_items_batch",
    "load_passages",
    "save_draft_item",
}

_VALIDATOR_SYMBOLS = {
    "ValidationReport",
    "validate_gold_dataset",
}

_SPLITTER_SYMBOLS = {
    "split_dataset",
    "stratified_split",
}

_AGREEMENT_SYMBOLS = {
    "AgreementReport",
    "evaluate_agreement",
}

_METRIC_SYMBOLS = {
    "AbstentionMetrics",
    "CitationMetrics",
    "RetrievalMetrics",
    "abstention_accuracy",
    "abstention_f1",
    "abstention_precision",
    "abstention_recall",
    "citation_f1",
    "citation_precision",
    "citation_recall",
    "compute_abstention_metrics",
    "compute_citation_metrics",
    "compute_retrieval_metrics",
    "hit_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
}

_RUNNER_SYMBOLS = {
    "CategoryEvalSummary",
    "EvalReport",
    "ItemEvalResult",
    "evaluate_retrieval",
    "log_to_mlflow",
}


def __getattr__(name: str) -> object:
    if name in _HELPER_SYMBOLS:
        import eval.labeling_helper as _helper

        return getattr(_helper, name)
    if name in _VALIDATOR_SYMBOLS:
        import eval.validate_gold as _validator

        return getattr(_validator, name)
    if name in _SPLITTER_SYMBOLS:
        import eval.split_eval as _splitter

        return getattr(_splitter, name)
    if name in _AGREEMENT_SYMBOLS:
        import eval.compute_self_agreement as _agreement

        return getattr(_agreement, name)
    if name in _METRIC_SYMBOLS:
        import eval.metrics as _metrics

        return getattr(_metrics, name)
    if name in _RUNNER_SYMBOLS:
        import eval.run_eval as _runner

        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
