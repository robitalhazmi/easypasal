"""Evaluation suite: metrics, judge calibration, offline runners, and labeling helpers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.chunk_mapper import ChunkSpanMapper
    from eval.compare_variants import (
        VariantComparisonReport,
        compare_judged_runs,
        compare_retrieval_runs,
        compare_variants,
        detect_report_type,
        format_comparison_table,
        load_report_json,
    )
    from eval.compute_judge_calibration import (
        CalibratedItem,
        CalibrationDiscrepancy,
        CategoryCalibrationSummary,
        JudgeCalibrationReport,
        evaluate_calibration,
        load_calibrated_items,
    )
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
        BaseJudgeBackend,
        BootstrapCI,
        CitationMetrics,
        CorrectnessResult,
        FaithfulnessResult,
        JudgedRecord,
        MockJudgeBackend,
        OpenAICompatibleJudgeBackend,
        PairedComparisonResult,
        RetrievalMetrics,
        TransformersJudgeBackend,
        abstention_accuracy,
        abstention_f1,
        abstention_precision,
        abstention_recall,
        bootstrap_ci,
        citation_f1,
        citation_precision,
        citation_recall,
        compute_abstention_metrics,
        compute_citation_metrics,
        compute_retrieval_metrics,
        evaluate_correctness,
        evaluate_faithfulness,
        evaluate_judged_item,
        hit_at_k,
        ndcg_at_k,
        paired_bootstrap_comparison,
        paired_permutation_test,
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
    from eval.smoke_eval import (
        CIGateFailureError,
        GateCheck,
        SmokeGateResult,
        evaluate_smoke_gate,
        load_smoke_baseline,
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
    "BaseJudgeBackend",
    "BootstrapCI",
    "CIGateFailureError",
    "CalibratedItem",
    "CalibrationDiscrepancy",
    "CategoryCalibrationSummary",
    "CategoryEvalSummary",
    "ChunkSpanMapper",
    "CitationMetrics",
    "CorrectnessResult",
    "EvalReport",
    "FaithfulnessResult",
    "GateCheck",
    "GoldItem",
    "ItemEvalResult",
    "JudgeCalibrationReport",
    "JudgedRecord",
    "MockJudgeBackend",
    "OpenAICompatibleJudgeBackend",
    "PairedComparisonResult",
    "PassageRecord",
    "RetrievalMetrics",
    "SmokeGateResult",
    "TransformersJudgeBackend",
    "ValidationReport",
    "VariantComparisonReport",
    "abstention_accuracy",
    "abstention_f1",
    "abstention_precision",
    "abstention_recall",
    "bootstrap_ci",
    "citation_f1",
    "citation_precision",
    "citation_recall",
    "compare_judged_runs",
    "compare_retrieval_runs",
    "compare_variants",
    "compute_abstention_metrics",
    "compute_citation_metrics",
    "compute_retrieval_metrics",
    "detect_report_type",
    "draft_candidate_questions",
    "evaluate_agreement",
    "evaluate_calibration",
    "evaluate_correctness",
    "evaluate_faithfulness",
    "evaluate_judged_item",
    "evaluate_retrieval",
    "evaluate_smoke_gate",
    "filter_passages",
    "format_comparison_table",
    "generate_blank_template",
    "generate_draft_items_batch",
    "hit_at_k",
    "load_calibrated_items",
    "load_passages",
    "load_report_json",
    "load_smoke_baseline",
    "log_to_mlflow",
    "ndcg_at_k",
    "paired_bootstrap_comparison",
    "paired_permutation_test",
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

_CALIBRATION_SYMBOLS = {
    "CalibratedItem",
    "CalibrationDiscrepancy",
    "CategoryCalibrationSummary",
    "JudgeCalibrationReport",
    "evaluate_calibration",
    "load_calibrated_items",
}

_METRIC_SYMBOLS = {
    "AbstentionMetrics",
    "BaseJudgeBackend",
    "BootstrapCI",
    "CitationMetrics",
    "CorrectnessResult",
    "FaithfulnessResult",
    "JudgedRecord",
    "MockJudgeBackend",
    "OpenAICompatibleJudgeBackend",
    "PairedComparisonResult",
    "RetrievalMetrics",
    "TransformersJudgeBackend",
    "abstention_accuracy",
    "abstention_f1",
    "abstention_precision",
    "abstention_recall",
    "bootstrap_ci",
    "citation_f1",
    "citation_precision",
    "citation_recall",
    "compute_abstention_metrics",
    "compute_citation_metrics",
    "compute_retrieval_metrics",
    "evaluate_correctness",
    "evaluate_faithfulness",
    "evaluate_judged_item",
    "hit_at_k",
    "ndcg_at_k",
    "paired_bootstrap_comparison",
    "paired_permutation_test",
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


_SMOKE_SYMBOLS = {
    "CIGateFailureError",
    "GateCheck",
    "SmokeGateResult",
    "evaluate_smoke_gate",
    "load_smoke_baseline",
}

_MAPPER_SYMBOLS = {
    "ChunkSpanMapper",
}

_COMPARISON_SYMBOLS = {
    "VariantComparisonReport",
    "compare_judged_runs",
    "compare_retrieval_runs",
    "compare_variants",
    "detect_report_type",
    "format_comparison_table",
    "load_report_json",
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
    if name in _CALIBRATION_SYMBOLS:
        import eval.compute_judge_calibration as _calibration

        return getattr(_calibration, name)
    if name in _METRIC_SYMBOLS:
        import eval.metrics as _metrics

        return getattr(_metrics, name)
    if name in _RUNNER_SYMBOLS:
        import eval.run_eval as _runner

        return getattr(_runner, name)
    if name in _SMOKE_SYMBOLS:
        import eval.smoke_eval as _smoke

        return getattr(_smoke, name)
    if name in _MAPPER_SYMBOLS:
        import eval.chunk_mapper as _mapper

        return getattr(_mapper, name)
    if name in _COMPARISON_SYMBOLS:
        import eval.compare_variants as _compare

        return getattr(_compare, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
