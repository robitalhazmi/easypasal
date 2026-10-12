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
    from eval.split_eval import (
        split_dataset,
        stratified_split,
    )
    from eval.validate_gold import (
        ValidationReport,
        validate_gold_dataset,
    )

__all__ = [
    "AgreementReport",
    "GoldItem",
    "PassageRecord",
    "ValidationReport",
    "draft_candidate_questions",
    "evaluate_agreement",
    "filter_passages",
    "generate_blank_template",
    "generate_draft_items_batch",
    "load_passages",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
