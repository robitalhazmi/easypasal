"""Evaluation suite: metrics, judge calibration, offline runners, and labeling helpers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

__all__ = [
    "GoldItem",
    "PassageRecord",
    "draft_candidate_questions",
    "filter_passages",
    "generate_blank_template",
    "generate_draft_items_batch",
    "load_passages",
    "save_draft_item",
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


def __getattr__(name: str) -> object:
    if name in _HELPER_SYMBOLS:
        import eval.labeling_helper as _helper

        return getattr(_helper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
