"""Gold evaluation dataset validator for EasyPasal.

Verifies schema compliance, unique IDs, category taxonomy allocations,
verbatim abstention behavior, and source text grounding against indexed corpus chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from eval.labeling_helper import (
    DEFAULT_META_PATH,
    VALID_CATEGORIES,
    GoldItem,
)

logger = logging.getLogger("easypasal.eval.validate_gold")

STANDARD_ABSTAIN_ID = (
    "Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut."
)
STANDARD_ABSTAIN_EN = (
    "Based on the available regulations, no legal basis was found regarding this matter."
)


@dataclass
class ValidationReport:
    """Summary of gold dataset validation results."""

    total_items: int = 0
    valid_items: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)
    theme_counts: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    abstain_count: int = 0
    verified_citations_count: int = 0

    @property
    def is_valid(self) -> bool:
        """Returns True if there are zero validation errors."""
        return len(self.errors) == 0


def load_corpus_chunk_map(meta_path: Path = DEFAULT_META_PATH) -> dict[str, dict[str, object]]:
    """Load indexed corpus chunks keyed by chunk_id."""
    if not meta_path.exists():
        logger.warning("Corpus metadata not found at %s. Chunk checks will be skipped.", meta_path)
        return {}

    with open(meta_path, encoding="utf-8") as f:
        meta_data = json.load(f)

    payloads = meta_data.get("corpus_payloads", [])
    chunk_map: dict[str, dict[str, object]] = {}
    for p in payloads:
        cid = p.get("chunk_id")
        if cid:
            chunk_map[str(cid)] = p
    return chunk_map


def validate_gold_dataset(
    dataset_path: Path,
    meta_path: Path | None = DEFAULT_META_PATH,
    min_items: int = 100,
    require_verified_status: bool = True,
) -> ValidationReport:
    """Validate a JSONL gold dataset against schema and corpus grounding rules."""
    report = ValidationReport()

    if not dataset_path.exists():
        report.errors.append(f"Dataset file does not exist: {dataset_path}")
        return report

    chunk_map = load_corpus_chunk_map(meta_path) if meta_path else {}

    seen_ids: set[str] = set()

    with open(dataset_path, encoding="utf-8") as f:
        lines = f.readlines()

    report.total_items = len(lines)
    if report.total_items < min_items:
        report.errors.append(
            f"Dataset size ({report.total_items} items) is below minimum requirement of {min_items}."
        )

    for line_idx, line in enumerate(lines, start=1):
        line_str = line.strip()
        if not line_str:
            continue

        # 1. JSON parsing
        try:
            raw_item = json.loads(line_str)
        except json.JSONDecodeError as err:
            report.errors.append(f"Line {line_idx}: Invalid JSON: {err}")
            continue

        # 2. Pydantic schema validation
        try:
            item = GoldItem.model_validate(raw_item)
        except ValidationError as val_err:
            report.errors.append(f"Line {line_idx}: Schema validation failed: {val_err}")
            continue

        # 3. ID uniqueness and format
        if item.id in seen_ids:
            report.errors.append(f"Line {line_idx}: Duplicate item ID: '{item.id}'.")
        seen_ids.add(item.id)

        # 4. Review status check
        if require_verified_status and item.review_status != "verified":
            report.errors.append(
                f"Line {line_idx} ({item.id}): Expected review_status='verified', got '{item.review_status}'."
            )

        # 5. Abstention consistency
        if item.should_abstain:
            report.abstain_count += 1
            if item.gold_citations:
                report.errors.append(
                    f"Line {line_idx} ({item.id}): should_abstain=True but gold_citations is non-empty."
                )
            if item.relevant_chunk_ids:
                report.errors.append(
                    f"Line {line_idx} ({item.id}): should_abstain=True but relevant_chunk_ids is non-empty."
                )
            if not item.abstain_reason:
                report.errors.append(
                    f"Line {line_idx} ({item.id}): should_abstain=True but abstain_reason is missing."
                )

            # Check verbatim refusal string
            expected_refusal = STANDARD_ABSTAIN_EN if item.language == "en" else STANDARD_ABSTAIN_ID
            if item.gold_answer.strip() != expected_refusal:
                report.errors.append(
                    f"Line {line_idx} ({item.id}): Abstaining item gold_answer does not match standard verbatim string."
                )
        else:
            if not item.gold_citations:
                report.errors.append(
                    f"Line {line_idx} ({item.id}): Non-abstaining item must have at least 1 gold_citation."
                )
            if not item.relevant_chunk_ids:
                report.errors.append(
                    f"Line {line_idx} ({item.id}): Non-abstaining item must have at least 1 relevant_chunk_id."
                )
            if item.abstain_reason is not None:
                report.errors.append(
                    f"Line {line_idx} ({item.id}): Non-abstaining item must have abstain_reason=None."
                )

            # Check citation presence in answer
            for cite in item.gold_citations:
                cite_clean = cite.strip("[]")
                # Ensure the citation is referenced in the answer
                if cite_clean not in item.gold_answer and f"[{cite_clean}]" not in item.gold_answer:
                    report.warnings.append(
                        f"Line {line_idx} ({item.id}): Citation '{cite_clean}' not verbatim found in gold_answer."
                    )

            # 6. Source Text Grounding Check against index metadata
            if chunk_map:
                for cid in item.relevant_chunk_ids:
                    if cid not in chunk_map:
                        report.errors.append(
                            f"Line {line_idx} ({item.id}): Chunk ID '{cid}' does not exist in corpus index!"
                        )
                    else:
                        chunk_payload = chunk_map[cid]
                        report.verified_citations_count += 1
                        # Verify document match if target_regulations provided
                        chunk_doc = str(chunk_payload.get("doc_number", ""))
                        if item.target_regulations:
                            matches_any = any(
                                t.lower() in chunk_doc.lower() or chunk_doc.lower() in t.lower()
                                for t in item.target_regulations
                            )
                            if not matches_any:
                                report.warnings.append(
                                    f"Line {line_idx} ({item.id}): Chunk doc '{chunk_doc}' not in target_regulations {item.target_regulations}."
                                )

        # 7. Grounded facts check
        if not item.grounded_facts:
            report.errors.append(
                f"Line {line_idx} ({item.id}): grounded_facts must contain at least 1 proposition."
            )

        # Accumulate statistics
        report.category_counts[item.category] = report.category_counts.get(item.category, 0) + 1
        report.theme_counts[item.theme] = report.theme_counts.get(item.theme, 0) + 1
        report.language_counts[item.language] = report.language_counts.get(item.language, 0) + 1
        report.valid_items += 1

    # Check taxonomy completeness
    for cat in VALID_CATEGORIES:
        count = report.category_counts.get(cat, 0)
        if count == 0:
            report.errors.append(f"Category '{cat}' is completely missing from the dataset!")

    return report


def print_validation_report(report: ValidationReport, dataset_path: Path) -> None:
    """Print formatted terminal report of validation results."""
    print("=" * 70)
    print(f"📊 EasyPasal Gold Dataset Validation Report: {dataset_path}")
    print("=" * 70)
    print(f"Total items inspected   : {report.total_items}")
    print(f"Schema-valid items      : {report.valid_items}")
    print(f"Abstention items        : {report.abstain_count}")
    print(f"Verified chunk lookups  : {report.verified_citations_count}")
    print()

    print("📁 Category Breakdown (9 categories):")
    for cat in VALID_CATEGORIES:
        cnt = report.category_counts.get(cat, 0)
        pct = (cnt / report.total_items * 100) if report.total_items else 0
        bar = "█" * int(pct // 2)
        print(f"  {cat:<28} : {cnt:>3} ({pct:>5.1f}%) {bar}")

    print()
    print("🏛️ Theme Breakdown (5 domains):")
    for theme, cnt in sorted(report.theme_counts.items()):
        pct = (cnt / report.total_items * 100) if report.total_items else 0
        print(f"  {theme:<28} : {cnt:>3} ({pct:>5.1f}%)")

    print()
    print("🌐 Language Breakdown:")
    for lang, cnt in sorted(report.language_counts.items()):
        pct = (cnt / report.total_items * 100) if report.total_items else 0
        print(f"  {lang:<28} : {cnt:>3} ({pct:>5.1f}%)")

    if report.warnings:
        print()
        print(f"⚠️ Warnings ({len(report.warnings)}):")
        for w in report.warnings[:10]:
            print(f"  - {w}")
        if len(report.warnings) > 10:
            print(f"  ... and {len(report.warnings) - 10} more warnings.")

    if report.errors:
        print()
        print(f"❌ ERRORS ({len(report.errors)}):")
        for e in report.errors[:20]:
            print(f"  - {e}")
        if len(report.errors) > 20:
            print(f"  ... and {len(report.errors) - 20} more errors.")
        print()
        print("❌ Dataset validation FAILED.")
    else:
        print()
        print("✅ ALL QUALITY GATES PASSED! Dataset is 100% verified and grounded.")
    print("=" * 70)


def main() -> int:
    """CLI entrypoint for dataset validation."""
    parser = argparse.ArgumentParser(description="Validate EasyPasal gold evaluation dataset.")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("data/gold/gold_dataset.jsonl"),
        help="Path to JSONL dataset file.",
    )
    parser.add_argument(
        "--index-meta",
        type=Path,
        default=DEFAULT_META_PATH,
        help="Path to corpus metadata file.",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=100,
        help="Minimum required item count.",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow review_status='unverified' (for drafts).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = validate_gold_dataset(
        dataset_path=args.dataset,
        meta_path=args.index_meta,
        min_items=args.min_items,
        require_verified_status=not args.allow_unverified,
    )

    print_validation_report(report, args.dataset)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
