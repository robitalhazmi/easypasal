"""Quantitative self-agreement evaluation protocol for EasyPasal gold datasets (T2.5).

Measures internal consistency of gold annotations on a 20% sample:
1. Abstention Accuracy (target >= 95%)
2. Categorical Agreement / Cohen's Kappa (target >= 0.85)
3. Citation Jaccard Similarity (target >= 0.85)
4. Answer Token F1 overlap (target >= 0.80)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from eval.labeling_helper import GoldItem
from eval.split_eval import load_gold_items

logger = logging.getLogger("easypasal.eval.agreement")

DEFAULT_ORIGINAL_PATH = Path("data/gold/gold_dataset.jsonl")
DEFAULT_RELABEL_PATH = Path("data/gold/relabel_annotations.jsonl")
DEFAULT_SAMPLE_PATH = Path("data/gold/relabel_sample.jsonl")
DEFAULT_SEED = 42


@dataclass
class AgreementReport:
    """Summary of self-agreement consistency metrics."""

    total_items: int = 0
    abstention_agreement: float = 0.0
    cohens_kappa: float = 0.0
    citation_jaccard: float = 0.0
    token_f1: float = 0.0
    category_jaccard: dict[str, float] = None  # type: ignore[assignment]

    @property
    def satisfies_thresholds(self) -> bool:
        """Check if all metrics satisfy acceptance thresholds in docs/eval-methodology.md."""
        return (
            self.abstention_agreement >= 0.95
            and self.cohens_kappa >= 0.85
            and self.citation_jaccard >= 0.85
            and self.token_f1 >= 0.80
        )


def sample_for_relabeling(
    items: list[GoldItem],
    sample_ratio: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> list[GoldItem]:
    """Sample approximately 20% of items stratified by category."""
    rng = random.Random(seed)
    by_category: dict[str, list[GoldItem]] = defaultdict(list)
    for it in items:
        by_category[it.category].append(it)

    sampled: list[GoldItem] = []
    for cat in sorted(by_category.keys()):
        cat_items = list(by_category[cat])
        cat_items.sort(key=lambda x: x.id)
        rng.shuffle(cat_items)

        k = max(1, int(round(len(cat_items) * sample_ratio)))
        sampled.extend(cat_items[:k])

    sampled.sort(key=lambda x: x.id)
    return sampled


def compute_abstention_agreement(
    original: list[GoldItem],
    reannotated: list[GoldItem],
) -> float:
    """Compute proportion of identical should_abstain determinations."""
    if not original:
        return 0.0
    matches = sum(
        1
        for o, r in zip(original, reannotated, strict=True)
        if o.should_abstain == r.should_abstain
    )
    return matches / len(original)


def compute_cohens_kappa(
    original: list[GoldItem],
    reannotated: list[GoldItem],
) -> float:
    """Compute Cohen's Kappa for categorical decisions (should_abstain classification)."""
    n = len(original)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = (
        sum(
            1
            for o, r in zip(original, reannotated, strict=True)
            if o.should_abstain == r.should_abstain
        )
        / n
    )

    # Marginal proportions for class True and False
    orig_true = sum(1 for o in original if o.should_abstain) / n
    orig_false = 1.0 - orig_true
    re_true = sum(1 for r in reannotated if r.should_abstain) / n
    re_false = 1.0 - re_true

    p_e = (orig_true * re_true) + (orig_false * re_false)

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def normalize_citation(cite: str) -> str:
    """Normalize citation tag for robust string matching."""
    norm = cite.strip().strip("[]")
    norm = re.sub(r"\s+", " ", norm)
    return norm.lower()


def compute_citation_jaccard(
    orig_item: GoldItem,
    re_item: GoldItem,
) -> float:
    """Compute Jaccard similarity between two citation sets."""
    orig_cites = {normalize_citation(c) for c in orig_item.gold_citations}
    re_cites = {normalize_citation(c) for c in re_item.gold_citations}

    if not orig_cites and not re_cites:
        # Both empty (e.g. abstaining items) = perfect agreement
        return 1.0
    if not orig_cites or not re_cites:
        return 0.0

    intersection = len(orig_cites & re_cites)
    union = len(orig_cites | re_cites)
    return intersection / union if union > 0 else 1.0


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


def compute_item_token_f1(text1: str, text2: str) -> float:
    """Compute token-level F1 overlap between two answers."""
    toks1 = tokenize(text1)
    toks2 = tokenize(text2)

    if not toks1 and not toks2:
        return 1.0
    if not toks1 or not toks2:
        return 0.0

    counts1 = Counter(toks1)
    counts2 = Counter(toks2)

    common = sum((counts1 & counts2).values())
    if common == 0:
        return 0.0

    precision = common / len(toks2)
    recall = common / len(toks1)
    return (2 * precision * recall) / (precision + recall)


def evaluate_agreement(
    original_path: Path = DEFAULT_ORIGINAL_PATH,
    relabel_path: Path = DEFAULT_RELABEL_PATH,
) -> AgreementReport:
    """Calculate all consistency metrics between original and re-annotated items."""
    orig_all = load_gold_items(original_path)
    orig_map = {item.id: item for item in orig_all}

    reannotated = load_gold_items(relabel_path)
    aligned_original: list[GoldItem] = []

    for re_item in reannotated:
        if re_item.id not in orig_map:
            raise KeyError(f"Item {re_item.id} in relabeled set missing from original dataset!")
        aligned_original.append(orig_map[re_item.id])

    n = len(reannotated)
    if n == 0:
        raise ValueError("Relabeled dataset is empty!")

    abstention_acc = compute_abstention_agreement(aligned_original, reannotated)
    kappa = compute_cohens_kappa(aligned_original, reannotated)

    # Per-item Jaccard and per-category Jaccard
    jaccards = [
        compute_citation_jaccard(o, r) for o, r in zip(aligned_original, reannotated, strict=True)
    ]
    mean_jaccard = sum(jaccards) / n

    cat_jaccards: dict[str, list[float]] = defaultdict(list)
    for o, _r, j in zip(aligned_original, reannotated, jaccards, strict=True):
        cat_jaccards[o.category].append(j)

    mean_cat_jaccard = {cat: sum(vals) / len(vals) for cat, vals in cat_jaccards.items()}

    # Token F1
    f1s = [
        compute_item_token_f1(o.gold_answer, r.gold_answer)
        for o, r in zip(aligned_original, reannotated, strict=True)
    ]
    mean_f1 = sum(f1s) / n

    return AgreementReport(
        total_items=n,
        abstention_agreement=abstention_acc,
        cohens_kappa=kappa,
        citation_jaccard=mean_jaccard,
        token_f1=mean_f1,
        category_jaccard=mean_cat_jaccard,
    )


def print_agreement_report(report: AgreementReport) -> None:
    """Print formatted terminal report of self-agreement evaluation."""
    print("=" * 75)
    print("🤝 EasyPasal Self-Agreement Protocol Report (T2.5)")
    print("=" * 75)
    print(f"Sample items re-annotated : {report.total_items} items (20% sample)")
    print()

    print(f"{'Metric':<32} | {'Observed':<10} | {'Threshold':<10} | {'Status':<8}")
    print("-" * 75)

    def status_badge(val: float, thresh: float) -> str:
        return "✅ PASS" if val >= thresh else "❌ FAIL"

    kappa_label = "Cohen's Kappa (κ)"
    print(
        f"{'Abstention Agreement (Acc_abstain)':<32} | {report.abstention_agreement:>8.1%}  | {'>= 95.0%':<10} | {status_badge(report.abstention_agreement, 0.95)}"
    )
    print(
        f"{kappa_label:<32} | {report.cohens_kappa:>8.3f}  | {'>= 0.850':<10} | {status_badge(report.cohens_kappa, 0.85)}"
    )
    print(
        f"{'Citation Jaccard Similarity':<32} | {report.citation_jaccard:>8.1%}  | {'>= 85.0%':<10} | {status_badge(report.citation_jaccard, 0.85)}"
    )
    print(
        f"{'Answer Lexical Overlap (Token F1)':<32} | {report.token_f1:>8.1%}  | {'>= 80.0%':<10} | {status_badge(report.token_f1, 0.80)}"
    )
    print("-" * 75)
    print()

    if report.category_jaccard:
        print("📁 Citation Jaccard by Category:")
        for cat, jacc in sorted(report.category_jaccard.items()):
            bar = "█" * int(jacc * 20)
            print(f"  {cat:<28} : {jacc:>5.1%} {bar}")
        print()

    if report.satisfies_thresholds:
        print("🎉 ALL SELF-AGREEMENT QUALITY CRITERIA MET! Evaluation set is verified.")
    else:
        print("⚠️ SOME QUALITY CRITERIA FAILED TO MEET THE TARGET THRESHOLDS.")
    print("=" * 75)


def main() -> int:
    """CLI entrypoint for self-agreement operations."""
    parser = argparse.ArgumentParser(
        description="Evaluate annotator self-agreement on a 20% sample."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Extract blind relabeling template (id, category, question, language).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate consistency between original and relabeled annotations.",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=DEFAULT_ORIGINAL_PATH,
        help="Path to full gold dataset.",
    )
    parser.add_argument(
        "--relabeled",
        type=Path,
        default=DEFAULT_RELABEL_PATH,
        help="Path to re-annotated dataset.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help="Path to save blind relabeling questions.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.sample:
        items = load_gold_items(args.original)
        sampled = sample_for_relabeling(items)
        args.sample_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.sample_output, "w", encoding="utf-8") as f:
            for it in sampled:
                blind_record = {
                    "id": it.id,
                    "category": it.category,
                    "theme": it.theme,
                    "language": it.language,
                    "question": it.question,
                }
                f.write(json.dumps(blind_record) + "\n")
        print(f"✅ Exported {len(sampled)} blind items to {args.sample_output}")
        return 0

    # Default to --evaluate
    report = evaluate_agreement(args.original, args.relabeled)
    print_agreement_report(report)
    return 0 if report.satisfies_thresholds else 1


if __name__ == "__main__":
    sys.exit(main())
