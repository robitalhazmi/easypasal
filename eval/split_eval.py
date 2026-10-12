"""Stratified train/dev/test dataset splitting with pinned random seed.

Splits data/gold/gold_dataset.jsonl into dev.jsonl and test.jsonl
with category and abstention stratification.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from eval.labeling_helper import VALID_CATEGORIES, GoldItem

logger = logging.getLogger("easypasal.eval.split_eval")

DEFAULT_INPUT_PATH = Path("data/gold/gold_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/gold")
DEFAULT_SEED = 42
DEFAULT_DEV_RATIO = 0.4


def load_gold_items(dataset_path: Path) -> list[GoldItem]:
    """Load and validate all GoldItems from a JSONL file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    items: list[GoldItem] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item_dict = json.loads(line_str)
                items.append(GoldItem.model_validate(item_dict))
            except Exception as err:
                raise ValueError(f"Error parsing line {line_num} in {dataset_path}: {err}") from err

    return items


def stratified_split(
    items: list[GoldItem],
    dev_ratio: float = DEFAULT_DEV_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[list[GoldItem], list[GoldItem]]:
    """Partition items into dev and test splits stratified by category and abstention status.

    Returns:
        tuple[list[GoldItem], list[GoldItem]]: (dev_items, test_items)
    """
    if not (0.0 < dev_ratio < 1.0):
        raise ValueError(f"dev_ratio must be between 0.0 and 1.0, got {dev_ratio}")

    rng = random.Random(seed)

    # Stratify by (category, should_abstain)
    strata: dict[tuple[str, bool], list[GoldItem]] = defaultdict(list)
    for it in items:
        strata[(it.category, it.should_abstain)].append(it)

    dev_items: list[GoldItem] = []
    test_items: list[GoldItem] = []

    # Sort keys for deterministic processing order
    for stratum_key in sorted(strata.keys()):
        stratum_list = list(strata[stratum_key])
        # Sort items by id first for canonical order, then shuffle with seeded RNG
        stratum_list.sort(key=lambda x: x.id)
        rng.shuffle(stratum_list)

        n_total = len(stratum_list)
        if n_total == 1:
            # If only 1 item in stratum, put it in test
            n_dev = 0
        else:
            n_dev = int(round(n_total * dev_ratio))
            # Ensure at least 1 in test and at least 1 in dev if n_total >= 2
            n_dev = max(1, min(n_total - 1, n_dev))

        dev_items.extend(stratum_list[:n_dev])
        test_items.extend(stratum_list[n_dev:])

    # Sort splits by item ID for clean, deterministic file serialization
    dev_items.sort(key=lambda x: x.id)
    test_items.sort(key=lambda x: x.id)

    return dev_items, test_items


def save_jsonl(items: list[GoldItem], output_file: Path) -> None:
    """Serialize list of GoldItems to a JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def print_split_summary(
    dev_items: list[GoldItem],
    test_items: list[GoldItem],
    dev_path: Path,
    test_path: Path,
) -> None:
    """Print formatted terminal tables showing category and theme distribution across splits."""
    total = len(dev_items) + len(test_items)
    dev_pct = (len(dev_items) / total * 100) if total else 0.0
    test_pct = (len(test_items) / total * 100) if total else 0.0

    print("=" * 75)
    print("✂️  EasyPasal Dataset Split Summary")
    print("=" * 75)
    print(f"Total items partitioned : {total}")
    print(f"Dev split ({dev_path}) : {len(dev_items):>3} items ({dev_pct:.1f}%)")
    print(f"Test split ({test_path}) : {len(test_items):>3} items ({test_pct:.1f}%)")
    print()

    print(f"{'Category':<30} | {'Dev':<6} | {'Test':<6} | {'Total':<6} | {'Dev %':<6}")
    print("-" * 75)

    dev_cat_counts: dict[str, int] = defaultdict(int)
    for it in dev_items:
        dev_cat_counts[it.category] += 1

    test_cat_counts: dict[str, int] = defaultdict(int)
    for it in test_items:
        test_cat_counts[it.category] += 1

    for cat in VALID_CATEGORIES:
        d_cnt = dev_cat_counts.get(cat, 0)
        t_cnt = test_cat_counts.get(cat, 0)
        tot = d_cnt + t_cnt
        pct = (d_cnt / tot * 100) if tot else 0.0
        print(f"{cat:<30} | {d_cnt:>6} | {t_cnt:>6} | {tot:>6} | {pct:>5.1f}%")

    print("-" * 75)
    print()

    print("🏛️ Domain Theme Breakdown:")
    dev_theme_counts: dict[str, int] = defaultdict(int)
    for it in dev_items:
        dev_theme_counts[it.theme] += 1
    test_theme_counts: dict[str, int] = defaultdict(int)
    for it in test_items:
        test_theme_counts[it.theme] += 1

    all_themes = sorted(set(dev_theme_counts.keys()) | set(test_theme_counts.keys()))
    print(f"{'Theme':<30} | {'Dev':<6} | {'Test':<6} | {'Total':<6}")
    print("-" * 55)
    for th in all_themes:
        d_cnt = dev_theme_counts.get(th, 0)
        t_cnt = test_theme_counts.get(th, 0)
        print(f"{th:<30} | {d_cnt:>6} | {t_cnt:>6} | {d_cnt + t_cnt:>6}")
    print("=" * 75)


def split_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    """Execute stratified dataset splitting, saving dev.jsonl and test.jsonl."""
    logger.info("Loading gold dataset from %s...", input_path)
    items = load_gold_items(input_path)

    dev_items, test_items = stratified_split(items, dev_ratio=dev_ratio, seed=seed)

    dev_path = output_dir / "dev.jsonl"
    test_path = output_dir / "test.jsonl"

    save_jsonl(dev_items, dev_path)
    save_jsonl(test_items, test_path)

    logger.info(
        "Successfully wrote %d dev items to %s and %d test items to %s",
        len(dev_items),
        dev_path,
        len(test_items),
        test_path,
    )

    print_split_summary(dev_items, test_items, dev_path, test_path)
    return dev_path, test_path


def main() -> int:
    """CLI entrypoint for dataset splitting."""
    parser = argparse.ArgumentParser(
        description="Stratified train/dev/test dataset splitting with pinned seed."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to input gold JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save dev.jsonl and test.jsonl.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=DEFAULT_DEV_RATIO,
        help="Proportion of items allocated to dev split (default: 0.4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic reproducibility (default: 42).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    split_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
