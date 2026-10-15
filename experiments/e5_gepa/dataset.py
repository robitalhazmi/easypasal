"""Dataset partitioning for Experiment E5 (DSPy + GEPA Prompt Optimization).

Partitions dev.jsonl into e5_train.jsonl (for GEPA reflective evolution) and
e5_val.jsonl (for Pareto frontier tracking and model selection), preserving
test.jsonl as strictly held-out for final benchmark and overfitting audit.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

from eval.labeling_helper import GoldItem
from eval.split_eval import load_gold_items, save_jsonl

logger = logging.getLogger("easypasal.experiments.e5_gepa.dataset")

DEFAULT_DEV_PATH = Path("data/gold/dev.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/gold")
DEFAULT_E5_TRAIN_RATIO = 0.6
DEFAULT_SEED = 42


def split_dev_for_gepa(
    dev_items: list[GoldItem],
    train_ratio: float = DEFAULT_E5_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[list[GoldItem], list[GoldItem]]:
    """Partition dev items into train and val splits stratified by category and abstention.

    Args:
        dev_items: List of 40 GoldItems from dev.jsonl.
        train_ratio: Proportion assigned to training (default: 0.6 -> 24 items).
        seed: Random seed for exact determinism.

    Returns:
        tuple of (train_items, val_items).
    """
    rng = random.Random(seed)

    strata: dict[tuple[str, bool], list[GoldItem]] = defaultdict(list)
    for it in dev_items:
        strata[(it.category, it.should_abstain)].append(it)

    train_items: list[GoldItem] = []
    val_items: list[GoldItem] = []

    for stratum_key in sorted(strata.keys()):
        group = list(strata[stratum_key])
        group.sort(key=lambda x: x.id)
        rng.shuffle(group)

        n_total = len(group)
        n_train = 1 if n_total == 1 else max(1, min(n_total - 1, int(round(n_total * train_ratio))))

        train_items.extend(group[:n_train])
        val_items.extend(group[n_train:])

    train_items.sort(key=lambda x: x.id)
    val_items.sort(key=lambda x: x.id)
    return train_items, val_items


def prepare_e5_splits(
    dev_path: Path = DEFAULT_DEV_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_ratio: float = DEFAULT_E5_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    """Load dev items, partition, and serialize e5_train.jsonl and e5_val.jsonl."""
    dev_items = load_gold_items(dev_path)
    train_items, val_items = split_dev_for_gepa(dev_items, train_ratio=train_ratio, seed=seed)

    train_path = output_dir / "e5_train.jsonl"
    val_path = output_dir / "e5_val.jsonl"

    save_jsonl(train_items, train_path)
    save_jsonl(val_items, val_path)

    logger.info(
        "Partitioned %d dev items into %d train (%s) and %d val (%s)",
        len(dev_items),
        len(train_items),
        train_path,
        len(val_items),
        val_path,
    )

    print("=" * 70)
    print("🧬 GEPA E5 Dataset Split Partition")
    print("=" * 70)
    print(f"Source dev items   : {len(dev_items)}")
    print(f"GEPA Train split   : {len(train_items)} items ({train_path})")
    print(f"GEPA Val split     : {len(val_items)} items ({val_path})")
    print("-" * 70)
    print(f"{'Category':<32} | {'Train':<6} | {'Val':<6} | {'Total':<6}")
    print("-" * 70)

    cats = sorted({it.category for it in dev_items})
    for c in cats:
        tr_c = sum(1 for it in train_items if it.category == c)
        va_c = sum(1 for it in val_items if it.category == c)
        print(f"{c:<32} | {tr_c:>6} | {va_c:>6} | {tr_c + va_c:>6}")
    print("=" * 70)

    return train_path, val_path


def main() -> None:
    """CLI entrypoint for E5 split preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare E5 GEPA train and val splits from dev.jsonl"
    )
    parser.add_argument("--dev-path", type=Path, default=DEFAULT_DEV_PATH, help="Path to dev.jsonl")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=DEFAULT_E5_TRAIN_RATIO, help="Train ratio (0.6)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (42)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare_e5_splits(
        dev_path=args.dev_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
