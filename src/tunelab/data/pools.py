from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from tunelab.data.loader import SpiderExample, TrainingPoolType

logger = logging.getLogger(__name__)

STANDARD_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


@dataclass
class SubsetMetadata:
    source_pool: TrainingPoolType
    fraction: float
    num_examples: int
    total_pool_examples: int
    seed: int
    generation_method: str
    example_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_pool": self.source_pool,
            "fraction": self.fraction,
            "num_examples": self.num_examples,
            "total_pool_examples": self.total_pool_examples,
            "seed": self.seed,
            "generation_method": self.generation_method,
            "example_ids": self.example_ids,
        }


def generate_deterministic_subsets(
    train_examples: List[SpiderExample],
    fractions: List[float] = STANDARD_FRACTIONS,
    seed: int = 42,
    source_pool: TrainingPoolType = "train_spider_only",
) -> Dict[str, Tuple[List[SpiderExample], SubsetMetadata]]:
    """Generates deterministic subsets across specified fractions from the training pool.
    
    Subsets are nested (a smaller fraction is a strict prefix/subset of a larger fraction)
    by shuffling the full training pool once with the fixed seed and taking prefixes.
    """
    if not train_examples:
        raise ValueError("Cannot generate subsets from empty training example list.")

    # Create a deterministic copy and shuffle once
    rng = random.Random(seed)
    shuffled = list(train_examples)
    rng.shuffle(shuffled)

    total_count = len(shuffled)
    subsets: Dict[str, Tuple[List[SpiderExample], SubsetMetadata]] = {}

    for frac in fractions:
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"Invalid fraction: {frac}. Must be in range (0.0, 1.0].")

        # Ensure at least 1 example is selected even for tiny fractions
        sample_count = max(1, int(round(total_count * frac)))
        if frac == 1.0:
            sample_count = total_count

        subset_examples = shuffled[:sample_count]
        meta = SubsetMetadata(
            source_pool=source_pool,
            fraction=frac,
            num_examples=len(subset_examples),
            total_pool_examples=total_count,
            seed=seed,
            generation_method="deterministic_prefix_shuffle",
            example_ids=[ex.example_id for ex in subset_examples],
        )

        frac_key = f"{int(frac * 100)}pct"
        subsets[frac_key] = (subset_examples, meta)

    return subsets


def create_internal_validation_split(
    train_examples: List[SpiderExample],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[SpiderExample], List[SpiderExample]]:
    """Partitions the training pool into internal-train and internal-val for hyperparameter tuning.
    
    CRITICAL: Partitioning is done at the DATABASE level so that internal validation
    also tests cross-database generalization without touching official dev databases.
    """
    if not train_examples:
        raise ValueError("Cannot partition empty training pool.")

    unique_dbs = sorted(list({ex.db_id for ex in train_examples}))
    if len(unique_dbs) < 2:
        raise ValueError("Need at least 2 distinct databases to create a cross-database split.")

    rng = random.Random(seed)
    shuffled_dbs = list(unique_dbs)
    rng.shuffle(shuffled_dbs)

    val_db_count = max(1, int(round(len(unique_dbs) * val_ratio)))
    val_dbs = set(shuffled_dbs[:val_db_count])
    train_dbs = set(shuffled_dbs[val_db_count:])

    internal_train = [ex for ex in train_examples if ex.db_id in train_dbs]
    internal_val = [ex for ex in train_examples if ex.db_id in val_dbs]

    return internal_train, internal_val
