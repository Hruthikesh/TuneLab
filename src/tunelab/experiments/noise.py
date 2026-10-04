import copy
import random
import re
from typing import List, Tuple

from tunelab.data.loader import SpiderExample

VALID_NOISE_TYPES = {"none", "sql_syntax_corruption", "mismatched_pair"}


def inject_training_noise(
    examples: List[SpiderExample],
    noise_level: float,
    noise_type: str = "none",
    seed: int = 42,
) -> Tuple[List[SpiderExample], List[str]]:
    """Deterministically corrupts a specified percentage of training examples.
    
    If noise_level is 0.0 or noise_type is 'none', returns clean unmodified examples.
    Records and returns the exact list of corrupted example IDs.
    """
    if noise_level == 0.0 or noise_type == "none":
        return list(examples), []

    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(f"Unknown noise_type '{noise_type}'. Must be one of {VALID_NOISE_TYPES}")

    if not (0.0 <= noise_level <= 1.0):
        raise ValueError(f"noise_level must be in [0.0, 1.0], got {noise_level}")

    num_to_corrupt = int(round(len(examples) * noise_level))
    if num_to_corrupt == 0:
        return list(examples), []

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    corrupt_indices = set(rng.sample(indices, num_to_corrupt))

    corrupted_examples: List[SpiderExample] = []
    affected_ids: List[str] = []

    for idx, ex in enumerate(examples):
        if idx not in corrupt_indices:
            corrupted_examples.append(ex)
            continue

        affected_ids.append(ex.example_id)
        corrupted_ex = copy.deepcopy(ex)

        if noise_type == "sql_syntax_corruption":
            # Malform SQL syntax deterministically by corrupting SELECT or FROM clause
            orig_sql = ex.query
            if orig_sql.upper().startswith("SELECT"):
                # Replace SELECT with malformed token
                corrupted_sql = "SELEKT" + orig_sql[6:]
            else:
                corrupted_sql = orig_sql + " SYNTAX_ERR_CORRUPT"
            corrupted_ex.query = corrupted_sql

        elif noise_type == "mismatched_pair":
            # Rotate query to assign an irrelevant query to the question
            other_idx = (idx + 1) % len(examples)
            corrupted_ex.query = examples[other_idx].query

        corrupted_examples.append(corrupted_ex)

    return corrupted_examples, affected_ids
