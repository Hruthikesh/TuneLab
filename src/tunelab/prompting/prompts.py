from typing import List, Optional

from tunelab.data.leakage import DataLeakageError
from tunelab.data.loader import SpiderExample
from tunelab.data.schema import DatabaseSchema

ZERO_SHOT_SYSTEM_PROMPT = (
    "You are an expert SQL engineer. Given the following database schema, generate a valid SQLite query "
    "that correctly answers the user's natural language question.\n"
    "Rules:\n"
    "- Return ONLY the raw SQL query. Do not wrap in markdown blocks, explanations, or prose.\n"
    "- Use only tables and columns described in the provided schema.\n"
    "- The query must be compatible with SQLite."
)


def build_zero_shot_prompt(question: str, schema: DatabaseSchema) -> str:
    """Builds a standardized zero-shot prompt for Text-to-SQL generation."""
    schema_text = schema.to_ddl_text()
    prompt = (
        f"{ZERO_SHOT_SYSTEM_PROMPT}\n\n"
        f"### Database Schema:\n"
        f"{schema_text}\n\n"
        f"### Question:\n"
        f"{question.strip()}\n\n"
        f"### SQL:\n"
    )
    return prompt


def build_few_shot_prompt(
    question: str,
    schema: DatabaseSchema,
    demonstrations: List[SpiderExample],
    k: int = 3,
) -> str:
    """Builds a few-shot prompt with K demonstrations from the training pool.
    
    CRITICAL: Validates that none of the demonstrations originate from the dev/eval split.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    # Leakage check: ensure no demonstration is from an evaluation split
    for idx, demo in enumerate(demonstrations):
        if demo.split.lower() in ("dev", "test", "eval"):
            raise DataLeakageError(
                f"Demonstration at index {idx} belongs to evaluation split '{demo.split}'. "
                f"Evaluation examples are strictly forbidden in few-shot prompts!"
            )

    selected_demos = demonstrations[:k]

    demo_blocks = []
    for idx, demo in enumerate(selected_demos, start=1):
        block = (
            f"Example {idx}:\n"
            f"Question: {demo.question}\n"
            f"SQL: {demo.query}"
        )
        demo_blocks.append(block)

    demos_text = "\n\n".join(demo_blocks)
    schema_text = schema.to_ddl_text()

    if selected_demos:
        prompt = (
            f"{ZERO_SHOT_SYSTEM_PROMPT}\n\n"
            f"### Demonstration Examples:\n"
            f"{demos_text}\n\n"
            f"### Target Database Schema:\n"
            f"{schema_text}\n\n"
            f"### Target Question:\n"
            f"{question.strip()}\n\n"
            f"### SQL:\n"
        )
    else:
        prompt = build_zero_shot_prompt(question, schema)

    return prompt


def select_deterministic_demonstrations(
    pool: List[SpiderExample],
    k: int = 3,
) -> List[SpiderExample]:
    """Deterministically selects K demonstrations from the training pool."""
    if not pool:
        return []
    # Verify no eval examples exist in pool
    for ex in pool:
        if ex.split.lower() in ("dev", "test", "eval"):
            raise DataLeakageError("Training demonstration pool contains an evaluation split example!")
    return pool[:k]
