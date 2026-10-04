from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tunelab.data.leakage import DataLeakageError
from tunelab.data.loader import SpiderExample
from tunelab.data.schema import DatabaseSchema

INSTRUCTION_HEADER = (
    "You are an expert SQL engineer. Given the database schema, write a valid SQLite query "
    "that answers the natural language question.\n"
    "Rules:\n"
    "- Return ONLY the raw SQL query.\n"
    "- Use only tables and columns from the provided schema."
)


@dataclass
class FormattedTrainingExample:
    """Represents a formatted training example with separated prompt and target segments."""
    example_id: str
    db_id: str
    prompt_text: str
    target_text: str
    full_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "db_id": self.db_id,
            "prompt_text": self.prompt_text,
            "target_text": self.target_text,
            "full_text": self.full_text,
        }


def format_single_example(example: SpiderExample, schema: DatabaseSchema) -> FormattedTrainingExample:
    """Formats a single Spider example into prompt and target components for instruction tuning."""
    if example.split.lower() in ("dev", "test", "eval"):
        raise DataLeakageError(
            f"Attempted to format example '{example.example_id}' belonging to evaluation split '{example.split}'. "
            "Evaluation examples must never enter the training dataset!"
        )

    schema_ddl = schema.to_ddl_text()
    prompt_text = (
        f"{INSTRUCTION_HEADER}\n\n"
        f"### Database Schema:\n"
        f"{schema_ddl}\n\n"
        f"### Question:\n"
        f"{example.question.strip()}\n\n"
        f"### SQL:\n"
    )

    target_text = f"{example.query.strip()}"
    full_text = f"{prompt_text}{target_text}"

    return FormattedTrainingExample(
        example_id=example.example_id,
        db_id=example.db_id,
        prompt_text=prompt_text,
        target_text=target_text,
        full_text=full_text,
    )


def format_dataset(
    examples: List[SpiderExample],
    schemas: Dict[str, DatabaseSchema],
) -> List[FormattedTrainingExample]:
    """Formats a list of Spider training examples, verifying zero evaluation split leakage."""
    formatted_list: List[FormattedTrainingExample] = []
    for ex in examples:
        schema = schemas.get(ex.db_id)
        if not schema:
            raise ValueError(f"Schema for database '{ex.db_id}' not found in schemas dictionary.")
        formatted_list.append(format_single_example(ex, schema))
    return formatted_list


def apply_loss_mask(
    prompt_tokens: List[int],
    target_tokens: List[int],
    max_seq_length: int = 1024,
    eos_token_id: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Constructs input_ids, attention_mask, and labels with proper loss masking.
    
    Prompt tokens receive label -100 (ignored in CrossEntropyLoss).
    Target SQL tokens receive their actual token IDs.
    """
    if not prompt_tokens:
        raise ValueError("prompt_tokens cannot be empty.")
    if not target_tokens:
        raise ValueError("target_tokens cannot be empty.")

    full_target = list(target_tokens)
    if eos_token_id is not None:
        full_target.append(eos_token_id)

    input_ids = list(prompt_tokens) + full_target
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(prompt_tokens)) + full_target

    # Truncate to max_seq_length
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
