from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tunelab.data.leakage import DataLeakageError
from tunelab.data.loader import SpiderExample


@dataclass
class RetrievedExample:
    """Represents a retrieved training demonstration with scoring and rank metadata."""
    example_id: str
    db_id: str
    question: str
    query: str
    split: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "db_id": self.db_id,
            "question": self.question,
            "query": self.query,
            "split": self.split,
            "score": round(self.score, 4),
            "rank": self.rank,
            "metadata": self.metadata,
        }

    def to_spider_example(self, source_file: str = "retrieval_index") -> SpiderExample:
        """Converts back to a SpiderExample for prompt builders."""
        return SpiderExample(
            example_id=self.example_id,
            db_id=self.db_id,
            question=self.question,
            query=self.query,
            split=self.split,
            source_file=source_file,
        )


class BaseRetriever(ABC):
    """Abstract base class for all Text-to-SQL retrieval algorithms."""

    def __init__(self):
        self.indexed_examples: List[SpiderExample] = []

    def validate_index_pool(self, examples: List[SpiderExample]) -> List[SpiderExample]:
        """Validates that the index pool contains strictly training examples.
        
        CRITICAL: Fails loudly with DataLeakageError if any evaluation example is present.
        Deduplicates examples by example_id deterministically.
        """
        if not examples:
            raise ValueError("Cannot index an empty list of examples.")

        seen_ids = set()
        deduped: List[SpiderExample] = []

        for ex in examples:
            if ex.split.lower() in ("dev", "test", "eval"):
                raise DataLeakageError(
                    f"Attempted to index example '{ex.example_id}' belonging to evaluation split '{ex.split}'. "
                    "The retrieval index must strictly contain training examples only!"
                )
            if ex.example_id not in seen_ids:
                seen_ids.add(ex.example_id)
                deduped.append(ex)

        return deduped

    @abstractmethod
    def index(self, examples: List[SpiderExample]) -> None:
        """Constructs the retrieval index from training examples."""
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[RetrievedExample]:
        """Retrieves top-K relevant demonstrations for a query string."""
        pass
