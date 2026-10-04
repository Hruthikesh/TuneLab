from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from tunelab.data.schema import DatabaseSchema, parse_spider_schema

logger = logging.getLogger(__name__)

TrainingPoolType = Literal["train_spider_only", "train_spider_and_others"]


@dataclass
class SpiderExample:
    example_id: str
    db_id: str
    question: str
    query: str
    split: str
    source_file: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "db_id": self.db_id,
            "question": self.question,
            "query": self.query,
            "split": self.split,
            "source_file": self.source_file,
        }


class SpiderDataLoader:
    """Reusable loader and validator for the official Spider v1.0 benchmark."""

    REQUIRED_RAW_FILES = [
        "tables.json",
        "train_spider.json",
        "dev.json",
    ]

    def __init__(self, raw_data_dir: str | Path):
        p = Path(raw_data_dir)
        if not (p / "tables.json").is_file() and (p / "spider" / "tables.json").is_file():
            p = p / "spider"
        self.raw_data_dir = p
        self.schemas: Dict[str, DatabaseSchema] = {}
        self.db_paths: Dict[str, Path] = {}

    def check_files_exist(self, require_others: bool = False) -> Tuple[bool, List[str]]:
        """Verifies if the required raw Spider files exist."""
        missing = []
        for filename in self.REQUIRED_RAW_FILES:
            if not (self.raw_data_dir / filename).is_file():
                missing.append(filename)
        if require_others and not (self.raw_data_dir / "train_others.json").is_file():
            missing.append("train_others.json")
        database_dir = self.raw_data_dir / "database"
        if not database_dir.is_dir():
            missing.append("database/")
        return len(missing) == 0, missing

    def load_schemas(self) -> Dict[str, DatabaseSchema]:
        """Loads and parses all schemas from tables.json."""
        tables_path = self.raw_data_dir / "tables.json"
        if not tables_path.is_file():
            raise FileNotFoundError(f"tables.json not found in {self.raw_data_dir}")

        with open(tables_path, "r", encoding="utf-8") as f:
            raw_schemas = json.load(f)

        if not isinstance(raw_schemas, list):
            raise ValueError(f"tables.json must be a JSON array, got {type(raw_schemas).__name__}")

        self.schemas = {}
        for entry in raw_schemas:
            schema = parse_spider_schema(entry)
            self.schemas[schema.db_id] = schema

        logger.info(f"Loaded {len(self.schemas)} schemas from {tables_path.name}")
        return self.schemas

    def scan_databases(self) -> Dict[str, Path]:
        """Scans raw_data_dir/database/ for .sqlite files matching db_ids."""
        db_dir = self.raw_data_dir / "database"
        if not db_dir.is_dir():
            raise FileNotFoundError(f"Database directory not found: {db_dir}")

        self.db_paths = {}
        for sub in db_dir.iterdir():
            if sub.is_dir():
                sqlite_file = sub / f"{sub.name}.sqlite"
                if sqlite_file.is_file():
                    self.db_paths[sub.name] = sqlite_file
                else:
                    # Also check for any .sqlite file in the subdirectory
                    any_sqlite = list(sub.glob("*.sqlite"))
                    if any_sqlite:
                        self.db_paths[sub.name] = any_sqlite[0]

        logger.info(f"Discovered {len(self.db_paths)} SQLite database files in {db_dir}")
        return self.db_paths

    def load_examples(
        self,
        training_pool: TrainingPoolType = "train_spider_only",
    ) -> Tuple[List[SpiderExample], List[SpiderExample]]:
        """Loads train and dev question-SQL examples based on the selected training pool."""
        if not self.schemas:
            self.load_schemas()
        if not self.db_paths:
            self.scan_databases()

        # Load Training Examples
        train_examples: List[SpiderExample] = []
        train_spider_path = self.raw_data_dir / "train_spider.json"
        self._load_file_examples(train_spider_path, "train", train_examples, prefix="train_spider")

        if training_pool == "train_spider_and_others":
            train_others_path = self.raw_data_dir / "train_others.json"
            self._load_file_examples(
                train_others_path, "train", train_examples, prefix="train_others"
            )

        # Load Dev Examples
        dev_examples: List[SpiderExample] = []
        dev_path = self.raw_data_dir / "dev.json"
        self._load_file_examples(dev_path, "dev", dev_examples, prefix="dev")

        logger.info(
            f"Loaded {len(train_examples)} train examples ({training_pool}) and {len(dev_examples)} dev examples."
        )
        return train_examples, dev_examples

    def _load_file_examples(
        self,
        file_path: Path,
        split: str,
        target_list: List[SpiderExample],
        prefix: str,
    ) -> None:
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{file_path.name} must contain a JSON array.")

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Entry {idx} in {file_path.name} is not a JSON object.")

            db_id = item.get("db_id")
            question = item.get("question")
            query = item.get("query")

            if not db_id or not question or not query:
                raise ValueError(
                    f"Entry {idx} in {file_path.name} is missing one of required fields: 'db_id', 'question', 'query'."
                )

            # Validate that the database ID exists in the loaded schemas
            if db_id not in self.schemas:
                raise ValueError(
                    f"Entry {idx} in {file_path.name} references unknown database ID '{db_id}' not found in tables.json."
                )

            # Validate that the SQLite file is present
            if db_id not in self.db_paths:
                raise ValueError(
                    f"Entry {idx} in {file_path.name} references database ID '{db_id}' which lacks a .sqlite file in database/."
                )

            example = SpiderExample(
                example_id=f"{prefix}_{idx}",
                db_id=db_id,
                question=question.strip(),
                query=query.strip(),
                split=split,
                source_file=file_path.name,
            )
            target_list.append(example)
