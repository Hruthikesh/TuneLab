from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tunelab.data.leakage import verify_split_integrity
from tunelab.data.loader import SpiderDataLoader, TrainingPoolType
from tunelab.data.pools import generate_deterministic_subsets
from tunelab.data.schema import validate_sqlite_database

logger = logging.getLogger(__name__)


def process_and_save_data(
    raw_dir: str | Path,
    processed_dir: str | Path,
    splits_dir: str | Path,
    training_pool: TrainingPoolType = "train_spider_only",
    seed: int = 42,
    validate_sqlite: bool = True,
) -> Dict[str, Any]:
    """Executes the complete data preparation and validation pipeline."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    splits_path = Path(splits_dir)

    processed_path.mkdir(parents=True, exist_ok=True)
    splits_path.mkdir(parents=True, exist_ok=True)

    loader = SpiderDataLoader(raw_path)
    ready, missing = loader.check_files_exist(require_others=(training_pool == "train_spider_and_others"))
    if not ready:
        raise FileNotFoundError(
            f"Missing required raw Spider files in '{raw_path}': {', '.join(missing)}"
        )

    # 1. Load and parse schemas
    schemas = loader.load_schemas()

    # 2. Discover SQLite databases and optionally validate
    db_paths = loader.scan_databases()
    sqlite_reports: Dict[str, Any] = {}
    if validate_sqlite:
        for db_id, path in db_paths.items():
            schema = schemas.get(db_id)
            report = validate_sqlite_database(path, schema=schema)
            sqlite_reports[db_id] = report

    # 3. Load and validate question-SQL examples
    train_examples, dev_examples = loader.load_examples(training_pool=training_pool)

    # 4. Strict leakage checks across split boundaries
    leakage_report = verify_split_integrity(train_examples, dev_examples)

    # 5. Generate deterministic training subsets (fractions)
    subsets = generate_deterministic_subsets(
        train_examples=train_examples,
        seed=seed,
        source_pool=training_pool,
    )

    # 6. Save intermediate representation
    # Save Schemas
    schemas_dict = {db_id: s.to_dict() for db_id, s in schemas.items()}
    with open(processed_path / "schemas.json", "w", encoding="utf-8") as f:
        json.dump(schemas_dict, f, indent=2)

    # Save Train Examples (JSONL)
    with open(processed_path / "train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex.to_dict()) + "\n")

    # Save Dev Examples (JSONL)
    with open(processed_path / "dev.jsonl", "w", encoding="utf-8") as f:
        for ex in dev_examples:
            f.write(json.dumps(ex.to_dict()) + "\n")

    # Save Subsets Metadata and ID maps
    subsets_metadata = {}
    for frac_key, (sub_exs, meta) in subsets.items():
        subsets_metadata[frac_key] = meta.to_dict()
        # Save individual subset JSONL
        subset_file = splits_path / f"train_{training_pool}_{frac_key}.jsonl"
        with open(subset_file, "w", encoding="utf-8") as f:
            for ex in sub_exs:
                f.write(json.dumps(ex.to_dict()) + "\n")

    with open(splits_path / f"subsets_metadata_{training_pool}.json", "w", encoding="utf-8") as f:
        json.dump(subsets_metadata, f, indent=2)

    # Pipeline summary metadata
    summary = {
        "benchmark": "spider",
        "version": "1.0",
        "training_pool": training_pool,
        "seed": seed,
        "num_schemas": len(schemas),
        "num_sqlite_dbs": len(db_paths),
        "num_train_examples": len(train_examples),
        "num_dev_examples": len(dev_examples),
        "leakage_check_passed": leakage_report.passed,
        "fractions_generated": list(subsets.keys()),
        "sqlite_validation": {
            "total_checked": len(sqlite_reports),
            "valid_count": sum(1 for r in sqlite_reports.values() if r.get("valid", False)),
        },
    }

    with open(processed_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
