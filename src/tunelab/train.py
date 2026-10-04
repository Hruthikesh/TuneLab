import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from tunelab.data.leakage import verify_split_integrity
from tunelab.data.loader import SpiderExample
from tunelab.data.pools import generate_deterministic_subsets
from tunelab.data.schema import DatabaseSchema, parse_spider_schema
from tunelab.training.format import format_dataset
from tunelab.training.hardware import detect_hardware
from tunelab.training.lora import (
    LoRAHyperparameters,
    compute_lora_parameter_accounting,
    save_run_metadata,
    train_lora_model,
)
from tunelab.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tunelab.train")


def load_examples_file(file_path: Path | str, split: str = "train") -> List[SpiderExample]:
    """Loads a list of SpiderExample objects from JSON or JSONL."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: List[SpiderExample] = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    examples.append(
                        SpiderExample(
                            example_id=item.get("example_id", f"{split}_{idx}"),
                            db_id=item["db_id"],
                            question=item["question"],
                            query=item["query"],
                            split=item.get("split", split),
                            source_file=path.name,
                        )
                    )
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        for idx, item in enumerate(raw_data):
            examples.append(
                SpiderExample(
                    example_id=item.get("example_id", f"{split}_{idx}"),
                    db_id=item["db_id"],
                    question=item["question"],
                    query=item["query"],
                    split=split,
                    source_file=path.name,
                )
            )
    return examples


def load_schemas_file(file_path: Path | str) -> Dict[str, DatabaseSchema]:
    """Loads schemas from tables.json or schemas.json."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Schemas file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_schemas = json.load(f)

    schemas: Dict[str, DatabaseSchema] = {}
    if isinstance(raw_schemas, list):
        for entry in raw_schemas:
            s = parse_spider_schema(entry)
            schemas[s.db_id] = s
    elif isinstance(raw_schemas, dict):
        for db_id, data in raw_schemas.items():
            schemas[db_id] = parse_spider_schema(data) if "table_names_original" in data else DatabaseSchema(db_id=db_id)
    return schemas


def run_training_pipeline(
    train_file: str | Path,
    schemas_file: str | Path,
    eval_file: Optional[str | Path] = None,
    output_dir: str | Path = "checkpoints/lora_run",
    data_fraction: float = 1.0,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    learning_rate: float = 0.0002,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_seq_length: int = 1024,
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Coordinates data loading, leakage checks, formatting, hardware detection, and LoRA setup."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Hardware Detection
    hw = detect_hardware()
    logger.info("=" * 60)
    logger.info(f"Hardware Profile: Device={hw.device}, CUDA={hw.cuda_available}, RAM={hw.ram_total_gb:.1f}GB")
    logger.info("=" * 60)

    # 2. Hyperparameter Configuration
    hyperparams = LoRAHyperparameters(
        r=r,
        alpha=alpha,
        dropout=dropout,
        learning_rate=learning_rate,
        num_epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        seed=seed,
    )

    # 3. Load Data & Schemas
    train_pool = load_examples_file(train_file, split="train")
    schemas = load_schemas_file(schemas_file)
    logger.info(f"Loaded {len(train_pool)} raw training examples and {len(schemas)} schemas.")

    # 4. Leakage Validation
    if eval_file:
        eval_examples = load_examples_file(eval_file, split="dev")
        leakage_report = verify_split_integrity(train_pool, eval_examples)
        logger.info(f"Split Integrity: PASSED (Train DBs: {leakage_report.train_db_count}, Eval DBs: {leakage_report.eval_db_count})")

    # 5. Data Fraction Slicing
    if data_fraction < 1.0:
        subsets = generate_deterministic_subsets(
            train_examples=train_pool,
            fractions=[data_fraction],
            seed=seed,
            source_pool="train_spider_only",
        )
        frac_key = f"{int(data_fraction * 100)}pct"
        active_train_examples = subsets[frac_key][0]
    else:
        active_train_examples = train_pool

    logger.info(f"Active training examples for fraction {data_fraction * 100:.0f}%: {len(active_train_examples)}")

    # 6. Format Dataset with Loss Masking Boundaries
    formatted_dataset = format_dataset(active_train_examples, schemas)
    logger.info(f"Formatted {len(formatted_dataset)} examples with instruction header and target SQL.")

    # 7. Parameter Accounting
    accounting = compute_lora_parameter_accounting(hyperparams)
    logger.info(f"LoRA Parameter Accounting (Rank {hyperparams.r}):")
    logger.info(f"  Base Parameters: {accounting['total_base_parameters']:,}")
    logger.info(f"  Trainable LoRA Parameters: {accounting['trainable_lora_parameters']:,}")
    logger.info(f"  Trainable Ratio: {accounting['trainable_parameter_percentage']:.4f}%")

    # 8. Execution Status Assessment
    metrics = None
    if dry_run:
        status = "DRY_RUN_ONLY"
        logger.info("Executing in DRY_RUN mode: Pipeline, formatting, and parameter accounting verified.")
    elif not hw.cuda_available:
        status = "TRAINING_PENDING_GPU"
        logger.warning(
            "CUDA is unavailable in this environment. Full LoRA fine-tuning cannot run on CPU. "
            "Marking status as TRAINING_PENDING_GPU."
        )
    else:
        logger.info("CUDA detected. Launching real LoRA training loop...")
        try:
            metrics = train_lora_model(
                formatted_dataset=formatted_dataset,
                hyperparameters=hyperparams,
                output_dir=out_path,
                device="cuda",
            )
            status = "COMPLETED"
            logger.info("LoRA training completed successfully!")
        except Exception as e:
            status = "FAILED"
            logger.error(f"LoRA training execution failed: {e}")
            metrics = {"error": str(e)}

    meta_file = save_run_metadata(
        output_dir=out_path,
        hyperparameters=hyperparams,
        hardware=hw,
        status=status,
        sample_count=len(active_train_examples),
        data_fraction=data_fraction,
        metrics=metrics,
    )

    return {
        "status": status,
        "data_fraction": data_fraction,
        "sample_count": len(active_train_examples),
        "parameter_accounting": accounting,
        "run_metadata_file": str(meta_file),
    }


def main():
    parser = argparse.ArgumentParser(description="TuneLab LoRA Supervised Fine-Tuning CLI")
    parser.add_argument("--config", type=str, default="configs/full.yaml", help="Path to YAML configuration")
    parser.add_argument("--train-file", type=str, required=True, help="Path to training examples (JSON/JSONL)")
    parser.add_argument("--schemas-file", type=str, required=True, help="Path to schemas (JSON)")
    parser.add_argument("--eval-file", type=str, default=None, help="Path to eval examples for split isolation check")
    parser.add_argument("--output-dir", type=str, default="checkpoints/lora_run", help="Output directory for metadata")
    parser.add_argument("--data-fraction", type=float, default=1.0, help="Training data fraction (0.01 to 1.00)")
    parser.add_argument("--r", type=int, choices=[4, 8, 16], default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--learning-rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Execute verification dry-run without training")
    args = parser.parse_args()

    summary = run_training_pipeline(
        train_file=args.train_file,
        schemas_file=args.schemas_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        data_fraction=args.data_fraction,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
