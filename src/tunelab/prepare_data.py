import argparse
import logging
from pathlib import Path
import sys

from tunelab.data.loader import SpiderDataLoader
from tunelab.data.pipeline import process_and_save_data
from tunelab.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tunelab.prepare_data")


def main():
    parser = argparse.ArgumentParser(description="TuneLab Dataset Preparation CLI for Spider v1.0")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base configuration YAML")
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory containing raw Spider benchmark files")
    parser.add_argument("--processed-dir", type=str, default=None, help="Output directory for processed artifacts")
    parser.add_argument("--splits-dir", type=str, default=None, help="Output directory for experiment splits")
    parser.add_argument(
        "--training-pool",
        choices=["train_spider_only", "train_spider_and_others"],
        default="train_spider_only",
        help="Training pool selection",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic subset generation")
    args = parser.parse_args()

    # Load base config if available
    config = {}
    config_path = Path(args.config)
    if config_path.is_file():
        config = load_config(config_path)

    raw_dir = args.raw_dir or config.get("paths", {}).get("raw_data_dir", "data/raw")
    processed_dir = args.processed_dir or config.get("paths", {}).get("processed_data_dir", "data/processed")
    splits_dir = args.splits_dir or config.get("paths", {}).get("splits_dir", "data/splits")
    seed = args.seed if args.seed is not None else config.get("seed", 42)

    # Allow subfolder raw/spider as a fallback location
    raw_path = Path(raw_dir)
    if not (raw_path / "tables.json").is_file() and (raw_path / "spider" / "tables.json").is_file():
        raw_path = raw_path / "spider"

    logger.info(f"Target raw data directory: {raw_path}")
    logger.info(f"Selected training pool: {args.training_pool}")

    loader = SpiderDataLoader(raw_path)
    ready, missing = loader.check_files_exist(require_others=(args.training_pool == "train_spider_and_others"))

    if not ready:
        logger.error("=" * 60)
        logger.error("OFFICIAL SPIDER DATASET FILES NOT FOUND")
        logger.error("=" * 60)
        logger.error(f"Missing required items in '{raw_path}':")
        for item in missing:
            logger.error(f"  - {item}")
        logger.error("\nTo prepare the real dataset:")
        logger.error("1. Place the official Spider v1.0 release files in: data/raw/spider/")
        logger.error("   Required: tables.json, train_spider.json, dev.json, database/")
        logger.error("2. Re-run: python -m tunelab.prepare_data\n")
        logger.error("Status: REAL DATA INGESTION PENDING (files not present).")
        sys.exit(1)

    try:
        summary = process_and_save_data(
            raw_dir=raw_path,
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            training_pool=args.training_pool,
            seed=seed,
            validate_sqlite=True,
        )
        logger.info("=" * 60)
        logger.info("DATA PREPARATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Schemas parsed: {summary['num_schemas']}")
        logger.info(f"SQLite DBs found: {summary['num_sqlite_dbs']}")
        logger.info(f"Train examples ({args.training_pool}): {summary['num_train_examples']}")
        logger.info(f"Dev examples: {summary['num_dev_examples']}")
        logger.info(f"Leakage checks: {'PASSED' if summary['leakage_check_passed'] else 'FAILED'}")
        logger.info(f"Generated data fractions: {summary['fractions_generated']}")
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
