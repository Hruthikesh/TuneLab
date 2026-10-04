import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Optional

from tunelab.experiments.config import ExperimentConfig, load_manifest
from tunelab.experiments.runner import execute_experiment_run
from tunelab.training.hardware import detect_hardware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tunelab.experiment")


def print_manifest_table(configs: List[ExperimentConfig], hw) -> None:
    """Prints a clear tabular view of planned experiments and their expected execution status."""
    header = f"{'RUN ID':<35} | {'METHOD':<10} | {'FRAC':<5} | {'K':<3} | {'RANK':<4} | {'NOISE':<5} | {'HW':<5} | {'EXPECTED STATUS'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for cfg in configs:
        exp_status = "READY"
        if cfg.hardware_requirements == "cuda" and not hw.cuda_available:
            exp_status = "PENDING_GPU"
        print(
            f"{cfg.run_id:<35} | {cfg.method:<10} | {cfg.data_fraction:<5} | {str(cfg.rag_k or '-'):<3} | "
            f"{str(cfg.lora_rank or '-'):<4} | {f'{int(cfg.noise_level*100)}%':<5} | {cfg.hardware_requirements:<5} | {exp_status}"
        )
    print("-" * len(header))
    print(f"Total planned runs: {len(configs)}")


def main():
    parser = argparse.ArgumentParser(description="TuneLab Controlled Experiment Runner")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest YAML file")
    parser.add_argument("--run-id", type=str, default=None, help="Filter to execute a specific run_id")
    parser.add_argument("--filter-method", type=str, default=None, help="Filter by method (ZERO_SHOT, FEW_SHOT, RAG, LORA, QLORA)")
    parser.add_argument("--list", action="store_true", help="List all planned experiments in manifest without executing")
    parser.add_argument("--dry-run", action="store_true", help="Execute dry-run validation without running models")
    parser.add_argument("--train-file", type=str, default="data/raw/spider/train_spider.json")
    parser.add_argument("--eval-file", type=str, default="data/raw/spider/dev.json")
    parser.add_argument("--schemas-file", type=str, default="data/raw/spider/tables.json")
    parser.add_argument("--db-dir", type=str, default="data/raw/spider/database")
    parser.add_argument("--output-dir", type=str, default="experiments/runs")
    args = parser.parse_args()

    hw = detect_hardware()
    manifest_path = Path(args.manifest)
    configs = load_manifest(manifest_path)

    # Filtering
    if args.run_id:
        configs = [c for c in configs if c.run_id == args.run_id or c.experiment_id == args.run_id]
        if not configs:
            logger.error(f"No experiment matched run_id '{args.run_id}'")
            sys.exit(1)

    if args.filter_method:
        target_m = args.filter_method.upper()
        configs = [c for c in configs if c.method == target_m]
        if not configs:
            logger.error(f"No experiment matched method '{target_m}'")
            sys.exit(1)

    if args.list:
        print_manifest_table(configs, hw)
        return

    logger.info("=" * 60)
    logger.info(f"TUNELAB EXPERIMENT RUNNER — Manifest: {manifest_path.name}")
    logger.info(f"Total planned runs: {len(configs)} (Dry Run: {args.dry_run})")
    logger.info("=" * 60)

    # Fallback to fixtures if raw spider files missing and running smoke
    train_p = Path(args.train_file)
    eval_p = Path(args.eval_file)
    schemas_p = Path(args.schemas_file)
    db_p = Path(args.db_dir)

    if "smoke" in manifest_path.name.lower() and not train_p.is_file():
        logger.info("Using synthetic test fixtures for smoke test.")
        fixtures_p = Path("tests/fixtures")
        train_p = fixtures_p / "train_spider.json"
        eval_p = fixtures_p / "dev.json"
        schemas_p = fixtures_p / "tables.json"
        db_p = fixtures_p / "database"

    results_summary = []
    for idx, cfg in enumerate(configs, start=1):
        logger.info(f"\n[{idx}/{len(configs)}] Processing Run: {cfg.run_id} ({cfg.method})")
        record = execute_experiment_run(
            config=cfg,
            train_file=train_p,
            eval_file=eval_p,
            schemas_file=schemas_p,
            db_dir=db_p,
            output_base_dir=Path(args.output_dir),
            dry_run=args.dry_run,
        )
        results_summary.append(record)
        logger.info(f"  Result: Status={record.status} | Acc={record.execution_accuracy} | Validity={record.sql_validity}")

    # Final summary count
    statuses = {}
    for r in results_summary:
        statuses[r.status] = statuses.get(r.status, 0) + 1

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT RUNNER FINISHED")
    logger.info("=" * 60)
    for stat, count in statuses.items():
        logger.info(f"  {stat}: {count} runs")
    logger.info(f"Master results written to: {args.output_dir}/master_results.jsonl and .csv")


if __name__ == "__main__":
    main()
