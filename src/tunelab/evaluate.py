import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from tunelab.data.loader import SpiderExample
from tunelab.data.schema import DatabaseSchema, parse_spider_schema
from tunelab.evaluation.evaluator import (
    EvaluationResult,
    compute_summary_metrics,
    evaluate_example,
)
from tunelab.models.base import HuggingFaceSQLGenerator, MockSQLGenerator, SQLGenerator
from tunelab.training.hardware import detect_hardware
from tunelab.prompting.prompts import (
    build_few_shot_prompt,
    build_zero_shot_prompt,
    select_deterministic_demonstrations,
)
from tunelab.prompting.rag import build_rag_prompt
from tunelab.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tunelab.evaluate")


def run_evaluation(
    eval_examples: List[SpiderExample],
    schemas: Dict[str, DatabaseSchema],
    db_dir: Path | str,
    generator: SQLGenerator,
    method: str = "zero_shot",
    demonstrations: Optional[List[SpiderExample]] = None,
    few_shot_k: int = 3,
    retriever: Optional[Any] = None,
    timeout_seconds: float = 5.0,
    output_file: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Runs end-to-end evaluation over a list of Spider examples."""
    db_root = Path(db_dir)
    results: List[EvaluationResult] = []

    for idx, ex in enumerate(eval_examples):
        schema = schemas.get(ex.db_id)
        if not schema:
            logger.warning(
                f"Schema for database '{ex.db_id}' not found; "
                f"skipping example {ex.example_id}"
            )
            continue

        sqlite_path = db_root / ex.db_id / f"{ex.db_id}.sqlite"
        if not sqlite_path.is_file():
            sqlite_path = db_root / f"{ex.db_id}.sqlite"
            if not sqlite_path.is_file():
                logger.warning(
                    f"SQLite DB file for '{ex.db_id}' not found at {sqlite_path}"
                )

        # Construct prompt
        retrieval_meta = None

        if method == "rag" and retriever is not None:
            prompt, retrieved_items = build_rag_prompt(
                question=ex.question,
                schema=schema,
                retriever=retriever,
                k=few_shot_k,
            )
            retrieval_meta = {
                "k": few_shot_k,
                "retrieved_ids": [item.example_id for item in retrieved_items],
                "scores": [item.score for item in retrieved_items],
                "ranks": [item.rank for item in retrieved_items],
            }
        elif method == "few_shot" and demonstrations:
            prompt = build_few_shot_prompt(
                ex.question,
                schema,
                demonstrations,
                k=few_shot_k,
            )
        else:
            prompt = build_zero_shot_prompt(ex.question, schema)

        # Generate SQL
        predicted_sql = generator.generate(
            prompt=prompt,
            question=ex.question,
            schema=schema,
        )

        # Evaluate against reference SQL
        res = evaluate_example(
            example_id=ex.example_id,
            database_id=ex.db_id,
            predicted_sql=predicted_sql,
            reference_sql=ex.query,
            sqlite_path=sqlite_path,
            timeout_seconds=timeout_seconds,
            retrieval_metadata=retrieval_meta,
        )

        results.append(res)

    summary = compute_summary_metrics(results)
    summary["method"] = method
    summary["few_shot_k"] = few_shot_k if method in ("few_shot", "rag") else 0

    if method == "rag":
        summary["rag_k"] = few_shot_k

    if output_file:
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Saved evaluation results to {out_p}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="TuneLab Evaluation CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="Path to evaluation JSONL or JSON file",
    )
    parser.add_argument(
        "--schemas-file",
        type=str,
        required=True,
        help="Path to schemas.json or tables.json",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        required=True,
        help="Directory holding database subfolders with .sqlite files",
    )
    parser.add_argument(
        "--method",
        choices=["zero_shot", "few_shot", "rag"],
        default="zero_shot",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Few-shot K",
    )
    parser.add_argument(
        "--train-pool-file",
        type=str,
        default=None,
        help="Path to training JSONL for few-shot demonstrations",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save evaluation JSON results",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run evaluation in smoke/test mode with MockSQLGenerator",
    )
    parser.add_argument(
        "--mock-mode",
        choices=["gold", "syntax_error", "timeout", "runtime_error"],
        default=None,
        help="Mock generator mode for testing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Hugging Face model identifier for real evaluation",
    )
    args = parser.parse_args()

    config = load_config(args.config) if Path(args.config).is_file() else {}
    timeout = config.get("evaluation", {}).get("timeout_seconds", 5.0)

    with open(args.schemas_file, "r", encoding="utf-8") as f:
        raw_schemas = json.load(f)

    schemas: Dict[str, DatabaseSchema] = {}

    if isinstance(raw_schemas, list):
        for entry in raw_schemas:
            s = parse_spider_schema(entry)
            schemas[s.db_id] = s
    elif isinstance(raw_schemas, dict):
        for db_id, data in raw_schemas.items():
            schemas[db_id] = DatabaseSchema(db_id=db_id)

    eval_examples: List[SpiderExample] = []
    eval_path = Path(args.eval_file)

    if eval_path.suffix == ".jsonl":
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    eval_examples.append(
                        SpiderExample(
                            example_id=item.get(
                                "example_id",
                                f"ex_{len(eval_examples)}",
                            ),
                            db_id=item["db_id"],
                            question=item["question"],
                            query=item["query"],
                            split=item.get("split", "dev"),
                            source_file=item.get("source_file", eval_path.name),
                        )
                    )
    else:
        with open(eval_path, "r", encoding="utf-8") as f:
            raw_eval = json.load(f)

        for idx, item in enumerate(raw_eval):
            eval_examples.append(
                SpiderExample(
                    example_id=f"dev_{idx}",
                    db_id=item["db_id"],
                    question=item["question"],
                    query=item["query"],
                    split="dev",
                    source_file=eval_path.name,
                )
            )

    demos: List[SpiderExample] = []
    retriever = None

    if args.method in ("few_shot", "rag") and args.train_pool_file:
        pool_p = Path(args.train_pool_file)

        if pool_p.suffix == ".jsonl":
            with open(pool_p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        demos.append(
                            SpiderExample(
                                example_id=item.get(
                                    "example_id",
                                    f"demo_{len(demos)}",
                                ),
                                db_id=item["db_id"],
                                question=item["question"],
                                query=item["query"],
                                split=item.get("split", "train"),
                                source_file=pool_p.name,
                            )
                        )
        else:
            with open(pool_p, "r", encoding="utf-8") as f:
                raw_demos = json.load(f)

            for idx, item in enumerate(raw_demos):
                demos.append(
                    SpiderExample(
                        example_id=f"demo_{idx}",
                        db_id=item["db_id"],
                        question=item["question"],
                        query=item["query"],
                        split="train",
                        source_file=pool_p.name,
                    )
                )

        if args.method == "rag":
            from tunelab.retrieval.bm25 import BM25Retriever

            retriever = BM25Retriever()
            retriever.index(demos)

    generator: SQLGenerator

    if args.smoke or args.mock_mode:
        mock_mode = args.mock_mode or "gold"
        reference_map = {
            ex.question.strip(): ex.query.strip()
            for ex in eval_examples
        }

        generator = MockSQLGenerator(
            mode=mock_mode,
            reference_map=reference_map,
        )

        logger.info(
            f"Initialized MockSQLGenerator (mode={mock_mode}) "
            "for smoke/test evaluation."
        )

    else:
        if os.environ.get("TUNELAB_TEST_NO_MODEL") == "1":
            logger.error(
                "MODEL_REQUIRED_FOR_RESEARCH: "
                "Test environment prevents model initialization."
            )
            sys.exit(1)

        # Research evaluation: Strictly require real HuggingFace model
        hw = detect_hardware()
        device = "cuda" if hw.cuda_available else "cpu"

        logger.info(
            f"Research evaluation mode: Initializing "
            f"HuggingFaceSQLGenerator for model '{args.model}' "
            f"on {device}..."
        )

        try:
            generator = HuggingFaceSQLGenerator(
                model_name=args.model,
                device=device,
                lazy_load=False,
            )
        except Exception as e:
            logger.error(
                "MODEL_REQUIRED_FOR_RESEARCH: "
                f"Real HuggingFaceSQLGenerator is required for research "
                f"evaluation, but model '{args.model}' could not be "
                f"initialized: {e}. Pass --smoke to run in test mode."
            )
            sys.exit(1)

    summary = run_evaluation(
        eval_examples=eval_examples,
        schemas=schemas,
        db_dir=args.db_dir,
        generator=generator,
        method=args.method,
        demonstrations=demos,
        few_shot_k=args.k,
        retriever=retriever,
        timeout_seconds=timeout,
        output_file=args.output_file,
    )

    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)

    for k, v in summary.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()