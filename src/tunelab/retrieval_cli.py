import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

from tunelab.data.loader import SpiderExample
from tunelab.retrieval.bm25 import BM25Retriever
from tunelab.retrieval.dense import BGEEmbeddingBackend, DenseRetriever
from tunelab.retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tunelab.retrieval")


def load_spider_examples_file(file_path: Path | str, default_split: str) -> List[SpiderExample]:
    """Loads a JSON or JSONL file into SpiderExample objects."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    examples: List[SpiderExample] = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    examples.append(
                        SpiderExample(
                            example_id=item.get("example_id", f"ex_{idx}"),
                            db_id=item["db_id"],
                            question=item["question"],
                            query=item["query"],
                            split=item.get("split", default_split),
                            source_file=path.name,
                        )
                    )
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for idx, item in enumerate(raw):
            examples.append(
                SpiderExample(
                    example_id=item.get("example_id", f"{default_split}_{idx}"),
                    db_id=item["db_id"],
                    question=item["question"],
                    query=item["query"],
                    split=item.get("split", default_split),
                    source_file=path.name,
                )
            )
    return examples


def main():
    parser = argparse.ArgumentParser(description="TuneLab Retrieval CLI")
    parser.add_argument("--train-file", type=str, required=True, help="Path to training examples (JSON or JSONL)")
    parser.add_argument("--eval-file", type=str, required=True, help="Path to evaluation queries (JSON or JSONL)")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25", help="Retrieval method")
    parser.add_argument("--k", type=int, choices=[1, 3, 5, 10], default=3, help="Top-K cutoff")
    parser.add_argument("--model-path", type=str, default=None, help="Local path to BGE model weights for dense retrieval")
    parser.add_argument("--output-file", type=str, default=None, help="Path to save retrieval results JSON")
    args = parser.parse_args()

    train_examples = load_spider_examples_file(args.train_file, default_split="train")
    eval_examples = load_spider_examples_file(args.eval_file, default_split="dev")

    logger.info(f"Loaded {len(train_examples)} training examples for index.")
    logger.info(f"Loaded {len(eval_examples)} evaluation queries.")
    logger.info(f"Selected retrieval method: {args.method} (K={args.k})")

    # Initialize retriever
    if args.method == "bm25":
        retriever = BM25Retriever()
        retriever.index(train_examples)
    elif args.method == "dense":
        backend = BGEEmbeddingBackend(local_model_path=args.model_path)
        if not backend.is_available():
            logger.error("=" * 60)
            logger.error("DENSE EMBEDDING MODEL UNAVAILABLE")
            logger.error("=" * 60)
            logger.error(f"Model weights for '{BGEEmbeddingBackend.MODEL_ID}' are not present on disk.")
            logger.error("In this offline environment, dense retrieval cannot run.")
            logger.error("Status: DENSE RETRIEVAL PENDING GPU/WEIGHT ENVIRONMENT.")
            sys.exit(1)
        retriever = DenseRetriever(backend=backend)
        retriever.index(train_examples)
    elif args.method == "hybrid":
        bge_backend = BGEEmbeddingBackend(local_model_path=args.model_path)
        if not bge_backend.is_available():
            logger.error("=" * 60)
            logger.error("HYBRID RETRIEVAL UNAVAILABLE")
            logger.error("=" * 60)
            logger.error("Hybrid retrieval requires both BM25 and Dense backends.")
            logger.error(f"Dense model '{BGEEmbeddingBackend.MODEL_ID}' weights are not present on disk.")
            logger.error("Status: HYBRID RETRIEVAL PENDING GPU/WEIGHT ENVIRONMENT.")
            sys.exit(1)
        retriever = HybridRetriever(
            bm25=BM25Retriever(),
            dense=DenseRetriever(backend=bge_backend),
        )
        retriever.index(train_examples)

    # Perform retrieval for each evaluation query
    retrieval_records = []
    for ex in eval_examples:
        retrieved_items = retriever.retrieve(ex.question, k=args.k)
        retrieval_records.append({
            "query_id": ex.example_id,
            "query_db_id": ex.db_id,
            "query_question": ex.question,
            "retrieved": [item.to_dict() for item in retrieved_items],
        })

    logger.info("=" * 50)
    logger.info(f"RETRIEVAL COMPLETE ({args.method.upper()}, K={args.k})")
    logger.info("=" * 50)
    for rec in retrieval_records[:3]:
        logger.info(f"Query: {rec['query_question']}")
        for item in rec["retrieved"]:
            logger.info(f"  [Rank {item['rank']}] Score: {item['score']} | DB: {item['db_id']} | ID: {item['example_id']} | Q: {item['question']}")

    if args.output_file:
        out_p = Path(args.output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": args.method,
            "k": args.k,
            "num_indexed_train_examples": len(train_examples),
            "num_queries": len(eval_examples),
            "results": retrieval_records,
        }
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved retrieval results to {out_p}")


if __name__ == "__main__":
    main()
