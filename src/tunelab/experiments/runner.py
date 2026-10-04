import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import yaml

from tunelab.data.leakage import verify_split_integrity
from tunelab.data.loader import SpiderExample
from tunelab.data.pools import generate_deterministic_subsets
from tunelab.data.schema import DatabaseSchema, parse_spider_schema
from tunelab.evaluate import run_evaluation
from tunelab.experiments.config import ExperimentConfig
from tunelab.experiments.noise import inject_training_noise
from tunelab.training.format import format_dataset
from tunelab.models.base import HuggingFaceSQLGenerator, MockSQLGenerator, SQLGenerator
from tunelab.models.lora_adapter import LoRAInferenceAdapter
from tunelab.models.qlora_adapter import QLoRAInferenceAdapter
from tunelab.retrieval.bm25 import BM25Retriever
from tunelab.training.hardware import detect_hardware
from tunelab.training.lora import (
    LoRAHyperparameters,
    compute_lora_parameter_accounting,
    train_lora_model,
)
from tunelab.training.qlora import (
    QLoRAConfig,
    check_qlora_dependencies,
    train_qlora_model,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResultRecord:
    run_id: str
    experiment_id: str
    method: str
    model: str
    dataset: str
    dataset_version: str
    data_fraction: float
    rag_k: Optional[int]
    lora_rank: Optional[int]
    noise_level: float
    seed: int
    total_examples: int
    sql_validity: Optional[float]
    exact_match: Optional[float]
    execution_accuracy: Optional[float]
    mean_latency_ms: Optional[float]
    training_time_s: Optional[float]
    inference_time_s: Optional[float]
    peak_vram_mb: Optional[float]
    trainable_parameters: Optional[int]
    total_parameters: Optional[int]
    status: str
    error_message: Optional[str]
    timestamp: str
    git_commit: Optional[str]
    hardware: Dict[str, Any]
    configuration_hash: str
    noise_metadata: Optional[Dict[str, Any]] = None
    model_type: str = "unknown"
    model_name: str = ""
    generator_type: str = "unknown"
    checkpoint_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "method": self.method,
            "model": self.model,
            "model_type": self.model_type,
            "model_name": self.model_name or self.model,
            "generator_type": self.generator_type,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "data_fraction": self.data_fraction,
            "rag_k": self.rag_k,
            "lora_rank": self.lora_rank,
            "noise_level": self.noise_level,
            "seed": self.seed,
            "total_examples": self.total_examples,
            "sql_validity": self.sql_validity,
            "exact_match": self.exact_match,
            "execution_accuracy": self.execution_accuracy,
            "mean_latency_ms": self.mean_latency_ms,
            "training_time_s": self.training_time_s,
            "inference_time_s": self.inference_time_s,
            "peak_vram_mb": self.peak_vram_mb,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "status": self.status,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "hardware": self.hardware,
            "configuration_hash": self.configuration_hash,
            "noise_metadata": self.noise_metadata,
            "checkpoint_dir": self.checkpoint_dir,
        }


MASTER_CSV_FIELDS = [
    "run_id",
    "experiment_id",
    "method",
    "model",
    "dataset",
    "dataset_version",
    "data_fraction",
    "rag_k",
    "lora_rank",
    "noise_level",
    "seed",
    "total_examples",
    "sql_validity",
    "exact_match",
    "execution_accuracy",
    "mean_latency_ms",
    "training_time_s",
    "inference_time_s",
    "peak_vram_mb",
    "trainable_parameters",
    "total_parameters",
    "status",
    "error_message",
    "timestamp",
    "git_commit",
    "configuration_hash",
    "hardware_device",
    "hardware_cuda",
]


def load_json_or_jsonl_examples(
    path: Path,
    split: str,
) -> List[SpiderExample]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    examples = []

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    examples.append(
                        SpiderExample(
                            example_id=item.get(
                                "example_id",
                                f"{split}_{idx}",
                            ),
                            db_id=item["db_id"],
                            question=item["question"],
                            query=item["query"],
                            split=item.get("split", split),
                            source_file=path.name,
                        )
                    )
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for idx, item in enumerate(raw):
            examples.append(
                SpiderExample(
                    example_id=item.get(
                        "example_id",
                        f"{split}_{idx}",
                    ),
                    db_id=item["db_id"],
                    question=item["question"],
                    query=item["query"],
                    split=split,
                    source_file=path.name,
                )
            )

    return examples


def append_to_master_table(
    master_path: Path,
    record: ExperimentResultRecord,
) -> None:
    master_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_file = master_path.with_suffix(".jsonl")
    csv_file = master_path.with_suffix(".csv")

    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")

    rec_dict = record.to_dict()
    rec_dict["hardware_device"] = record.hardware.get(
        "device",
        "unknown",
    )
    rec_dict["hardware_cuda"] = record.hardware.get(
        "cuda_available",
        False,
    )
    rec_dict.pop("hardware", None)
    rec_dict.pop("model_type", None)
    rec_dict.pop("model_name", None)
    rec_dict.pop("generator_type", None)
    rec_dict.pop("checkpoint_dir", None)

    if "noise_metadata" in rec_dict:
        rec_dict.pop("noise_metadata", None)

    with open(
        csv_file,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=MASTER_CSV_FIELDS,
            extrasaction="ignore",
        )

        if not csv_file.exists() or csv_file.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(
            {
                field: rec_dict.get(field)
                for field in MASTER_CSV_FIELDS
            }
        )


def execute_experiment_run(
    config: ExperimentConfig,
    train_file: Path,
    eval_file: Path,
    schemas_file: Path,
    db_dir: Path,
    output_base_dir: Path = Path("experiments/runs"),
    mock_generator: Optional[SQLGenerator] = None,
    dry_run: bool = False,
) -> ExperimentResultRecord:

    run_dir = output_base_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f)

    hw = detect_hardware()
    timestamp = datetime.now(timezone.utc).isoformat()
    cfg_hash = config.compute_hash()

    is_smoke = (
        getattr(config, "is_smoke", False)
        or config.experiment_id.upper().startswith("SMOKE")
        or config.dataset.lower() == "fixtures"
    )

    base_record_args = {
        "run_id": config.run_id,
        "experiment_id": config.experiment_id,
        "method": config.method,
        "model": config.model,
        "dataset": config.dataset,
        "dataset_version": config.dataset_version,
        "data_fraction": config.data_fraction,
        "rag_k": config.rag_k,
        "lora_rank": config.lora_rank,
        "noise_level": config.noise_level,
        "seed": config.seed,
        "total_examples": 0,
        "sql_validity": None,
        "exact_match": None,
        "execution_accuracy": None,
        "mean_latency_ms": None,
        "training_time_s": None,
        "inference_time_s": None,
        "peak_vram_mb": None,
        "trainable_parameters": None,
        "total_parameters": None,
        "status": "PENDING",
        "error_message": None,
        "timestamp": timestamp,
        "git_commit": None,
        "hardware": hw.to_dict(),
        "configuration_hash": cfg_hash,
        "noise_metadata": None,
        "model_type": "mock" if is_smoke else "huggingface_causal_lm",
        "model_name": config.model,
        "generator_type": "MockSQLGenerator" if is_smoke else "pending",
    }

    if not (
        train_file.is_file()
        and eval_file.is_file()
        and schemas_file.is_file()
    ):
        base_record_args["status"] = "PENDING_DATA"
        base_record_args["error_message"] = (
            f"Required dataset files not found on disk: "
            f"{train_file}, {eval_file}, or {schemas_file}."
        )
        rec = ExperimentResultRecord(**base_record_args)
        append_to_master_table(
            output_base_dir / "master_results",
            rec,
        )
        return rec

    train_pool = load_json_or_jsonl_examples(
        train_file,
        split="train",
    )

    eval_pool = load_json_or_jsonl_examples(
        eval_file,
        split="dev",
    )

    with open(schemas_file, "r", encoding="utf-8") as f:
        raw_schemas = json.load(f)

    schemas: Dict[str, DatabaseSchema] = {
        s["db_id"]: parse_spider_schema(s)
        for s in (
            raw_schemas
            if isinstance(raw_schemas, list)
            else raw_schemas.values()
        )
    }

    verify_split_integrity(
        train_pool,
        eval_pool,
    )

    if config.data_fraction < 1.0:
        subsets = generate_deterministic_subsets(
            train_pool,
            fractions=[config.data_fraction],
            seed=config.seed,
        )
        active_train = subsets[
            f"{int(config.data_fraction * 100)}pct"
        ][0]
    else:
        active_train = train_pool

    affected_ids = []

    if config.noise_level > 0.0:
        active_train, affected_ids = inject_training_noise(
            active_train,
            noise_level=config.noise_level,
            noise_type=config.noise_type,
            seed=config.seed,
        )

    noise_meta = {
        "noise_type": config.noise_type,
        "noise_rate": config.noise_level,
        "random_seed": config.seed,
        "affected_example_ids": affected_ids,
        "corruption_rule": (
            "none"
            if config.noise_level == 0.0
            or config.noise_type == "none"
            else f"deterministic_{config.noise_type}"
        ),
    }

    base_record_args["noise_metadata"] = noise_meta

    with open(
        run_dir / "noise_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            noise_meta,
            f,
            indent=2,
        )

    base_record_args["total_examples"] = len(eval_pool)

    if config.method in ("LORA", "QLORA"):
        lora_params = LoRAHyperparameters(
            r=config.lora_rank or 8,
            seed=config.seed,
        )

        accounting = compute_lora_parameter_accounting(
            lora_params
        )

        base_record_args["trainable_parameters"] = (
            accounting["trainable_lora_parameters"]
        )
        base_record_args["total_parameters"] = (
            accounting["total_base_parameters"]
        )

    if dry_run:
        base_record_args["status"] = "DRY_RUN_ONLY"
        base_record_args["generator_type"] = "none_dry_run"

        rec = ExperimentResultRecord(**base_record_args)

        with open(
            run_dir / "summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                rec.to_dict(),
                f,
                indent=2,
            )

        append_to_master_table(
            output_base_dir / "master_results",
            rec,
        )

        return rec

    if (
        config.hardware_requirements == "cuda"
        and not hw.cuda_available
    ):
        base_record_args["status"] = "PENDING_GPU"
        base_record_args["error_message"] = (
            "CUDA GPU hardware required for execution but unavailable."
        )
        base_record_args["generator_type"] = "none_pending_gpu"

        rec = ExperimentResultRecord(**base_record_args)

        with open(
            run_dir / "summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                rec.to_dict(),
                f,
                indent=2,
            )

        append_to_master_table(
            output_base_dir / "master_results",
            rec,
        )

        return rec

    generator: Optional[SQLGenerator] = None

    if is_smoke:
        generator = mock_generator

        if generator is None:
            ref_map = {
                ex.question.strip(): ex.query.strip()
                for ex in eval_pool
            }

            generator = MockSQLGenerator(
                mode="gold",
                reference_map=ref_map,
            )

        base_record_args["model_type"] = "mock"
        base_record_args["generator_type"] = type(generator).__name__

    else:
        if isinstance(mock_generator, MockSQLGenerator):
            base_record_args["status"] = "FAILED"
            base_record_args["error_message"] = (
                f"MockSQLGenerator is strictly forbidden for research "
                f"experiment '{config.experiment_id}'. "
                "Real model execution required."
            )
            base_record_args["generator_type"] = "REJECTED_MOCK"

            rec = ExperimentResultRecord(**base_record_args)

            with open(
                run_dir / "summary.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    rec.to_dict(),
                    f,
                    indent=2,
                )

            append_to_master_table(
                output_base_dir / "master_results",
                rec,
            )

            return rec

        if (
            hw.torch_version == "NOT_INSTALLED"
            or hw.transformers_version == "NOT_INSTALLED"
        ):
            base_record_args["status"] = "PENDING_DEPENDENCY"
            base_record_args["error_message"] = (
                "PyTorch and Transformers dependencies are required "
                "for real model execution."
            )
            base_record_args["generator_type"] = (
                "none_pending_dependency"
            )

            rec = ExperimentResultRecord(**base_record_args)

            with open(
                run_dir / "summary.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    rec.to_dict(),
                    f,
                    indent=2,
                )

            append_to_master_table(
                output_base_dir / "master_results",
                rec,
            )

            return rec

        if config.method in (
            "ZERO_SHOT",
            "FEW_SHOT",
            "RAG",
        ):
            try:
                device = (
                    "cuda"
                    if hw.cuda_available
                    else "cpu"
                )

                generator = HuggingFaceSQLGenerator(
                    model_name=config.model,
                    device=device,
                )

                base_record_args["model_type"] = (
                    "huggingface_causal_lm"
                )
                base_record_args["generator_type"] = (
                    "HuggingFaceSQLGenerator"
                )

            except Exception as e:
                base_record_args["status"] = "PENDING_MODEL"
                base_record_args["error_message"] = (
                    f"Failed to instantiate HuggingFace model "
                    f"'{config.model}': {e}"
                )
                base_record_args["generator_type"] = (
                    "none_pending_model"
                )

                rec = ExperimentResultRecord(**base_record_args)

                with open(
                    run_dir / "summary.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        rec.to_dict(),
                        f,
                        indent=2,
                    )

                append_to_master_table(
                    output_base_dir / "master_results",
                    rec,
                )

                return rec

        elif config.method == "LORA":
            base_record_args["model_type"] = "lora_adapter"
            base_record_args["generator_type"] = (
                "LoRAInferenceAdapter"
            )

            lora_params = LoRAHyperparameters(
                model_name=config.model,
                r=config.lora_rank or 8,
                seed=config.seed,
                learning_rate=config.training_settings.get(
                    "learning_rate",
                    0.0002,
                ),
                num_epochs=config.training_settings.get(
                    "num_epochs",
                    3,
                ),
                batch_size=config.training_settings.get(
                    "batch_size",
                    2,
                ),
                gradient_accumulation_steps=config.training_settings.get(
                    "gradient_accumulation_steps",
                    8,
                ),
                max_seq_length=config.training_settings.get(
                    "max_seq_length",
                    1024,
                ),
            )

            accounting = compute_lora_parameter_accounting(
                lora_params
            )

            base_record_args["trainable_parameters"] = (
                accounting["trainable_lora_parameters"]
            )
            base_record_args["total_parameters"] = (
                accounting["total_base_parameters"]
            )

            if not hw.cuda_available:
                base_record_args["status"] = "PENDING_GPU"
                base_record_args["error_message"] = (
                    "CUDA GPU hardware required for LoRA "
                    "training and inference."
                )

                rec = ExperimentResultRecord(**base_record_args)

                with open(
                    run_dir / "summary.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        rec.to_dict(),
                        f,
                        indent=2,
                    )

                append_to_master_table(
                    output_base_dir / "master_results",
                    rec,
                )

                return rec

            formatted_dataset = format_dataset(
                active_train,
                schemas,
            )

            checkpoint_dir = (
                run_dir / "adapter_checkpoint"
            )

            train_start = time.perf_counter()

            try:
                train_lora_model(
                    formatted_dataset=formatted_dataset,
                    hyperparameters=lora_params,
                    output_dir=checkpoint_dir,
                    device="cuda",
                )

                train_time = (
                    time.perf_counter()
                    - train_start
                )

                base_record_args["training_time_s"] = round(
                    train_time,
                    2,
                )

                base_record_args["checkpoint_dir"] = str(
                    checkpoint_dir
                )

            except Exception as e:
                base_record_args["status"] = "FAILED"
                base_record_args["error_message"] = (
                    f"LoRA training failed: {e}"
                )

                rec = ExperimentResultRecord(**base_record_args)

                with open(
                    run_dir / "summary.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        rec.to_dict(),
                        f,
                        indent=2,
                    )

                append_to_master_table(
                    output_base_dir / "master_results",
                    rec,
                )

                return rec

            try:
                generator = LoRAInferenceAdapter(
                    checkpoint_dir=checkpoint_dir,
                    base_model_name=config.model,
                )

            except Exception as e:
                base_record_args["status"] = "PENDING_MODEL"
                base_record_args["error_message"] = (
                    f"LoRA adapter initialization failed: {e}"
                )

                rec = ExperimentResultRecord(**base_record_args)

                with open(
                    run_dir / "summary.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        rec.to_dict(),
                        f,
                        indent=2,
                    )

                append_to_master_table(
                    output_base_dir / "master_results",
                    rec,
                )

                return rec

        elif config.method == "QLORA":
            base_record_args["model_type"] = "qlora_adapter"
            base_record_args["generator_type"] = (
                "QLoRAInferenceAdapter"
            )

            dep_check = check_qlora_dependencies()

            if not dep_check["ready"]:
                base_record_args["status"] = (
                    "PENDING_GPU"
                    if not hw.cuda_available
                    else "PENDING_DEPENDENCY"
                )
                base_record_args["error_message"] = (
                    dep_check["reason"]
                )

                rec = ExperimentResultRecord(**base_record_args)

                with open(
                    run_dir / "summary.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        rec.to_dict(),
                        f,
                        indent=2,
                    )

                append_to_master_table(
                    output_base_dir / "master_results",
                    rec,
                )

                return rec

            q_cfg = QLoRAConfig(
                **(
                    config.qlora_config
                    or {"load_in_4bit": True}
                )
            )

            lora_params = LoRAHyperparameters(
                model_name=config.model,
                r=config.lora_rank or 8,
                seed=config.seed,
                learning_rate=config.training_settings.get(
                    "learning_rate",
                    0.0002,
                ),
                num_epochs=config.training_settings.get(
                    "num_epochs",
                    3,
                ),
                batch_size=config.training_settings.get(
                    "batch_size",
                    2,
                ),
                gradient_accumulation_steps=config.training_settings.get(
                    "gradient_accumulation_steps",
                    8,
                ),
                max_seq_length=config.training_settings.get(
                    "max_seq_length",
                    1024,
                ),
            )

            accounting = compute_lora_parameter_accounting(
                lora_params
            )

            base_record_args["trainable_parameters"] = (
                accounting["trainable_lora_parameters"]
            )
            base_record_args["total_parameters"] = (
                accounting["total_base_parameters"]
            )

            formatted_dataset = format_dataset(
                active_train,
                schemas,
            )

            checkpoint_dir = (
                run_dir / "adapter_checkpoint"
            )

            train_start = time.perf_counter()

            try:
                train_qlora_model(
                    formatted_dataset=formatted_dataset,
                    hyperparameters=lora_params,
                    qlora_config=q_cfg,
                    output_dir=checkpoint_dir,
                )

                train_time = (
                    time.perf_counter()
                    - train_start
                )

                base_record_args["training_time_s"] = round(
                    train_time,
                    2,
                )

                base_record_args["checkpoint_dir"] = str(
                    checkpoint_dir
                )

            except Exception as e:
                base_record_args["status"] = "FAILED"
                base_record_args["error_message"] = (
                    f"QLoRA training failed: {e}"
                )

                rec = ExperimentResultRecord(**base_record_args)

                with open(
                    run_dir / "summary.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        rec.to_dict(),
                        f,
                        indent=2,
                    )

                append_to_master_table(
                    output_base_dir / "master_results",
                    rec,
                )

                return rec

            try:
                generator = QLoRAInferenceAdapter(
                    checkpoint_dir=checkpoint_dir,
                    base_model_name=config.model,
                    qlora_config=q_cfg,
                )

            except Exception as e:
                base_record_args["status"] = "PENDING_MODEL"
                base_record_args["error_message"] = (
                    f"QLoRA adapter initialization failed: {e}"
                )

                rec = ExperimentResultRecord(**base_record_args)

                with open(
                    run_dir / "summary.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        rec.to_dict(),
                        f,
                        indent=2,
                    )

                append_to_master_table(
                    output_base_dir / "master_results",
                    rec,
                )

                return rec

    retriever = None
    few_shot_demos = None
    few_shot_k = 0

    if config.method == "FEW_SHOT":
        few_shot_demos = active_train
        few_shot_k = config.training_settings.get(
            "few_shot_k",
            3,
        )

    elif config.method == "RAG":
        retriever = BM25Retriever()
        retriever.index(active_train)
        few_shot_k = config.rag_k or 3

    elif config.method == "LORA" and is_smoke:
        generator = LoRAInferenceAdapter(
            checkpoint_dir=run_dir,
            mock_generator=generator,
        )

        if base_record_args["training_time_s"] is None:
            base_record_args["training_time_s"] = 0.0

    elif config.method == "QLORA" and is_smoke:
        q_cfg = QLoRAConfig(
            **(
                config.qlora_config
                or {"load_in_4bit": True}
            )
        )

        generator = QLoRAInferenceAdapter(
            checkpoint_dir=run_dir,
            qlora_config=q_cfg,
            mock_generator=generator,
        )

        if base_record_args["training_time_s"] is None:
            base_record_args["training_time_s"] = 0.0

    inf_start = time.perf_counter()

    summary = run_evaluation(
        eval_examples=eval_pool,
        schemas=schemas,
        db_dir=db_dir,
        generator=generator,
        method=(
            "rag"
            if config.method == "RAG"
            else (
                "few_shot"
                if config.method == "FEW_SHOT"
                else "zero_shot"
            )
        ),
        demonstrations=few_shot_demos,
        few_shot_k=few_shot_k,
        retriever=retriever,
        output_file=run_dir / "evaluation_results.json",
    )

    inf_time = time.perf_counter() - inf_start

    eval_json_p = run_dir / "evaluation_results.json"

    if eval_json_p.is_file():
        with open(
            eval_json_p,
            "r",
            encoding="utf-8",
        ) as f:
            eval_payload = json.load(f)

        with open(
            run_dir / "per_example.jsonl",
            "w",
            encoding="utf-8",
        ) as f:
            for item in eval_payload.get(
                "results",
                [],
            ):
                f.write(
                    json.dumps(item) + "\n"
                )

    base_record_args["status"] = "COMPLETED"
    base_record_args["sql_validity"] = summary.get(
        "sql_validity"
    )
    base_record_args["exact_match"] = summary.get(
        "exact_match"
    )
    base_record_args["execution_accuracy"] = summary.get(
        "execution_accuracy"
    )
    base_record_args["mean_latency_ms"] = summary.get(
        "mean_latency_ms"
    )
    base_record_args["inference_time_s"] = round(
        inf_time,
        2,
    )

    if base_record_args["training_time_s"] is None:
        base_record_args["training_time_s"] = 0.0

    rec = ExperimentResultRecord(**base_record_args)

    with open(
        run_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rec.to_dict(),
            f,
            indent=2,
        )

    append_to_master_table(
        output_base_dir / "master_results",
        rec,
    )

    return rec