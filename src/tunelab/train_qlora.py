import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from tunelab.data.leakage import verify_split_integrity
from tunelab.data.pools import generate_deterministic_subsets
from tunelab.train import load_examples_file, load_schemas_file
from tunelab.training.format import format_dataset
from tunelab.training.lora import (
    LoRAHyperparameters,
    compute_lora_parameter_accounting,
)
from tunelab.training.memory import get_gpu_memory_stats
from tunelab.training.qlora import (
    QLoRAConfig,
    check_qlora_dependencies,
    save_qlora_run_metadata,
    train_qlora_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("tunelab.train_qlora")


def run_qlora_pipeline(
    train_file: str | Path,
    schemas_file: str | Path,
    eval_file: Optional[str | Path] = None,
    output_dir: str | Path = "checkpoints/qlora_run",
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
    quant_type: str = "nf4",
    double_quant: bool = True,
    compute_dtype: str = "bfloat16",
    dry_run: bool = False,
) -> Dict[str, Any]:
    if not 0 < data_fraction <= 1.0:
        raise ValueError("data_fraction must be between 0 and 1.")

    if quant_type not in {"nf4", "fp4"}:
        raise ValueError("quant_type must be 'nf4' or 'fp4'.")

    if compute_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(
            "compute_dtype must be 'bfloat16', 'float16', or 'float32'."
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dep_check = check_qlora_dependencies()
    hw = dep_check["hardware_profile"]

    logger.info(
        "QLoRA Hardware Profile: Device=%s, CUDA=%s, RAM=%.1fGB",
        hw.device,
        hw.cuda_available,
        hw.ram_total_gb,
    )

    logger.info(
        "Dependencies: bitsandbytes=%s, PEFT=%s, PyTorch=%s",
        hw.bitsandbytes_available,
        hw.peft_version,
        hw.torch_version,
    )

    lora_params = LoRAHyperparameters(
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

    qlora_cfg = QLoRAConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    train_pool = load_examples_file(
        train_file,
        split="train",
    )

    schemas = load_schemas_file(schemas_file)

    if eval_file is not None:
        eval_examples = load_examples_file(
            eval_file,
            split="dev",
        )

        verify_split_integrity(
            train_pool,
            eval_examples,
        )

        logger.info(
            "Split Integrity: PASSED"
        )

    if data_fraction < 1.0:
        subsets = generate_deterministic_subsets(
            train_examples=train_pool,
            fractions=[data_fraction],
            seed=seed,
            source_pool="train_spider_only",
        )

        fraction_key = f"{int(data_fraction * 100)}pct"

        if fraction_key not in subsets:
            raise ValueError(
                f"Generated subset '{fraction_key}' was not found."
            )

        active_train_examples = subsets[fraction_key][0]
    else:
        active_train_examples = train_pool

    logger.info(
        "Active training examples: %d (%.0f%%)",
        len(active_train_examples),
        data_fraction * 100,
    )

    formatted_dataset = format_dataset(
        active_train_examples,
        schemas,
    )

    accounting = compute_lora_parameter_accounting(
        lora_params
    )

    logger.info(
        "QLoRA Parameter Accounting: rank=%d, quantization=%s",
        lora_params.r,
        qlora_cfg.bnb_4bit_quant_type,
    )

    logger.info(
        "Base Parameters: %s",
        f"{accounting['total_base_parameters']:,}",
    )

    logger.info(
        "Trainable LoRA Parameters: %s",
        f"{accounting['trainable_lora_parameters']:,}",
    )

    logger.info(
        "Trainable Ratio: %.4f%%",
        accounting["trainable_parameter_percentage"],
    )

    memory_stats = get_gpu_memory_stats()

    metrics = None

    if dry_run:
        status = "DRY_RUN_ONLY"

        logger.info(
            "QLoRA dry run completed. No model training was started."
        )

    elif not dep_check["ready"]:
        if not hw.cuda_available:
            status = "QLORA_PENDING_GPU"
        else:
            status = "QLORA_DEPENDENCIES_UNAVAILABLE"

        logger.warning(
            "QLoRA training unavailable: %s",
            dep_check.get("reason"),
        )

    else:
        logger.info(
            "CUDA and QLoRA dependencies verified. "
            "Starting real QLoRA training."
        )

        try:
            metrics = train_qlora_model(
                formatted_dataset=formatted_dataset,
                hyperparameters=lora_params,
                qlora_config=qlora_cfg,
                output_dir=out_path,
            )

            status = "COMPLETED"

            logger.info(
                "QLoRA training completed successfully."
            )

        except Exception as exc:
            status = "FAILED"
            metrics = {
                "error": str(exc),
            }

            logger.exception(
                "QLoRA training failed."
            )

    metadata_file = save_qlora_run_metadata(
        output_dir=out_path,
        hyperparameters=lora_params,
        qlora_config=qlora_cfg,
        hardware=hw,
        status=status,
        sample_count=len(active_train_examples),
        data_fraction=data_fraction,
        memory_stats=memory_stats,
        metrics=metrics,
    )

    return {
        "method": "qlora",
        "status": status,
        "data_fraction": data_fraction,
        "sample_count": len(active_train_examples),
        "qlora_config": qlora_cfg.to_dict(),
        "parameter_accounting": accounting,
        "memory_stats": memory_stats,
        "run_metadata_file": str(metadata_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="TuneLab QLoRA 4-bit supervised fine-tuning CLI"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/full.yaml",
    )

    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--schemas-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--eval-file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/qlora_run",
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--r",
        type=int,
        choices=[4, 8, 16],
        default=8,
    )

    parser.add_argument(
        "--alpha",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0002,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
    )

    double_quant_group = parser.add_mutually_exclusive_group()

    double_quant_group.add_argument(
        "--double-quant",
        dest="double_quant",
        action="store_true",
    )

    double_quant_group.add_argument(
        "--no-double-quant",
        dest="double_quant",
        action="store_false",
    )

    parser.set_defaults(
        double_quant=True
    )

    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    summary = run_qlora_pipeline(
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
        quant_type=args.quant_type,
        double_quant=args.double_quant,
        compute_dtype=args.compute_dtype,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("QLORA PIPELINE SUMMARY")
    logger.info("=" * 60)

    for key, value in summary.items():
        logger.info(
            "%s: %s",
            key,
            value,
        )


if __name__ == "__main__":
    main()