from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tunelab.training.hardware import HardwareProfile

logger = logging.getLogger(__name__)

TARGET_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

VALID_QWEN_TARGET_MODULES: Set[str] = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

QWEN_1_5B_HIDDEN_SIZE = 1536
QWEN_1_5B_INTERMEDIATE_SIZE = 8960
QWEN_1_5B_NUM_LAYERS = 28
QWEN_1_5B_TOTAL_PARAMS = 1543714816


@dataclass
class LoRAHyperparameters:
    model_name: str = TARGET_MODEL_ID
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    learning_rate: float = 0.0002
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    seed: int = 42
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03

    def __post_init__(self):
        if self.r not in (4, 8, 16):
            raise ValueError(f"LoRA rank r must be one of [4, 8, 16], got {self.r}")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"Dropout must be in range [0.0, 1.0), got {self.dropout}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        validate_target_modules(self.target_modules)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "seed": self.seed,
            "target_modules": self.target_modules,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "warmup_ratio": self.warmup_ratio,
        }


def validate_target_modules(target_modules: List[str]) -> None:
    if not target_modules:
        raise ValueError("target_modules cannot be empty.")
    for module in target_modules:
        if module not in VALID_QWEN_TARGET_MODULES:
            raise ValueError(
                f"Target module '{module}' is invalid for Qwen2.5. "
                f"Valid modules are: {sorted(list(VALID_QWEN_TARGET_MODULES))}"
            )


def compute_lora_parameter_accounting(
    config: LoRAHyperparameters,
    total_base_params: int = QWEN_1_5B_TOTAL_PARAMS,
    hidden_size: int = QWEN_1_5B_HIDDEN_SIZE,
    intermediate_size: int = QWEN_1_5B_INTERMEDIATE_SIZE,
    num_layers: int = QWEN_1_5B_NUM_LAYERS,
) -> Dict[str, Any]:
    trainable_params_per_layer = 0
    rank = config.r

    for mod in config.target_modules:
        if mod in ("q_proj", "k_proj", "v_proj", "o_proj"):
            trainable_params_per_layer += 2 * rank * hidden_size
        elif mod in ("gate_proj", "up_proj"):
            trainable_params_per_layer += rank * (hidden_size + intermediate_size)
        elif mod == "down_proj":
            trainable_params_per_layer += rank * (intermediate_size + hidden_size)

    total_trainable_params = trainable_params_per_layer * num_layers
    percentage = (total_trainable_params / total_base_params) * 100.0

    return {
        "total_base_parameters": total_base_params,
        "trainable_lora_parameters": total_trainable_params,
        "trainable_parameter_percentage": round(percentage, 4),
        "rank": rank,
        "target_modules": config.target_modules,
        "num_layers": num_layers,
    }


def save_run_metadata(
    output_dir: Path | str,
    hyperparameters: LoRAHyperparameters,
    hardware: HardwareProfile,
    status: str,
    sample_count: int,
    data_fraction: float,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta_file = out_path / "run_metadata.json"

    param_accounting = compute_lora_parameter_accounting(hyperparameters)

    payload = {
        "status": status,
        "model_id": hyperparameters.model_name,
        "hyperparameters": hyperparameters.to_dict(),
        "parameter_accounting": param_accounting,
        "data_fraction": data_fraction,
        "training_sample_count": sample_count,
        "hardware_profile": hardware.to_dict(),
        "metrics": metrics or {},
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Saved run metadata to {meta_file} (Status: {status})")
    return meta_file


def train_lora_model(
    formatted_dataset: List[Dict[str, Any]],
    hyperparameters: LoRAHyperparameters,
    output_dir: Path | str,
    device: str = "cuda",
) -> Dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise RuntimeError(f"Missing required training dependencies: {e}. PyTorch, Transformers, and PEFT are required.")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    device_map = "auto" if device == "cuda" else {"": "cpu"}

    tokenizer = AutoTokenizer.from_pretrained(
        hyperparameters.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        hyperparameters.model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=hyperparameters.r,
        lora_alpha=hyperparameters.alpha,
        lora_dropout=hyperparameters.dropout,
        target_modules=hyperparameters.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_config)

    tokenized_items = []
    for item in formatted_dataset:
        prompt_text = item["prompt"]
        target_text = item["target_sql"] + tokenizer.eos_token

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)

        input_ids = (prompt_ids + target_ids)[:hyperparameters.max_seq_length]
        labels = [-100] * min(len(prompt_ids), hyperparameters.max_seq_length) + target_ids[:max(0, hyperparameters.max_seq_length - len(prompt_ids))]
        attention_mask = [1] * len(input_ids)

        tokenized_items.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    from torch.utils.data import Dataset
    class SQLTrainingDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_ds = SQLTrainingDataset(tokenized_items)
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    training_args = TrainingArguments(
        output_dir=str(out_p),
        per_device_train_batch_size=hyperparameters.batch_size,
        gradient_accumulation_steps=hyperparameters.gradient_accumulation_steps,
        learning_rate=hyperparameters.learning_rate,
        num_train_epochs=hyperparameters.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        seed=hyperparameters.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    model.save_pretrained(str(out_p))
    tokenizer.save_pretrained(str(out_p))

    return {
        "training_loss": train_result.training_loss,
        "global_step": train_result.global_step,
    }
