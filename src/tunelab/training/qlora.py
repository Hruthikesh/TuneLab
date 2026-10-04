from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tunelab.training.hardware import HardwareProfile, detect_hardware
from tunelab.training.lora import LoRAHyperparameters, compute_lora_parameter_accounting

logger = logging.getLogger(__name__)

VALID_QUANT_TYPES = {"nf4", "fp4"}
VALID_COMPUTE_DTYPES = {"bfloat16", "float16", "float32"}


@dataclass
class QLoRAConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    def __init__(
        self,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: Optional[str] = None,
        bnb_4bit_use_double_quant: Optional[bool] = None,
        bnb_4bit_compute_dtype: Optional[str] = None,
        quant_type: Optional[str] = None,
        double_quant: Optional[bool] = None,
        compute_dtype: Optional[str] = None,
    ):
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = (bnb_4bit_quant_type or quant_type or "nf4").lower()
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant if bnb_4bit_use_double_quant is not None else (double_quant if double_quant is not None else True)
        self.bnb_4bit_compute_dtype = (bnb_4bit_compute_dtype or compute_dtype or "bfloat16").lower()
        self._validate()

    def _validate(self):
        if not self.load_in_4bit:
            raise ValueError("QLoRA requires load_in_4bit=True. For unquantized LoRA, use Phase 5.")
        if self.bnb_4bit_quant_type.lower() not in VALID_QUANT_TYPES:
            raise ValueError(
                f"Invalid bnb_4bit_quant_type: '{self.bnb_4bit_quant_type}'. Must be one of {sorted(list(VALID_QUANT_TYPES))}"
            )
        if self.bnb_4bit_compute_dtype.lower() not in VALID_COMPUTE_DTYPES:
            raise ValueError(
                f"Invalid bnb_4bit_compute_dtype: '{self.bnb_4bit_compute_dtype}'. Must be one of {sorted(list(VALID_COMPUTE_DTYPES))}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type.lower(),
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype.lower(),
        }


def check_qlora_dependencies() -> Dict[str, Any]:
    hw = detect_hardware()
    missing: List[str] = []

    if hw.torch_version == "NOT_INSTALLED":
        missing.append("torch")
    if hw.transformers_version == "NOT_INSTALLED":
        missing.append("transformers")
    if hw.peft_version == "NOT_INSTALLED":
        missing.append("peft")
    if not hw.bitsandbytes_available:
        missing.append("bitsandbytes")
    if not hw.cuda_available:
        missing.append("CUDA (GPU with compute capability >= 7.0)")

    ready = len(missing) == 0
    reason = None if ready else f"Missing required QLoRA components: {', '.join(missing)}"

    return {
        "ready": ready,
        "missing": missing,
        "reason": reason,
        "hardware_profile": hw,
    }


def load_qlora_model(
    model_name: str,
    qlora_config: QLoRAConfig,
    lora_params: LoRAHyperparameters,
):
    dep_check = check_qlora_dependencies()
    if not dep_check["ready"]:
        raise RuntimeError(
            f"Cannot initialize QLoRA model: {dep_check['reason']}. "
            "No silent fallback permitted. QLoRA execution requires GPU and bitsandbytes."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    compute_dtype = torch.bfloat16 if qlora_config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        r=lora_params.r,
        lora_alpha=lora_params.alpha,
        lora_dropout=lora_params.dropout,
        target_modules=lora_params.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, peft_config)
    return model, tokenizer


def save_qlora_run_metadata(
    output_dir: Path | str,
    hyperparameters: LoRAHyperparameters,
    qlora_config: QLoRAConfig,
    hardware: HardwareProfile,
    status: str,
    sample_count: int,
    data_fraction: float,
    memory_stats: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta_file = out_path / "run_metadata.json"

    accounting = compute_lora_parameter_accounting(hyperparameters)

    payload = {
        "method": "qlora",
        "status": status,
        "model_id": hyperparameters.model_name,
        "hyperparameters": hyperparameters.to_dict(),
        "qlora_config": qlora_config.to_dict(),
        "parameter_accounting": accounting,
        "data_fraction": data_fraction,
        "training_sample_count": sample_count,
        "hardware_profile": hardware.to_dict(),
        "memory_stats": memory_stats or {},
        "metrics": metrics or {},
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Saved QLoRA run metadata to {meta_file} (Status: {status})")
    return meta_file


def train_qlora_model(
    formatted_dataset: List[Dict[str, Any]],
    hyperparameters: LoRAHyperparameters,
    qlora_config: QLoRAConfig,
    output_dir: Path | str,
) -> Dict[str, Any]:
    model, tokenizer = load_qlora_model(
        model_name=hyperparameters.model_name,
        qlora_config=qlora_config,
        lora_params=hyperparameters,
    )

    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from torch.utils.data import Dataset

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

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
        fp16=(qlora_config.bnb_4bit_compute_dtype == "float16" or not torch.cuda.is_bf16_supported()),
        bf16=(qlora_config.bnb_4bit_compute_dtype == "bfloat16" and torch.cuda.is_bf16_supported()),
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
