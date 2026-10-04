from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

VALID_METHODS = {"ZERO_SHOT", "FEW_SHOT", "RAG", "LORA", "QLORA"}
VALID_RAG_K = {1, 3, 5, 10}
VALID_LORA_RANKS = {4, 8, 16}
VALID_NOISE_LEVELS = {0.0, 0.05, 0.10, 0.20}


@dataclass
class ExperimentConfig:
    experiment_id: str
    method: str
    model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    dataset: str = "spider"
    dataset_version: str = "1.0"
    data_fraction: float = 1.0
    rag_k: Optional[int] = None
    lora_rank: Optional[int] = None
    qlora_config: Optional[Dict[str, Any]] = None
    noise_level: float = 0.0
    noise_type: str = "none"
    seed: int = 42
    eval_split: str = "dev"
    training_settings: Dict[str, Any] = field(default_factory=dict)
    hardware_requirements: str = "cpu"
    run_id: Optional[str] = None
    is_smoke: bool = False

    def __post_init__(self):
        self.method = self.method.upper()
        if self.experiment_id.upper().startswith("SMOKE") or self.dataset.lower() == "fixtures":
            self.is_smoke = True
        elif self.method in ("LORA", "QLORA") and self.hardware_requirements == "cpu":
            self.hardware_requirements = "cuda"
        self.validate()
        if not self.run_id:
            cfg_hash = self.compute_hash()[:8]
            self.run_id = f"{self.experiment_id}_{self.method.lower()}_s{self.seed}_{cfg_hash}"

    def validate(self) -> None:
        """Validates configuration parameters and prevents invalid combinations."""
        if self.method not in VALID_METHODS:
            raise ValueError(f"Invalid method '{self.method}'. Must be one of {sorted(list(VALID_METHODS))}")

        if not (0.0 < self.data_fraction <= 1.0):
            raise ValueError(f"data_fraction must be in range (0.0, 1.0], got {self.data_fraction}")

        if round(self.noise_level, 2) not in VALID_NOISE_LEVELS:
            raise ValueError(f"noise_level must be one of {VALID_NOISE_LEVELS}, got {self.noise_level}")

        if self.noise_level > 0.0 and self.noise_type == "none":
            raise ValueError("noise_type must be specified when noise_level > 0.0")

        # Method-specific parameter validation
        if self.method == "ZERO_SHOT":
            if self.rag_k is not None:
                raise ValueError("rag_k cannot be specified for ZERO_SHOT experiments.")
            if self.lora_rank is not None:
                raise ValueError("lora_rank cannot be specified for ZERO_SHOT experiments.")
            if self.qlora_config is not None:
                raise ValueError("qlora_config cannot be specified for ZERO_SHOT experiments.")

        elif self.method == "FEW_SHOT":
            if self.rag_k is not None:
                raise ValueError("rag_k applies only to RAG; use training_settings['few_shot_k'] for FEW_SHOT.")
            if self.lora_rank is not None:
                raise ValueError("lora_rank cannot be specified for FEW_SHOT experiments.")
            if self.qlora_config is not None:
                raise ValueError("qlora_config cannot be specified for FEW_SHOT experiments.")

        elif self.method == "RAG":
            if self.rag_k not in VALID_RAG_K:
                raise ValueError(f"RAG experiments require rag_k in {sorted(list(VALID_RAG_K))}, got {self.rag_k}")
            if self.lora_rank is not None:
                raise ValueError("lora_rank cannot be specified for RAG experiments.")
            if self.qlora_config is not None:
                raise ValueError("qlora_config cannot be specified for RAG experiments.")

        elif self.method == "LORA":
            if self.lora_rank not in VALID_LORA_RANKS:
                raise ValueError(f"LORA experiments require lora_rank in {sorted(list(VALID_LORA_RANKS))}, got {self.lora_rank}")
            if self.rag_k is not None:
                raise ValueError("rag_k cannot be specified for LORA experiments.")
            if self.qlora_config is not None:
                raise ValueError("qlora_config cannot be specified for unquantized LORA experiments.")

        elif self.method == "QLORA":
            if self.lora_rank not in VALID_LORA_RANKS:
                raise ValueError(f"QLORA experiments require lora_rank in {sorted(list(VALID_LORA_RANKS))}, got {self.lora_rank}")
            if self.rag_k is not None:
                raise ValueError("rag_k cannot be specified for QLORA experiments.")
            if not isinstance(self.qlora_config, dict):
                raise ValueError("QLORA experiments require a qlora_config dictionary (e.g. quant_type='nf4').")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "method": self.method,
            "model": self.model,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "data_fraction": self.data_fraction,
            "rag_k": self.rag_k,
            "lora_rank": self.lora_rank,
            "qlora_config": self.qlora_config,
            "noise_level": self.noise_level,
            "noise_type": self.noise_type,
            "seed": self.seed,
            "eval_split": self.eval_split,
            "training_settings": self.training_settings,
            "hardware_requirements": self.hardware_requirements,
            "is_smoke": self.is_smoke,
        }

    def compute_hash(self) -> str:
        """Computes a deterministic SHA256 configuration hash."""
        d = self.to_dict()
        d.pop("run_id", None)
        serialized = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_manifest(manifest_path: Path | str) -> List[ExperimentConfig]:
    """Loads an experiment manifest YAML file into a list of ExperimentConfig objects."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "experiments" not in data:
        raise ValueError(f"Manifest at {path} must be a dictionary containing an 'experiments' list.")

    configs: List[ExperimentConfig] = []
    for idx, item in enumerate(data["experiments"]):
        try:
            cfg = ExperimentConfig(**item)
            configs.append(cfg)
        except Exception as e:
            raise ValueError(f"Failed parsing experiment {idx} ('{item.get('experiment_id')}') in {path.name}: {e}")

    return configs
