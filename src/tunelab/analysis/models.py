import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PerExampleResult:
    example_id: str
    database_id: str
    predicted_sql: str
    reference_sql: str
    exact_match: bool
    execution_match: bool
    error_type: str
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "database_id": self.database_id,
            "predicted_sql": self.predicted_sql,
            "reference_sql": self.reference_sql,
            "exact_match": self.exact_match,
            "execution_match": self.execution_match,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ErrorAnalysisRecord:
    example_id: str
    database_id: str
    coarse_error: str
    fine_error: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "database_id": self.database_id,
            "coarse_error": self.coarse_error,
            "fine_error": self.fine_error,
            "explanation": self.explanation,
        }


@dataclass
class ExperimentRun:
    run_id: str
    experiment_id: str
    method: str
    model: str
    dataset: str
    dataset_version: str
    data_fraction: float
    model_type: str = "unknown"
    model_name: str = ""
    generator_type: str = "unknown"
    rag_k: Optional[int] = None
    lora_rank: Optional[int] = None
    noise_level: float = 0.0
    noise_type: str = "none"
    seed: int = 42
    status: str = "COMPLETED"
    total_examples: int = 0
    sql_validity: Optional[float] = None
    exact_match: Optional[float] = None
    execution_accuracy: Optional[float] = None
    mean_latency_ms: Optional[float] = None
    median_latency_ms: Optional[float] = None
    training_time_s: Optional[float] = None
    inference_time_s: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    trainable_parameters: Optional[int] = None
    total_parameters: Optional[int] = None
    error_message: Optional[str] = None
    noise_metadata: Optional[Dict[str, Any]] = None
    hardware: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    git_commit: Optional[str] = None
    configuration_hash: str = ""
    run_dir: Optional[str] = None
    per_example_results: Optional[List[PerExampleResult]] = None
    is_research_result: bool = False
    is_pipeline_test: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "method": self.method,
            "model": self.model,
            "model_type": self.model_type,
            "generator_type": self.generator_type,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "data_fraction": self.data_fraction,
            "rag_k": self.rag_k,
            "lora_rank": self.lora_rank,
            "noise_level": self.noise_level,
            "noise_type": self.noise_type,
            "seed": self.seed,
            "status": self.status,
            "total_examples": self.total_examples,
            "sql_validity": self.sql_validity,
            "exact_match": self.exact_match,
            "execution_accuracy": self.execution_accuracy,
            "mean_latency_ms": self.mean_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "training_time_s": self.training_time_s,
            "inference_time_s": self.inference_time_s,
            "peak_vram_mb": self.peak_vram_mb,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "error_message": self.error_message,
            "noise_metadata": self.noise_metadata,
            "hardware": self.hardware,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "configuration_hash": self.configuration_hash,
            "run_dir": self.run_dir,
            "is_research_result": self.is_research_result,
            "is_pipeline_test": self.is_pipeline_test,
        }


@dataclass
class ExperimentSummary:
    config_key: str
    method: str
    model: str
    data_fraction: float
    rag_k: Optional[int]
    lora_rank: Optional[int]
    noise_level: float
    n_runs: int
    seeds: List[int]
    status: str
    execution_accuracy_mean: Optional[float] = None
    execution_accuracy_std: Optional[float] = None
    exact_match_mean: Optional[float] = None
    exact_match_std: Optional[float] = None
    sql_validity_mean: Optional[float] = None
    sql_validity_std: Optional[float] = None
    mean_latency_ms: Optional[float] = None
    median_latency_ms: Optional[float] = None
    training_time_mean_s: Optional[float] = None
    peak_vram_mean_mb: Optional[float] = None
    trainable_parameters: Optional[int] = None
    total_parameters: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_key": self.config_key,
            "method": self.method,
            "model": self.model,
            "data_fraction": self.data_fraction,
            "rag_k": self.rag_k,
            "lora_rank": self.lora_rank,
            "noise_level": self.noise_level,
            "n_runs": self.n_runs,
            "seeds": self.seeds,
            "status": self.status,
            "execution_accuracy_mean": self.execution_accuracy_mean,
            "execution_accuracy_std": self.execution_accuracy_std,
            "exact_match_mean": self.exact_match_mean,
            "exact_match_std": self.exact_match_std,
            "sql_validity_mean": self.sql_validity_mean,
            "sql_validity_std": self.sql_validity_std,
            "mean_latency_ms": self.mean_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "training_time_mean_s": self.training_time_mean_s,
            "peak_vram_mean_mb": self.peak_vram_mean_mb,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
        }
