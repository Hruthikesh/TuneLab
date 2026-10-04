import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from tunelab.analysis.models import ExperimentRun

logger = logging.getLogger(__name__)


def compare_methods(runs: List[ExperimentRun]) -> pd.DataFrame:
    """Generates a matched method comparison table across the 5 canonical methods.
    
    Compares ZERO_SHOT, FEW_SHOT, RAG, LORA, QLORA under matched experimental conditions:
    (clean 0% noise, full data 1.0 / matched fraction, same model and dataset).
    """
    columns = [
        "Method",
        "Data Fraction",
        "RAG K",
        "LoRA Rank",
        "Noise",
        "EM",
        "EX",
        "SQL Validity",
        "Latency (ms)",
        "Training Time (s)",
        "Peak VRAM (MB)",
        "Status",
    ]

    target_methods = ["ZERO_SHOT", "FEW_SHOT", "RAG", "LORA", "QLORA"]
    rows = []

    # Group runs by method where noise is 0.0
    clean_runs = [r for r in runs if r.noise_level == 0.0]

    for method in target_methods:
        method_runs = [r for r in clean_runs if r.method.upper() == method]
        # Sort by completed status first, then highest fraction
        method_runs.sort(key=lambda r: (r.status == "COMPLETED", r.data_fraction), reverse=True)

        if not method_runs:
            rows.append({
                "Method": method,
                "Data Fraction": None,
                "RAG K": None,
                "LoRA Rank": None,
                "Noise": 0.0,
                "EM": None,
                "EX": None,
                "SQL Validity": None,
                "Latency (ms)": None,
                "Training Time (s)": None,
                "Peak VRAM (MB)": None,
                "Status": "MISSING",
            })
            continue

        r = method_runs[0]
        rows.append({
            "Method": r.method,
            "Data Fraction": r.data_fraction,
            "RAG K": r.rag_k,
            "LoRA Rank": r.lora_rank,
            "Noise": r.noise_level,
            "EM": r.exact_match,
            "EX": r.execution_accuracy,
            "SQL Validity": r.sql_validity,
            "Latency (ms)": r.mean_latency_ms,
            "Training Time (s)": r.training_time_s,
            "Peak VRAM (MB)": r.peak_vram_mb,
            "Status": r.status,
        })

    return pd.DataFrame(rows, columns=columns)


def analyze_data_scaling(runs: List[ExperimentRun]) -> pd.DataFrame:
    """Analyzes data scaling across fractions 1%, 5%, 10%, 25%, 50%, 100% for LoRA and QLoRA.
    
    Missing fractions are preserved explicitly without interpolation.
    """
    target_fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    target_methods = ["LORA", "QLORA"]
    rows = []

    for method in target_methods:
        method_runs = [r for r in runs if r.method.upper() == method and r.noise_level == 0.0]
        for frac in target_fractions:
            matched = [r for r in method_runs if abs(r.data_fraction - frac) < 1e-4]
            if not matched:
                rows.append({
                    "Method": method,
                    "Data Fraction": frac,
                    "EX": None,
                    "EM": None,
                    "SQL Validity": None,
                    "Status": "MISSING",
                    "Is Missing": True,
                })
            else:
                r = matched[0]
                rows.append({
                    "Method": r.method,
                    "Data Fraction": frac,
                    "EX": r.execution_accuracy,
                    "EM": r.exact_match,
                    "SQL Validity": r.sql_validity,
                    "Status": r.status,
                    "Is Missing": r.status != "COMPLETED",
                })

    return pd.DataFrame(rows)


def analyze_rag_k(runs: List[ExperimentRun]) -> pd.DataFrame:
    """Analyzes RAG retrieval depth across K in {1, 3, 5, 10}.
    
    Reports observed values without declaring an optimal K or ranking methods.
    """
    target_k = [1, 3, 5, 10]
    rag_runs = [r for r in runs if r.method.upper() == "RAG" and r.noise_level == 0.0]
    rows = []

    for k in target_k:
        matched = [r for r in rag_runs if r.rag_k == k]
        if not matched:
            rows.append({
                "RAG K": k,
                "EX": None,
                "EM": None,
                "Latency (ms)": None,
                "Status": "MISSING",
                "Is Missing": True,
            })
        else:
            r = matched[0]
            rows.append({
                "RAG K": k,
                "EX": r.execution_accuracy,
                "EM": r.exact_match,
                "Latency (ms)": r.mean_latency_ms,
                "Status": r.status,
                "Is Missing": r.status != "COMPLETED",
            })

    return pd.DataFrame(rows)


def analyze_lora_rank(runs: List[ExperimentRun]) -> pd.DataFrame:
    """Analyzes parameter rank ablation across r in {4, 8, 16} for LoRA and QLoRA.
    
    Reports observations rather than declaring a universally best rank.
    """
    target_ranks = [4, 8, 16]
    target_methods = ["LORA", "QLORA"]
    rows = []

    for method in target_methods:
        method_runs = [r for r in runs if r.method.upper() == method and r.noise_level == 0.0]
        for rank in target_ranks:
            matched = [r for r in method_runs if r.lora_rank == rank]
            if not matched:
                rows.append({
                    "Method": method,
                    "LoRA Rank": rank,
                    "EX": None,
                    "EM": None,
                    "Training Time (s)": None,
                    "Peak VRAM (MB)": None,
                    "Status": "MISSING",
                    "Is Missing": True,
                })
            else:
                r = matched[0]
                rows.append({
                    "Method": r.method,
                    "LoRA Rank": rank,
                    "EX": r.execution_accuracy,
                    "EM": r.exact_match,
                    "Training Time (s)": r.training_time_s,
                    "Peak VRAM (MB)": r.peak_vram_mb,
                    "Status": r.status,
                    "Is Missing": r.status != "COMPLETED",
                })

    return pd.DataFrame(rows)


def validate_noise_metadata(run: ExperimentRun) -> bool:
    """Validates that a noise run has complete and unambiguous corruption semantics."""
    if run.noise_level == 0.0:
        return True
    meta = run.noise_metadata
    if not meta or not isinstance(meta, dict):
        return False
    required_keys = {"noise_type", "noise_rate", "random_seed", "affected_example_ids", "corruption_rule"}
    if not required_keys.issubset(meta.keys()):
        return False
    if meta.get("noise_type") in (None, "", "none"):
        return False
    return True


def analyze_noise_sensitivity(runs: List[ExperimentRun]) -> Tuple[pd.DataFrame, Optional[str]]:
    """Analyzes noise sensitivity across rates 0%, 5%, 10%, 20% for RAG, LoRA, QLoRA.
    
    If corruption semantics are insufficiently specified, returns an empty table and error message.
    """
    target_rates = [0.0, 0.05, 0.10, 0.20]
    target_methods = ["RAG", "LORA", "QLORA"]

    # Check validity of metadata on noise runs
    noise_runs = [r for r in runs if r.noise_level > 0.0 and r.method.upper() in target_methods]
    for nr in noise_runs:
        if not validate_noise_metadata(nr):
            msg = "Noise analysis unavailable because corruption semantics are insufficiently specified."
            logger.warning(msg)
            return pd.DataFrame(), msg

    rows = []
    for method in target_methods:
        method_runs = [r for r in runs if r.method.upper() == method]
        for rate in target_rates:
            matched = [r for r in method_runs if abs(r.noise_level - rate) < 1e-4]
            if not matched:
                rows.append({
                    "Method": method,
                    "Noise Rate": rate,
                    "Noise Type": "none" if rate == 0.0 else "unspecified",
                    "EX": None,
                    "EM": None,
                    "SQL Validity": None,
                    "Status": "MISSING",
                    "Is Missing": True,
                })
            else:
                r = matched[0]
                rows.append({
                    "Method": r.method,
                    "Noise Rate": rate,
                    "Noise Type": r.noise_type,
                    "EX": r.execution_accuracy,
                    "EM": r.exact_match,
                    "SQL Validity": r.sql_validity,
                    "Status": r.status,
                    "Is Missing": r.status != "COMPLETED",
                })

    return pd.DataFrame(rows), None


def analyze_crossover(runs: List[ExperimentRun]) -> Dict[str, Any]:
    """Identifies observed configurations where fine-tuning execution accuracy exceeds RAG.
    
    Does NOT extrapolate or claim crossover on insufficient or unmatched runs.
    """
    completed_runs = [r for r in runs if r.status == "COMPLETED" and r.noise_level == 0.0]
    rag_runs = [r for r in completed_runs if r.method.upper() == "RAG"]

    if not rag_runs:
        return {
            "status": "UNAVAILABLE",
            "message": "Crossover analysis unavailable: insufficient matched experiments.",
            "crossover_points": [],
        }

    # Reference RAG (canonical K=3, 100% data)
    ref_rag = None
    for r in rag_runs:
        if r.rag_k == 3:
            ref_rag = r
            break
    if not ref_rag:
        ref_rag = rag_runs[0]

    if ref_rag.execution_accuracy is None:
        return {
            "status": "UNAVAILABLE",
            "message": "Crossover analysis unavailable: insufficient matched experiments.",
            "crossover_points": [],
        }

    rag_ex = ref_rag.execution_accuracy
    crossovers = []

    ft_runs = [r for r in completed_runs if r.method.upper() in ("LORA", "QLORA")]
    for ft in ft_runs:
        if ft.execution_accuracy is not None and ft.execution_accuracy > rag_ex:
            crossovers.append({
                "fine_tuning_method": ft.method,
                "data_fraction": ft.data_fraction,
                "lora_rank": ft.lora_rank,
                "ft_execution_accuracy": ft.execution_accuracy,
                "rag_execution_accuracy": rag_ex,
                "observed_ex_difference": round(ft.execution_accuracy - rag_ex, 4),
                "rag_configuration": f"RAG (K={ref_rag.rag_k})",
            })

    if not crossovers:
        return {
            "status": "COMPLETED_NO_CROSSOVER_OBSERVED",
            "message": f"Fine-tuning execution accuracy did not exceed reference RAG EX ({rag_ex}) in any tested completed configuration.",
            "crossover_points": [],
        }

    return {
        "status": "COMPLETED",
        "reference_rag_ex": rag_ex,
        "reference_rag_k": ref_rag.rag_k,
        "crossover_points": crossovers,
    }
