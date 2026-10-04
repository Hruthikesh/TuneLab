import math
import statistics
from typing import Any, Dict, List, Optional
import pandas as pd

from tunelab.analysis.models import ExperimentRun, ExperimentSummary


def compute_seed_statistics(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    """Computes mean and standard deviation for a list of values.
    
    If n == 1: returns (val, None) - no fake standard deviation or confidence interval.
    If n >= 2: returns (mean, sample_std).
    """
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return None, None
    if len(valid) == 1:
        return round(valid[0], 4), None
    mean_val = statistics.mean(valid)
    std_val = statistics.stdev(valid)
    return round(mean_val, 4), round(std_val, 4)


def aggregate_runs_by_config(runs: List[ExperimentRun]) -> List[ExperimentSummary]:
    """Aggregates multiple experiment runs sharing identical configuration parameters across seeds."""
    groups: Dict[tuple, List[ExperimentRun]] = {}

    for run in runs:
        key = (
            run.method,
            run.model,
            run.data_fraction,
            run.rag_k,
            run.lora_rank,
            run.noise_level,
        )
        if key not in groups:
            groups[key] = []
        groups[key].append(run)

    summaries: List[ExperimentSummary] = []

    for key, run_list in groups.items():
        method, model, data_fraction, rag_k, lora_rank, noise_level = key
        completed_runs = [r for r in run_list if r.status == "COMPLETED"]
        seeds = [r.seed for r in run_list]

        status = "COMPLETED" if completed_runs else (run_list[0].status if run_list else "UNKNOWN")
        n_completed = len(completed_runs)

        if n_completed == 0:
            summaries.append(
                ExperimentSummary(
                    config_key=str(key),
                    method=method,
                    model=model,
                    data_fraction=data_fraction,
                    rag_k=rag_k,
                    lora_rank=lora_rank,
                    noise_level=noise_level,
                    n_runs=len(run_list),
                    seeds=seeds,
                    status=status,
                )
            )
            continue

        ex_vals = [r.execution_accuracy for r in completed_runs if r.execution_accuracy is not None]
        em_vals = [r.exact_match for r in completed_runs if r.exact_match is not None]
        val_vals = [r.sql_validity for r in completed_runs if r.sql_validity is not None]
        lat_vals = [r.mean_latency_ms for r in completed_runs if r.mean_latency_ms is not None]
        med_lat_vals = [r.median_latency_ms for r in completed_runs if r.median_latency_ms is not None]
        train_time_vals = [r.training_time_s for r in completed_runs if r.training_time_s is not None]
        vram_vals = [r.peak_vram_mb for r in completed_runs if r.peak_vram_mb is not None]

        ex_m, ex_s = compute_seed_statistics(ex_vals)
        em_m, em_s = compute_seed_statistics(em_vals)
        val_m, val_s = compute_seed_statistics(val_vals)
        lat_m, _ = compute_seed_statistics(lat_vals)
        med_lat_m, _ = compute_seed_statistics(med_lat_vals)
        train_m, _ = compute_seed_statistics(train_time_vals)
        vram_m, _ = compute_seed_statistics(vram_vals)

        # Parameter counts from first completed run
        trainable_p = completed_runs[0].trainable_parameters
        total_p = completed_runs[0].total_parameters

        summaries.append(
            ExperimentSummary(
                config_key=str(key),
                method=method,
                model=model,
                data_fraction=data_fraction,
                rag_k=rag_k,
                lora_rank=lora_rank,
                noise_level=noise_level,
                n_runs=n_completed,
                seeds=seeds,
                status="COMPLETED",
                execution_accuracy_mean=ex_m,
                execution_accuracy_std=ex_s,
                exact_match_mean=em_m,
                exact_match_std=em_s,
                sql_validity_mean=val_m,
                sql_validity_std=val_s,
                mean_latency_ms=lat_m,
                median_latency_ms=med_lat_m,
                training_time_mean_s=train_m,
                peak_vram_mean_mb=vram_m,
                trainable_parameters=trainable_p,
                total_parameters=total_p,
            )
        )

    return summaries


def format_cost_table(runs: List[ExperimentRun]) -> pd.DataFrame:
    """Generates a latency/cost table cleanly separating one-time training costs from per-query inference costs."""
    rows = []
    for r in runs:
        rows.append({
            "run_id": r.run_id,
            "method": r.method,
            "data_fraction": r.data_fraction,
            "lora_rank": r.lora_rank,
            "status": r.status,
            # Per-query inference metrics
            "mean_latency_ms": r.mean_latency_ms,
            "median_latency_ms": r.median_latency_ms,
            "inference_time_s": r.inference_time_s,
            # One-time training costs
            "training_time_s": r.training_time_s,
            "peak_vram_mb": r.peak_vram_mb,
            "trainable_parameters": r.trainable_parameters,
            "total_parameters": r.total_parameters,
        })
    return pd.DataFrame(rows)
