from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from tunelab.analysis.comparator import (
    analyze_crossover,
    analyze_data_scaling,
    analyze_lora_rank,
    analyze_noise_sensitivity,
    analyze_rag_k,
    compare_methods,
)
from tunelab.analysis.error_analysis import compute_error_distribution
from tunelab.analysis.metrics import aggregate_runs_by_config, format_cost_table
from tunelab.analysis.models import ExperimentRun

logger = logging.getLogger(__name__)


def export_analysis_tables(
    runs: List[ExperimentRun],
    output_dir: Path | str = "reports/analysis",
    include_smoke: bool = False,
    git_commit: Optional[str] = None,
) -> Dict[str, Path]:
    """Generates and exports all machine-readable analysis tables to output_dir.
    
    Preserves raw experiment results without modification.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    generated_files: Dict[str, Path] = {}

    target_runs = runs if include_smoke else [r for r in runs if r.is_research_result]

    # 1. Method Comparison
    method_df = compare_methods(target_runs)
    m_file = out_p / "method_comparison.csv"
    method_df.to_csv(m_file, index=False)
    generated_files["method_comparison"] = m_file

    # 2. Data Scaling
    scaling_df = analyze_data_scaling(target_runs)
    s_file = out_p / "data_scaling.csv"
    scaling_df.to_csv(s_file, index=False)
    generated_files["data_scaling"] = s_file

    # 3. RAG K
    rag_k_df = analyze_rag_k(target_runs)
    k_file = out_p / "rag_k.csv"
    rag_k_df.to_csv(k_file, index=False)
    generated_files["rag_k"] = k_file

    # 4. LoRA Rank
    rank_df = analyze_lora_rank(target_runs)
    r_file = out_p / "lora_rank.csv"
    rank_df.to_csv(r_file, index=False)
    generated_files["lora_rank"] = r_file

    # 5. Noise Analysis
    noise_df, noise_msg = analyze_noise_sensitivity(target_runs)
    n_file = out_p / "noise_analysis.csv"
    if not noise_df.empty:
        noise_df.to_csv(n_file, index=False)
    else:
        # Export empty placeholder table with notice
        pd.DataFrame([{"status": "UNAVAILABLE", "message": noise_msg or "No noise data"}]).to_csv(n_file, index=False)
    generated_files["noise_analysis"] = n_file

    # 6. Error Analysis
    err_df = compute_error_distribution(target_runs, group_by="method", include_smoke=include_smoke)
    e_file = out_p / "error_analysis.csv"
    err_df.to_csv(e_file, index=False)
    generated_files["error_analysis"] = e_file

    # 7. Crossover Analysis
    crossover_data = analyze_crossover(target_runs)
    c_file = out_p / "crossover_analysis.json"
    with open(c_file, "w", encoding="utf-8") as f:
        json.dump(crossover_data, f, indent=2)
    generated_files["crossover_analysis"] = c_file

    # 8. Latency and Cost
    cost_df = format_cost_table(target_runs)
    cost_file = out_p / "latency_and_costs.csv"
    cost_df.to_csv(cost_file, index=False)
    generated_files["latency_and_costs"] = cost_file

    # 9. Analysis Metadata & Provenance Tracking
    meta_file = out_p / "analysis_metadata.json"
    dataset_versions = list(set([r.dataset_version for r in target_runs if r.dataset_version]))
    metadata = {
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "include_smoke": include_smoke,
        "total_runs_analyzed": len(target_runs),
        "source_run_ids": [r.run_id for r in target_runs],
        "dataset_versions": dataset_versions,
        "git_commit": git_commit,
        "classification_breakdown": {
            "research_results": len([r for r in target_runs if r.is_research_result]),
            "pipeline_tests": len([r for r in target_runs if r.is_pipeline_test]),
            "completed": len([r for r in target_runs if r.status == 'COMPLETED']),
            "pending_or_failed": len([r for r in target_runs if r.status != 'COMPLETED']),
        },
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    generated_files["analysis_metadata"] = meta_file

    logger.info(f"Exported {len(generated_files)} analysis tables and metadata to {out_p}")
    return generated_files
