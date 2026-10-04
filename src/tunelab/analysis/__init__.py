from tunelab.analysis.comparator import (
    analyze_crossover,
    analyze_data_scaling,
    analyze_lora_rank,
    analyze_noise_sensitivity,
    analyze_rag_k,
    compare_methods,
)
from tunelab.analysis.error_analysis import (
    analyze_errors_for_run,
    classify_error,
    compute_error_distribution,
)
from tunelab.analysis.loader import (
    filter_by_experiment,
    filter_by_method,
    filter_by_status,
    load_all_runs,
    load_per_example_results,
    load_run_from_directory,
    load_runs_from_master,
)
from tunelab.analysis.metrics import (
    aggregate_runs_by_config,
    compute_seed_statistics,
    format_cost_table,
)
from tunelab.analysis.models import (
    ErrorAnalysisRecord,
    ExperimentRun,
    ExperimentSummary,
    PerExampleResult,
)
from tunelab.analysis.plots import (
    plot_accuracy_vs_data_fraction,
    plot_accuracy_vs_latency,
    plot_accuracy_vs_noise,
    plot_error_distribution_by_method,
    plot_exact_match_vs_data_fraction,
    plot_lora_accuracy_vs_rank,
    plot_rag_accuracy_vs_k,
)
from tunelab.analysis.tables import export_analysis_tables
from tunelab.analysis.report import generate_research_report

__all__ = [
    "analyze_crossover",
    "analyze_data_scaling",
    "analyze_errors_for_run",
    "analyze_lora_rank",
    "analyze_noise_sensitivity",
    "analyze_rag_k",
    "classify_error",
    "compare_methods",
    "compute_error_distribution",
    "compute_seed_statistics",
    "export_analysis_tables",
    "generate_research_report",
    "filter_by_experiment",
    "filter_by_method",
    "filter_by_status",
    "format_cost_table",
    "load_all_runs",
    "load_per_example_results",
    "load_run_from_directory",
    "load_runs_from_master",
    "plot_accuracy_vs_data_fraction",
    "plot_accuracy_vs_latency",
    "plot_accuracy_vs_noise",
    "plot_error_distribution_by_method",
    "plot_exact_match_vs_data_fraction",
    "plot_lora_accuracy_vs_rank",
    "plot_rag_accuracy_vs_k",
    "ErrorAnalysisRecord",
    "ExperimentRun",
    "ExperimentSummary",
    "PerExampleResult",
    "aggregate_runs_by_config",
]
