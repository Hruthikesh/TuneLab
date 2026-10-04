import argparse
import logging
from pathlib import Path
import sys

from tunelab.analysis.comparator import (
    analyze_crossover,
    analyze_data_scaling,
    analyze_lora_rank,
    analyze_noise_sensitivity,
    analyze_rag_k,
    compare_methods,
)
from tunelab.analysis.error_analysis import compute_error_distribution
from tunelab.analysis.loader import filter_by_experiment, filter_by_method, load_all_runs
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tunelab.analysis")


def main():
    parser = argparse.ArgumentParser(description="TuneLab Result & Error Analysis CLI")
    parser.add_argument("--results-dir", type=str, default="experiments/runs", help="Directory containing run directories and master results")
    parser.add_argument("--output-dir", type=str, default="reports/analysis", help="Directory to export analysis tables and plots")
    parser.add_argument("--experiment-id", type=str, default=None, help="Filter runs by experiment_id")
    parser.add_argument("--method", type=str, default=None, help="Filter runs by method")
    parser.add_argument("--include-smoke", action="store_true", help="Explicitly include smoke/pipeline-test runs (diagnostic only)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"

    logger.info("=" * 60)
    logger.info("TUNELAB EXPERIMENT ANALYSIS")
    logger.info(f"Results Directory: {results_dir}")
    logger.info(f"Output Directory:  {output_dir}")
    logger.info(f"Include Smoke:     {args.include_smoke}")
    logger.info("=" * 60)

    # 1. Load runs
    # First load all runs including tests to inspect total repository state
    all_runs = load_all_runs(results_dir=results_dir, include_smoke=True)

    if not all_runs:
        logger.warning(f"No experiment runs found in {results_dir}.")
        print("\n[NOTICE] No experiment runs found in specified directory.")
        return

    # Filter by user options if provided
    if args.experiment_id:
        all_runs = filter_by_experiment(all_runs, args.experiment_id)
    if args.method:
        all_runs = filter_by_method(all_runs, args.method)

    research_runs = [r for r in all_runs if r.is_research_result]
    pipeline_runs = [r for r in all_runs if r.is_pipeline_test]

    logger.info(f"Found {len(all_runs)} total runs:")
    logger.info(f"  - Research Runs (Completed, Real Data): {len(research_runs)}")
    logger.info(f"  - Pipeline Tests / Smoke / Dry-Run / Pending: {len(pipeline_runs)}")

    # 2. Status Guard: Core Principle Check
    if not research_runs and not args.include_smoke:
        print("\n" + "!" * 60)
        print("TUNELAB RESEARCH STATUS NOTICE")
        print("!" * 60)
        print("Substantive benchmark conclusions cannot yet be computed.")
        print("Current repository contains only pipeline tests, smoke runs, dry-runs, or pending records.")
        print(f"Statuses observed: {sorted(list(set(r.status for r in all_runs)))}")
        print("To inspect pipeline diagnostics, re-run with --include-smoke.")
        print("!" * 60 + "\n")
        # Still generate empty/status-preserving tables in output_dir
        export_analysis_tables(all_runs, output_dir=output_dir, include_smoke=False)
        return

    target_runs = all_runs if args.include_smoke else research_runs
    mode_label = "DIAGNOSTIC PIPELINE TEST ANALYSIS" if args.include_smoke else "FORMAL RESEARCH BENCHMARK ANALYSIS"

    print(f"\nExecuting {mode_label} over {len(target_runs)} runs...\n")

    # 3. Export tables
    tables = export_analysis_tables(target_runs, output_dir=output_dir, include_smoke=args.include_smoke)
    for name, path in tables.items():
        logger.info(f"  Table generated: {name} -> {path}")

    # 4. Generate plots
    figures_dir.mkdir(parents=True, exist_ok=True)
    scaling_df = analyze_data_scaling(target_runs)
    rag_k_df = analyze_rag_k(target_runs)
    rank_df = analyze_lora_rank(target_runs)
    noise_df, _ = analyze_noise_sensitivity(target_runs)
    error_df = compute_error_distribution(target_runs, group_by="method", include_smoke=args.include_smoke)

    plot_accuracy_vs_data_fraction(scaling_df, figures_dir / "acc_vs_data_fraction.png")
    plot_exact_match_vs_data_fraction(scaling_df, figures_dir / "em_vs_data_fraction.png")
    plot_rag_accuracy_vs_k(rag_k_df, figures_dir / "rag_accuracy_vs_k.png")
    plot_lora_accuracy_vs_rank(rank_df, figures_dir / "lora_accuracy_vs_rank.png")
    if not noise_df.empty:
        plot_accuracy_vs_noise(noise_df, figures_dir / "accuracy_vs_noise.png")
    plot_error_distribution_by_method(error_df, figures_dir / "error_distribution_by_method.png")

    # 5. Display Summary Tables
    print("\n--- METHOD COMPARISON TABLE ---")
    method_df = compare_methods(target_runs)
    print(method_df.to_string(index=False))

    print("\n--- DATA SCALING ANALYSIS ---")
    print(scaling_df.to_string(index=False))

    print("\n--- RAG K ANALYSIS ---")
    print(rag_k_df.to_string(index=False))

    print("\n--- LORA RANK ANALYSIS ---")
    print(rank_df.to_string(index=False))

    print("\n--- CROSSOVER ANALYSIS ---")
    crossover = analyze_crossover(target_runs)
    print(f"Status: {crossover.get('status')}")
    if crossover.get("message"):
        print(f"Note: {crossover.get('message')}")
    for cp in crossover.get("crossover_points", []):
        print(f"  * {cp['fine_tuning_method']} (frac={cp['data_fraction']}) EX={cp['ft_execution_accuracy']} > {cp['rag_configuration']} EX={cp['rag_execution_accuracy']} (Diff: +{cp['observed_ex_difference']})")

    print("\nAnalysis completed successfully.\n")


if __name__ == "__main__":
    main()
