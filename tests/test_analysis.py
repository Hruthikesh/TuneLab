import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd

from tunelab.analysis.comparator import (
    analyze_crossover,
    analyze_data_scaling,
    analyze_lora_rank,
    analyze_noise_sensitivity,
    analyze_rag_k,
    compare_methods,
    validate_noise_metadata,
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


class TestTuneLabAnalysis(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures" / "analysis_fixtures"

    def test_result_loading_and_status_parsing(self):
        runs = load_all_runs(self.fixtures_dir, include_smoke=True)
        self.assertGreater(len(runs), 0)

        run_map = {r.run_id: r for r in runs}

        # Check smoke run
        self.assertIn("SMOKE_01_ZERO_SHOT", run_map)
        smoke = run_map["SMOKE_01_ZERO_SHOT"]
        self.assertEqual(smoke.status, "COMPLETED")
        self.assertTrue(smoke.is_pipeline_test)
        self.assertFalse(smoke.is_research_result)

        # Check dry run
        self.assertIn("DRY_01_LORA", run_map)
        dry = run_map["DRY_01_LORA"]
        self.assertEqual(dry.status, "DRY_RUN_ONLY")
        self.assertIsNone(dry.execution_accuracy)
        self.assertFalse(dry.is_research_result)

        # Check pending run
        self.assertIn("PENDING_01_QLORA", run_map)
        pending = run_map["PENDING_01_QLORA"]
        self.assertEqual(pending.status, "PENDING_GPU")
        self.assertIsNone(pending.execution_accuracy)
        self.assertFalse(pending.is_research_result)

        # Check failed run
        self.assertIn("FAIL_01_RAG", run_map)
        failed = run_map["FAIL_01_RAG"]
        self.assertEqual(failed.status, "FAILED")
        self.assertIsNone(failed.execution_accuracy)
        self.assertFalse(failed.is_research_result)

    def test_completed_run_filtering_excludes_non_research(self):
        # Default load_all_runs excludes smoke, dry, pending, failed, and fixtures
        research_runs = load_all_runs(self.fixtures_dir, include_smoke=False)
        for r in research_runs:
            self.assertTrue(r.is_research_result)
            self.assertFalse(r.is_pipeline_test)
            self.assertEqual(r.status, "COMPLETED")
            self.assertNotIn("SMOKE", r.run_id)
            self.assertNotIn("DRY", r.run_id)
            self.assertNotIn("PENDING", r.run_id)
            self.assertNotIn("FAIL", r.run_id)

    def test_per_example_loading_and_median_latency(self):
        err_run_dir = self.fixtures_dir / "RES_ERROR_DIAGNOSTICS"
        per_ex = load_per_example_results(err_run_dir)
        self.assertEqual(len(per_ex), 15)
        self.assertEqual(per_ex[0].example_id, "ex1")
        self.assertTrue(per_ex[0].execution_match)

    def test_error_categorization_and_fine_grained_diagnostics(self):
        err_run_dir = self.fixtures_dir / "RES_ERROR_DIAGNOSTICS"
        per_ex = load_per_example_results(err_run_dir)
        records = [classify_error(r) for r in per_ex]

        fine_categories = [r.fine_error for r in records]
        # Assert each fine error category was identified
        self.assertIn("none", fine_categories)
        self.assertIn("syntax_error", fine_categories)
        self.assertIn("missing_table", fine_categories)
        self.assertIn("missing_column", fine_categories)
        self.assertIn("timeout", fine_categories)
        self.assertIn("invalid_statement", fine_categories)
        self.assertIn("wrong_table", fine_categories)
        self.assertIn("wrong_join", fine_categories)
        self.assertIn("wrong_aggregation", fine_categories)
        self.assertIn("wrong_group_by", fine_categories)
        self.assertIn("wrong_ordering", fine_categories)
        self.assertIn("nested_query", fine_categories)
        self.assertIn("wrong_filter", fine_categories)
        self.assertIn("UNCLASSIFIED_RESULT_MISMATCH", fine_categories)

    def test_per_method_error_distribution(self):
        diag_run = load_run_from_directory(self.fixtures_dir / "RES_ERROR_DIAGNOSTICS")
        dist_df = compute_error_distribution([diag_run], group_by="method", include_smoke=True)
        self.assertFalse(dist_df.empty)
        self.assertIn("error_type", dist_df.columns)
        self.assertIn("percentage", dist_df.columns)

    def test_statistical_handling_single_vs_multi_seed(self):
        # Single value
        mean_1, std_1 = compute_seed_statistics([0.65])
        self.assertEqual(mean_1, 0.65)
        self.assertIsNone(std_1)

        # Multi seed values
        mean_m, std_m = compute_seed_statistics([0.70, 0.72, 0.74])
        self.assertAlmostEqual(mean_m, 0.72, places=3)
        self.assertIsNotNone(std_m)
        self.assertGreater(std_m, 0.0)

    def test_multi_seed_aggregation(self):
        runs = load_all_runs(self.fixtures_dir, include_smoke=True)
        summaries = aggregate_runs_by_config(runs)
        lora_100_summary = [s for s in summaries if s.method == "LORA" and s.data_fraction == 1.0 and s.lora_rank == 8 and s.status == "COMPLETED"]
        self.assertTrue(len(lora_100_summary) > 0)
        s = lora_100_summary[0]
        self.assertEqual(s.n_runs, 3)
        self.assertEqual(set(s.seeds), {42, 43, 44})
        self.assertAlmostEqual(s.execution_accuracy_mean, 0.72, places=2)
        self.assertIsNotNone(s.execution_accuracy_std)

    def test_data_scaling_preserves_missing_data(self):
        runs = [
            ExperimentRun(
                run_id="lora_10pct",
                experiment_id="data_scaling",
                method="LORA",
                model="qwen",
                dataset="spider",
                dataset_version="1.0",
                data_fraction=0.10,
                lora_rank=8,
                status="COMPLETED",
                execution_accuracy=0.55,
                exact_match=0.45,
                sql_validity=0.90,
            ),
            ExperimentRun(
                run_id="lora_100pct",
                experiment_id="data_scaling",
                method="LORA",
                model="qwen",
                dataset="spider",
                dataset_version="1.0",
                data_fraction=1.00,
                lora_rank=8,
                status="COMPLETED",
                execution_accuracy=0.72,
                exact_match=0.64,
                sql_validity=0.96,
            ),
        ]
        df = analyze_data_scaling(runs)
        self.assertEqual(len(df), 12)  # 6 fractions x 2 methods (LoRA, QLoRA)
        # 25% LoRA is not in runs: must be marked MISSING
        row_25_lora = df[(df["Method"] == "LORA") & (df["Data Fraction"] == 0.25)].iloc[0]
        self.assertTrue(row_25_lora["Is Missing"])
        self.assertEqual(row_25_lora["Status"], "MISSING")
        self.assertTrue(pd.isna(row_25_lora["EX"]))

    def test_rag_k_analysis(self):
        runs = [
            ExperimentRun(
                run_id="rag_k1",
                experiment_id="rag_k",
                method="RAG",
                model="qwen",
                dataset="spider",
                dataset_version="1.0",
                data_fraction=1.0,
                rag_k=1,
                status="COMPLETED",
                execution_accuracy=0.62,
                exact_match=0.52,
                mean_latency_ms=90.0,
            ),
            ExperimentRun(
                run_id="rag_k3",
                experiment_id="rag_k",
                method="RAG",
                model="qwen",
                dataset="spider",
                dataset_version="1.0",
                data_fraction=1.0,
                rag_k=3,
                status="COMPLETED",
                execution_accuracy=0.68,
                exact_match=0.60,
                mean_latency_ms=120.0,
            ),
        ]
        df = analyze_rag_k(runs)
        self.assertEqual(len(df), 4)  # K in {1, 3, 5, 10}
        k5_row = df[df["RAG K"] == 5].iloc[0]
        self.assertTrue(k5_row["Is Missing"])
        self.assertEqual(k5_row["Status"], "MISSING")

    def test_lora_rank_analysis(self):
        runs = [
            ExperimentRun(
                run_id="lora_r4",
                experiment_id="lora_rank",
                method="LORA",
                model="qwen",
                dataset="spider",
                dataset_version="1.0",
                data_fraction=1.0,
                lora_rank=4,
                status="COMPLETED",
                execution_accuracy=0.69,
                exact_match=0.59,
            )
        ]
        df = analyze_lora_rank(runs)
        self.assertEqual(len(df), 6)  # 3 ranks x 2 methods
        r16_row = df[(df["Method"] == "LORA") & (df["LoRA Rank"] == 16)].iloc[0]
        self.assertTrue(r16_row["Is Missing"])

    def test_noise_metadata_validation(self):
        # 1. Incomplete noise run missing required fields
        invalid_run = ExperimentRun(
            run_id="noise_invalid",
            experiment_id="noise",
            method="LORA",
            model="qwen",
            dataset="spider",
            dataset_version="1.0",
            data_fraction=1.0,
            noise_level=0.10,
            noise_metadata={"noise_rate": 0.10},  # Missing noise_type, random_seed, etc.
        )
        self.assertFalse(validate_noise_metadata(invalid_run))
        df, msg = analyze_noise_sensitivity([invalid_run])
        self.assertTrue(df.empty)
        self.assertIn("insufficiently specified", msg)

        # 2. Valid noise metadata
        valid_run = ExperimentRun(
            run_id="noise_valid",
            experiment_id="noise",
            method="LORA",
            model="qwen",
            dataset="spider",
            dataset_version="1.0",
            data_fraction=1.0,
            noise_level=0.10,
            noise_type="sql_syntax_corruption",
            noise_metadata={
                "noise_type": "sql_syntax_corruption",
                "noise_rate": 0.10,
                "random_seed": 42,
                "affected_example_ids": ["1", "2"],
                "corruption_rule": "deterministic_sql_syntax_corruption",
            },
            status="COMPLETED",
            execution_accuracy=0.65,
            exact_match=0.55,
            sql_validity=0.92,
        )
        self.assertTrue(validate_noise_metadata(valid_run))
        df, msg = analyze_noise_sensitivity([valid_run])
        self.assertFalse(df.empty)
        self.assertIsNone(msg)

    def test_crossover_analysis_scenarios(self):
        # Scenario A: Insufficient runs (no RAG)
        res_a = analyze_crossover([])
        self.assertEqual(res_a["status"], "UNAVAILABLE")

        # Scenario B: Matched RAG and LoRA where LoRA outperforms RAG
        rag_run = ExperimentRun(
            run_id="rag_base",
            experiment_id="matched",
            method="RAG",
            model="qwen",
            dataset="spider",
            dataset_version="1.0",
            data_fraction=1.0,
            rag_k=3,
            status="COMPLETED",
            execution_accuracy=0.65,
        )
        lora_run = ExperimentRun(
            run_id="lora_win",
            experiment_id="matched",
            method="LORA",
            model="qwen",
            dataset="spider",
            dataset_version="1.0",
            data_fraction=1.0,
            lora_rank=8,
            status="COMPLETED",
            execution_accuracy=0.72,
        )
        res_b = analyze_crossover([rag_run, lora_run])
        self.assertEqual(res_b["status"], "COMPLETED")
        self.assertEqual(len(res_b["crossover_points"]), 1)
        cp = res_b["crossover_points"][0]
        self.assertEqual(cp["fine_tuning_method"], "LORA")
        self.assertAlmostEqual(cp["observed_ex_difference"], 0.07, places=3)

    def test_cost_separation_inference_vs_training(self):
        run = ExperimentRun(
            run_id="lora_cost",
            experiment_id="cost",
            method="LORA",
            model="qwen",
            dataset="spider",
            dataset_version="1.0",
            data_fraction=1.0,
            lora_rank=8,
            status="COMPLETED",
            mean_latency_ms=50.0,
            median_latency_ms=48.0,
            inference_time_s=12.5,
            training_time_s=3600.0,
            peak_vram_mb=6500.0,
            trainable_parameters=1376256,
            total_parameters=1543714816,
        )
        cost_df = format_cost_table([run])
        self.assertEqual(len(cost_df), 1)
        row = cost_df.iloc[0]
        self.assertEqual(row["mean_latency_ms"], 50.0)
        self.assertEqual(row["training_time_s"], 3600.0)
        self.assertEqual(row["peak_vram_mb"], 6500.0)

    def test_plot_generation_and_graceful_missing_behavior(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_p = Path(tmp_dir) / "test_plot.png"

            # 1. Empty dataframe -> plot skipped gracefully without crash
            empty_df = pd.DataFrame()
            res = plot_accuracy_vs_data_fraction(empty_df, fig_p)
            self.assertFalse(res)
            self.assertFalse(fig_p.exists())

            # 2. Valid data -> plot generated cleanly
            valid_df = pd.DataFrame([
                {"Method": "LORA", "Data Fraction": 0.1, "EX": 0.5, "Status": "COMPLETED"},
                {"Method": "LORA", "Data Fraction": 0.5, "EX": 0.7, "Status": "COMPLETED"},
            ])
            res_valid = plot_accuracy_vs_data_fraction(valid_df, fig_p)
            self.assertTrue(res_valid)
            self.assertTrue(fig_p.exists())
            self.assertGreater(fig_p.stat().st_size, 0)

    def test_export_analysis_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs = load_all_runs(self.fixtures_dir, include_smoke=True)
            files = export_analysis_tables(runs, output_dir=tmp_dir, include_smoke=True)

            self.assertIn("method_comparison", files)
            self.assertIn("data_scaling", files)
            self.assertIn("rag_k", files)
            self.assertIn("lora_rank", files)
            self.assertIn("noise_analysis", files)
            self.assertIn("error_analysis", files)
            self.assertIn("crossover_analysis", files)
            self.assertIn("analysis_metadata", files)

            for p in files.values():
                self.assertTrue(p.exists())

            # Verify analysis metadata records provenance
            with open(files["analysis_metadata"], "r") as f:
                meta = json.load(f)
            self.assertIn("generation_timestamp", meta)
            self.assertIn("source_run_ids", meta)
            self.assertIn("classification_breakdown", meta)


if __name__ == "__main__":
    unittest.main()
