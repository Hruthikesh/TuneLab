from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tunelab.analysis.report import generate_research_report


class TestResearchReportGeneration(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures"
        self.analysis_fixtures = self.fixtures_dir / "analysis_fixtures"

    def test_report_generation_contains_all_13_sections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "report.md"
            report_text = generate_research_report(
                results_dir="experiments/runs",
                output_file=out_file,
                include_smoke=False,
            )
            self.assertTrue(out_file.is_file())
            self.assertGreater(len(report_text), 5000)

            # Check that all 13 sections are present
            required_sections = [
                "## 1. Abstract",
                "## 2. Introduction",
                "## 3. Dataset & Data Integrity",
                "## 4. Methodology & Model Configurations",
                "## 5. Experimental Design & Protocol",
                "## 6. Evaluation Metrics",
                "## 7. Experimental Results",
                "## 8. Error Analysis & Diagnostics",
                "## 9. Crossover Analysis: When Does Fine-Tuning Beat RAG?",
                "## 10. Discussion",
                "## 11. Limitations & Execution Integrity",
                "## 12. Reproducibility & Environment Profile",
                "## 13. Conclusion & Next Steps",
            ]
            for sec in required_sections:
                self.assertIn(sec, report_text, f"Missing required section: {sec}")

    def test_report_unexecuted_integrity_guard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "report.md"
            report_text = generate_research_report(
                results_dir="experiments/runs",
                output_file=out_file,
                include_smoke=False,
            )
            # Must contain unexecuted notice and cannot invent crossover
            self.assertIn("RESEARCH EXECUTION STATUS NOTICE", report_text)
            self.assertIn("PENDING GPU", report_text)
            self.assertIn("NOT RUN / INSUFFICIENT DATA", report_text)

    def test_report_with_completed_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "fixture_report.md"
            report_text = generate_research_report(
                results_dir=self.analysis_fixtures,
                output_file=out_file,
                include_smoke=False,
            )
            self.assertTrue(out_file.is_file())
            # Fixtures contain completed LORA and RAG runs
            self.assertIn("FORMAL BENCHMARK EXECUTION RESULTS", report_text)
            self.assertIn("COMPLETED", report_text)
            self.assertIn("LORA", report_text)
            self.assertIn("RAG (K=3)", report_text)

    def test_report_cli_entry_point(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "cli_report.md"
            cmd = [
                sys.executable,
                "-m",
                "tunelab.analysis.report",
                "--results-dir",
                "experiments/runs",
                "--output-file",
                str(out_file),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
            self.assertEqual(res.returncode, 0, f"CLI failed: {res.stderr}")
            self.assertTrue(out_file.is_file())
            self.assertIn("TuneLab Research Report generated successfully", res.stdout)


if __name__ == "__main__":
    unittest.main()
