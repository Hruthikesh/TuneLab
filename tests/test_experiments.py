import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tunelab.data.loader import SpiderExample
from tunelab.experiments.config import ExperimentConfig, load_manifest
from tunelab.experiments.noise import inject_training_noise
from tunelab.experiments.runner import execute_experiment_run
from tunelab.models.base import MockSQLGenerator


class TestExperimentOrchestration(unittest.TestCase):

    def setUp(self):
        self.fixtures_dir = (
            Path(__file__).resolve().parent / "fixtures"
        )

        self.manifest_dir = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "manifests"
        )

        self.u_db = self.fixtures_dir / "database"

        self.eval_file = self.fixtures_dir / "dev.json"
        self.schemas_file = self.fixtures_dir / "tables.json"
        self.db_dir = self.fixtures_dir / "database"

    def test_manifest_parsing(self):
        smoke_cfgs = load_manifest(
            self.manifest_dir / "smoke.yaml"
        )

        self.assertEqual(len(smoke_cfgs), 5)
        self.assertEqual(smoke_cfgs[0].method, "ZERO_SHOT")
        self.assertEqual(smoke_cfgs[1].method, "FEW_SHOT")
        self.assertEqual(smoke_cfgs[2].method, "RAG")
        self.assertEqual(smoke_cfgs[3].method, "LORA")
        self.assertEqual(smoke_cfgs[4].method, "QLORA")

        pilot_cfgs = load_manifest(
            self.manifest_dir / "pilot.yaml"
        )

        self.assertEqual(len(pilot_cfgs), 5)

        main_cfgs = load_manifest(
            self.manifest_dir / "main.yaml"
        )

        self.assertEqual(len(main_cfgs), 31)

    def test_config_validation_invalid_combinations(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(
                experiment_id="inv1",
                method="ZERO_SHOT",
                rag_k=3,
            )

        with self.assertRaises(ValueError):
            ExperimentConfig(
                experiment_id="inv2",
                method="ZERO_SHOT",
                lora_rank=8,
            )

        with self.assertRaises(ValueError):
            ExperimentConfig(
                experiment_id="inv3",
                method="LORA",
                lora_rank=8,
                rag_k=3,
            )

        with self.assertRaises(ValueError):
            ExperimentConfig(
                experiment_id="inv4",
                method="LORA",
            )

        with self.assertRaises(ValueError):
            ExperimentConfig(
                experiment_id="inv5",
                method="ZERO_SHOT",
                noise_level=0.10,
                noise_type="none",
            )

    def test_deterministic_configuration_hash(self):
        cfg1 = ExperimentConfig(
            experiment_id="test",
            method="ZERO_SHOT",
            seed=42,
        )

        cfg2 = ExperimentConfig(
            experiment_id="test",
            method="ZERO_SHOT",
            seed=42,
        )

        cfg3 = ExperimentConfig(
            experiment_id="test",
            method="ZERO_SHOT",
            seed=43,
        )

        self.assertEqual(
            cfg1.compute_hash(),
            cfg2.compute_hash(),
        )

        self.assertNotEqual(
            cfg1.compute_hash(),
            cfg3.compute_hash(),
        )

    def test_deterministic_noise_injection(self):
        examples = [
            SpiderExample(
                "1",
                "m",
                "What is gpa?",
                "SELECT gpa FROM student",
                "train",
                "f",
            ),
            SpiderExample(
                "2",
                "m",
                "List names.",
                "SELECT name FROM student",
                "train",
                "f",
            ),
            SpiderExample(
                "3",
                "m",
                "Count courses.",
                "SELECT count(*) FROM course",
                "train",
                "f",
            ),
            SpiderExample(
                "4",
                "m",
                "List titles.",
                "SELECT title FROM course",
                "train",
                "f",
            ),
        ]

        clean, affected_0 = inject_training_noise(
            examples,
            noise_level=0.0,
        )

        self.assertEqual(
            len(affected_0),
            0,
        )

        self.assertEqual(
            [e.query for e in clean],
            [e.query for e in examples],
        )

        c1, aff1 = inject_training_noise(
            examples,
            noise_level=0.50,
            noise_type="sql_syntax_corruption",
            seed=42,
        )

        c2, aff2 = inject_training_noise(
            examples,
            noise_level=0.50,
            noise_type="sql_syntax_corruption",
            seed=42,
        )

        self.assertEqual(len(aff1), 2)
        self.assertEqual(aff1, aff2)

        self.assertEqual(
            [e.query for e in c1],
            [e.query for e in c2],
        )

        for e in c1:
            if e.example_id in aff1:
                self.assertTrue(
                    e.query.startswith("SELEKT")
                )

    @patch("tunelab.experiments.runner.detect_hardware")
    def test_pending_gpu_status_when_cuda_required(
        self,
        mock_detect_hardware,
    ):
        from tunelab.training.hardware import detect_hardware

        hw = detect_hardware()

        hw.cuda_available = False
        hw.device = "cpu"
        hw.gpu_count = 0
        hw.gpu_name = None
        hw.gpu_vram_gb = None
        hw.status = "TRAINING_PENDING_GPU"

        mock_detect_hardware.return_value = hw

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExperimentConfig(
                experiment_id="test_gpu_check",
                method="LORA",
                lora_rank=8,
                hardware_requirements="cuda",
            )

            rec = execute_experiment_run(
                config=cfg,
                train_file=self.fixtures_dir / "train_spider.json",
                eval_file=self.fixtures_dir / "dev.json",
                schemas_file=self.fixtures_dir / "tables.json",
                db_dir=self.fixtures_dir / "database",
                output_base_dir=Path(tmp_dir),
            )

            self.assertEqual(
                rec.status,
                "PENDING_GPU",
            )

            self.assertIn(
                "CUDA GPU hardware required",
                str(rec.error_message),
            )

    def test_smoke_execution_across_all_five_methods(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            smoke_cfgs = load_manifest(
                self.manifest_dir / "smoke.yaml"
            )

            records = []

            for cfg in smoke_cfgs:
                rec = execute_experiment_run(
                    config=cfg,
                    train_file=self.fixtures_dir / "train_spider.json",
                    eval_file=self.fixtures_dir / "dev.json",
                    schemas_file=self.fixtures_dir / "tables.json",
                    db_dir=self.fixtures_dir / "database",
                    output_base_dir=Path(tmp_dir),
                )

                records.append(rec)

                self.assertEqual(
                    rec.status,
                    "COMPLETED",
                )

                self.assertIsNotNone(
                    rec.execution_accuracy
                )

                run_p = Path(tmp_dir) / rec.run_id

                self.assertTrue(
                    (run_p / "config.yaml").is_file()
                )

                self.assertTrue(
                    (run_p / "summary.json").is_file()
                )

                self.assertTrue(
                    (run_p / "evaluation_results.json").is_file()
                )

            master_jsonl = (
                Path(tmp_dir) / "master_results.jsonl"
            )

            master_csv = (
                Path(tmp_dir) / "master_results.csv"
            )

            self.assertTrue(
                master_jsonl.is_file()
            )

            self.assertTrue(
                master_csv.is_file()
            )

            with open(master_jsonl, "r") as f:
                lines = [
                    line
                    for line in f
                    if line.strip()
                ]

            self.assertEqual(len(lines), 5)

    def test_research_experiment_rejects_mock_generator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExperimentConfig(
                experiment_id="REAL_RESEARCH_EXP01",
                method="ZERO_SHOT",
                dataset="spider",
            )

            mock = MockSQLGenerator(
                mode="gold"
            )

            rec = execute_experiment_run(
                config=cfg,
                train_file=self.fixtures_dir / "train_spider.json",
                eval_file=self.fixtures_dir / "dev.json",
                schemas_file=self.fixtures_dir / "tables.json",
                db_dir=self.fixtures_dir / "database",
                output_base_dir=Path(tmp_dir),
                mock_generator=mock,
            )

            self.assertEqual(
                rec.status,
                "FAILED",
            )

            self.assertEqual(
                rec.generator_type,
                "REJECTED_MOCK",
            )

            self.assertIn(
                "MockSQLGenerator is strictly forbidden",
                str(rec.error_message),
            )

    @patch(
        "tunelab.experiments.runner.HuggingFaceSQLGenerator"
    )
    @patch(
        "tunelab.experiments.runner.detect_hardware"
    )
    def test_research_experiment_requires_real_model_without_silent_mock(
        self,
        mock_detect_hardware,
        mock_model_generator,
    ):
        from tunelab.training.hardware import detect_hardware

        hw = detect_hardware()

        hw.cuda_available = True
        hw.device = "cuda"
        hw.gpu_count = 1
        hw.gpu_name = "TEST_GPU"
        hw.gpu_vram_gb = 6.0
        hw.status = "READY"

        mock_detect_hardware.return_value = hw

        mock_model_generator.side_effect = RuntimeError(
            "Model weights unavailable for unit test"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExperimentConfig(
                experiment_id="REAL_RESEARCH_EXP02",
                method="ZERO_SHOT",
                dataset="spider",
            )

            rec = execute_experiment_run(
                config=cfg,
                train_file=self.fixtures_dir / "train_spider.json",
                eval_file=self.fixtures_dir / "dev.json",
                schemas_file=self.fixtures_dir / "tables.json",
                db_dir=self.fixtures_dir / "database",
                output_base_dir=Path(tmp_dir),
                mock_generator=None,
            )

            self.assertIn(
                rec.status,
                (
                    "PENDING_DEPENDENCY",
                    "PENDING_MODEL",
                ),
            )

            self.assertNotEqual(
                rec.status,
                "COMPLETED",
            )

            self.assertNotEqual(
                rec.model_type,
                "mock",
            )

            mock_model_generator.assert_called_once()

    def test_rag_demonstrations_affect_model_prompt_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExperimentConfig(
                experiment_id="SMOKE_RAG_PROMPT_TEST",
                method="RAG",
                rag_k=3,
                dataset="fixtures",
            )

            captured_prompts = []

            class CapturingGenerator(MockSQLGenerator):
                def generate(
                    self,
                    prompt,
                    question,
                    schema,
                ):
                    captured_prompts.append(prompt)

                    return super().generate(
                        prompt,
                        question,
                        schema,
                    )

            capturing = CapturingGenerator(
                mode="gold"
            )

            rec = execute_experiment_run(
                config=cfg,
                train_file=self.fixtures_dir / "train_spider.json",
                eval_file=self.fixtures_dir / "dev.json",
                schemas_file=self.fixtures_dir / "tables.json",
                db_dir=self.fixtures_dir / "database",
                output_base_dir=Path(tmp_dir),
                mock_generator=capturing,
            )

            self.assertEqual(
                rec.status,
                "COMPLETED",
            )

            self.assertGreater(
                len(captured_prompts),
                0,
            )

            self.assertIn(
                "### Demonstration Examples:",
                captured_prompts[0],
            )

            self.assertIn(
                "Example 1:",
                captured_prompts[0],
            )

            eval_json_p = (
                Path(tmp_dir)
                / rec.run_id
                / "evaluation_results.json"
            )

            self.assertTrue(
                eval_json_p.is_file()
            )

            with open(eval_json_p) as f:
                eval_data = json.load(f)

            self.assertIn(
                "rag_k",
                eval_data["summary"],
            )

            self.assertEqual(
                eval_data["summary"]["rag_k"],
                3,
            )

            first_res = eval_data["results"][0]

            self.assertIn(
                "retrieval_metadata",
                first_res,
            )

            self.assertEqual(
                first_res["retrieval_metadata"]["k"],
                3,
            )

            self.assertGreater(
                len(
                    first_res[
                        "retrieval_metadata"
                    ]["retrieved_ids"]
                ),
                0,
            )

    def test_result_provenance_metadata_recorded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExperimentConfig(
                experiment_id="SMOKE_PROVENANCE_TEST",
                method="ZERO_SHOT",
                dataset="fixtures",
            )

            rec = execute_experiment_run(
                config=cfg,
                train_file=self.fixtures_dir / "train_spider.json",
                eval_file=self.fixtures_dir / "dev.json",
                schemas_file=self.fixtures_dir / "tables.json",
                db_dir=self.fixtures_dir / "database",
                output_base_dir=Path(tmp_dir),
            )

            self.assertEqual(
                rec.model_type,
                "mock",
            )

            self.assertEqual(
                rec.model_name,
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            )

            self.assertEqual(
                rec.generator_type,
                "MockSQLGenerator",
            )

            d = rec.to_dict()

            self.assertIn(
                "model_type",
                d,
            )

            self.assertIn(
                "generator_type",
                d,
            )

            self.assertIn(
                "model_name",
                d,
            )

    @patch(
        "tunelab.experiments.runner.train_lora_model"
    )
    @patch(
        "tunelab.experiments.runner.LoRAInferenceAdapter"
    )
    @patch(
        "tunelab.experiments.runner.detect_hardware"
    )
    def test_real_lora_experiment_calls_train_lora_model(
        self,
        mock_detect_hw,
        mock_adapter_cls,
        mock_train_lora,
    ):
        from tunelab.training.hardware import detect_hardware

        hw = detect_hardware()

        hw.cuda_available = True
        hw.device = "cuda"
        hw.torch_version = "2.1.0"
        hw.transformers_version = "4.38.0"
        hw.peft_version = "0.9.0"

        mock_detect_hw.return_value = hw

        mock_train_lora.return_value = {
            "training_loss": 0.42,
            "global_step": 100,
        }

        adapter_inst = MockSQLGenerator(
            mode="gold"
        )

        mock_adapter_cls.return_value = adapter_inst

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExperimentConfig(
                experiment_id="REAL_LORA_EXP_01",
                method="LORA",
                dataset="spider",
                lora_rank=8,
                model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            )

            rec = execute_experiment_run(
                config=cfg,
                train_file=self.fixtures_dir / "train_spider.json",
                eval_file=self.fixtures_dir / "dev.json",
                schemas_file=self.fixtures_dir / "tables.json",
                db_dir=self.fixtures_dir / "database",
                output_base_dir=Path(tmp_dir),
                mock_generator=None,
            )

            mock_train_lora.assert_called_once()

            kwargs = mock_train_lora.call_args.kwargs

            self.assertEqual(
                kwargs["device"],
                "cuda",
            )

            self.assertEqual(
                kwargs["hyperparameters"].r,
                8,
            )

            self.assertEqual(
                kwargs["output_dir"],
                Path(tmp_dir)
                / rec.run_id
                / "adapter_checkpoint",
            )

            self.assertGreater(
                len(kwargs["formatted_dataset"]),
                0,
            )

            mock_adapter_cls.assert_called_once_with(
                checkpoint_dir=(
                    Path(tmp_dir)
                    / rec.run_id
                    / "adapter_checkpoint"
                ),
                base_model_name=(
                    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
                ),
            )

            self.assertEqual(
                rec.status,
                "COMPLETED",
            )

            self.assertEqual(
                rec.method,
                "LORA",
            )

            self.assertIsNotNone(
                rec.training_time_s
            )

            self.assertIsNotNone(
                rec.trainable_parameters
            )

            self.assertIsNotNone(
                rec.total_parameters
            )

    @patch(
        "tunelab.experiments.runner.train_qlora_model"
    )
    @patch(
        "tunelab.experiments.runner.QLoRAInferenceAdapter"
    )
    @patch(
        "tunelab.experiments.runner.check_qlora_dependencies"
    )
    @patch(
        "tunelab.experiments.runner.detect_hardware"
    )
    def test_real_qlora_experiment_calls_train_qlora_model(
        self,
        mock_detect_hw,
        mock_check_dep,
        mock_adapter_cls,
        mock_train_qlora,
    ):
        from tunelab.training.hardware import detect_hardware

        hw = detect_hardware()

        hw.cuda_available = True
        hw.device = "cuda"
        hw.torch_version = "2.1.0"
        hw.transformers_version = "4.38.0"
        hw.peft_version = "0.9.0"
        hw.bitsandbytes_available = True

        mock_detect_hw.return_value = hw

        mock_check_dep.return_value = {
            "ready": True,
            "missing": [],
            "reason": None,
            "hardware_profile": hw,
        }

        mock_train_qlora.return_value = {
            "training_loss": 0.38,
            "global_step": 100,
        }

        adapter_inst = MockSQLGenerator(
            mode="gold"
        )

        mock_adapter_cls.return_value = adapter_inst

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExperimentConfig(
                experiment_id="REAL_QLORA_EXP_01",
                method="QLORA",
                dataset="spider",
                lora_rank=8,
                model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                qlora_config={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                },
            )

            rec = execute_experiment_run(
                config=cfg,
                train_file=self.fixtures_dir / "train_spider.json",
                eval_file=self.fixtures_dir / "dev.json",
                schemas_file=self.fixtures_dir / "tables.json",
                db_dir=self.fixtures_dir / "database",
                output_base_dir=Path(tmp_dir),
                mock_generator=None,
            )

            mock_train_qlora.assert_called_once()

            kwargs = mock_train_qlora.call_args.kwargs

            self.assertTrue(
                kwargs["qlora_config"].load_in_4bit
            )

            self.assertEqual(
                kwargs["qlora_config"].bnb_4bit_quant_type,
                "nf4",
            )

            self.assertEqual(
                kwargs["output_dir"],
                Path(tmp_dir)
                / rec.run_id
                / "adapter_checkpoint",
            )

            self.assertGreater(
                len(kwargs["formatted_dataset"]),
                0,
            )

            self.assertEqual(
                rec.status,
                "COMPLETED",
            )

            self.assertEqual(
                rec.method,
                "QLORA",
            )

            self.assertIsNotNone(
                rec.training_time_s
            )

    def test_standalone_evaluation_cli_requires_real_model_or_smoke(
        self,
    ):
        eval_script = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "tunelab"
            / "evaluate.py"
        )

        cmd_fail = [
            os.environ.get(
                "PYTHON",
                __import__("sys").executable,
            ),
            str(eval_script),
            "--eval-file",
            str(self.eval_file),
            "--schemas-file",
            str(self.schemas_file),
            "--db-dir",
            str(self.db_dir),
        ]

        offline_env = os.environ.copy()
        offline_env["HF_HUB_OFFLINE"] = "1"
        offline_env["TRANSFORMERS_OFFLINE"] = "1"
        offline_env["TUNELAB_TEST_NO_MODEL"] = "1"

        res_fail = subprocess.run(
            cmd_fail,
            capture_output=True,
            text=True,
            env=offline_env,
            timeout=30,
        )

        self.assertNotEqual(
            res_fail.returncode,
            0,
        )

        self.assertIn(
            "MODEL_REQUIRED_FOR_RESEARCH",
            res_fail.stderr,
        )

        cmd_smoke = cmd_fail + ["--smoke"]

        res_smoke = subprocess.run(
            cmd_smoke,
            capture_output=True,
            text=True,
            env=offline_env,
            timeout=30,
        )

        self.assertEqual(
            res_smoke.returncode,
            0,
        )

        self.assertIn(
            "EVALUATION SUMMARY",
            res_smoke.stderr,
        )


if __name__ == "__main__":
    unittest.main()