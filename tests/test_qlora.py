import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tunelab.evaluation.evaluator import evaluate_example
from tunelab.models.base import MockSQLGenerator
from tunelab.models.qlora_adapter import QLoRAInferenceAdapter
from tunelab.train_qlora import run_qlora_pipeline
from tunelab.training.lora import LoRAHyperparameters
from tunelab.training.memory import (
    get_gpu_memory_stats,
    reset_peak_memory_stats,
)
from tunelab.training.qlora import (
    QLoRAConfig,
    check_qlora_dependencies,
    load_qlora_model,
)


class TestQLoRATraining(unittest.TestCase):

    def setUp(self):
        self.fixtures_dir = (
            Path(__file__).resolve().parent / "fixtures"
        )

        self.u_db = (
            self.fixtures_dir
            / "database"
            / "mock_university"
            / "mock_university.sqlite"
        )

    def test_qlora_config_valid(self):
        cfg = QLoRAConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )

        self.assertTrue(cfg.load_in_4bit)
        self.assertEqual(
            cfg.bnb_4bit_quant_type,
            "nf4",
        )
        self.assertTrue(
            cfg.bnb_4bit_use_double_quant
        )
        self.assertEqual(
            cfg.bnb_4bit_compute_dtype,
            "bfloat16",
        )

    def test_qlora_config_invalid_options(self):
        with self.assertRaises(ValueError):
            QLoRAConfig(load_in_4bit=False)

        with self.assertRaises(ValueError):
            QLoRAConfig(
                bnb_4bit_quant_type="int8"
            )

        with self.assertRaises(ValueError):
            QLoRAConfig(
                bnb_4bit_compute_dtype="int4"
            )

    def test_qlora_dependency_check_in_sandbox(self):
        unavailable = {
            "ready": False,
            "missing": ["CUDA", "bitsandbytes"],
            "reason": "CUDA and bitsandbytes are unavailable",
        }

        with patch(
            "tests.test_qlora.check_qlora_dependencies",
            return_value=unavailable,
        ):
            dep_check = check_qlora_dependencies()

        self.assertFalse(dep_check["ready"])
        self.assertIn(
            "CUDA",
            str(dep_check["missing"]),
        )

    def test_no_silent_fallback_guard(self):
        cfg = QLoRAConfig()
        lora_params = LoRAHyperparameters(r=8)

        unavailable = {
            "ready": False,
            "missing": ["CUDA"],
            "reason": "CUDA is unavailable",
        }

        with patch(
            "tunelab.training.qlora.check_qlora_dependencies",
            return_value=unavailable,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                load_qlora_model(
                    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    cfg,
                    lora_params,
                )

        self.assertIn(
            "No silent fallback permitted",
            str(ctx.exception),
        )

    def test_memory_instrumentation(self):
        mem = get_gpu_memory_stats()

        self.assertIn(
            "cuda_available",
            mem,
        )
        self.assertIn(
            "allocated_mb",
            mem,
        )
        self.assertIn(
            "peak_allocated_mb",
            mem,
        )

        reset_peak_memory_stats()

    def test_qlora_dry_run_pipeline_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_qlora_pipeline(
                train_file=(
                    self.fixtures_dir
                    / "train_spider.json"
                ),
                schemas_file=(
                    self.fixtures_dir
                    / "tables.json"
                ),
                eval_file=(
                    self.fixtures_dir
                    / "dev.json"
                ),
                output_dir=tmp_dir,
                data_fraction=0.50,
                r=8,
                quant_type="nf4",
                double_quant=True,
                compute_dtype="bfloat16",
                dry_run=True,
            )

            self.assertEqual(
                summary["status"],
                "DRY_RUN_ONLY",
            )
            self.assertEqual(
                summary["method"],
                "qlora",
            )
            self.assertEqual(
                summary["qlora_config"][
                    "bnb_4bit_quant_type"
                ],
                "nf4",
            )

            meta_file = (
                Path(tmp_dir)
                / "run_metadata.json"
            )

            self.assertTrue(
                meta_file.is_file()
            )

            with open(
                meta_file,
                "r",
                encoding="utf-8",
            ) as f:
                meta = json.load(f)

            self.assertEqual(
                meta["status"],
                "DRY_RUN_ONLY",
            )
            self.assertEqual(
                meta["method"],
                "qlora",
            )
            self.assertIn(
                "qlora_config",
                meta,
            )
            self.assertIn(
                "memory_stats",
                meta,
            )

    def test_qlora_pending_status_without_gpu(self):
        hardware_profile = type(
            "HardwareProfileStub",
            (),
            {
                "cuda_available": False,
                "device": "cpu",
                "ram_total_gb": 16.0,
                "bitsandbytes_available": False,
                "peft_version": "test",
                "torch_version": "test",
                "to_dict": lambda self: {
                    "cuda_available": self.cuda_available,
                    "device": self.device,
                    "ram_total_gb": self.ram_total_gb,
                    "bitsandbytes_available": self.bitsandbytes_available,
                    "peft_version": self.peft_version,
                    "torch_version": self.torch_version,
                },
            },
        )()

        unavailable = {
            "ready": False,
            "missing": ["CUDA"],
            "reason": "CUDA is unavailable",
            "hardware_profile": hardware_profile,
        }

        with patch(
            "tunelab.train_qlora.check_qlora_dependencies",
            return_value=unavailable,
        ):
            with tempfile.TemporaryDirectory() as tmp_dir:
                summary = run_qlora_pipeline(
                    train_file=(
                        self.fixtures_dir
                        / "train_spider.json"
                    ),
                    schemas_file=(
                        self.fixtures_dir
                        / "tables.json"
                    ),
                    eval_file=(
                        self.fixtures_dir
                        / "dev.json"
                    ),
                    output_dir=tmp_dir,
                    data_fraction=1.00,
                    dry_run=False,
                )

        self.assertEqual(
            summary["status"],
            "QLORA_PENDING_GPU",
        )

    def test_qlora_inference_adapter_and_eval_integration(self):
        gold_sql = (
            "SELECT avg(gpa) FROM student"
        )

        question = (
            "What is the average GPA of all students?"
        )

        mock = MockSQLGenerator(
            mode="gold",
            reference_map={
                question: gold_sql,
            },
        )

        adapter = QLoRAInferenceAdapter(
            checkpoint_dir="/tmp/dummy_qlora_ckpt",
            mock_generator=mock,
        )

        prompt = (
            "### Schema ... "
            "### Question ... "
            "### SQL:"
        )

        from tunelab.data.schema import parse_spider_schema

        with open(
            self.fixtures_dir / "tables.json",
            "r",
            encoding="utf-8",
        ) as f:
            raw_schemas = json.load(f)

        schema = parse_spider_schema(
            next(
                schema
                for schema in raw_schemas
                if schema["db_id"]
                == "mock_university"
            )
        )

        predicted_sql = adapter.generate(
            prompt,
            question,
            schema,
        )

        result = evaluate_example(
            "qlora_eval_1",
            "mock_university",
            predicted_sql,
            gold_sql,
            self.u_db,
        )

        self.assertTrue(
            result.sql_valid
        )
        self.assertTrue(
            result.exact_match
        )
        self.assertTrue(
            result.execution_accuracy
        )
        self.assertEqual(
            result.error_type,
            "CORRECT",
        )


if __name__ == "__main__":
    unittest.main()