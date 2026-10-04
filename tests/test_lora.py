import json
from pathlib import Path
import tempfile
import unittest

from tunelab.data.leakage import DataLeakageError
from tunelab.data.loader import SpiderExample
from tunelab.data.schema import parse_spider_schema
from tunelab.evaluation.evaluator import evaluate_example
from tunelab.models.base import MockSQLGenerator
from tunelab.models.lora_adapter import LoRAInferenceAdapter
from tunelab.train import run_training_pipeline
from tunelab.training.format import (
    apply_loss_mask,
    format_dataset,
    format_single_example,
)
from tunelab.training.hardware import detect_hardware
from tunelab.training.lora import (
    LoRAHyperparameters,
    compute_lora_parameter_accounting,
    validate_target_modules,
)


class TestLoRATraining(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures"

        with open(self.fixtures_dir / "tables.json", "r") as f:
            raw_schemas = json.load(f)

        self.schemas = {
            s["db_id"]: parse_spider_schema(s)
            for s in raw_schemas
        }

        self.u_schema = self.schemas["mock_university"]

        self.u_db = (
            self.fixtures_dir
            / "database"
            / "mock_university"
            / "mock_university.sqlite"
        )

        self.train_example = SpiderExample(
            example_id="t1",
            db_id="mock_university",
            question="What is the average GPA?",
            query="SELECT avg(gpa) FROM student",
            split="train",
            source_file="train_spider.json",
        )

    def test_example_formatting_determinism(self):
        f1 = format_single_example(
            self.train_example,
            self.u_schema,
        )
        f2 = format_single_example(
            self.train_example,
            self.u_schema,
        )

        self.assertEqual(f1.full_text, f2.full_text)
        self.assertIn("### Database Schema:", f1.prompt_text)
        self.assertIn("CREATE TABLE student", f1.prompt_text)
        self.assertIn(
            "### Question:\nWhat is the average GPA?",
            f1.prompt_text,
        )
        self.assertEqual(
            f1.target_text,
            "SELECT avg(gpa) FROM student",
        )
        self.assertTrue(
            f1.full_text.endswith(
                "SELECT avg(gpa) FROM student"
            )
        )

    def test_loss_masking_correctness(self):
        prompt_tokens = [101, 2054, 2003, 102]
        target_tokens = [3001, 4002, 5003]
        eos_id = 999

        masked = apply_loss_mask(
            prompt_tokens,
            target_tokens,
            max_seq_length=512,
            eos_token_id=eos_id,
        )

        input_ids = masked["input_ids"]
        labels = masked["labels"]

        self.assertEqual(
            len(input_ids),
            len(labels),
        )
        self.assertEqual(
            labels[:len(prompt_tokens)],
            [-100] * len(prompt_tokens),
        )
        self.assertEqual(
            labels[len(prompt_tokens):],
            target_tokens + [eos_id],
        )

    def test_training_leakage_protection(self):
        dev_example = SpiderExample(
            example_id="dev_leak",
            db_id="mock_inventory",
            question="Dev question",
            query="SELECT 1",
            split="dev",
            source_file="dev.json",
        )

        with self.assertRaises(DataLeakageError):
            format_single_example(
                dev_example,
                self.schemas["mock_inventory"],
            )

        with self.assertRaises(DataLeakageError):
            format_dataset(
                [dev_example],
                self.schemas,
            )

    def test_lora_hyperparameter_validation(self):
        c4 = LoRAHyperparameters(r=4)
        self.assertEqual(c4.r, 4)

        c8 = LoRAHyperparameters(r=8)
        self.assertEqual(c8.r, 8)

        c16 = LoRAHyperparameters(r=16)
        self.assertEqual(c16.r, 16)

        with self.assertRaises(ValueError):
            LoRAHyperparameters(r=32)

        with self.assertRaises(ValueError):
            LoRAHyperparameters(dropout=1.5)

        with self.assertRaises(ValueError):
            LoRAHyperparameters(learning_rate=-0.01)

    def test_target_module_validation(self):
        validate_target_modules(
            ["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        with self.assertRaises(ValueError):
            validate_target_modules(
                ["q_proj", "invalid_layer_name"]
            )

        with self.assertRaises(ValueError):
            validate_target_modules([])

    def test_parameter_accounting_math(self):
        config = LoRAHyperparameters(
            r=8,
            target_modules=["q_proj", "v_proj"],
        )

        accounting = compute_lora_parameter_accounting(config)

        self.assertEqual(
            accounting["trainable_lora_parameters"],
            1376256,
        )
        self.assertAlmostEqual(
            accounting["trainable_parameter_percentage"],
            0.0892,
            places=3,
        )
        self.assertEqual(
            accounting["rank"],
            8,
        )

    def test_hardware_detection(self):
        hw = detect_hardware()

        self.assertIn(
            hw.device,
            ("cpu", "cuda"),
        )
        self.assertGreater(
            hw.cpu_count,
            0,
        )

        if hw.cuda_available:
            self.assertEqual(hw.device, "cuda")
            self.assertEqual(hw.status, "READY")
            self.assertGreater(hw.gpu_count, 0)
            self.assertIsNotNone(hw.gpu_name)
            self.assertIsNotNone(hw.gpu_vram_gb)
        else:
            self.assertEqual(hw.device, "cpu")
            self.assertEqual(hw.status, "TRAINING_PENDING_GPU")
            self.assertEqual(hw.gpu_count, 0)

    def test_training_pipeline_dry_run_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_training_pipeline(
                train_file=self.fixtures_dir / "train_spider.json",
                schemas_file=self.fixtures_dir / "tables.json",
                eval_file=self.fixtures_dir / "dev.json",
                output_dir=tmp_dir,
                data_fraction=0.50,
                r=8,
                dry_run=True,
            )

            self.assertEqual(
                summary["status"],
                "DRY_RUN_ONLY",
            )
            self.assertEqual(
                summary["data_fraction"],
                0.50,
            )
            self.assertGreater(
                summary["sample_count"],
                0,
            )

            meta_path = Path(tmp_dir) / "run_metadata.json"

            self.assertTrue(
                meta_path.is_file()
            )

            with open(meta_path, "r") as f:
                saved_meta = json.load(f)

            self.assertEqual(
                saved_meta["status"],
                "DRY_RUN_ONLY",
            )
            self.assertEqual(
                saved_meta["hyperparameters"]["r"],
                8,
            )
            self.assertIn(
                "trainable_lora_parameters",
                saved_meta["parameter_accounting"],
            )

    def test_evaluation_integration_with_lora_adapter(self):
        gold_sql = "SELECT avg(gpa) FROM student"
        question = "What is the average GPA of all students?"

        mock = MockSQLGenerator(
            mode="gold",
            reference_map={
                question: gold_sql,
            },
        )

        adapter = LoRAInferenceAdapter(
            checkpoint_dir="/tmp/dummy_checkpoint",
            mock_generator=mock,
        )

        prompt = "### Schema ... ### Question ... ### SQL:"

        predicted_sql = adapter.generate(
            prompt,
            question,
            self.u_schema,
        )

        res = evaluate_example(
            "lora_eval_1",
            "mock_university",
            predicted_sql,
            gold_sql,
            self.u_db,
        )

        self.assertTrue(res.sql_valid)
        self.assertTrue(res.exact_match)
        self.assertTrue(res.execution_accuracy)
        self.assertEqual(
            res.error_type,
            "CORRECT",
        )


if __name__ == "__main__":
    unittest.main()