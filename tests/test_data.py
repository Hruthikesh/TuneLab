import json
from pathlib import Path
import tempfile
import unittest

from tunelab.data.leakage import DataLeakageError, verify_split_integrity
from tunelab.data.loader import SpiderDataLoader, SpiderExample
from tunelab.data.pipeline import process_and_save_data
from tunelab.data.pools import generate_deterministic_subsets, create_internal_validation_split
from tunelab.data.schema import parse_spider_schema, validate_sqlite_database


class TestDataLayer(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures"
        self.assertTrue(self.fixtures_dir.is_dir(), "Test fixtures directory missing.")

    def test_schema_parsing(self):
        with open(self.fixtures_dir / "tables.json", "r") as f:
            raw_schemas = json.load(f)

        u_schema_entry = next(s for s in raw_schemas if s["db_id"] == "mock_university")
        schema = parse_spider_schema(u_schema_entry)

        self.assertEqual(schema.db_id, "mock_university")
        self.assertIn("student", schema.tables)
        self.assertIn("course", schema.tables)
        self.assertIn("enrollment", schema.tables)

        # Check column properties
        student_id_col = schema.tables["student"].columns["student_id"]
        self.assertTrue(student_id_col.is_primary_key)
        self.assertEqual(student_id_col.column_type, "NUMBER")

        # Check resolved foreign keys
        enrollment_student_col = schema.tables["enrollment"].columns["student_id"]
        self.assertEqual(enrollment_student_col.foreign_key_to, ("student", "student_id"))

        ddl = schema.to_ddl_text()
        self.assertIn("CREATE TABLE student", ddl)
        self.assertIn("REFERENCES student(student_id)", ddl)

    def test_sqlite_validation(self):
        db_path = self.fixtures_dir / "database" / "mock_university" / "mock_university.sqlite"
        self.assertTrue(db_path.is_file())

        with open(self.fixtures_dir / "tables.json", "r") as f:
            raw_schemas = json.load(f)
        schema = parse_spider_schema(next(s for s in raw_schemas if s["db_id"] == "mock_university"))

        report = validate_sqlite_database(db_path, schema=schema)
        self.assertTrue(report["readable"])
        self.assertTrue(report["valid"])
        self.assertEqual(report["table_count"], 3)
        self.assertEqual(len(report["discrepancies"]), 0)

    def test_loader_valid_fixtures(self):
        loader = SpiderDataLoader(self.fixtures_dir)
        ready, missing = loader.check_files_exist(require_others=True)
        self.assertTrue(ready, f"Missing files in fixtures: {missing}")

        schemas = loader.load_schemas()
        self.assertEqual(len(schemas), 3)

        dbs = loader.scan_databases()
        self.assertEqual(len(dbs), 3)

        # Test train_spider_only
        train_ex, dev_ex = loader.load_examples(training_pool="train_spider_only")
        self.assertEqual(len(train_ex), 4)
        self.assertEqual(len(dev_ex), 2)

        # Test train_spider_and_others
        train_all, _ = loader.load_examples(training_pool="train_spider_and_others")
        self.assertEqual(len(train_all), 5)

    def test_loader_malformed_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Missing required fields in example
            with open(tmp_path / "tables.json", "w") as f:
                json.dump([{"db_id": "test_db", "table_names_original": ["t"], "table_names": ["t"],
                            "column_names_original": [[-1, "*"], [0, "id"]], "column_names": [[-1, "*"], [0, "id"]],
                            "column_types": ["text", "number"], "primary_keys": [1], "foreign_keys": []}], f)

            (tmp_path / "database" / "test_db").mkdir(parents=True)
            (tmp_path / "database" / "test_db" / "test_db.sqlite").touch()

            # Malformed example (missing 'query')
            with open(tmp_path / "train_spider.json", "w") as f:
                json.dump([{"db_id": "test_db", "question": "What is id?"}], f)
            with open(tmp_path / "dev.json", "w") as f:
                json.dump([], f)

            loader = SpiderDataLoader(tmp_path)
            with self.assertRaises(ValueError):
                loader.load_examples()

    def test_leakage_detection(self):
        loader = SpiderDataLoader(self.fixtures_dir)
        train_ex, dev_ex = loader.load_examples()

        # Legitimate split has zero overlap
        report = verify_split_integrity(train_ex, dev_ex)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.db_id_overlap), 0)

        # Introduce artificial database leakage
        leaked_dev_ex = list(dev_ex)
        leaked_dev_ex.append(
            SpiderExample(
                example_id="leak_1",
                db_id="mock_university",  # DB exists in train!
                question="Leaked question",
                query="SELECT * FROM student",
                split="dev",
                source_file="dev.json"
            )
        )
        with self.assertRaises(DataLeakageError):
            verify_split_integrity(train_ex, leaked_dev_ex)

    def test_deterministic_subset_generation(self):
        loader = SpiderDataLoader(self.fixtures_dir)
        train_ex, _ = loader.load_examples(training_pool="train_spider_and_others")
        self.assertEqual(len(train_ex), 5)

        # Generate twice with seed 42
        subsets_run1 = generate_deterministic_subsets(train_ex, fractions=[0.25, 0.50, 1.00], seed=42)
        subsets_run2 = generate_deterministic_subsets(train_ex, fractions=[0.25, 0.50, 1.00], seed=42)

        # Check exact determinism of example IDs
        ids_1 = [e.example_id for e in subsets_run1["50pct"][0]]
        ids_2 = [e.example_id for e in subsets_run2["50pct"][0]]
        self.assertEqual(ids_1, ids_2)

        # Verify subset nesting: 25% subset must be a strict prefix of 50% subset
        ids_25 = [e.example_id for e in subsets_run1["25pct"][0]]
        self.assertEqual(ids_25, ids_1[:len(ids_25)])

    def test_internal_cross_database_validation_split(self):
        loader = SpiderDataLoader(self.fixtures_dir)
        train_ex, _ = loader.load_examples()

        internal_train, internal_val = create_internal_validation_split(train_ex, val_ratio=0.5, seed=42)
        train_dbs = {e.db_id for e in internal_train}
        val_dbs = {e.db_id for e in internal_val}

        # Databases must be strictly disjoint
        self.assertEqual(len(train_dbs.intersection(val_dbs)), 0)
        self.assertEqual(len(train_dbs) + len(val_dbs), 2)

    def test_end_to_end_pipeline_with_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp_out:
            processed_dir = Path(tmp_out) / "processed"
            splits_dir = Path(tmp_out) / "splits"

            summary = process_and_save_data(
                raw_dir=self.fixtures_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                training_pool="train_spider_only",
                seed=42,
                validate_sqlite=True,
            )

            self.assertEqual(summary["num_schemas"], 3)
            self.assertEqual(summary["num_train_examples"], 4)
            self.assertEqual(summary["num_dev_examples"], 2)
            self.assertTrue(summary["leakage_check_passed"])

            # Verify saved files
            self.assertTrue((processed_dir / "schemas.json").is_file())
            self.assertTrue((processed_dir / "train.jsonl").is_file())
            self.assertTrue((processed_dir / "dev.jsonl").is_file())
            self.assertTrue((processed_dir / "metadata.json").is_file())
            self.assertTrue((splits_dir / "subsets_metadata_train_spider_only.json").is_file())


if __name__ == "__main__":
    unittest.main()
