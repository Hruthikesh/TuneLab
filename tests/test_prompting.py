import sqlite3
import unittest
from pathlib import Path

from tunelab.data.loader import SpiderExample
from tunelab.data.schema import Column, DatabaseSchema, Table
from tunelab.evaluation.evaluator import evaluate_example
from tunelab.models.base import (
    MockSQLGenerator,
    extract_sql_from_generation,
)
from tunelab.prompting.prompts import (
    build_few_shot_prompt,
    build_zero_shot_prompt,
    select_deterministic_demonstrations,
)


class TestPromptingAndPipeline(unittest.TestCase):

    def setUp(self):
        employees = Table(
            name="employees",
            original_name="employees",
            columns={
                "id": Column(
                    name="id",
                    original_name="id",
                    column_type="number",
                    is_primary_key=True,
                ),
                "name": Column(
                    name="name",
                    original_name="name",
                    column_type="text",
                ),
                "department": Column(
                    name="department",
                    original_name="department",
                    column_type="text",
                ),
            },
        )

        self.schema = DatabaseSchema(
            db_id="prompting_test",
            tables={
                "employees": employees,
            },
            foreign_keys=[],
        )

        self.db_path = (
            Path(__file__).parent
            / "fixtures"
            / "database"
            / f"{self.schema.db_id}.sqlite"
        )

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(self.db_path)

        connection.execute("DROP TABLE IF EXISTS employees")

        connection.execute(
            """
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT,
                department TEXT
            )
            """
        )

        connection.executemany(
            "INSERT INTO employees (id, name, department) VALUES (?, ?, ?)",
            [
                (1, "Alice", "Mining"),
                (2, "Bob", "Computer Science"),
                (3, "Charlie", "Mining"),
            ],
        )

        connection.commit()
        connection.close()

    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()

    def test_zero_shot_prompt_structure(self):
        question = "How many employees are there?"

        prompt = build_zero_shot_prompt(
            question=question,
            schema=self.schema,
        )

        self.assertIn(question, prompt)
        self.assertIn("employees", prompt)
        self.assertIn("SQL", prompt)

    def test_few_shot_prompt_structure(self):
        question = "How many employees work in Mining?"

        demonstrations = [
            SpiderExample(
                example_id="demo_1",
                db_id="prompting_test",
                question="How many employees are there?",
                query="SELECT COUNT(*) FROM employees",
                split="train",
                source_file="test",
            )
        ]

        prompt = build_few_shot_prompt(
            question=question,
            schema=self.schema,
            demonstrations=demonstrations,
        )

        self.assertIn(question, prompt)
        self.assertIn(
            "How many employees are there?",
            prompt,
        )
        self.assertIn(
            "SELECT COUNT(*) FROM employees",
            prompt,
        )
        self.assertIn("employees", prompt)

    def test_leakage_prevention_in_demonstrations(self):
        evaluation_example = SpiderExample(
            example_id="target",
            db_id="prompting_test",
            question="How many employees are there?",
            query="SELECT COUNT(*) FROM employees",
            split="dev",
            source_file="test",
        )

        training_example = SpiderExample(
            example_id="demo_1",
            db_id="prompting_test",
            question="List all employees",
            query="SELECT name FROM employees",
            split="train",
            source_file="test",
        )

        selected = select_deterministic_demonstrations(
            pool=[training_example],
            k=1,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0].example_id,
            "demo_1",
        )

        with self.assertRaises(Exception):
            select_deterministic_demonstrations(
                pool=[
                    training_example,
                    evaluation_example,
                ],
                k=1,
            )

    def test_sql_extraction_from_inline_explanation(self):
        generated = (
            "SELECT count(*) FROM singer "
            "This query will return the total number of singers."
        )

        extracted = extract_sql_from_generation(generated)

        self.assertEqual(
            extracted,
            "SELECT count(*) FROM singer",
        )

    def test_sql_extraction_from_markdown_block(self):
        generated = "```sql\nSELECT count(*) FROM singer\n```"

        extracted = extract_sql_from_generation(generated)

        self.assertEqual(
            extracted,
            "SELECT count(*) FROM singer",
        )

    def test_end_to_end_mock_pipeline_accuracy(self):
        question = "How many employees are there?"
        reference_sql = "SELECT COUNT(*) FROM employees"

        generator = MockSQLGenerator(
            mode="gold",
            reference_map={
                question: reference_sql,
            },
        )

        prompt = build_zero_shot_prompt(
            question=question,
            schema=self.schema,
        )

        generated_sql = generator.generate(
            prompt=prompt,
            question=question,
            schema=self.schema,
        )

        result = evaluate_example(
            example_id="prompting_test_1",
            database_id=self.schema.db_id,
            predicted_sql=generated_sql,
            reference_sql=reference_sql,
            sqlite_path=self.db_path,
        )

        self.assertTrue(result.exact_match)
        self.assertTrue(result.execution_accuracy)


if __name__ == "__main__":
    unittest.main()