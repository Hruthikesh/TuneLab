from pathlib import Path
import sqlite3
import unittest

from tunelab.evaluation.canonicalize import canonicalize_sql, compute_exact_match
from tunelab.evaluation.executor import (
    check_sql_safety,
    compare_results,
    execute_sqlite_query,
    reference_requires_order_by,
    UnsafeSQLStatementError,
)
from tunelab.evaluation.evaluator import (
    compute_summary_metrics,
    evaluate_example,
)


class TestEvaluationFramework(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures"
        self.u_db = self.fixtures_dir / "database" / "mock_university" / "mock_university.sqlite"
        self.assertTrue(self.u_db.is_file(), f"Test DB missing at {self.u_db}")

    def test_sql_canonicalization(self):
        raw = "SELECT  /* comment */ name, gpa\n-- line comment\nFROM student WHERE gpa > 3.0;\n"
        canonical = canonicalize_sql(raw)
        self.assertEqual(canonical, "select name, gpa from student where gpa > 3.0")

    def test_exact_match(self):
        q1 = "SELECT name, gpa FROM student;"
        q2 = "select   name,   gpa   from   student"
        q3 = "SELECT gpa, name FROM student"
        self.assertTrue(compute_exact_match(q1, q2))
        self.assertFalse(compute_exact_match(q1, q3))

    def test_correct_query_execution(self):
        gold = "SELECT name FROM student WHERE gpa > 3.5"
        pred = "SELECT name FROM student WHERE gpa > 3.5"
        res = evaluate_example("ex1", "mock_university", pred, gold, self.u_db)
        self.assertTrue(res.sql_valid)
        self.assertTrue(res.exact_match)
        self.assertTrue(res.execution_accuracy)
        self.assertEqual(res.error_type, "CORRECT")

    def test_incorrect_query_result_mismatch(self):
        gold = "SELECT name FROM student WHERE gpa > 3.5"
        pred = "SELECT name FROM student WHERE gpa > 3.0"
        res = evaluate_example("ex2", "mock_university", pred, gold, self.u_db)
        self.assertTrue(res.sql_valid)
        self.assertFalse(res.execution_accuracy)
        self.assertEqual(res.error_type, "RESULT_MISMATCH")

    def test_equivalent_query_different_formatting(self):
        gold = "SELECT name FROM student WHERE gpa > 3.5"
        pred = "SELECT T.name FROM student AS T WHERE T.gpa > 3.5"
        res = evaluate_example("ex3", "mock_university", pred, gold, self.u_db)
        self.assertTrue(res.sql_valid)
        self.assertFalse(res.exact_match)  # Strings differ
        self.assertTrue(res.execution_accuracy)  # Execution matches
        self.assertEqual(res.error_type, "CORRECT")

    def test_unordered_results_without_order_by(self):
        # Bag equality test
        gold_rows = [("Alice", 3.8), ("Bob", 3.2), ("Charlie", 3.9)]
        pred_rows = [("Bob", 3.2), ("Charlie", 3.9), ("Alice", 3.8)]
        # Order by NOT required
        matches = compare_results(pred_rows, gold_rows, order_by_required=False)
        self.assertTrue(matches)

    def test_ordered_results_with_order_by(self):
        gold_rows = [("Bob", 3.2), ("Alice", 3.8), ("Charlie", 3.9)]
        pred_rows = [("Alice", 3.8), ("Bob", 3.2), ("Charlie", 3.9)]
        # Order by required: sequence must match strictly
        matches = compare_results(pred_rows, gold_rows, order_by_required=True)
        self.assertFalse(matches)

        gold_sql = "SELECT name FROM student ORDER BY gpa ASC"
        self.assertTrue(reference_requires_order_by(gold_sql))

    def test_duplicate_rows_preservation(self):
        # Duplicate row counts must match
        gold_rows = [("A",), ("A",), ("B",)]
        pred_rows_missing_dup = [("A",), ("B",)]
        matches = compare_results(pred_rows_missing_dup, gold_rows, order_by_required=False)
        self.assertFalse(matches)

        pred_rows_correct = [("B",), ("A",), ("A",)]
        matches_correct = compare_results(pred_rows_correct, gold_rows, order_by_required=False)
        self.assertTrue(matches_correct)

    def test_null_values_handling(self):
        gold_rows = [("Alice", None), ("Bob", 3.2)]
        pred_rows = [("Alice", None), ("Bob", 3.2)]
        self.assertTrue(compare_results(pred_rows, gold_rows, order_by_required=False))

        # None != 0 and None != ""
        pred_rows_wrong = [("Alice", 0.0), ("Bob", 3.2)]
        self.assertFalse(compare_results(pred_rows_wrong, gold_rows, order_by_required=False))

    def test_numeric_tolerance(self):
        gold_rows = [(3.6333333333333335,)]
        pred_rows = [(3.6333,)]  # Within 1e-4 tolerance
        self.assertTrue(compare_results(pred_rows, gold_rows, order_by_required=False))

        pred_rows_far = [(3.7,)]  # Outside tolerance
        self.assertFalse(compare_results(pred_rows_far, gold_rows, order_by_required=False))

    def test_empty_result_sets(self):
        gold = "SELECT name FROM student WHERE gpa > 5.0"  # Empty
        pred = "SELECT name FROM student WHERE gpa < 0.0"  # Empty
        res = evaluate_example("empty_match", "mock_university", pred, gold, self.u_db)
        self.assertTrue(res.execution_accuracy)

        pred_non_empty = "SELECT name FROM student"
        res_mismatch = evaluate_example("empty_diff", "mock_university", pred_non_empty, gold, self.u_db)
        self.assertFalse(res_mismatch.execution_accuracy)

    def test_syntax_error_classification(self):
        gold = "SELECT name FROM student"
        pred = "SELECCT name FORM student"
        res = evaluate_example("err_syn", "mock_university", pred, gold, self.u_db)
        self.assertFalse(res.sql_valid)
        self.assertFalse(res.execution_accuracy)
        self.assertEqual(res.error_type, "SYNTAX_ERROR")

    def test_missing_table_classification(self):
        gold = "SELECT name FROM student"
        pred = "SELECT * FROM non_existent_table"
        res = evaluate_example("err_tbl", "mock_university", pred, gold, self.u_db)
        self.assertFalse(res.sql_valid)
        self.assertEqual(res.error_type, "MISSING_TABLE")

    def test_missing_column_classification(self):
        gold = "SELECT name FROM student"
        pred = "SELECT ghost_column FROM student"
        res = evaluate_example("err_col", "mock_university", pred, gold, self.u_db)
        self.assertFalse(res.sql_valid)
        self.assertEqual(res.error_type, "MISSING_COLUMN")

    def test_timeout_handling(self):
        # Infinite recursive query
        gold = "SELECT name FROM student"
        pred = "WITH RECURSIVE r(i) AS (VALUES(0) UNION ALL SELECT i+1 FROM r) SELECT count(*) FROM r"
        res = evaluate_example("err_time", "mock_university", pred, gold, self.u_db, timeout_seconds=0.1)
        self.assertFalse(res.sql_valid)
        self.assertEqual(res.error_type, "TIMEOUT")

    def test_destructive_sql_rejection(self):
        destructive_queries = [
            "DROP TABLE student",
            "DELETE FROM student WHERE student_id = 1",
            "UPDATE student SET gpa = 4.0",
            "INSERT INTO student VALUES (99, 'Hacker', 4.0)",
            "ALTER TABLE student ADD COLUMN hacked TEXT",
        ]
        gold = "SELECT count(*) FROM student"
        initial_count = execute_sqlite_query(self.u_db, gold).rows[0][0]

        for dq in destructive_queries:
            with self.subTest(query=dq):
                res = evaluate_example("sec_test", "mock_university", dq, gold, self.u_db)
                self.assertFalse(res.sql_valid)
                self.assertEqual(res.error_type, "INVALID_STATEMENT")

        # Verify DB remained completely unmodified
        post_count = execute_sqlite_query(self.u_db, gold).rows[0][0]
        self.assertEqual(initial_count, post_count)

    def test_summary_metrics(self):
        results = [
            evaluate_example("1", "mock_university", "SELECT name FROM student", "SELECT name FROM student", self.u_db),
            evaluate_example("2", "mock_university", "SELECT 1", "SELECT name FROM student", self.u_db),
        ]
        metrics = compute_summary_metrics(results)
        self.assertEqual(metrics["total_samples"], 2)
        self.assertEqual(metrics["execution_accuracy"], 0.5)
        self.assertEqual(metrics["sql_validity"], 1.0)
        self.assertIn("CORRECT", metrics["error_breakdown"])
        self.assertIn("RESULT_MISMATCH", metrics["error_breakdown"])


if __name__ == "__main__":
    unittest.main()
