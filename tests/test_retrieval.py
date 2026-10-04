import json
from pathlib import Path
import unittest

from tunelab.data.leakage import DataLeakageError
from tunelab.data.loader import SpiderExample
from tunelab.data.schema import parse_spider_schema
from tunelab.evaluation.evaluator import evaluate_example
from tunelab.models.base import MockSQLGenerator
from tunelab.prompting.rag import build_rag_prompt
from tunelab.retrieval.base import RetrievedExample
from tunelab.retrieval.bm25 import BM25Retriever, tokenize
from tunelab.retrieval.dense import (
    BGEEmbeddingBackend,
    DenseRetriever,
    MockDenseEmbeddingBackend,
)
from tunelab.retrieval.hybrid import HybridRetriever
from tunelab.retrieval.metrics import (
    compute_hit_rate,
    compute_mrr,
    compute_recall_at_k,
    evaluate_retrieval_batch,
)


class TestRetrievalLayer(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures"
        self.train_corpus = [
            SpiderExample("ex_1", "mock_university", "What is the average GPA of all students?", "SELECT avg(gpa) FROM student", "train", "t1"),
            SpiderExample("ex_2", "mock_university", "Find the names of students enrolled in Algorithms.", "SELECT T1.name FROM student AS T1 JOIN enrollment AS T2 ON T1.student_id = T2.student_id JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.title = 'Algorithms'", "train", "t1"),
            SpiderExample("ex_3", "mock_ecommerce", "List all customers from India.", "SELECT name FROM customer WHERE country = 'India'", "train", "t1"),
            SpiderExample("ex_4", "mock_ecommerce", "How many orders were placed by customer 1?", "SELECT count(*) FROM orders WHERE customer_id = 1", "train", "t1"),
            SpiderExample("ex_5", "mock_ecommerce", "What is the total order amount across all orders?", "SELECT sum(total_amount) FROM orders", "train", "t1"),
        ]
        self.u_db = self.fixtures_dir / "database" / "mock_university" / "mock_university.sqlite"
        with open(self.fixtures_dir / "tables.json", "r") as f:
            raw_schemas = json.load(f)
        self.schema = parse_spider_schema(next(s for s in raw_schemas if s["db_id"] == "mock_university"))

    def test_bm25_tokenization(self):
        tokens = tokenize("What is the average GPA, for students?")
        self.assertEqual(tokens, ["what", "is", "the", "average", "gpa", "for", "students"])

        empty_tokens = tokenize("")
        self.assertEqual(empty_tokens, [])

    def test_bm25_ranking(self):
        bm25 = BM25Retriever()
        bm25.index(self.train_corpus)

        results = bm25.retrieve("average GPA of students", k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].example_id, "ex_1")
        self.assertGreater(results[0].score, 0.0)

    def test_bm25_deterministic_tie_breaking(self):
        bm25 = BM25Retriever()
        bm25.index(self.train_corpus)

        # Query with completely unrelated terms -> all scores are 0.0
        results1 = bm25.retrieve("unrelated xylophone astronaut", k=5)
        results2 = bm25.retrieve("unrelated xylophone astronaut", k=5)

        self.assertEqual([r.example_id for r in results1], [r.example_id for r in results2])
        # Tie-breaker must sort by example_id ascending
        expected_ids = sorted([ex.example_id for ex in self.train_corpus])
        self.assertEqual([r.example_id for r in results1], expected_ids)

    def test_top_k_behavior(self):
        bm25 = BM25Retriever()
        bm25.index(self.train_corpus)

        for k in (1, 3, 5, 10):
            res = bm25.retrieve("orders placed by customer", k=k)
            expected_count = min(k, len(self.train_corpus))
            self.assertEqual(len(res), expected_count)
            self.assertEqual([r.rank for r in res], list(range(1, expected_count + 1)))

    def test_duplicate_handling_in_index(self):
        corpus_with_dups = list(self.train_corpus) + [self.train_corpus[0]]
        bm25 = BM25Retriever()
        bm25.index(corpus_with_dups)
        self.assertEqual(len(bm25.indexed_examples), 5)

    def test_leakage_prevention_at_index_time(self):
        contaminated_corpus = list(self.train_corpus)
        contaminated_corpus.append(
            SpiderExample("leak_dev", "mock_inventory", "Dev question", "SELECT 1", "dev", "dev.json")
        )

        bm25 = BM25Retriever()
        with self.assertRaises(DataLeakageError):
            bm25.index(contaminated_corpus)

        dense = DenseRetriever(backend=MockDenseEmbeddingBackend())
        with self.assertRaises(DataLeakageError):
            dense.index(contaminated_corpus)

    def test_bge_backend_availability_and_pending_status(self):
        backend = BGEEmbeddingBackend()
        # In current offline environment without weights on disk, backend must be unavailable
        self.assertFalse(backend.is_available())

        with self.assertRaises(RuntimeError) as ctx:
            backend.embed_texts(["sample text"])
        self.assertIn("unavailable in this offline environment", str(ctx.exception))
        self.assertIn("PENDING", str(ctx.exception))

    def test_mock_dense_retrieval(self):
        backend = MockDenseEmbeddingBackend(dimension=384)
        retriever = DenseRetriever(backend=backend)
        retriever.index(self.train_corpus)

        res = retriever.retrieve("orders by customer", k=3)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].rank, 1)
        self.assertEqual(res[1].rank, 2)
        self.assertEqual(res[2].rank, 3)

    def test_hybrid_rrf_ranking_and_metadata(self):
        bm25 = BM25Retriever()
        mock_dense = DenseRetriever(backend=MockDenseEmbeddingBackend())

        hybrid = HybridRetriever(bm25=bm25, dense=mock_dense, rrf_k=60)
        hybrid.index(self.train_corpus)

        results = hybrid.retrieve("average GPA of students", k=3)
        self.assertEqual(len(results), 3)

        top_item = results[0]
        self.assertIn("rrf_score", top_item.metadata)
        self.assertIn("bm25_rank", top_item.metadata)
        self.assertIn("dense_rank", top_item.metadata)
        self.assertEqual(top_item.metadata["retriever"], "hybrid")

    def test_retrieval_metrics(self):
        retrieved_ids = ["ex_1", "ex_2", "ex_3", "ex_4", "ex_5"]
        gold_relevant = {"ex_1", "ex_4"}

        # Recall@K
        self.assertAlmostEqual(compute_recall_at_k(retrieved_ids, gold_relevant, k=1), 0.5)
        self.assertAlmostEqual(compute_recall_at_k(retrieved_ids, gold_relevant, k=4), 1.0)
        self.assertAlmostEqual(compute_recall_at_k(retrieved_ids, set(), k=3), 0.0)

        # MRR
        self.assertAlmostEqual(compute_mrr(retrieved_ids, gold_relevant, k=5), 1.0)  # first at rank 1
        self.assertAlmostEqual(compute_mrr(retrieved_ids, {"ex_2"}, k=5), 0.5)  # first at rank 2
        self.assertAlmostEqual(compute_mrr(retrieved_ids, {"missing"}, k=5), 0.0)

        # Hit Rate
        self.assertAlmostEqual(compute_hit_rate(retrieved_ids, {"ex_3"}, k=3), 1.0)
        self.assertAlmostEqual(compute_hit_rate(retrieved_ids, {"ex_5"}, k=3), 0.0)

        # Batch evaluation
        batch_retrievals = {
            "q1": [RetrievedExample("ex_1", "m", "q", "s", "train", 1.0, 1)],
            "q2": [RetrievedExample("ex_3", "m", "q", "s", "train", 1.0, 1)],
        }
        batch_gold = {
            "q1": {"ex_1"},
            "q2": {"ex_2"},  # Miss
        }
        batch_summary = evaluate_retrieval_batch(batch_retrievals, batch_gold, k_values=[1])
        self.assertEqual(batch_summary["K=1"]["recall"], 0.5)
        self.assertEqual(batch_summary["K=1"]["mrr"], 0.5)
        self.assertEqual(batch_summary["K=1"]["hit_rate"], 0.5)

    def test_rag_prompt_construction(self):
        bm25 = BM25Retriever()
        bm25.index(self.train_corpus)

        question = "What is the average GPA of students?"
        prompt, retrieved = build_rag_prompt(question, self.schema, retriever=bm25, k=3)

        self.assertEqual(len(retrieved), 3)
        self.assertIn("### Demonstration Examples:", prompt)
        self.assertIn("Example 1:", prompt)
        self.assertIn("Example 2:", prompt)
        self.assertIn("Example 3:", prompt)
        self.assertIn("### Target Database Schema:", prompt)
        self.assertIn("CREATE TABLE student", prompt)
        self.assertIn("### Target Question:", prompt)
        self.assertIn(question, prompt)

    def test_end_to_end_rag_mock_pipeline(self):
        bm25 = BM25Retriever()
        bm25.index(self.train_corpus)

        eval_q = "What is the average GPA of all students?"
        gold_sql = "SELECT avg(gpa) FROM student"

        # 1. RAG Prompt construction
        prompt, retrieved = build_rag_prompt(eval_q, self.schema, retriever=bm25, k=2)
        self.assertEqual(len(retrieved), 2)

        # 2. Mock Generator (Gold mode)
        generator = MockSQLGenerator(mode="gold", reference_map={eval_q: gold_sql})
        predicted_sql = generator.generate(prompt, eval_q, self.schema)

        # 3. Phase 3 Evaluator
        res = evaluate_example("rag_pipe_1", "mock_university", predicted_sql, gold_sql, self.u_db)
        self.assertTrue(res.sql_valid)
        self.assertTrue(res.exact_match)
        self.assertTrue(res.execution_accuracy)
        self.assertEqual(res.error_type, "CORRECT")


if __name__ == "__main__":
    unittest.main()
