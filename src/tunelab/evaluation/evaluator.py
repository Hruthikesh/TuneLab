from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from tunelab.evaluation.canonicalize import compute_exact_match
from tunelab.evaluation.executor import (
    compare_results,
    execute_sqlite_query,
    reference_requires_order_by,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    example_id: str
    database_id: str
    predicted_sql: str
    reference_sql: str
    sql_valid: bool
    exact_match: bool
    execution_accuracy: bool
    latency_ms: float
    error_type: str
    error_message: Optional[str] = None
    retrieval_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "example_id": self.example_id,
            "database_id": self.database_id,
            "predicted_sql": self.predicted_sql,
            "reference_sql": self.reference_sql,
            "sql_valid": self.sql_valid,
            "exact_match": self.exact_match,
            "execution_accuracy": self.execution_accuracy,
            "latency_ms": round(self.latency_ms, 2),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
        if self.retrieval_metadata is not None:
            d["retrieval_metadata"] = self.retrieval_metadata
        return d


def evaluate_example(
    example_id: str,
    database_id: str,
    predicted_sql: str,
    reference_sql: str,
    sqlite_path: Path | str,
    timeout_seconds: float = 5.0,
    float_tolerance: float = 1e-4,
    retrieval_metadata: Optional[Dict[str, Any]] = None,
) -> EvaluationResult:
    """Evaluates a single predicted SQL against reference SQL on target SQLite database."""
    # 1. Exact Match computation
    is_exact_match = compute_exact_match(predicted_sql, reference_sql)

    # 2. Execute Gold SQL
    gold_out = execute_sqlite_query(sqlite_path, reference_sql, timeout_seconds=timeout_seconds)
    if not gold_out.success:
        return EvaluationResult(
            example_id=example_id,
            database_id=database_id,
            predicted_sql=predicted_sql,
            reference_sql=reference_sql,
            sql_valid=False,
            exact_match=is_exact_match,
            execution_accuracy=False,
            latency_ms=gold_out.latency_ms,
            error_type="REFERENCE_SQL_ERROR",
            error_message=f"Gold reference SQL failed execution: {gold_out.error_message}",
            retrieval_metadata=retrieval_metadata,
        )

    # 3. Execute Predicted SQL
    pred_out = execute_sqlite_query(sqlite_path, predicted_sql, timeout_seconds=timeout_seconds)
    if not pred_out.success:
        return EvaluationResult(
            example_id=example_id,
            database_id=database_id,
            predicted_sql=predicted_sql,
            reference_sql=reference_sql,
            sql_valid=False,
            exact_match=is_exact_match,
            execution_accuracy=False,
            latency_ms=pred_out.latency_ms,
            error_type=pred_out.error_type or "EXECUTION_ERROR",
            error_message=pred_out.error_message,
            retrieval_metadata=retrieval_metadata,
        )

    # 4. Compare Results
    order_required = reference_requires_order_by(reference_sql)
    exec_acc = compare_results(
        pred_rows=pred_out.rows,
        gold_rows=gold_out.rows,
        order_by_required=order_required,
        float_tolerance=float_tolerance,
    )

    error_type = "CORRECT" if exec_acc else "RESULT_MISMATCH"
    error_msg = None if exec_acc else "Predicted SQL returned different result rows than reference SQL."

    return EvaluationResult(
        example_id=example_id,
        database_id=database_id,
        predicted_sql=predicted_sql,
        reference_sql=reference_sql,
        sql_valid=True,
        exact_match=is_exact_match,
        execution_accuracy=exec_acc,
        latency_ms=pred_out.latency_ms,
        error_type=error_type,
        error_message=error_msg,
        retrieval_metadata=retrieval_metadata,
    )


def compute_summary_metrics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Computes aggregate benchmark metrics from a list of evaluation results."""
    total = len(results)
    if total == 0:
        return {
            "total_samples": 0,
            "sql_validity": 0.0,
            "exact_match": 0.0,
            "execution_accuracy": 0.0,
            "mean_latency_ms": 0.0,
            "error_breakdown": {},
        }

    valid_count = sum(1 for r in results if r.sql_valid)
    em_count = sum(1 for r in results if r.exact_match)
    ex_count = sum(1 for r in results if r.execution_accuracy)
    mean_lat = sum(r.latency_ms for r in results) / total

    error_breakdown: Dict[str, int] = {}
    for r in results:
        error_breakdown[r.error_type] = error_breakdown.get(r.error_type, 0) + 1

    return {
        "total_samples": total,
        "sql_validity": round(valid_count / total, 4),
        "exact_match": round(em_count / total, 4),
        "execution_accuracy": round(ex_count / total, 4),
        "mean_latency_ms": round(mean_lat, 2),
        "error_breakdown": error_breakdown,
    }
