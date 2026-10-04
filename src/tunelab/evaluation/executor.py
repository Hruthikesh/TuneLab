from dataclasses import dataclass
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, List, Optional, Tuple

DESTRUCTIVE_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
    "ATTACH", "DETACH", "PRAGMA", "REPLACE", "TRUNCATE", "VACUUM"
}

DESTRUCTIVE_PATTERN = re.compile(
    r"\b(" + "|".join(DESTRUCTIVE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


class SQLExecutionTimeoutError(Exception):
    """Raised when query execution exceeds the configured timeout."""
    pass


class UnsafeSQLStatementError(Exception):
    """Raised when generated SQL contains non-SELECT or destructive operations."""
    pass


@dataclass
class ExecutionOutput:
    success: bool
    rows: List[Tuple[Any, ...]]
    latency_ms: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def check_sql_safety(sql: str) -> None:
    """Verifies that the SQL string does not attempt destructive/write operations."""
    stripped = sql.strip()
    if not stripped:
        raise UnsafeSQLStatementError("SQL query is empty.")

    # Match forbidden destructive operations
    match = DESTRUCTIVE_PATTERN.search(stripped)
    if match:
        raise UnsafeSQLStatementError(
            f"Query contains forbidden keyword '{match.group(0)}'. Only read-only queries are permitted."
        )


def execute_sqlite_query(
    sqlite_path: Path | str,
    sql: str,
    timeout_seconds: float = 5.0,
) -> ExecutionOutput:
    """Executes a SQL query on a SQLite database with safety and timeout guards."""
    path = Path(sqlite_path)
    if not path.is_file():
        return ExecutionOutput(
            success=False,
            rows=[],
            latency_ms=0.0,
            error_type="DATABASE_NOT_FOUND",
            error_message=f"Database file does not exist: {path}",
        )

    # 1. Safety verification
    try:
        check_sql_safety(sql)
    except UnsafeSQLStatementError as e:
        return ExecutionOutput(
            success=False,
            rows=[],
            latency_ms=0.0,
            error_type="INVALID_STATEMENT",
            error_message=str(e),
        )

    # 2. Connect in read-only mode
    uri = f"file:{path.resolve()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(str(path.resolve()), timeout=timeout_seconds)

    start_time = time.perf_counter()
    timed_out = False

    def progress_handler() -> int:
        nonlocal timed_out
        if (time.perf_counter() - start_time) > timeout_seconds:
            timed_out = True
            return 1  # Aborts SQLite execution immediately
        return 0

    # Check timeout every 500 instructions
    conn.set_progress_handler(progress_handler, 500)

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        raw_rows = cursor.fetchall()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Normalize tuple values for consistent comparison
        normalized_rows = [_normalize_row(row) for row in raw_rows]
        return ExecutionOutput(
            success=True,
            rows=normalized_rows,
            latency_ms=elapsed_ms,
        )

    except sqlite3.OperationalError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_str = str(e).lower()
        if timed_out or "interrupted" in err_str:
            return ExecutionOutput(
                success=False,
                rows=[],
                latency_ms=elapsed_ms,
                error_type="TIMEOUT",
                error_message=f"Query exceeded execution timeout of {timeout_seconds}s",
            )
        if "no such table" in err_str:
            return ExecutionOutput(
                success=False,
                rows=[],
                latency_ms=elapsed_ms,
                error_type="MISSING_TABLE",
                error_message=str(e),
            )
        if "no such column" in err_str:
            return ExecutionOutput(
                success=False,
                rows=[],
                latency_ms=elapsed_ms,
                error_type="MISSING_COLUMN",
                error_message=str(e),
            )
        if "syntax error" in err_str or "incomplete input" in err_str or "near " in err_str:
            return ExecutionOutput(
                success=False,
                rows=[],
                latency_ms=elapsed_ms,
                error_type="SYNTAX_ERROR",
                error_message=str(e),
            )
        return ExecutionOutput(
            success=False,
            rows=[],
            latency_ms=elapsed_ms,
            error_type="EXECUTION_ERROR",
            error_message=str(e),
        )

    except sqlite3.Error as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return ExecutionOutput(
            success=False,
            rows=[],
            latency_ms=elapsed_ms,
            error_type="EXECUTION_ERROR",
            error_message=str(e),
        )

    finally:
        conn.close()


def _normalize_row(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Normalizes elements of a row tuple."""
    normalized = []
    for item in row:
        if item is None:
            normalized.append(None)
        elif isinstance(item, float):
            normalized.append(item)
        elif isinstance(item, int):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(item.strip())
        else:
            normalized.append(str(item))
    return tuple(normalized)


def _compare_values(val1: Any, val2: Any, float_tol: float = 1e-4) -> bool:
    """Compares two individual cell values with NULL handling and floating point tolerance."""
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False

    is_num1 = isinstance(val1, (int, float))
    is_num2 = isinstance(val2, (int, float))

    if is_num1 and is_num2:
        return math.isclose(float(val1), float(val2), rel_tol=float_tol, abs_tol=1e-5)

    return str(val1).strip().lower() == str(val2).strip().lower()


def _compare_rows(row1: Tuple[Any, ...], row2: Tuple[Any, ...], float_tol: float = 1e-4) -> bool:
    """Compares two row tuples element by element."""
    if len(row1) != len(row2):
        return False
    return all(_compare_values(v1, v2, float_tol) for v1, v2 in zip(row1, row2))


def compare_results(
    pred_rows: List[Tuple[Any, ...]],
    gold_rows: List[Tuple[Any, ...]],
    order_by_required: bool,
    float_tolerance: float = 1e-4,
) -> bool:
    """Compares candidate and reference execution results."""
    if len(pred_rows) != len(gold_rows):
        return False

    if len(pred_rows) == 0:
        return True

    if order_by_required:
        for r_pred, r_gold in zip(pred_rows, gold_rows):
            if not _compare_rows(r_pred, r_gold, float_tolerance):
                return False
        return True

    unmatched_gold = list(gold_rows)
    for r_pred in pred_rows:
        matched_idx = -1
        for idx, r_gold in enumerate(unmatched_gold):
            if _compare_rows(r_pred, r_gold, float_tolerance):
                matched_idx = idx
                break
        if matched_idx == -1:
            return False
        unmatched_gold.pop(matched_idx)

    return len(unmatched_gold) == 0


def reference_requires_order_by(reference_sql: str) -> bool:
    """Detects whether the reference query specifies an explicit ORDER BY."""
    return bool(ORDER_BY_PATTERN.search(reference_sql))
