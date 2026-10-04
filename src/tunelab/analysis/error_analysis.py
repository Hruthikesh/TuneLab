import collections
import re
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from tunelab.analysis.models import ErrorAnalysisRecord, ExperimentRun, PerExampleResult

COARSE_CATEGORIES = [
    "correct",
    "syntax_error",
    "missing_table",
    "missing_column",
    "timeout",
    "invalid_statement",
    "execution_error",
    "result_mismatch",
]

FINE_CATEGORIES = [
    "none",
    "syntax_error",
    "missing_table",
    "missing_column",
    "timeout",
    "invalid_statement",
    "execution_error",
    "wrong_table",
    "wrong_column",
    "wrong_join",
    "wrong_filter",
    "wrong_aggregation",
    "wrong_group_by",
    "wrong_ordering",
    "nested_query",
    "schema_misunderstanding",
    "UNCLASSIFIED_RESULT_MISMATCH",
]


def _extract_sql_features(sql: str) -> Dict[str, Any]:
    """Extracts deterministic syntactic features from an SQL query."""
    cleaned = re.sub(r"\s+", " ", sql.strip())
    tokens = [t.upper() for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", cleaned)]
    token_set = set(tokens)

    # Tables following FROM or JOIN
    from_matches = re.findall(r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", cleaned, re.IGNORECASE)
    join_matches = re.findall(r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)", cleaned, re.IGNORECASE)
    tables = set([t.lower() for t in from_matches + join_matches])

    # Projection columns
    proj_match = re.search(r"\bSELECT\s+(.+?)\s+\bFROM\b", cleaned, re.IGNORECASE)
    proj_cols = set()
    if proj_match:
        raw_cols = proj_match.group(1).split(",")
        proj_cols = set([c.strip().lower() for c in raw_cols])

    # Aggregations
    aggs = set([t for t in tokens if t in {"COUNT", "SUM", "AVG", "MIN", "MAX"}])

    # Join keywords
    has_join = "JOIN" in token_set

    # Group by
    has_group = "GROUP" in token_set and "BY" in token_set

    # Order by
    has_order = "ORDER" in token_set and "BY" in token_set
    has_desc = "DESC" in token_set

    # Nested queries (multiple SELECTs)
    select_count = tokens.count("SELECT")

    # Where clause presence
    has_where = "WHERE" in token_set

    return {
        "tables": tables,
        "proj_cols": proj_cols,
        "aggs": aggs,
        "has_join": has_join,
        "has_group": has_group,
        "has_order": has_order,
        "has_desc": has_desc,
        "select_count": select_count,
        "has_where": has_where,
        "tokens": tokens,
        "token_set": token_set,
    }


def classify_error(result: PerExampleResult) -> ErrorAnalysisRecord:
    """Classifies an example result into coarse and fine research error categories."""
    if result.execution_match:
        return ErrorAnalysisRecord(
            example_id=result.example_id,
            database_id=result.database_id,
            coarse_error="correct",
            fine_error="none",
            explanation="Execution match confirmed.",
        )

    coarse = result.error_type.lower() if result.error_type else "execution_error"
    if coarse not in COARSE_CATEGORIES:
        coarse = "execution_error"

    # Coarse errors with execution failure map directly
    if coarse in ("syntax_error", "missing_table", "missing_column", "timeout", "invalid_statement"):
        return ErrorAnalysisRecord(
            example_id=result.example_id,
            database_id=result.database_id,
            coarse_error=coarse,
            fine_error=coarse,
            explanation=result.error_message or f"Evaluator reported {coarse}",
        )

    # If result_mismatch, inspect query structures deterministically
    pred = result.predicted_sql or ""
    gold = result.reference_sql or ""

    if not pred.strip():
        return ErrorAnalysisRecord(
            example_id=result.example_id,
            database_id=result.database_id,
            coarse_error="invalid_statement",
            fine_error="invalid_statement",
            explanation="Empty prediction returned.",
        )

    try:
        p_feat = _extract_sql_features(pred)
        g_feat = _extract_sql_features(gold)

        # 1. Nested query divergence
        if (p_feat["select_count"] > 1) != (g_feat["select_count"] > 1):
            return ErrorAnalysisRecord(
                example_id=result.example_id,
                database_id=result.database_id,
                coarse_error="result_mismatch",
                fine_error="nested_query",
                explanation="Divergence in subquery/nested SELECT structure.",
            )

        # 2. Join divergence
        if p_feat["has_join"] != g_feat["has_join"]:
            return ErrorAnalysisRecord(
                example_id=result.example_id,
                database_id=result.database_id,
                coarse_error="result_mismatch",
                fine_error="wrong_join",
                explanation="Predicted omitted or improperly introduced a JOIN clause.",
            )

        # 3. Table divergence
        if p_feat["tables"] and g_feat["tables"] and p_feat["tables"] != g_feat["tables"]:
            return ErrorAnalysisRecord(
                example_id=result.example_id,
                database_id=result.database_id,
                coarse_error="result_mismatch",
                fine_error="wrong_table",
                explanation=f"Predicted queried {p_feat['tables']}, reference queried {g_feat['tables']}",
            )

        # 4. Aggregation divergence
        if p_feat["aggs"] != g_feat["aggs"]:
            return ErrorAnalysisRecord(
                example_id=result.example_id,
                database_id=result.database_id,
                coarse_error="result_mismatch",
                fine_error="wrong_aggregation",
                explanation=f"Aggregation mismatch: predicted {p_feat['aggs']}, reference {g_feat['aggs']}",
            )

        # 5. Group by divergence
        if p_feat["has_group"] != g_feat["has_group"]:
            return ErrorAnalysisRecord(
                example_id=result.example_id,
                database_id=result.database_id,
                coarse_error="result_mismatch",
                fine_error="wrong_group_by",
                explanation="GROUP BY clause mismatch between predicted and reference.",
            )

        # 6. Order by / Direction divergence
        if p_feat["has_order"] != g_feat["has_order"] or p_feat["has_desc"] != g_feat["has_desc"]:
            return ErrorAnalysisRecord(
                example_id=result.example_id,
                database_id=result.database_id,
                coarse_error="result_mismatch",
                fine_error="wrong_ordering",
                explanation="ORDER BY clause or sort direction (ASC/DESC) mismatch.",
            )

        # 7. Column projection divergence
        if p_feat["proj_cols"] and g_feat["proj_cols"] and p_feat["proj_cols"] != g_feat["proj_cols"]:
            return ErrorAnalysisRecord(
                example_id=result.example_id,
                database_id=result.database_id,
                coarse_error="result_mismatch",
                fine_error="wrong_column",
                explanation=f"Column mismatch: predicted {p_feat['proj_cols']}, reference {g_feat['proj_cols']}",
            )

        # 8. Where clause / Filter divergence
        if p_feat["has_where"] != g_feat["has_where"]:
            return ErrorAnalysisRecord(
                example_id=result.example_id,
                database_id=result.database_id,
                coarse_error="result_mismatch",
                fine_error="wrong_filter",
                explanation="WHERE filter presence mismatch.",
            )

        # Default fallback to UNCLASSIFIED_RESULT_MISMATCH
        return ErrorAnalysisRecord(
            example_id=result.example_id,
            database_id=result.database_id,
            coarse_error="result_mismatch",
            fine_error="UNCLASSIFIED_RESULT_MISMATCH",
            explanation="Result set mismatch without single unambiguous structural discriminator.",
        )
    except Exception as e:
        return ErrorAnalysisRecord(
            example_id=result.example_id,
            database_id=result.database_id,
            coarse_error="result_mismatch",
            fine_error="UNCLASSIFIED_RESULT_MISMATCH",
            explanation=f"Fallback unclassified mismatch: {e}",
        )


def analyze_errors_for_run(run: ExperimentRun) -> List[ErrorAnalysisRecord]:
    """Generates error analysis records for all examples in a run."""
    if not run.per_example_results:
        return []
    return [classify_error(r) for r in run.per_example_results]


def compute_error_distribution(
    runs: List[ExperimentRun],
    group_by: str = "method",
    use_fine_categories: bool = True,
    include_smoke: bool = False,
) -> pd.DataFrame:
    """Computes error counts and percentage distribution by method or experimental condition."""
    target_runs = runs if include_smoke else [r for r in runs if r.is_research_result]

    records = []
    for run in target_runs:
        if not run.per_example_results:
            continue
        analyzed = analyze_errors_for_run(run)
        group_val = getattr(run, group_by, run.method)
        total = len(analyzed)
        if total == 0:
            continue

        counts: Dict[str, int] = collections.defaultdict(int)
        for rec in analyzed:
            cat = rec.fine_error if use_fine_categories else rec.coarse_error
            counts[cat] += 1

        for cat, cnt in counts.items():
            pct = round((cnt / total) * 100.0, 2)
            records.append({
                group_by: group_val,
                "error_type": cat,
                "count": cnt,
                "total_examples": total,
                "percentage": pct,
            })

    if not records:
        return pd.DataFrame(columns=[group_by, "error_type", "count", "total_examples", "percentage"])

    df = pd.DataFrame(records)
    return df.sort_values(by=[group_by, "count"], ascending=[True, False]).reset_index(drop=True)
