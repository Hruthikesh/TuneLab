import re


def canonicalize_sql(sql: str) -> str:
    """Conservatively canonicalizes a SQL string for Exact Match comparison.
    
    Operations performed:
    1. Strip leading/trailing whitespace.
    2. Remove single-line comments (-- ...) and block comments (/* ... */).
    3. Remove trailing semicolons.
    4. Collapse all consecutive whitespace characters (newlines, tabs, spaces) into a single space.
    5. Convert to lowercase.
    """
    if not sql or not isinstance(sql, str):
        return ""

    # Strip block comments
    cleaned = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # Strip line comments
    cleaned = re.sub(r"--[^\n]*", " ", cleaned)
    # Strip trailing semicolons and whitespace
    cleaned = cleaned.strip().rstrip(";").strip()
    # Collapse multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Lowercase
    return cleaned.lower()


def compute_exact_match(pred_sql: str, gold_sql: str) -> bool:
    """Computes exact match between predicted and gold SQL after canonicalization."""
    return canonicalize_sql(pred_sql) == canonicalize_sql(gold_sql)
