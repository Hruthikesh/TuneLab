from dataclasses import dataclass
from typing import List, Set
from tunelab.data.loader import SpiderExample


class DataLeakageError(Exception):
    """Raised when data leakage or cross-split contamination is detected."""
    pass


@dataclass
class LeakageCheckReport:
    passed: bool
    train_db_count: int
    eval_db_count: int
    db_id_overlap: List[str]
    question_exact_matches: List[str]
    query_exact_matches: List[str]
    pair_exact_matches: List[str]

    def summary(self) -> str:
        lines = [
            f"Leakage Check: {'PASSED' if self.passed else 'FAILED'}",
            f"Train DBs: {self.train_db_count}, Eval DBs: {self.eval_db_count}",
            f"Overlapping DB IDs: {len(self.db_id_overlap)}",
            f"Identical Question-SQL Pairs: {len(self.pair_exact_matches)}",
            f"Identical Questions: {len(self.question_exact_matches)}",
        ]
        return "\n".join(lines)


def verify_split_integrity(
    train_examples: List[SpiderExample],
    eval_examples: List[SpiderExample],
    allow_empty: bool = False,
) -> LeakageCheckReport:
    """Rigorous leakage verification between training and evaluation splits."""
    if not train_examples or not eval_examples:
        if allow_empty:
            return LeakageCheckReport(
                passed=True,
                train_db_count=len({ex.db_id for ex in train_examples}),
                eval_db_count=len({ex.db_id for ex in eval_examples}),
                db_id_overlap=[],
                question_exact_matches=[],
                query_exact_matches=[],
                pair_exact_matches=[],
            )
        raise ValueError("Cannot perform leakage check on empty split lists.")

    train_dbs: Set[str] = {ex.db_id for ex in train_examples}
    eval_dbs: Set[str] = {ex.db_id for ex in eval_examples}

    # 1. Database ID Overlap Check
    db_overlap = sorted(list(train_dbs.intersection(eval_dbs)))

    # 2. Exact Question-SQL Pair Check
    train_pairs: Set[tuple] = {(ex.db_id, ex.question.strip().lower(), ex.query.strip().lower()) for ex in train_examples}
    pair_matches = []
    for ex in eval_examples:
        key = (ex.db_id, ex.question.strip().lower(), ex.query.strip().lower())
        if key in train_pairs:
            pair_matches.append(f"DB '{ex.db_id}' | Q: '{ex.question}' | SQL: '{ex.query}'")

    # 3. Exact Question Matching across splits
    train_questions: Set[tuple] = {(ex.db_id, ex.question.strip().lower()) for ex in train_examples}
    question_matches = []
    for ex in eval_examples:
        key = (ex.db_id, ex.question.strip().lower())
        if key in train_questions:
            question_matches.append(f"DB '{ex.db_id}' | Q: '{ex.question}'")

    passed = len(db_overlap) == 0 and len(pair_matches) == 0 and len(question_matches) == 0

    report = LeakageCheckReport(
        passed=passed,
        train_db_count=len(train_dbs),
        eval_db_count=len(eval_dbs),
        db_id_overlap=db_overlap,
        question_exact_matches=question_matches,
        query_exact_matches=[],
        pair_exact_matches=pair_matches,
    )

    if not passed:
        err_msg = [
            "DATA LEAKAGE DETECTED across train and evaluation splits!",
            f"Overlapping Database IDs ({len(db_overlap)}): {db_overlap}",
            f"Overlapping Question-SQL Pairs ({len(pair_matches)})",
            f"Overlapping Questions ({len(question_matches)})",
        ]
        raise DataLeakageError("\n".join(err_msg))

    return report
