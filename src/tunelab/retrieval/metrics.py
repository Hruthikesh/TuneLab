from typing import Dict, List, Optional, Set

from tunelab.retrieval.base import RetrievedExample


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Computes Recall@K: fraction of gold relevant examples retrieved in top-K."""
    if not relevant_ids:
        return 0.0
    top_k_ids = set(retrieved_ids[:k])
    hits = top_k_ids.intersection(relevant_ids)
    return len(hits) / len(relevant_ids)


def compute_mrr(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Computes Mean Reciprocal Rank at cutoff K."""
    if not relevant_ids:
        return 0.0
    for rank, ex_id in enumerate(retrieved_ids[:k], start=1):
        if ex_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_hit_rate(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Computes Hit Rate at cutoff K: 1.0 if at least one relevant item is in top-K, else 0.0."""
    if not relevant_ids:
        return 0.0
    top_k_ids = set(retrieved_ids[:k])
    return 1.0 if bool(top_k_ids.intersection(relevant_ids)) else 0.0


def evaluate_retrieval_batch(
    retrieval_results: Dict[str, List[RetrievedExample]],
    ground_truth_relevance: Dict[str, Set[str]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, Dict[str, float]]:
    """Evaluates retrieval quality across a batch of queries for multiple K values.
    
    Returns:
        Dict mapping K (e.g. 'K=1') to dict of mean metrics: 'recall', 'mrr', 'hit_rate'.
    """
    total_queries = len(retrieval_results)
    if total_queries == 0:
        return {}

    summary: Dict[str, Dict[str, float]] = {}

    for k in k_values:
        total_recall = 0.0
        total_mrr = 0.0
        total_hits = 0.0
        evaluated_count = 0

        for q_id, retrieved in retrieval_results.items():
            relevant_ids = ground_truth_relevance.get(q_id, set())
            if not relevant_ids:
                continue

            r_ids = [r.example_id for r in retrieved]
            total_recall += compute_recall_at_k(r_ids, relevant_ids, k)
            total_mrr += compute_mrr(r_ids, relevant_ids, k)
            total_hits += compute_hit_rate(r_ids, relevant_ids, k)
            evaluated_count += 1

        if evaluated_count > 0:
            summary[f"K={k}"] = {
                "recall": round(total_recall / evaluated_count, 4),
                "mrr": round(total_mrr / evaluated_count, 4),
                "hit_rate": round(total_hits / evaluated_count, 4),
                "num_queries_evaluated": evaluated_count,
            }
        else:
            summary[f"K={k}"] = {
                "recall": 0.0,
                "mrr": 0.0,
                "hit_rate": 0.0,
                "num_queries_evaluated": 0,
            }

    return summary
