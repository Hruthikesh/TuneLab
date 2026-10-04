from typing import Dict, List, Optional

from tunelab.data.loader import SpiderExample
from tunelab.retrieval.base import BaseRetriever, RetrievedExample
from tunelab.retrieval.bm25 import BM25Retriever
from tunelab.retrieval.dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid Retriever combining BM25 and Dense Retrieval via Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_Score(d) = 1 / (rrf_k + rank_bm25(d)) + 1 / (rrf_k + rank_dense(d))
    
    Preserves independently inspectable rankings and scores for BM25 and Dense.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        super().__init__()
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def index(self, examples: List[SpiderExample]) -> None:
        """Indexes examples in both BM25 and Dense sub-retrievers."""
        self.indexed_examples = self.validate_index_pool(examples)
        self.bm25.index(self.indexed_examples)
        if self.dense.backend.is_available():
            self.dense.index(self.indexed_examples)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedExample]:
        """Retrieves top-K demonstrations using Reciprocal Rank Fusion."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if not self.dense.backend.is_available():
            raise RuntimeError(
                "Cannot perform hybrid retrieval: dense embedding backend is unavailable in this environment."
            )

        total_corpus = len(self.indexed_examples)
        if total_corpus == 0:
            return []

        # Retrieve full candidate rankings from both methods
        bm25_results = self.bm25.retrieve(query, k=total_corpus)
        dense_results = self.dense.retrieve(query, k=total_corpus)

        bm25_map: Dict[str, RetrievedExample] = {r.example_id: r for r in bm25_results}
        dense_map: Dict[str, RetrievedExample] = {r.example_id: r for r in dense_results}

        example_map: Dict[str, SpiderExample] = {ex.example_id: ex for ex in self.indexed_examples}

        # Compute RRF score for all candidates
        rrf_scores: Dict[str, float] = {}
        for ex_id in example_map.keys():
            b_item = bm25_map.get(ex_id)
            d_item = dense_map.get(ex_id)

            score = 0.0
            if b_item:
                score += 1.0 / (self.rrf_k + b_item.rank)
            if d_item:
                score += 1.0 / (self.rrf_k + d_item.rank)
            rrf_scores[ex_id] = score

        # Sort deterministically: rrf_score DESC, example_id ASC
        sorted_ids = sorted(
            example_map.keys(),
            key=lambda eid: (-rrf_scores[eid], eid),
        )

        top_k_ids = sorted_ids[:k]

        results = []
        for rank, ex_id in enumerate(top_k_ids, start=1):
            ex = example_map[ex_id]
            b_item = bm25_map.get(ex_id)
            d_item = dense_map.get(ex_id)

            metadata = {
                "retriever": "hybrid",
                "rrf_k": self.rrf_k,
                "bm25_rank": b_item.rank if b_item else None,
                "bm25_score": round(b_item.score, 4) if b_item else None,
                "dense_rank": d_item.rank if d_item else None,
                "dense_score": round(d_item.score, 4) if d_item else None,
                "rrf_score": round(rrf_scores[ex_id], 6),
            }

            results.append(
                RetrievedExample(
                    example_id=ex.example_id,
                    db_id=ex.db_id,
                    question=ex.question,
                    query=ex.query,
                    split=ex.split,
                    score=rrf_scores[ex_id],
                    rank=rank,
                    metadata=metadata,
                )
            )

        return results
