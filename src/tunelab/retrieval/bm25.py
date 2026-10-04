from collections import Counter
import math
import re
from typing import Dict, List, Optional

from tunelab.data.loader import SpiderExample
from tunelab.retrieval.base import BaseRetriever, RetrievedExample


def tokenize(text: str) -> List[str]:
    """Transparent word tokenizer for Okapi BM25."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class BM25Retriever(BaseRetriever):
    """Okapi BM25 Lexical Retriever for Text-to-SQL demonstration selection.
    
    Implements standard Robertson-Spärck Jones Okapi BM25 scoring with deterministic
    lexicographical tie-breaking by example_id.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__()
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_term_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}

    def index(self, examples: List[SpiderExample]) -> None:
        """Indexes the training examples for BM25 retrieval."""
        self.indexed_examples = self.validate_index_pool(examples)
        self.corpus_size = len(self.indexed_examples)

        tokenized_corpus = [tokenize(ex.question) for ex in self.indexed_examples]
        self.doc_lens = [len(tokens) for tokens in tokenized_corpus]
        total_tokens = sum(self.doc_lens)
        self.avg_doc_len = (total_tokens / self.corpus_size) if self.corpus_size > 0 else 0.0
        self.doc_term_freqs = [Counter(tokens) for tokens in tokenized_corpus]

        # Calculate document frequency n(w)
        doc_freqs: Counter = Counter()
        for tf in self.doc_term_freqs:
            for term in tf.keys():
                doc_freqs[term] += 1

        # Calculate Robertson-Spärck Jones IDF with smoothing (+1.0 ensures non-negative)
        self.idf = {}
        for term, df in doc_freqs.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedExample]:
        """Retrieves top-K training demonstrations matching the query question."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if self.corpus_size == 0 or not self.indexed_examples:
            return []

        query_tokens = tokenize(query)
        scores: List[float] = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            for doc_idx, tf in enumerate(self.doc_term_freqs):
                count = tf.get(term, 0)
                if count > 0:
                    denom = count + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_idx] / self.avg_doc_len))
                    term_score = idf_val * (count * (self.k1 + 1.0)) / denom
                    scores[doc_idx] += term_score

        # Combine into candidates for sorting
        candidates = []
        for idx, (ex, score) in enumerate(zip(self.indexed_examples, scores)):
            candidates.append((score, ex.example_id, ex))

        # Deterministic sorting: primary score DESC, secondary example_id ASC
        candidates.sort(key=lambda item: (-item[0], item[1]))

        # Slice top-K
        top_k_candidates = candidates[:k]

        results = []
        for rank, (score, _, ex) in enumerate(top_k_candidates, start=1):
            results.append(
                RetrievedExample(
                    example_id=ex.example_id,
                    db_id=ex.db_id,
                    question=ex.question,
                    query=ex.query,
                    split=ex.split,
                    score=score,
                    rank=rank,
                    metadata={"retriever": "bm25"},
                )
            )

        return results
