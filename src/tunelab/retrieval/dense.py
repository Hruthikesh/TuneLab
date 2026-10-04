from abc import ABC, abstractmethod
import hashlib
from pathlib import Path
from typing import List, Optional
import numpy as np

from tunelab.data.loader import SpiderExample
from tunelab.retrieval.base import BaseRetriever, RetrievedExample


class DenseEmbeddingBackend(ABC):
    """Abstract interface for dense sentence embedding backends."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the model weights and deep-learning runtime are available."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of texts into an L2-normalized 2D numpy array (N, D)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a search query into an L2-normalized 1D numpy array (D,)."""
        pass


class BGEEmbeddingBackend(DenseEmbeddingBackend):
    """Real BAAI/bge-small-en-v1.5 dense embedding backend.
    
    Status in offline sandbox: PENDING (weights not cached locally, network is offline).
    """

    MODEL_ID = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM = 384
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, local_model_path: Optional[str | Path] = None):
        self.local_model_path = Path(local_model_path) if local_model_path else None
        self._model = None
        self._tokenizer = None

    @property
    def dimension(self) -> int:
        return self.EMBEDDING_DIM

    def is_available(self) -> bool:
        """Checks if transformers/torch and local model weights are actually present."""
        try:
            import torch
            import transformers
        except ImportError:
            return False

        if self.local_model_path and self.local_model_path.is_dir():
            return True
        return False

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError(
                f"Model weights for '{self.MODEL_ID}' are unavailable in this offline environment. "
                "Dense embedding execution is PENDING until run in a GPU/connected environment."
            )
        raise NotImplementedError("Real BGE forward pass is pending GPU environment execution.")

    def embed_query(self, query: str) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError(
                f"Model weights for '{self.MODEL_ID}' are unavailable in this offline environment. "
                "Dense embedding execution is PENDING until run in a GPU/connected environment."
            )
        raise NotImplementedError("Real BGE forward pass is pending GPU environment execution.")


class MockDenseEmbeddingBackend(DenseEmbeddingBackend):
    """TEST FIXTURE ONLY: Deterministic pseudo-semantic embedding backend for unit testing.
    
    WARNING: This backend produces synthetic vectors for testing retrieval logic and cosine math.
    It MUST NEVER be used to report benchmark results or research numbers.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def is_available(self) -> bool:
        return True

    def _hash_text_to_vector(self, text: str) -> np.ndarray:
        """Deterministically hashes text tokens to a unit-normalized vector."""
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec

        for word in words:
            # Deterministic MD5 hash mapped into vector indices
            h = hashlib.md5(word.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self._dim
            sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = [self._hash_text_to_vector(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._hash_text_to_vector(query).astype(np.float32)


class DenseRetriever(BaseRetriever):
    """Dense Semantic Retriever using vector embeddings and cosine similarity."""

    def __init__(self, backend: DenseEmbeddingBackend):
        super().__init__()
        self.backend = backend
        self.embeddings: Optional[np.ndarray] = None

    def index(self, examples: List[SpiderExample]) -> None:
        """Indexes training examples by embedding their questions."""
        self.indexed_examples = self.validate_index_pool(examples)
        if not self.backend.is_available():
            raise RuntimeError(
                "Cannot construct dense index: embedding backend is unavailable in this environment."
            )

        texts = [ex.question for ex in self.indexed_examples]
        self.embeddings = self.backend.embed_texts(texts)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedExample]:
        """Retrieves top-K demonstrations using cosine similarity."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if self.embeddings is None or len(self.indexed_examples) == 0:
            return []

        query_vec = self.backend.embed_query(query)
        # Cosine similarity via dot product (both vectors are L2-normalized)
        scores = np.dot(self.embeddings, query_vec).tolist()

        candidates = []
        for ex, score in zip(self.indexed_examples, scores):
            candidates.append((score, ex.example_id, ex))

        # Deterministic sorting: score DESC, example_id ASC
        candidates.sort(key=lambda item: (-item[0], item[1]))
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
                    metadata={"retriever": "dense"},
                )
            )

        return results
