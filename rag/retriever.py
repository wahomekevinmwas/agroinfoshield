"""
rag/retriever.py
----------------
Responsibility: Take a user query, embed it, and return the most
semantically similar KBDocuments from the FAISS index.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer

from rag.loader import KBDocument
from rag.embedder import KBEmbedder, MODEL_NAME


@dataclass
class RetrievalResult:
    """A single retrieved document with its similarity score."""
    document: KBDocument
    score: float
    rank: int


class KBRetriever:
    """
    Embeds a user query and retrieves the top-k most similar
    KBDocuments from the FAISS index.

    Usage:
        retriever = KBRetriever()
        results = retriever.search("Are GMO crops safe?", top_k=3)
        for r in results:
            print(r.score, r.document.myth, r.document.verdict)
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self._model = None
        self._index = None
        self._documents = None

    def _ensure_loaded(self):
        """Load model and index on first search call."""
        if self._index is None:
            embedder = KBEmbedder()
            self._index, self._documents = embedder.load_or_build()
            self._model = SentenceTransformer(MODEL_NAME)

    def search(
        self,
        query: str,
        top_k: int = None,
    ) -> List[RetrievalResult]:
        """
        Search the knowledge base for documents similar to the query.

        Args:
            query:  The user's question or claim in any language.
            top_k:  Number of results to return. Defaults to self.top_k.

        Returns:
            List of RetrievalResult objects sorted by similarity score
            (highest first).
        """
        self._ensure_loaded()

        k = top_k or self.top_k

        # Embed the query with the same normalization as the index
        query_vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS search — returns (scores, indices)
        scores, indices = self._index.search(query_vector, k)

        results = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:
                continue  # FAISS returns -1 for empty slots
            results.append(RetrievalResult(
                document=self._documents[idx],
                score=float(score),
                rank=rank,
            ))

        return results

    def search_with_threshold(
        self,
        query: str,
        threshold: float = 0.3,
        top_k: int = None,
    ) -> List[RetrievalResult]:
        """
        Search and filter results below a minimum similarity threshold.

        Args:
            query:      The user query.
            threshold:  Minimum cosine similarity score (0.0 to 1.0).
                        0.3 is a reasonable default — below this the
                        match is likely irrelevant.
            top_k:      Max results before threshold filtering.

        Returns:
            Filtered list of RetrievalResult objects.
        """
        results = self.search(query, top_k=top_k)
        return [r for r in results if r.score >= threshold]