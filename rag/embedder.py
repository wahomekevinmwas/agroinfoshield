"""
rag/embedder.py
---------------
Responsibility: Generate sentence embeddings for all KBDocument chunks
and build a FAISS index that persists to disk.

Why this exists:
    The retriever needs to search documents by semantic meaning, not
    keywords. This module converts each chunk's text into a dense vector
    using a local sentence-transformers model, then stores those vectors
    in a FAISS index for fast similarity search.

Design decisions:
    1. Model: all-MiniLM-L6-v2 — small (80MB), fast, good multilingual
       performance for English and Swahili content. Runs fully locally
       with no API key or internet connection after first download.

    2. Persistence: The FAISS index and document metadata are saved to
       disk after first build. On subsequent startups the embedder loads
       from disk instead of recomputing — cold start goes from ~30s to
       under 1s.

    3. Rebuild detection: If myth_facts.json is newer than the saved
       index, the embedder automatically rebuilds. This means adding new
       entries to the knowledge base triggers a rebuild on next startup
       with no manual intervention.

    4. Metadata storage: FAISS stores vectors only — not the original
       text or metadata. We store document metadata separately as a
       pickle file alongside the index so the retriever can return full
       KBDocument objects, not just similarity scores.
"""

import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.loader import KBDocument, KBLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "data" / "faiss_index"
INDEX_FILE = DEFAULT_INDEX_DIR / "index.faiss"
DOCS_FILE = DEFAULT_INDEX_DIR / "documents.pkl"
KB_FILE = Path(__file__).parent.parent / "data" / "myth_facts.json"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class KBEmbedder:
    """
    Generates embeddings for KBDocument chunks and manages a persistent
    FAISS index.

    Usage:
        embedder = KBEmbedder()
        index, documents = embedder.load_or_build()

    After the first run, subsequent calls load from disk in under 1s.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_dir: Path = DEFAULT_INDEX_DIR,
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_file = self.index_dir / "index.faiss"
        self.docs_file = self.index_dir / "documents.pkl"
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first access."""
        if self._model is None:
            print(f"[KBEmbedder] Loading model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            print(f"[KBEmbedder] Model loaded.")
        return self._model

    def load_or_build(self) -> Tuple[faiss.Index, List[KBDocument]]:
        """
        Load the FAISS index from disk if it is current, otherwise rebuild.

        Returns:
            Tuple of (faiss_index, list_of_KBDocuments)

        The index and documents list are parallel — index.search() returns
        integer positions that map directly to documents[position].
        """
        if self._index_is_current():
            print("[KBEmbedder] Loading existing index from disk...")
            return self._load_from_disk()
        else:
            print("[KBEmbedder] Building new index...")
            return self._build_and_save()

    def rebuild(self) -> Tuple[faiss.Index, List[KBDocument]]:
        """Force a rebuild regardless of whether the index is current."""
        print("[KBEmbedder] Forcing index rebuild...")
        return self._build_and_save()

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _index_is_current(self) -> bool:
        """
        Check if the persisted index exists and is newer than myth_facts.json.
        """
        if not self.index_file.exists() or not self.docs_file.exists():
            return False

        index_mtime = self.index_file.stat().st_mtime
        kb_mtime = KB_FILE.stat().st_mtime if KB_FILE.exists() else 0

        is_current = index_mtime > kb_mtime
        if not is_current:
            print("[KBEmbedder] Knowledge base updated — index will rebuild.")
        return is_current

    def _build_and_save(self) -> Tuple[faiss.Index, List[KBDocument]]:
        """
        Load documents, generate embeddings, build FAISS index, save to disk.
        """
        loader = KBLoader()
        documents = loader.load()

        if not documents:
            raise ValueError(
                "[KBEmbedder] No documents to embed. "
                "Check that myth_facts.json has valid entries."
            )

        print(f"[KBEmbedder] Embedding {len(documents)} documents...")
        start = time.time()

        chunk_texts = [doc.chunk_text for doc in documents]
        embeddings = self.model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        elapsed = time.time() - start
        print(f"[KBEmbedder] Embeddings generated in {elapsed:.1f}s. "
              f"Shape: {embeddings.shape}")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))

        print(f"[KBEmbedder] FAISS index built. "
              f"Total vectors: {index.ntotal}")

        self._save_to_disk(index, documents)

        return index, documents

    def _save_to_disk(
        self,
        index: faiss.Index,
        documents: List[KBDocument],
    ) -> None:
        """Save FAISS index and document metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(self.index_file))

        with open(self.docs_file, "wb") as f:
            pickle.dump(documents, f)

        print(f"[KBEmbedder] Index saved to {self.index_dir}/")

    def _load_from_disk(self) -> Tuple[faiss.Index, List[KBDocument]]:
        """Load FAISS index and document metadata from disk."""
        index = faiss.read_index(str(self.index_file))

        with open(self.docs_file, "rb") as f:
            documents = pickle.load(f)

        print(f"[KBEmbedder] Loaded index with {index.ntotal} vectors "
              f"and {len(documents)} documents.")
        return index, documents