"""
AI Personal Intelligence Copilot
Vector Database Layer — 100% FREE Stack

Embedding Engine:
  HuggingFace sentence-transformers/all-MiniLM-L6-v2
  • Runs 100% LOCALLY — no API calls, no cost, no rate limits
  • 384-dimensional vectors
  • Excellent semantic similarity for RAG retrieval
  • 22MB model, downloads once, cached locally
  • ~50ms per batch on CPU

Vector Stores:
  FAISS  — Facebook AI Similarity Search (local, free, blazing fast)
  ChromaDB — embeddable vector DB (local, free, metadata filtering)
"""

from __future__ import annotations

import os
import uuid
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Embedding Engine (FREE — runs locally)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Local embedding engine using HuggingFace sentence-transformers.

    Why all-MiniLM-L6-v2?
      • State-of-the-art semantic similarity performance
      • 22MB download — tiny, fast
      • 384 dimensions — ideal balance of quality vs. speed
      • MIT License — completely free for commercial use
      • No API key, no rate limits, no cost ever

    First run will download the model (~22MB) and cache it locally.
    Subsequent runs load from cache instantly.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = "./models/hf_cache",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence-transformer model (downloads on first run)."""
        try:
            from sentence_transformers import SentenceTransformer
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
            )
            logger.info(
                f"EmbeddingEngine: {self.model_name} loaded "
                f"(dim={self.dimension}, FREE local inference)"
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install: pip install sentence-transformers\n"
                "Falling back to random vectors (demo only)."
            )
            self._model = None
        except Exception as e:
            logger.error(f"Model load error: {e}. Using random vectors.")
            self._model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        if not texts:
            return []

        if self._model is not None:
            try:
                vectors = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=32,
                )
                return vectors.tolist()
            except Exception as e:
                logger.error(f"Embedding error: {e}")

        # Fallback: random unit vectors (demo only — not for production)
        logger.warning("Using random vectors — install sentence-transformers for real embeddings")
        dim = self.dimension
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        # Normalize to unit sphere
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        return vecs.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """all-MiniLM-L6-v2 produces 384-dimensional vectors."""
        return 384

    @property
    def is_real(self) -> bool:
        """True if using real HuggingFace model, False if random fallback."""
        return self._model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Document Chunk
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunk:
    """A text chunk with its embedding and metadata."""

    def __init__(
        self,
        text: str,
        metadata: Dict,
        embedding: Optional[List[float]] = None,
        chunk_id: str = "",
    ):
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.text = text
        self.metadata = metadata
        self.embedding = embedding

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Vector Store (FREE, local)
# ─────────────────────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    FAISS-backed vector store. Everything runs locally — no cloud, no cost.

    Uses IndexFlatIP (inner product) with L2-normalized vectors,
    which gives exact cosine similarity. Perfect for RAG retrieval.
    """

    def __init__(self, index_path: str = "./data/faiss_index", dimension: int = 384):
        self.index_path = index_path
        self.dimension = dimension
        self._index = None
        self._id_to_chunk: Dict[int, DocumentChunk] = {}
        self._load()

    def _load(self):
        idx_file = f"{self.index_path}.index"
        meta_file = f"{self.index_path}.meta"

        if os.path.exists(idx_file) and os.path.exists(meta_file):
            try:
                import faiss
                self._index = faiss.read_index(idx_file)
                with open(meta_file, "rb") as f:
                    self._id_to_chunk = pickle.load(f)
                logger.info(f"FAISS: Loaded {self._index.ntotal} vectors from disk")
                return
            except Exception as e:
                logger.warning(f"FAISS load failed: {e}. Creating fresh index.")

        self._create_fresh()

    def _create_fresh(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            self._id_to_chunk = {}
            logger.info(f"FAISS: New IndexFlatIP (dim={self.dimension})")
        except ImportError:
            logger.warning("faiss-cpu not installed: pip install faiss-cpu")
            self._index = None

    def save(self):
        if self._index is None:
            return
        try:
            import faiss
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, f"{self.index_path}.index")
            with open(f"{self.index_path}.meta", "wb") as f:
                pickle.dump(self._id_to_chunk, f)
        except Exception as e:
            logger.error(f"FAISS save error: {e}")

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        if self._index is None or not chunks:
            return 0

        valid = [c for c in chunks if c.embedding]
        if not valid:
            return 0

        import faiss
        vectors = []
        for chunk in valid:
            vec = np.array(chunk.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        matrix = np.stack(vectors)
        start_id = self._index.ntotal
        self._index.add(matrix)

        for i, chunk in enumerate(valid):
            self._id_to_chunk[start_id + i] = chunk

        self.save()
        logger.info(f"FAISS: Added {len(valid)} chunks (total: {self._index.ntotal})")
        return len(valid)

    def search(
        self, query_vector: List[float], top_k: int = 5, threshold: float = 0.35
    ) -> List[Tuple[DocumentChunk, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vec = vec.reshape(1, -1)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold and idx in self._id_to_chunk:
                results.append((self._id_to_chunk[idx], float(score)))

        return results

    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    def clear(self):
        self._create_fresh()
        self.save()


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Vector Store (FREE, local)
# ─────────────────────────────────────────────────────────────────────────────

class ChromaDBVectorStore:
    """ChromaDB — local persistent vector DB with metadata filtering."""

    def __init__(self, persist_dir: str = "./data/chroma_db",
                 collection_name: str = "copilot_knowledge"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init()

    def _init(self):
        try:
            import chromadb
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB: {self.collection_name} ({self._collection.count()} docs)")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        if self._collection is None:
            return 0
        valid = [c for c in chunks if c.embedding]
        if not valid:
            return 0
        self._collection.add(
            ids=[c.chunk_id for c in valid],
            embeddings=[c.embedding for c in valid],
            documents=[c.text for c in valid],
            metadatas=[c.metadata for c in valid],
        )
        return len(valid)

    def search(
        self, query_vector: List[float], top_k: int = 5, threshold: float = 0.3
    ) -> List[Tuple[DocumentChunk, float]]:
        if self._collection is None or self._collection.count() == 0:
            return []

        k = min(top_k, self._collection.count())
        results = self._collection.query(
            query_embeddings=[query_vector], n_results=k
        )

        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            score = 1 - dist
            if score >= threshold:
                chunk = DocumentChunk(
                    text=doc, metadata=meta,
                    chunk_id=results["ids"][0][i]
                )
                output.append((chunk, score))

        return output

    def total_vectors(self) -> int:
        return self._collection.count() if self._collection else 0

    def clear(self):
        if self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )


# ─────────────────────────────────────────────────────────────────────────────
# VectorStoreManager — unified interface
# ─────────────────────────────────────────────────────────────────────────────

class VectorStoreManager:
    """
    Single interface to whichever vector backend is configured.
    Owns the EmbeddingEngine and the embed→store pipeline.
    """

    def __init__(
        self,
        store_type: str = "faiss",
        embedding_engine: Optional[EmbeddingEngine] = None,
        faiss_index_path: str = "./data/faiss_index",
        chroma_dir: str = "./data/chroma_db",
        chroma_collection: str = "copilot_knowledge",
    ):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.store_type = store_type

        if store_type == "chromadb":
            self._store = ChromaDBVectorStore(chroma_dir, chroma_collection)
        else:
            self._store = FAISSVectorStore(
                faiss_index_path, dimension=self.embedding_engine.dimension
            )

        logger.info(
            f"VectorStoreManager: {store_type.upper()} | "
            f"Embeddings: HuggingFace {self.embedding_engine.model_name} (FREE local)"
        )

    def ingest_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> int:
        if not texts:
            return 0

        metadatas = metadatas or [{}] * len(texts)
        logger.info(f"Embedding {len(texts)} chunks with HuggingFace (free, local)…")

        embeddings = self.embedding_engine.embed(texts)
        chunks = [
            DocumentChunk(text=t, metadata=m, embedding=e)
            for t, m, e in zip(texts, metadatas, embeddings)
        ]

        count = self._store.add_chunks(chunks)
        logger.info(f"Ingested {count} chunks into {self.store_type.upper()}")
        return count

    def search(
        self, query: str, top_k: int = 5, threshold: float = 0.35
    ) -> List[Tuple[str, Dict, float]]:
        if self.total_documents() == 0:
            return []

        query_vec = self.embedding_engine.embed_single(query)
        raw = self._store.search(query_vec, top_k=top_k, threshold=threshold)
        return [(chunk.text, chunk.metadata, score) for chunk, score in raw]

    def total_documents(self) -> int:
        return self._store.total_vectors()

    def clear(self):
        self._store.clear()
        logger.info("VectorStore cleared")
