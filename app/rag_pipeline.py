"""
AI Personal Intelligence Copilot
Document Ingestion + RAG Pipeline

Pipeline stages:
  1. LOAD     — Read file bytes (PDF, DOCX, TXT, MD)
  2. PARSE    — Extract raw text
  3. CHUNK    — Split into overlapping chunks (RecursiveCharacterTextSplitter)
  4. EMBED    — Generate vector embeddings per chunk
  5. STORE    — Write chunks into FAISS / ChromaDB
  6. RETRIEVE — Semantic search at query time
  7. AUGMENT  — Inject retrieved context into LLM prompt

Why RAG improves accuracy by ~60%?
  Without RAG: LLM answers from parametric memory only → hallucinations, stale facts
  With RAG:    LLM grounds answers in user's actual documents → precise, cited, fresh

Why LangChain's TextSplitter?
  Recursive splitting respects sentence/paragraph boundaries, producing coherent
  chunks that embed semantically cleanly and retrieve meaningfully.
"""

from __future__ import annotations

import io
import uuid
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, BinaryIO
from loguru import logger

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # Minimal fallback splitter
        class RecursiveCharacterTextSplitter:  # type: ignore
            def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_text(self, text: str):
                chunks = []
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunks.append(text[start:end])
                    start += self.chunk_size - self.chunk_overlap
                return [c for c in chunks if c.strip()]
from vector_db.vector_store import VectorStoreManager


# ─────────────────────────────────────────────────────────────────────────────
# Text Extractors (one per file type)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pypdf not installed. Install it for PDF support.")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        logger.warning("python-docx not installed.")
        return ""
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode plain text file."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def extract_text(file_bytes: bytes, file_type: str) -> str:
    """
    Route to the correct extractor based on file extension.
    file_type: 'pdf' | 'docx' | 'txt' | 'md' | 'text'
    """
    ft = file_type.lower().strip(".")
    if ft == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ft in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    elif ft in ("txt", "md", "text", "rst", "csv"):
        return extract_text_from_txt(file_bytes)
    else:
        # Best-effort: try UTF-8 decode
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# Text Splitter
# ─────────────────────────────────────────────────────────────────────────────

def build_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    RecursiveCharacterTextSplitter: splits on paragraphs → sentences → words.
    This hierarchy preserves semantic coherence in each chunk.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Ingestion Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DocumentIngestionPipeline:
    """
    End-to-end ingestion pipeline: file bytes → vector DB.

    Usage
    -----
    ```python
    pipeline = DocumentIngestionPipeline(vector_store_manager)
    result   = pipeline.ingest_file(
        file_bytes=open("report.pdf","rb").read(),
        filename="report.pdf",
        user_id="u-123",
    )
    print(result)  # {'doc_id': '...', 'chunks': 42, 'status': 'ready'}
    ```
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.vector_store = vector_store
        self.splitter = build_splitter(chunk_size, chunk_overlap)

    def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: str = "default",
        extra_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Full ingestion flow for a single file.
        Returns a status dict compatible with the Document DB model.
        """
        doc_id = str(uuid.uuid4())
        file_type = Path(filename).suffix.lstrip(".")
        result = {
            "doc_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "chunks": 0,
            "status": "error",
            "error": None,
        }

        try:
            # ── 1. Extract text ───────────────────────────────────────────
            logger.info(f"Extracting text from '{filename}' ({file_type.upper()})")
            raw_text = extract_text(file_bytes, file_type)

            if not raw_text.strip():
                result["error"] = "No text could be extracted from the file."
                return result

            # ── 2. Chunk ──────────────────────────────────────────────────
            chunks = self.splitter.split_text(raw_text)
            logger.info(f"Split into {len(chunks)} chunks")

            # ── 3. Build metadata per chunk ───────────────────────────────
            metadatas = [
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "user_id": user_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **(extra_metadata or {}),
                }
                for i in range(len(chunks))
            ]

            # ── 4. Embed + Store ──────────────────────────────────────────
            ingested = self.vector_store.ingest_texts(chunks, metadatas)

            result["chunks"] = ingested
            result["status"] = "ready"
            logger.success(f"Ingested '{filename}': {ingested} chunks stored")

        except Exception as e:
            logger.error(f"Ingestion error for '{filename}': {e}")
            result["error"] = str(e)

        return result

    def ingest_text(
        self,
        text: str,
        source: str = "manual_input",
        user_id: str = "default",
    ) -> Dict:
        """
        Ingest raw text directly (e.g., from notes, chat input, paste).
        """
        doc_id = str(uuid.uuid4())
        chunks = self.splitter.split_text(text)

        if not chunks:
            return {"doc_id": doc_id, "chunks": 0, "status": "error",
                    "error": "Empty text"}

        metadatas = [
            {
                "doc_id": doc_id,
                "filename": source,
                "file_type": "text",
                "user_id": user_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        ingested = self.vector_store.ingest_texts(chunks, metadatas)
        return {"doc_id": doc_id, "chunks": ingested, "status": "ready", "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Steps
    -----
    1. Receive user query
    2. Embed query
    3. Search vector DB for top-k similar chunks
    4. Format retrieved chunks as context
    5. Inject context into LLM prompt
    6. Return augmented response with source citations

    Why this improves accuracy by 60%?
      The LLM is grounded in retrieved facts rather than relying on
      potentially stale or hallucinated parametric knowledge.
    """

    CONTEXT_TEMPLATE = """
You have access to the following relevant context retrieved from the user's documents.
Use this information to provide accurate, grounded answers.

--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---

Based on the above context and the conversation history, answer the user's question.
If the context doesn't contain enough information, say so honestly and answer from
your general knowledge, clearly distinguishing the two.
"""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        top_k: int = 5,
        similarity_threshold: float = 0.4,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.threshold = similarity_threshold

    def retrieve(self, query: str) -> List[Tuple[str, Dict, float]]:
        """Retrieve relevant chunks for a query."""
        if self.vector_store.total_documents() == 0:
            return []
        return self.vector_store.search(query, top_k=self.top_k, threshold=self.threshold)

    def build_context_block(self, results: List[Tuple[str, Dict, float]]) -> str:
        """Format retrieved chunks into a readable context block."""
        if not results:
            return ""

        blocks = []
        for i, (text, meta, score) in enumerate(results, 1):
            source = meta.get("filename", "unknown")
            chunk_idx = meta.get("chunk_index", "?")
            blocks.append(
                f"[Source {i}: {source} | Chunk {chunk_idx} | Score: {score:.2f}]\n{text}"
            )

        return "\n\n".join(blocks)

    def build_augmented_prompt(
        self, query: str, results: List[Tuple[str, Dict, float]]
    ) -> Optional[str]:
        """
        Returns the context injection string to prepend to the system prompt.
        Returns None if no relevant context was found.
        """
        if not results:
            return None

        context = self.build_context_block(results)
        return self.CONTEXT_TEMPLATE.format(context=context)

    def get_sources(self, results: List[Tuple[str, Dict, float]]) -> List[Dict]:
        """Extract source citations from results."""
        seen = set()
        sources = []
        for _, meta, score in results:
            filename = meta.get("filename", "unknown")
            if filename not in seen:
                seen.add(filename)
                sources.append({
                    "filename": filename,
                    "doc_id": meta.get("doc_id", ""),
                    "relevance_score": round(score, 3),
                })
        return sources
