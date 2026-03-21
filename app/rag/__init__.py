"""RAG: предобработка, индексация, поиск."""
from app.rag.ingest import (
    build_faiss_index,
    build_knowledge_base,
    load_faiss_index,
    load_or_build_faiss_index,
)
from app.rag.retriever import retrieve_context
from app.rag.preprocess import preprocess_text
from app.rag.embeddings import get_embeddings

__all__ = [
    "build_knowledge_base",
    "build_faiss_index",
    "load_faiss_index",
    "load_or_build_faiss_index",
    "retrieve_context",
    "preprocess_text",
    "get_embeddings",
]
