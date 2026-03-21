"""
Обратная совместимость: переадресация на app.rag.
Вся логика RAG перенесена в app/rag/ (preprocess, ingest, retriever, embeddings).
"""
from app.rag import (
    build_faiss_index,
    build_knowledge_base,
    load_faiss_index,
    load_or_build_faiss_index,
    retrieve_context,
)

__all__ = [
    "build_knowledge_base",
    "build_faiss_index",
    "load_faiss_index",
    "load_or_build_faiss_index",
    "retrieve_context",
]
