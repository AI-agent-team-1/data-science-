"""
Обратная совместимость: реэкспорт публичного API из ``app.rag``.

Где используется (ответ на вопрос ревьюера «зачем rag.py»):
- Скрипты и ноутбуки вне пакета ``app`` могут писать
  ``from rag import build_faiss_index, retrieve_context, ...`` без префикса ``app``.
- Основной код бота и агента импортирует напрямую ``app.rag`` / ``app.rag.ingest``;
  корневой ``rag.py`` для них не обязателен.

Вся реализация — в ``app/rag/`` (preprocess, ingest, retriever, embeddings).
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
