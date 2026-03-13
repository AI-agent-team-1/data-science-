"""Инструменты агента: RAG и веб-поиск (LangChain tools для LangGraph)."""
from __future__ import annotations

from langchain_core.tools import tool

from app.web_search import search_web
from rag import (
    build_knowledge_base,
    load_or_build_faiss_index,
    retrieve_context,
)

# При первом импорте загружаем базу знаний и FAISS — один раз на процесс
_knowledge_chunks = build_knowledge_base()
_faiss_store = load_or_build_faiss_index(_knowledge_chunks)


@tool
def rag_search(query: str) -> str:
    """Инструмент для модели: поиск в локальной базе (docs). Возвращает текст выдержек для контекста."""
    return retrieve_context(
        _knowledge_chunks,
        query,
        vectorstore=_faiss_store,
    )


@tool
def web_search(query: str) -> str:
    """Инструмент для модели: веб-поиск через Tavily. Возвращает текст результатов для контекста."""
    return search_web(query, max_results=3)
