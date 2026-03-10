"""Инструменты агента: RAG и веб-поиск (LangChain tools для LangGraph)."""
from __future__ import annotations

from langchain_core.tools import tool

from app.web_search import search_web
from rag import (
    build_knowledge_base,
    load_or_build_faiss_index,
    retrieve_context,
)

# Инициализация RAG при первом импорте (один раз при старте бота)
_knowledge_chunks = build_knowledge_base()
_faiss_store = load_or_build_faiss_index(_knowledge_chunks)


@tool
def rag_search(query: str) -> str:
    """Ищет в локальной базе знаний (документы по нефтегазовой тематике). Используй для ответов по регламентам, процедурам и внутренней документации."""
    return retrieve_context(
        _knowledge_chunks,
        query,
        vectorstore=_faiss_store,
    )


@tool
def web_search(query: str) -> str:
    """Ищет актуальную информацию в интернете. Используй, когда в базе знаний нет ответа или нужны свежие данные."""
    return search_web(query, max_results=3)
