"""Инструменты агента: RAG и веб-поиск (LangChain tools для LangGraph)."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import os

from langchain_core.tools import tool

from app.web_search import search_web
from rag import (
    build_knowledge_base,
    load_or_build_faiss_index,
    retrieve_context,
)

# Таймаут на выполнение tool-функций (чтобы модель не зависала/не разгоняла сеть).
TOOL_TIMEOUT_SEC = int(os.getenv("TOOL_TIMEOUT_SEC", "20"))

# Пул потоков для таймаутов инструментов.
_executor = ThreadPoolExecutor(max_workers=4)

# Ленивая инициализация RAG: запускаем только при первом обращении к rag_search.
_rag_init_lock = threading.Lock()
_knowledge_chunks = None
_faiss_store = None


def _get_rag_resources():
    """Лениво загружает knowledge_chunks + FAISS store и кэширует в модуле."""
    global _knowledge_chunks, _faiss_store

    # Если уже инициализировано — просто вернём.
    if _knowledge_chunks is not None:
        return _knowledge_chunks, _faiss_store

    with _rag_init_lock:
        # Проверяем снова, чтобы не инициализировать дважды при гонке.
        if _knowledge_chunks is None:
            _knowledge_chunks = build_knowledge_base()
            _faiss_store = load_or_build_faiss_index(_knowledge_chunks)

    return _knowledge_chunks, _faiss_store


def _invoke_with_timeout(func, op_name: str, *args, timeout_sec: int = TOOL_TIMEOUT_SEC) -> str:
    """Выполняет функцию в отдельном потоке и ограничивает по времени."""
    future = _executor.submit(func, *args)
    try:
        result = future.result(timeout=timeout_sec)
        return result if isinstance(result, str) else str(result)
    except FuturesTimeoutError:
        print(f"[tools] {op_name} timed out after {timeout_sec}s")
        return ""
    except Exception as e:
        print(f"[tools] {op_name} failed: {e}")
        return ""


@tool
def rag_search(query: str) -> str:
    """Инструмент для модели: поиск в локальной базе (docs). Возвращает текст выдержек для контекста."""
    def _impl():
        knowledge_chunks, faiss_store = _get_rag_resources()
        return retrieve_context(
            knowledge_chunks,
            query,
            vectorstore=faiss_store,
        )

    return _invoke_with_timeout(_impl, "rag_search", timeout_sec=TOOL_TIMEOUT_SEC)


@tool
def web_search(query: str) -> str:
    """Инструмент для модели: веб-поиск через Tavily. Возвращает текст результатов для контекста."""
    return _invoke_with_timeout(
        lambda: search_web(query, max_results=3),
        "web_search",
        timeout_sec=TOOL_TIMEOUT_SEC,
    )
