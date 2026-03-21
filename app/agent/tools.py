"""Инструменты агента: RAG и веб-поиск (LangChain tools для LangGraph)."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import os

from langchain_core.tools import tool

from app.web_search import search_web
from app.rag import (
    build_knowledge_base,
    load_or_build_faiss_index,
    retrieve_context,
)

# Таймаут на выполнение tool-функций (чтобы модель не зависала/не разгоняла сеть).
TOOL_TIMEOUT_SEC = int(os.getenv("TOOL_TIMEOUT_SEC", "20"))

# Пул потоков для таймаутов инструментов.
_executor = ThreadPoolExecutor(max_workers=4)

# Инициализация RAG при старте контейнера (без "ленивой" загрузки).
# Так мы гарантируем, что после изменений в `docs/` индекс/чанки будут обновлены.
_init_rag_lock = threading.Lock()
_knowledge_chunks = []
_faiss_store = None

with _init_rag_lock:
    try:
        _knowledge_chunks = build_knowledge_base()
        _faiss_store = load_or_build_faiss_index(_knowledge_chunks)
    except Exception as e:
        # Если на старте не получилось загрузить/построить FAISS, работаем с keyword-fallback.
        print(f"[RAG] Ошибка инициализации ресурсов: {e}")


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
    # Таймаут убран: RAG может долго собираться/искать, особенно после изменения docs.
    return retrieve_context(
        _knowledge_chunks,
        query,
        vectorstore=_faiss_store,
    )


@tool
def web_search(query: str) -> str:
    """Инструмент для модели: веб-поиск через Tavily. Возвращает текст результатов для контекста."""
    return _invoke_with_timeout(
        lambda: search_web(query, max_results=3),
        "web_search",
        timeout_sec=TOOL_TIMEOUT_SEC,
    )
