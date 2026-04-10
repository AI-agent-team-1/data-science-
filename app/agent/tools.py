"""Инструменты агента: RAG и веб-поиск (LangChain tools для LangGraph)."""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from langchain_core.tools import tool

from app.web_search import search_web
from app.rag import (
    build_knowledge_base,
    load_or_build_faiss_index,
    retrieve_context,
)

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_SEC = int(os.getenv("TOOL_TIMEOUT_SEC", "20"))

_executor = ThreadPoolExecutor(max_workers=4)

_init_rag_lock = threading.Lock()
_rag_initialized = False
_knowledge_chunks: list = []
_faiss_store = None


def init_rag() -> None:
    """Явная загрузка базы знаний и FAISS. Вызывать до первого запроса (bot / FastAPI lifespan)."""
    global _rag_initialized, _knowledge_chunks, _faiss_store
    with _init_rag_lock:
        if _rag_initialized:
            return
        try:
            _knowledge_chunks = build_knowledge_base()
            _faiss_store = load_or_build_faiss_index(_knowledge_chunks)
        except OSError as e:
            logger.error("[RAG] Ошибка инициализации (файлы/диск): %s", e, exc_info=True)
            _knowledge_chunks = []
            _faiss_store = None
        except Exception as e:
            logger.exception("[RAG] Ошибка инициализации ресурсов: %s", e)
            _knowledge_chunks = []
            _faiss_store = None
        _rag_initialized = True


def _invoke_with_timeout(func, op_name: str, *args, timeout_sec: int = TOOL_TIMEOUT_SEC) -> str:
    future = _executor.submit(func, *args)
    try:
        result = future.result(timeout=timeout_sec)
        return result if isinstance(result, str) else str(result)
    except FuturesTimeoutError:
        logger.warning("[tools] %s timed out after %ss", op_name, timeout_sec)
        return ""
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("[tools] %s failed: %s", op_name, e, exc_info=True)
        return ""


def _rag_retrieve(query: str) -> str:
    return retrieve_context(
        _knowledge_chunks,
        query,
        vectorstore=_faiss_store,
    )


@tool
def rag_search(query: str) -> str:
    """Инструмент для модели: поиск в локальной базе (docs). Возвращает текст выдержек для контекста."""
    if not _rag_initialized:
        init_rag()
    return _invoke_with_timeout(
        lambda: _rag_retrieve(query),
        "rag_search",
        timeout_sec=TOOL_TIMEOUT_SEC,
    )


@tool
def web_search(query: str) -> str:
    """Инструмент для модели: веб-поиск через Tavily. Возвращает текст результатов для контекста."""
    return _invoke_with_timeout(
        lambda: search_web(query, max_results=3),
        "web_search",
        timeout_sec=TOOL_TIMEOUT_SEC,
    )
