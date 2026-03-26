"""Langfuse: CallbackHandler для LangGraph + flush. Один запрос пользователя — один handler."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

from app.config import settings
from app.observability.sanitize import hash_user_id, langfuse_mask

_init_lock = threading.Lock()
_client_ready = False


def _langfuse_configured() -> bool:
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key.strip()
        and settings.langfuse_secret_key.strip()
    )


def _ensure_langfuse_client() -> None:
    """Langfuse SDK 3.x: регистрирует клиент по public_key до get_client / CallbackHandler / propagate_attributes."""
    global _client_ready
    if not _langfuse_configured():
        return
    if _client_ready:
        return
    with _init_lock:
        if _client_ready:
            return
        try:
            from langfuse import Langfuse
        except ImportError:
            return
        Langfuse(
            public_key=settings.langfuse_public_key.strip(),
            secret_key=settings.langfuse_secret_key.strip(),
            host=settings.langfuse_host.rstrip("/"),
            # SDK 3.x будет применять mask-функцию к input/output/metadata
            # перед отправкой в Langfuse. Это покрывает PII в payload'ах модели.
            mask=langfuse_mask,
        )
        _client_ready = True


def langfuse_graph_invoke_config(chat_id: int) -> Optional[Dict[str, Any]]:
    """
    Полный RunnableConfig для graph.invoke, как в callback-first-проектах (run_name + metadata для Langfuse).

    Ключи langfuse_* в metadata обрабатывает Langfuse CallbackHandler (сессия, user, теги).
    """
    if not _langfuse_configured():
        return None
    _ensure_langfuse_client()
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        print("[langfuse] Пакет langfuse не установлен. Установите: pip install 'langfuse>=3,<4'")
        return None

    trace_id = uuid4().hex
    # Хэшируем user/session id, чтобы не хранить в Langfuse прямые
    # идентификаторы Telegram-чата (PII).
    sid = str(chat_id)
    hashed_sid = hash_user_id(sid)
    handler = CallbackHandler(
        public_key=settings.langfuse_public_key.strip(),
        update_trace=True,
    )
    return {
        "callbacks": [handler],
        "run_name": "llm_bot",
        "metadata": {
            "app": "llm_bot",
            "langfuse_trace_id": trace_id,
            "langfuse_session_id": hashed_sid,
            "langfuse_user_id": hashed_sid,
            "langfuse_tags": ["telegram", "llm_bot"],
        },
    }


def langfuse_callbacks_for_chat(chat_id: int) -> List[Any]:
    """
    Только список handlers (совместимость). Предпочтительно использовать langfuse_graph_invoke_config.
    """
    cfg = langfuse_graph_invoke_config(chat_id)
    if not cfg:
        return []
    cbs = cfg.get("callbacks")
    return list(cbs) if isinstance(cbs, list) else []


@contextmanager
def langfuse_chat_context(chat_id: int) -> Generator[None, None, None]:
    """
    Опционально: propagate_attributes для OTel (session/user). Основной путь — metadata в invoke config.
    """
    if not _langfuse_configured():
        yield
        return
    _ensure_langfuse_client()
    try:
        from langfuse import propagate_attributes
    except ImportError:
        yield
        return

    sid = str(chat_id)
    with propagate_attributes(
        session_id=sid,
        user_id=sid,
        metadata={"telegram_chat_id": sid, "app": "llm_bot"},
    ):
        yield


def flush_langfuse() -> None:
    """Сбрасывает буфер событий Langfuse (важно после запроса в долгоживущем процессе)."""
    if not _langfuse_configured():
        return
    _ensure_langfuse_client()
    try:
        from langfuse import get_client
    except ImportError:
        return
    try:
        get_client(public_key=settings.langfuse_public_key.strip()).flush()
    except Exception as e:
        print(f"[langfuse] flush: {e}")
