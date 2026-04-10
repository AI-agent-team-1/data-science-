"""Запуск агента: граф LangGraph с инструментами RAG и веб-поиск."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from typing import Any, List
from urllib.parse import urlparse
import html

from langchain_core.messages import AIMessage, HumanMessage

from app import chat_history_sqlite
from app.config import settings
from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.observability.langfuse_tracing import (
    flush_langfuse,
    langfuse_graph_invoke_config,
)

logger = logging.getLogger(__name__)

#
# История диалога: SQLite (таблица chat_history) + TTL + LRU
# - TTL: 24 часа без активности — сессия удаляется при следующем обращении / prune
# - Максимум: 100 chat_id (LRU по last_access)
# - Потокобезопасность: SQLite под lock; per-chat lock для последовательности одного chat_id
#
CHAT_HISTORY_TTL_SECONDS = 24 * 60 * 60
MAX_CHAT_HISTORIES = 100

# пер-чат блокировки (чтобы один chat_id обрабатывался последовательно)
_hist_lock = threading.Lock()
_chat_locks: dict[int, threading.Lock] = {}


def session_id_to_chat_id(session_id: str) -> int:
    """Стабильное целое для веб-сессии (UUID в строке → int для run_agent)."""
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def clear_chat_history(chat_id: int) -> None:
    """Удаляет историю диалога для данного chat_id (Telegram или веб-сессия)."""
    chat_history_sqlite.delete_session(chat_id)
    with _hist_lock:
        _chat_locks.pop(chat_id, None)


def _format_source(source: str, max_len: int = 80) -> str:
    """Форматирует источник для HTML-сообщения.

    Для URL делаем короткий текст (например, домен) и сохраняем ссылку полной в href,
    чтобы Telegram оставался кликабельным на правильный адрес.
    """
    s = source.strip()
    if s.startswith("http://") or s.startswith("https://"):
        u = urlparse(s)
        shortened = f"{u.scheme}://{u.netloc}/"
        return (
            f'<a href="{html.escape(s, quote=True)}">'
            f'{html.escape(shortened[:max_len], quote=True)}'
            f"</a>"
        )

    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return html.escape(s, quote=True)


def run_agent(user_text: str, chat_id: int) -> str:
    """Запускает граф агента с историей чата.

    Возвращает текст последнего ответа модели.
    """
    with _hist_lock:
        lock = _chat_locks.get(chat_id)
        if lock is None:
            lock = threading.Lock()
            _chat_locks[chat_id] = lock

    with lock:
        now = time.time()
        removed = chat_history_sqlite.prune_stale_and_lru(
            now, CHAT_HISTORY_TTL_SECONDS, MAX_CHAT_HISTORIES
        )
        with _hist_lock:
            for cid in removed:
                _chat_locks.pop(cid, None)

        messages: List[Any] = chat_history_sqlite.load_messages(
            chat_id, now, CHAT_HISTORY_TTL_SECONDS
        )
        if len(messages) > settings.max_history_messages:
            messages = messages[-settings.max_history_messages :]
        messages.append(HumanMessage(content=user_text))

        input_messages_len = len(messages)
        state: AgentState = {"messages": messages}

        logger.info("[%s] USER: %s", chat_id, user_text)
        graph = get_graph()
        lf_config = langfuse_graph_invoke_config(chat_id)
        try:
            if lf_config:
                result = graph.invoke(state, config=lf_config)
            else:
                result = graph.invoke(state)
        finally:
            flush_langfuse()

        out_messages = result["messages"]

        last = out_messages[-1] if out_messages else None
        if isinstance(last, AIMessage) and last.content:
            response_text = last.content if isinstance(last.content, str) else str(last.content)
        else:
            response_text = (
                str(last.content) if last and getattr(last, "content", None) else "Не удалось сформировать ответ."
            )

        logger.info("[%s] BOT: %s", chat_id, response_text[:500] + ("..." if len(response_text) > 500 else ""))

        sources: list[str] = []
        seen_sources: set[str] = set()
        new_messages = out_messages[input_messages_len:] if isinstance(out_messages, list) else []
        for m in new_messages:
            content = getattr(m, "content", None)
            if not isinstance(content, str):
                continue
            matches = re.findall(r"(?m)^Источник:\s*(.+)\s*$", content)
            for s in matches:
                s = s.strip()
                if s and s not in seen_sources:
                    seen_sources.add(s)
                    sources.append(s)

        if sources:
            sources = sources[:3]
            response_text_plain = html.escape(response_text.strip(), quote=False)
            response_text = (
                response_text_plain
                + "\n\n"
                + "\n".join(f"Источник: {_format_source(s)}" for s in sources)
            )

        to_store = list(out_messages)[-settings.max_history_messages :]
        chat_history_sqlite.save_messages(chat_id, time.time(), to_store)

        removed2 = chat_history_sqlite.prune_stale_and_lru(
            time.time(), CHAT_HISTORY_TTL_SECONDS, MAX_CHAT_HISTORIES
        )
        with _hist_lock:
            for cid in removed2:
                _chat_locks.pop(cid, None)

        return response_text
