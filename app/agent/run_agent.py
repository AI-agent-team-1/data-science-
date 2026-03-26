"""Запуск агента: граф LangGraph с инструментами RAG и веб-поиск."""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Deque, List
from urllib.parse import urlparse
import html

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.observability.langfuse_tracing import (
    flush_langfuse,
    langfuse_graph_invoke_config,
)

#
# История диалога (TTL + ограничение по числу чатов)
# - TTL: 24 часа (очистка старых чатов)
# - Максимум: 100 chat_id (LRU eviction)
# - Потокобезопасность: сериализуем обновление истории на chat_id уровне
#
CHAT_HISTORY_TTL_SECONDS = 24 * 60 * 60
MAX_CHAT_HISTORIES = 100

# chat_id -> deque(messages), order == LRU
chat_histories: "OrderedDict[int, Deque[Any]]" = OrderedDict()
# chat_id -> last_access_timestamp
chat_last_access: dict[int, float] = {}

# глобальная блокировка для структуры словарей/OrderedDict
_hist_lock = threading.Lock()
# пер-чат блокировки (чтобы один chat_id обрабатывался последовательно)
_chat_locks: dict[int, threading.Lock] = {}

def _format_source(source: str, max_len: int = 80) -> str:
    """Форматирует источник для HTML-сообщения.

    Для URL делаем короткий текст (например, домен) и сохраняем ссылку полной в href,
    чтобы Telegram оставался кликабельным на правильный адрес.
    """
    s = source.strip()
    if s.startswith("http://") or s.startswith("https://"):
        u = urlparse(s)
        shortened = f"{u.scheme}://{u.netloc}/"
        # Важно: href оставляем полным URL, чтобы клик вёл на правильный адрес.
        return (
            f'<a href="{html.escape(s, quote=True)}">'
            f'{html.escape(shortened[:max_len], quote=True)}'
            f"</a>"
        )

    # Для RAG источник обычно это имя документа
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return html.escape(s, quote=True)


def run_agent(user_text: str, chat_id: int) -> str:
    """Запускает граф агента с историей чата.

    Возвращает текст последнего ответа модели.
    """
    # Гарантируем наличие lock для конкретного chat_id
    with _hist_lock:
        lock = _chat_locks.get(chat_id)
        if lock is None:
            lock = threading.Lock()
            _chat_locks[chat_id] = lock

    # Сериализуем обработку одного chat_id (важно для консистентности истории)
    with lock:
        now = time.time()
        with _hist_lock:
            # Прочистим просроченные чаты и сделаем LRU-ограничение
            expired = [
                cid for cid, ts in chat_last_access.items()
                if now - ts > CHAT_HISTORY_TTL_SECONDS
            ]
            for cid in expired:
                chat_last_access.pop(cid, None)
                chat_histories.pop(cid, None)

            # Если чата ещё нет — создаём deque с maxlen
            history = chat_histories.get(chat_id)
            if history is None:
                history = deque(maxlen=settings.max_history_messages)
                chat_histories[chat_id] = history

            # LRU обновление: данный chat_id становится "самым свежим"
            chat_histories.move_to_end(chat_id, last=True)
            chat_last_access[chat_id] = now

            messages = list(history)
            messages.append(HumanMessage(content=user_text))

            # Ограничение по числу чатов (LRU eviction)
            while len(chat_histories) > MAX_CHAT_HISTORIES:
                oldest_cid, _ = chat_histories.popitem(last=False)
                chat_last_access.pop(oldest_cid, None)

        state: AgentState = {"messages": messages}
        # Чтобы не брать "источники" из прошлой истории, запоминаем сколько
        # сообщений мы передали в граф (history + новое HumanMessage).
        input_messages_len = len(messages)

        print(f"[{chat_id}] USER:", user_text)
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

    # Финальный ответ — последнее сообщение ассистента (после всех вызовов инструментов)
    last = out_messages[-1] if out_messages else None
    if isinstance(last, AIMessage) and last.content:
        response_text = last.content if isinstance(last.content, str) else str(last.content)
    else:
        response_text = str(last.content) if last and getattr(last, "content", None) else "Не удалось сформировать ответ."

    print(f"[{chat_id}] BOT:", response_text)

    # Добавляем источники в конец ответа (deterministic):
    # - для web_search инструмент уже содержит "Источник: <url>"
    # - для rag_search мы помечаем источник как "Источник: <doc_name>"
    sources: list[str] = []
    seen_sources: set[str] = set()
    # Парсим только сообщения, которые были добавлены в текущем вызове графа.
    new_messages = out_messages[input_messages_len:] if isinstance(out_messages, list) else []
    for m in new_messages:
        content = getattr(m, "content", None)
        if not isinstance(content, str):
            continue
        # Берём все строки вида "Источник: ..." и сохраняем уникальные значения.
        matches = re.findall(r"(?m)^Источник:\s*(.+)\s*$", content)
        for s in matches:
            s = s.strip()
            if s and s not in seen_sources:
                seen_sources.add(s)
                sources.append(s)

    if sources:
        # Ограничим количество источников, чтобы не раздувать сообщения.
        sources = sources[:3]
        # Экранируем основной ответ, чтобы в Telegram HTML режиме ничего не ломалось.
        response_text_plain = html.escape(response_text.strip(), quote=False)
        response_text = (
            response_text_plain
            + "\n\n"
            + "\n".join(f"Источник: {_format_source(s)}" for s in sources)
        )

    # Обновляем историю под тем же chat_id lock, чтобы не было гонок
    with _hist_lock:
        now = time.time()
        history_deque = chat_histories.get(chat_id)
        if history_deque is None:
            history_deque = deque(maxlen=settings.max_history_messages)
            chat_histories[chat_id] = history_deque

        chat_histories.move_to_end(chat_id, last=True)
        chat_last_access[chat_id] = now

        # В LangGraph `out_messages` может включать промежуточные сообщения — нам нужна вся цепочка,
        # но обрезаем её maxlen'ом у deque.
        history_deque.clear()
        history_deque.extend(list(out_messages)[-settings.max_history_messages:])

        # Ещё раз ограничим число чатов на всякий случай
        while len(chat_histories) > MAX_CHAT_HISTORIES:
            oldest_cid, _ = chat_histories.popitem(last=False)
            chat_last_access.pop(oldest_cid, None)

    return response_text
