"""Запуск агента: граф LangGraph с инструментами RAG и веб-поиск."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.graph import get_graph
from app.state import AgentState

chat_histories: dict[int, List[Any]] = defaultdict(list)


def run_agent(user_text: str, chat_id: int) -> str:
    history = chat_histories[chat_id]
    messages = list(history)
    messages.append(HumanMessage(content=user_text))
    state: AgentState = {"messages": messages}

    print(f"[{chat_id}] USER:", user_text)
    graph = get_graph()
    result = graph.invoke(state)

    out_messages = result["messages"]
    if len(out_messages) > settings.max_history_messages:
        chat_histories[chat_id] = list(out_messages)[-settings.max_history_messages:]
    else:
        chat_histories[chat_id] = list(out_messages)

    last = out_messages[-1] if out_messages else None
    if isinstance(last, AIMessage) and last.content:
        response_text = last.content if isinstance(last.content, str) else str(last.content)
    else:
        response_text = str(last.content) if last and getattr(last, "content", None) else "Не удалось сформировать ответ."

    print(f"[{chat_id}] BOT:", response_text)
    return response_text
