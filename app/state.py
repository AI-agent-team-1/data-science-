"""Состояние агента для графа LangGraph."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Сообщения диалога (system, human, assistant, tool). Reducer добавляет новые к списку."""
    messages: Annotated[list, add_messages]
