"""Состояние агента для графа LangGraph."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Состояние графа LangGraph: список сообщений. add_messages объединяет новые с существующими."""
    messages: Annotated[list, add_messages]
