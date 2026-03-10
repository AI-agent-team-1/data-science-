"""Граф агента LangGraph: модель решает, когда вызывать инструменты и когда дать ответ."""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings
from app.prompts import SYSTEM_PROMPT
from app.state import AgentState
from app.tools import rag_search, web_search

TOOLS = [rag_search, web_search]


def _create_agent_graph():
    llm = ChatOpenAI(
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        model_name=settings.model_name,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        # Системный промпт в начале каждого вызова
        full = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = llm_with_tools.invoke(full)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "agent")
    return builder.compile()


# Один граф на процесс
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _create_agent_graph()
    return _compiled_graph
