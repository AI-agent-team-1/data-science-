"""Запуск агента: RAG + LLM с историей диалога."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.prompts import SYSTEM_PROMPT
from rag import (
    build_knowledge_base,
    load_or_build_faiss_index,
    retrieve_context,
)

llm = ChatOpenAI(
    openai_api_key=settings.openrouter_api_key,
    openai_api_base=settings.openrouter_base_url,
    model_name=settings.model_name,
)

knowledge_chunks = build_knowledge_base()
faiss_store = load_or_build_faiss_index(knowledge_chunks)
chat_histories: dict[int, List[Any]] = defaultdict(list)


def run_agent(user_text: str, chat_id: int) -> str:
    history = chat_histories[chat_id]
    history.append(HumanMessage(content=user_text))

    rag_context = retrieve_context(
        knowledge_chunks, user_text, vectorstore=faiss_store
    )
    messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    if rag_context:
        messages.append(SystemMessage(content=rag_context))
    messages.extend(history[-settings.max_history_messages :])

    print(f"[{chat_id}] USER:", user_text)
    response_msg = llm.invoke(messages)
    response_text = response_msg.content
    print(f"[{chat_id}] BOT:", response_text)

    if isinstance(response_msg, AIMessage):
        history.append(response_msg)
    else:
        history.append(AIMessage(content=response_text))

    if len(history) > settings.max_history_messages:
        chat_histories[chat_id] = history[-settings.max_history_messages :]

    return response_text
