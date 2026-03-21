"""Эмбеддинги для RAG (OpenRouter API)."""
from __future__ import annotations

import os
from typing import Any, Optional

from app.config import rag_settings

OPENROUTER_EMBEDDINGS_BASE = "https://openrouter.ai/api/v1"


def get_embeddings() -> Optional[Any]:
    """Создаёт клиент эмбеддингов OpenRouter (для FAISS). При отсутствии ключа — None."""
    try:
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_EMBEDDINGS_BASE,
            model=rag_settings.embedding_model,
        )
    except Exception as e:
        print(f"[RAG] Эмбеддинги недоступны: {e}")
        return None
