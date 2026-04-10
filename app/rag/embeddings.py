"""Эмбеддинги для RAG (OpenRouter API)."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from app.config import rag_settings

logger = logging.getLogger(__name__)

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
    except ImportError as e:
        logger.error("[RAG] langchain_openai недоступен: %s", e)
        return None
    except (ValueError, TypeError) as e:
        logger.error("[RAG] Эмбеддинги не настроены: %s", e, exc_info=True)
        return None
