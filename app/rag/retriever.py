"""Поиск по RAG: FAISS или keyword-fallback."""
from __future__ import annotations

from typing import Any, List, Optional

from app.config import rag_settings

RAG_HEADER = (
    "Вот выдержки из локальной базы знаний (документы из папки docs). "
    "Отвечай, опираясь прежде всего на них. Если информации недостаточно, "
    "честно скажи, чего не хватает.\n\n"
)


def _normalize_word(w: str) -> str:
    """Приводит слово к нижнему регистру, оставляет буквы, цифры, дефис и ё."""
    return "".join(c for c in w.lower() if c.isalnum() or c in "ё-")


def _format_context_parts(items: List[tuple[str, str]], limit: int) -> List[str]:
    """Форматирует список (источник, текст) в блоки для контекста с лимитом по символам."""
    parts = []
    total = 0
    for idx, (source, text) in enumerate(items, start=1):
        part = f"[{idx}]\nИсточник: {source}\n{text}\n"
        if total + len(part) > limit:
            break
        parts.append(part)
        total += len(part)
    return parts


def _score_chunk(query_tokens: List[str], chunk_text: str) -> int:
    """Оценка релевантности чанка запросу: совпадение слов даёт баллы (fallback без FAISS)."""
    text_lower = chunk_text.lower()
    chunk_words = [_normalize_word(w) for w in text_lower.split() if len(w) > 1]
    score = 0
    for token in query_tokens:
        if not token:
            continue
        t = _normalize_word(token)
        if len(t) < 2:
            continue
        if t in text_lower:
            score += 2
        elif any(t in w or w in t for w in chunk_words):
            score += 1
    return score


def _fallback_chunks(
    chunks: List[dict[str, Any]],
    n: int,
) -> List[dict[str, Any]]:
    """Выбирает до n чанков из разных источников, если по ключевым словам ничего не подошло."""
    seen_sources: set[str] = set()
    result: List[dict[str, Any]] = []
    for c in chunks:
        if c["source"] not in seen_sources and len(result) < n:
            result.append(c)
            seen_sources.add(c["source"])
        if len(result) >= n:
            break
    if len(result) < n:
        for c in chunks:
            if c not in result and len(result) < n:
                result.append(c)
    return result


def retrieve_context(
    knowledge_chunks: List[dict[str, Any]],
    query: str,
    k: int | None = None,
    max_chars: int | None = None,
    vectorstore: Optional[Any] = None,
) -> str:
    """
    Ищет по запросу: при наличии vectorstore — FAISS, иначе по ключевым словам + fallback.
    Возвращает текст для контекста LLM.
    """
    k = k if k is not None else rag_settings.top_k
    limit = max_chars or rag_settings.max_context_chars

    if vectorstore is not None:
        try:
            docs = vectorstore.similarity_search(query, k=k)
            items = [(doc.metadata.get("source", "?"), doc.page_content) for doc in docs]
            parts = _format_context_parts(items, limit)
            if parts:
                return RAG_HEADER + "\n\n".join(parts)
        except Exception as e:
            print(f"[RAG] Ошибка FAISS, fallback на ключевые слова: {e}")

    if not knowledge_chunks:
        return ""

    tokens = [w for t in query.split() if t.strip() for w in [_normalize_word(t)] if len(w) >= 2]
    scored = [(_score_chunk(tokens, c["text"]), c) for c in knowledge_chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = (
        [c for _, c in scored[:k]]
        if scored and scored[0][0] > 0
        else _fallback_chunks(knowledge_chunks, n=k)
    )
    parts = _format_context_parts([(c["source"], c["text"]) for c in top_chunks], limit)
    if not parts:
        return ""
    return RAG_HEADER + "\n\n".join(parts)
