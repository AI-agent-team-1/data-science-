"""Веб-поиск через Tavily API."""
from __future__ import annotations

from app.config import settings

_tavily_client = None


def _get_client():
    global _tavily_client
    if _tavily_client is None and settings.tavily_api_key:
        try:
            from tavily import TavilyClient
            _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
        except Exception as e:
            print(f"[web_search] Tavily недоступен: {e}")
    return _tavily_client


def search_web(query: str, max_results: int = 3) -> str:
    """
    Поиск в интернете через Tavily. Возвращает текст для контекста LLM или пустую строку.
    """
    client = _get_client()
    if not client or not query.strip():
        return ""

    try:
        response = client.search(query, max_results=max_results)
        results = response.get("results") or []
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results[:max_results], start=1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            parts.append(f"[{i}] {title}\n{content}\nИсточник: {url}")
        header = "Результаты веб-поиска (актуальная информация из интернета):\n\n"
        return header + "\n\n".join(parts)
    except Exception as e:
        print(f"[web_search] Ошибка: {e}")
        return ""
