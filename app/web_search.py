"""Веб-поиск через Tavily API."""
from __future__ import annotations

from app.config import settings

_tavily_client = None


def _get_client():
    """Создаёт и кэширует клиент Tavily при первом вызове (если задан TAVILY_API_KEY)."""
    global _tavily_client
    if _tavily_client is None and settings.tavily_api_key:
        try:
            from tavily import TavilyClient
            _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
        except Exception as e:
            print(f"[web_search] Tavily недоступен: {e}")
    return _tavily_client


def search_web(query: str, max_results: int = 3) -> str:
    """Поиск в интернете через Tavily. Возвращает отформатированный текст для контекста LLM или сообщение об ошибке."""
    client = _get_client()
    if not query.strip():
        return ""
    if not client:
        return "[Веб-поиск недоступен: не настроен TAVILY_API_KEY. Ответь по базе знаний или скажи, что актуальных данных из интернета нет.]"

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
        return "[Веб-поиск временно недоступен. Опирайся на базу знаний или скажи пользователю попробовать позже.]"
