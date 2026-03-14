# Этот код делает гибридный поиск контекста для ответа пользователя

import os
from tavily import TavilyClient         # клиент для веб-поиска через Tavily API
from rag_search import get_rag_result   # функция локального поиска по базе знаний

import re
from prompts import RAG_PATTERNS, WEB_PATTERNS


# Эта функция нужна, чтобы безопасно подключиться к Tavily
def get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("Не найден TAVILY_API_KEY.")
    return TavilyClient(api_key=api_key)

# Эта функция превращает сырой ответ Tavily в один текстовый блок, который потом можно передать модели
def format_web_context(search_result: dict) -> str:
    # список кусков текста
    parts = []

    # Берёт краткий ответ Tavily
    answer = search_result.get("answer")
    if answer:
        parts.append(f"Краткий веб-ответ: {answer}")

    # Проходит по найденным источникам
    for i, item in enumerate(search_result.get("results", []), 1):
        title = item.get("title", "Без названия")
        content = item.get("content", "")
        url = item.get("url", "")
        parts.append(f"Источник {i}: {title}\n{content}\nURL: {url}")

    # Возвращает итоговый текст
    return "\n\n".join(parts) if parts else "Информация в интернете не найдена."

# Эта функция делает веб-поиск по запросу пользователя.
def get_web_result(user_query: str) -> dict:
    try:
        client = get_tavily_client()

        # Вызов поиска
        result = client.search(
            query=f"{user_query} нефтегазовая отрасль",
            max_results=3,
            topic="general",
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
        )

        return {
            "source": "web",
            "context": format_web_context(result),
            "reason": "web_fallback",
        }

    except Exception as e:
        return {
            "source": "web",
            "context": f"Веб-поиск сейчас недоступен: {e}",
            "reason": "web_exception",
        }


def detect_query_route(user_query: str) -> str:
    query = user_query.lower()

    rag_score = sum(bool(re.search(p, query)) for p in RAG_PATTERNS)
    web_score = sum(bool(re.search(p, query)) for p in WEB_PATTERNS)

    if web_score > rag_score and web_score > 0:
        return "web"

    if rag_score > web_score and rag_score > 0:
        return "rag"

    return "hybrid"


# Выбирает, откуда брать контекст для ответа
def get_best_context(user_query: str, k: int = 4, distance_threshold: float = 1.2) -> dict:
    
    route = detect_query_route(user_query)

    if route == "web":
        web_result = get_web_result(user_query)
        return {
            "source": web_result["source"],
            "context": web_result["context"],
            "reason": "router_web",
        }

    if route == "rag":
        rag_result = get_rag_result(
            user_query=user_query,
            k=k,
            distance_threshold=distance_threshold,
        )

        if rag_result["source"] == "rag":
            return {
                **rag_result,
                "reason": "router_rag",
            }

        web_result = get_web_result(user_query)
        return {
            "source": web_result["source"],
            "context": web_result["context"],
            "reason": f"web_fallback_after_{rag_result['reason']}",
        }

    # hybrid
    rag_result = get_rag_result(
        user_query=user_query,
        k=k,
        distance_threshold=distance_threshold,
    )

    if rag_result["source"] == "rag":
        return {
            **rag_result,
            "reason": f"hybrid_rag_{rag_result['reason']}",
        }

    web_result = get_web_result(user_query)
    return {
        "source": web_result["source"],
        "context": web_result["context"],
        "reason": f"hybrid_web_after_{rag_result['reason']}",
    }


    # # Сначала вызывается RAG
    # rag_result = get_rag_result(user_query=user_query,
    #                             k=k,
    #                             distance_threshold=distance_threshold,)

    # # Если локальная база дала хороший ответ, сразу возвращаем его
    # if rag_result["source"] == "rag":
    #     return rag_result

    # # Если RAG не справился, тогда запускается веб-поиск
    # web_result = get_web_result(user_query)


    # return {
    #     "source": web_result["source"],
    #     "context": web_result["context"],
    #     "reason": f"web_fallback_after_{rag_result['reason']}",
    # }