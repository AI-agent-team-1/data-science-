from dotenv import load_dotenv
import os
from tavily import TavilyClient

load_dotenv()

MAX_SEARCH_RESULTS = 3  # сколько результатов поиска показывать

# Инициализация Tavily клиента
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search(query: str) -> str:
    try:
        print(f"[SEARCH] Запрос: {query}")
        
        # Выполняем поиск
        search_result = tavily_client.search(
            query=query,
            search_depth="advanced",  # Более глубокий поиск
            max_results=MAX_SEARCH_RESULTS,
            include_answer=False,  # Не включаем сводный ответ
            include_raw_content=False  # Не включаем сырой контент для экономии токенов
        )
        
        # Форматируем результаты
        formatted_results = "Вот что удалось найти в интернете:\n\n"
        
        for i, result in enumerate(search_result.get('results', []), 1):
            title = result.get('title', 'Без заголовка')
            content = result.get('content', 'Нет описания')
            url = result.get('url', '')
            
            formatted_results += f"{i}. {title}\n"
            formatted_results += f"   {content[:300]}..." if len(content) > 300 else f"   {content}"
            if url:
                formatted_results += f"\n   Источник: {url}"
            formatted_results += "\n\n"
        
        if not search_result.get('results'):
            return "По вашему запросу ничего не найдено."
        
        return formatted_results
        
    except Exception as e:
        print(f"[SEARCH ERROR] {str(e)}")
        return f"Ошибка при поиске: {str(e)}"


def keys_for_search(text: str) -> bool:
    search_keywords = [
        'найди', 'поищи', 'погугли', 'найти', 'поиск',
        'новости', 'последние', 'актуальный', 'свежий',
        'что происходит', 'что нового', 'в интернете',
        'в сети', 'check', 'search', 'find', 'google',
        'latest', 'news', 'current', 'update'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in search_keywords)
