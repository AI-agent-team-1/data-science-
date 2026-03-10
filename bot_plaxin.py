import telebot
from dotenv import load_dotenv
import os
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient
import json

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "z-ai/glm-4.5-air:free"
MAX_HISTORY_MESSAGES = 20  # сколько последних сообщений диалога помнить
MAX_SEARCH_RESULTS = 3  # сколько результатов поиска показывать

# Инициализация клиентов
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=OPENROUTER_BASE,
    model_name=MODEL_NAME,
)

# Инициализация Tavily клиента
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

# история сообщений для каждого чата
chat_histories = defaultdict(list)

SYSTEM_PROMPT = (
    "Ты дружелюбный русскоязычный помощник в Telegram. "
    "Веди себя как собеседник, помни контекст предыдущих сообщений "
    "и отвечай кратко и по делу.\n\n"
    "У тебя есть доступ к веб-поиску. Если пользователь просит найти актуальную информацию, "
    "новости, данные из интернета или использует слова 'найди', 'поищи', 'погугли', 'что нового', "
    "'актуальная информация' — используй результаты поиска, которые будут предоставлены в сообщении. "
    "Отвечай на основе найденной информации, указывая источники, если это уместно."
)

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


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(
        message,
        "Привет! Я ИИ‑бот с доступом к интернету. Я могу:\n"
        "• Отвечать на вопросы, помня контекст беседы\n"
        "• Искать актуальную информацию в интернете (просто скажи 'найди...')\n"
        "• Рассказывать последние новости\n\n"
        "Попробуй спросить меня о чём-нибудь или попроси найти информацию!"
    )

@bot.message_handler(commands=["search"])
def handle_search_command(message):
    query = message.text.replace('/search', '', 1).strip()
    if not query:
        bot.reply_to(message, "Напиши что искать после команды /search. Например: /search последние новости технологий")
        return
    
    perform_search(message, query)

def perform_search(message, query):
    """Выполняет поиск и отправляет результаты"""
    try:
        chat_id = message.chat.id
        
        # Отправляем уведомление о поиске
        waiting_msg = bot.reply_to(message, "🔍 Ищу информацию в интернете...")
        
        # Выполняем поиск
        search_results = search_web(query)
        
        # Добавляем поиск в историю
        history = chat_histories[chat_id]
        
        # Создаем сообщение с результатами поиска
        search_context = f"Поисковый запрос: {query}\n\n{search_results}\n\nНа основе этой информации дай ответ пользователю."
        
        # Формируем сообщения для LLM с контекстом поиска
        messages_for_llm = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Вот результаты поиска по запросу '{query}':\n{search_results}\n\nДай краткий ответ на основе этой информации, указав источники.")
        ]
        
        # Получаем ответ от модели
        response_msg = llm.invoke(messages_for_llm)
        response_text = response_msg.content
        
        # Удаляем сообщение "ищу информацию"
        bot.delete_message(chat_id, waiting_msg.message_id)
        
        # Отправляем ответ с результатами
        bot.reply_to(message, f"🔍 Результаты поиска по запросу '{query}':\n\n{response_text}")
        
        # Сохраняем в историю
        history.append(HumanMessage(content=f"[ПОИСК] {query}"))
        history.append(AIMessage(content=response_text))
        
        # Ограничиваем историю
        if len(history) > MAX_HISTORY_MESSAGES:
            chat_histories[chat_id] = history[-MAX_HISTORY_MESSAGES:]
            
    except Exception as e:
        bot.reply_to(message, f"Ошибка при поиске: {str(e)}. Попробуйте позже.")

@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        chat_id = message.chat.id
        history = chat_histories[chat_id]
        user_text = message.text

        print(f"[{chat_id}] USER:", user_text)

        # Проверяем, нужен ли веб-поиск
        if needs_web_search(user_text):
            perform_search(message, user_text)
            return

        # Обычный ответ без поиска
        user_msg = HumanMessage(content=user_text)
        history.append(user_msg)

        messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + history[-MAX_HISTORY_MESSAGES:]

        response_msg = llm.invoke(messages_for_llm)
        response_text = response_msg.content

        print(f"[{chat_id}] BOT:", response_text)

        if isinstance(response_msg, AIMessage):
            history.append(response_msg)
        else:
            history.append(AIMessage(content=response_text))

        if len(history) > MAX_HISTORY_MESSAGES:
            chat_histories[chat_id] = history[-MAX_HISTORY_MESSAGES:]

        bot.reply_to(message, response_text)

    except Exception as e:
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")

if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling()