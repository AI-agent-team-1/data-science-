import os
from collections import defaultdict

import telebot
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from web_search import web_search, keys_for_search

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "z-ai/glm-4.5-air:free"
MAX_HISTORY_MESSAGES = 20  # сколько последних сообщений диалога помнить

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=OPENROUTER_BASE,
    model_name=MODEL_NAME,
)

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

# история сообщений для каждого чата
chat_histories = defaultdict(list)

SYSTEM_PROMPT = (
    "Ты дружелюбный русскоязычный помощник в Telegram. "
    "Веди себя как собеседник, помни контекст предыдущих сообщений "
    "и отвечай кратко и по делу."
)


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(
        message,
        "Привет! Я ИИ‑бот с доступом к интернету. Я могу:\n"
        "• Отвечать на вопросы, помня контекст беседы\n"
        "• Искать актуальную информацию в интернете (просто скажи 'найди...')\n"
        "• Рассказывать последние новости\n\n"
        "Попробуй спросить меня о чём-нибудь или попроси найти информацию!",
    )


@bot.message_handler(commands=["search"])
def handle_search_command(message):
    query = message.text.replace("/search", "", 1).strip()
    if not query:
        bot.reply_to(
            message,
            "Напиши что искать после команды /search. Например: /search последние новости технологий",
        )
        return

    perform_search(message, query)


def perform_search(message, query: str) -> None:
    try:
        chat_id = message.chat.id

        # Отправляем уведомление о поиске
        waiting_msg = bot.reply_to(message, "🔍 Ищу информацию в интернете...")

        # Выполняем поиск
        search_results = web_search(query)

        # Добавляем поиск в историю
        history = chat_histories[chat_id]

        # Формируем сообщения для LLM с контекстом поиска
        messages_for_llm = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Вот результаты поиска по запросу '{query}':\n"
                    f"{search_results}\n\n"
                    "Дай краткий ответ на основе этой информации, указав источники."
                )
            ),
        ]

        # Получаем ответ от модели
        response_msg = llm.invoke(messages_for_llm)
        response_text = response_msg.content

        # Удаляем сообщение 'ищу информацию'
        bot.delete_message(chat_id, waiting_msg.message_id)

        # Отправляем ответ с результатами
        bot.reply_to(
            message,
            f"🔍 Результаты поиска по запросу '{query}':\n\n{response_text}",
        )

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
        if keys_for_search(user_text):
            perform_search(message, user_text)
            return

        # Обычный ответ без поиска
        user_msg = HumanMessage(content=user_text)
        history.append(user_msg)

        messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + history[
            -MAX_HISTORY_MESSAGES:
        ]

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
