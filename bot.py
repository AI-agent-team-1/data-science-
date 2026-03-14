import os                                    # чтобы читать переменные окружения, например токены
import telebot                               # библиотека для Telegram-бота
from dotenv import load_dotenv               # загружает переменные из файла .env

from langchain_openai import ChatOpenAI      # интерфейс для общения с языковой моделью
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # специальные типы сообщений для истории диалога

from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE, HELP_MESSAGE # системный промпт
from web_search import get_best_context      # ищет лучший контекст: из RAG или из веба
import re


# Загрузка .env
load_dotenv()

# Настройки модели
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "z-ai/glm-4.5-air:free"

# Создание объекта LLM
llm = ChatOpenAI(openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                 openai_api_base=OPENROUTER_BASE,
                 model_name=MODEL_NAME,)

# Создание Telegram-бота
BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

# Словари для состояния
chat_history = {}

# Откуда пришёл контекст: RAG, веба
def get_source_prefix(context_source: str) -> str:
    if context_source == "rag":
        return "📚 Источник: локальная база документов"
    if context_source == "web":
        return "🌐 Источник: интернет"
    return ""

# Список команд, которые Telegram показывает пользователю в меню бота
bot.set_my_commands([telebot.types.BotCommand("start", "Перезапуск бота"),
                     telebot.types.BotCommand("help", "Что умеет бот"),])


@bot.message_handler(commands=["start"])
def handle_start(message):
    chat_id = message.chat.id
    chat_history.pop(chat_id, None)
    # bot.reply_to(message, WELCOME_MESSAGE)
    bot.reply_to(message, WELCOME_MESSAGE, parse_mode=None)
    # bot.reply_to(message, WELCOME_MESSAGE, reply_markup=telebot.types.ReplyKeyboardRemove())


@bot.message_handler(commands=["help"])
def handle_help(message):
    # bot.reply_to(message, HELP_MESSAGE)
    bot.reply_to(message, HELP_MESSAGE, parse_mode=None)


def get_system_prompt(context: str) -> str:
    return SYSTEM_PROMPT.format(context=context)


def cleanup_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("`", "")
    return text.strip()


# Основной обработчик сообщений
@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        # Получение данных сообщения
        chat_id = message.chat.id
        user_message = message.text

        # Проверка, что сообщение текстовое
        if not user_message:
            bot.reply_to(message, "Пришлите текстовый вопрос.")
            return

        # Игнорирование команд, значит это команда
        if user_message.startswith("/"):
            return

        # Инициализация истории чата
        if chat_id not in chat_history:
            chat_history[chat_id] = []

        # Получение лучшего контекста
        context_data = get_best_context(user_query=user_message,
                                        k=4,
                                        distance_threshold=1.2,)

        # Извлечение данных из результата
        final_context = context_data["context"]
        context_source = context_data["source"]

        # Добавление сообщения пользователя в историю
        chat_history[chat_id].append(HumanMessage(content=user_message))

        # Ограничение длины истории
        if len(chat_history[chat_id]) > 10:
            chat_history[chat_id] = chat_history[chat_id][-10:]

        # Получение системного промпта
        system_prompt = get_system_prompt(final_context)

        # Формирование полного списка сообщений
        messages = [SystemMessage(content=system_prompt)] + chat_history[chat_id]

        # print(f"User ({chat_id}): {user_message}")
        # print(f"Context source: {context_source}")
        # print(f"Context reason: {context_data.get('reason')}")
        # print(f"Context preview: {final_context[:300]}...")

        # Вызов модели. Вся собранная информация уходит в LLM, и модель возвращает ответ
        # response = llm.invoke(messages).content
        raw_response = llm.invoke(messages).content
        response = cleanup_markdown(raw_response)

        # Сохранение ответа бота в историю
        chat_history[chat_id].append(AIMessage(content=response))

        # Ограничение истории
        if len(chat_history[chat_id]) > 10:
            chat_history[chat_id] = chat_history[chat_id][-10:]

        # Подготовка ответа пользователю
        source_prefix = get_source_prefix(context_source)
        # source_prefix = ''
        display_response = f"{source_prefix}\n\n{response}" if source_prefix else response

        # Ответ пользователю в Telegram
        # bot.reply_to(message, display_response)
        bot.reply_to(message, display_response, parse_mode=None)

    # Обработка ошибок
    except Exception as e:
        # print(error_msg)
        bot.reply_to(message, f"Ошибка: {e}. Попробуйте позже.")

# Запуск бота
bot.polling()