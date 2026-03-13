"""
Telegram-бот: приём сообщений, история диалога, вызов LLM (или агента из app).
"""
import telebot
from dotenv import load_dotenv
import os
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# Подключение к OpenRouter и лимит истории
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "z-ai/glm-4.5-air:free"
MAX_HISTORY_MESSAGES = 20  # сколько последних сообщений диалога помнить

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=OPENROUTER_BASE,
    model_name=MODEL_NAME,
)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
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
    """Отправляет приветствие на команды /start и /help."""
    bot.reply_to(
        message,
        "Привет! Я ИИ‑бот. Пиши мне сообщения — я буду отвечать, "
        "помня контекст нашей беседы.",
    )


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    """Любое текстовое сообщение: добавляет в историю, вызывает LLM, отправляет ответ."""
    try:
        chat_id = message.chat.id
        history = chat_histories[chat_id]

        user_msg = HumanMessage(content=message.text)
        history.append(user_msg)

        messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + history[-MAX_HISTORY_MESSAGES:]

        print(f"[{chat_id}] USER:", message.text)

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


bot.polling()
