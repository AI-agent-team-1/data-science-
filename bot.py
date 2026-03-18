"""
Telegram-бот: приём сообщений, вызов агента (app.run_agent) с RAG и web_search.
Системный промпт и приветствие — в app.prompts.
"""
import telebot
from telebot import apihelper
from telebot.apihelper import ApiTelegramException

from app.config import settings
from app.prompts import WELCOME_MESSAGE
from app.run_agent import run_agent

# Увеличенные таймауты: при медленной сети поднимаем CONNECT/READ
apihelper.CONNECT_TIMEOUT = 30
apihelper.READ_TIMEOUT = 60

bot = telebot.TeleBot(settings.telegram_token)


def _send_reply(message, text: str) -> bool:
    """Отправляет ответ. При 403 (пользователь заблокировал бота) логирует и возвращает False."""
    try:
        bot.reply_to(message, text)
        return True
    except ApiTelegramException as e:
        if e.error_code == 403 and "blocked by the user" in (e.description or ""):
            print(f"[{message.chat.id}] Пользователь заблокировал бота, ответ не отправлен.")
        else:
            print(f"[{message.chat.id}] Telegram API: {e}")
        return False


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    """Отправляет приветствие из app.prompts."""
    _send_reply(message, WELCOME_MESSAGE)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    """Текстовые сообщения передаём агенту (граф с rag_search, web_search), отправляем ответ."""
    try:
        if not message.text:
            return
        answer = run_agent(message.text, message.chat.id)
        _send_reply(message, answer)
    except Exception as e:
        _send_reply(message, f"Ошибка: {e}. Попробуйте позже.")


if __name__ == "__main__":
    bot.infinity_polling()