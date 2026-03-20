"""
Telegram-бот: приём сообщений, вызов агента (app.run_agent) с RAG и web_search.
Системный промпт и приветствие — в app.prompts.
"""
import telebot
import re
from telebot import apihelper
from telebot.apihelper import ApiTelegramException

from app.config import settings
from app.prompts import HELP_MESSAGE, WELCOME_MESSAGE
from app.run_agent import run_agent

# Увеличенные таймауты: при медленной сети поднимаем CONNECT/READ
apihelper.CONNECT_TIMEOUT = 30
apihelper.READ_TIMEOUT = 60

bot = telebot.TeleBot(settings.telegram_token)

def cleanup_markdown(text: str) -> str:
    """
    Мягкая очистка ответа от наиболее частых Markdown-паттернов,
    которые могут визуально искажать Telegram (например, **жирный**).
    """
    if not isinstance(text, str):
        text = str(text)

    # Убираем только bold-формат: **...** и __...__
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.*?)__", r"\1", text, flags=re.DOTALL)

    # Убираем inline-code бэктики
    text = text.replace("`", "")

    return text.strip()


def _send_reply(message, text: str) -> bool:
    """Отправляет ответ. При 403 (пользователь заблокировал бота) логирует и возвращает False."""
    try:
        # Используем HTML mode, чтобы кликабельные ссылки из "Источник:" работали.
        bot.reply_to(message, text, parse_mode="HTML")
        return True
    except ApiTelegramException as e:
        if e.error_code == 403 and "blocked by the user" in (e.description or ""):
            print(f"[{message.chat.id}] Пользователь заблокировал бота, ответ не отправлен.")
        else:
            print(f"[{message.chat.id}] Telegram API: {e}")
        return False


@bot.message_handler(commands=["start"])
def handle_start(message):
    """Отправляет приветствие из app.prompts."""
    _send_reply(message, WELCOME_MESSAGE)


@bot.message_handler(commands=["help"])
def handle_help(message):
    """Отправляет справку из app.prompts."""
    _send_reply(message, HELP_MESSAGE)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    """Текстовые сообщения передаём агенту (граф с rag_search, web_search), отправляем ответ."""
    try:
        if not message.text:
            return
        answer = run_agent(message.text, message.chat.id)
        _send_reply(message, cleanup_markdown(answer))
    except Exception as e:
        # Пользователю не показываем детали исключения (снижаем риск утечек).
        print(f"[bot] handle_text failed: {e}")
        _send_reply(message, "Ошибка. Попробуйте позже.")


if __name__ == "__main__":
    bot.infinity_polling()