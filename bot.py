"""
Telegram-бот: приём сообщений, вызов агента (app.agent.run_agent) с RAG и web_search.
Системный промпт и приветствие — в app.agent.prompts.
"""
from __future__ import annotations

import logging
import re

import telebot
from telebot import apihelper
from telebot.apihelper import ApiTelegramException

from app.config import settings, validate_required_for_telegram_bot
from app.agent.prompts import HELP_MESSAGE, WELCOME_MESSAGE
from app.agent.run_agent import clear_chat_history, run_agent
from app.agent.tools import init_rag
from app.chat_history_sqlite import init_schema
from app.guardrails_ai import guard_input, guard_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

validate_required_for_telegram_bot()
init_schema()
init_rag()

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

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.*?)__", r"\1", text, flags=re.DOTALL)
    text = text.replace("`", "")
    return text.strip()


def _send_reply(message, text: str) -> bool:
    """Отправляет ответ. При 403 (пользователь заблокировал бота) логирует и возвращает False."""
    try:
        bot.reply_to(message, text, parse_mode="HTML")
        return True
    except ApiTelegramException as e:
        if e.error_code == 403 and "blocked by the user" in (e.description or ""):
            logger.warning("[%s] Пользователь заблокировал бота, ответ не отправлен.", message.chat.id)
        else:
            logger.warning("[%s] Telegram API: %s", message.chat.id, e)
        return False


@bot.message_handler(commands=["start"])
def handle_start(message):
    """Отправляет приветствие из app.agent.prompts."""
    _send_reply(message, WELCOME_MESSAGE)


@bot.message_handler(commands=["help"])
def handle_help(message):
    """Отправляет справку из app.agent.prompts."""
    _send_reply(message, HELP_MESSAGE)


@bot.message_handler(commands=["reset"])
def handle_reset(message):
    """Очищает историю диалога для текущего чата."""
    clear_chat_history(message.chat.id)
    _send_reply(message, "История диалога очищена. Можете продолжить с чистого листа.")


@bot.message_handler(content_types=["text"])
def handle_text(message):
    """Текстовые сообщения передаём агенту (граф с rag_search, web_search), отправляем ответ."""
    try:
        if not message.text:
            return
        input_check = guard_input(message.text)
        if not input_check.allowed:
            _send_reply(message, input_check.text or "")
            return
        answer = run_agent(message.text, message.chat.id)
        output_check = guard_output(answer)
        if not output_check.allowed:
            answer = output_check.text or ""
        _send_reply(message, cleanup_markdown(answer))
    except Exception:
        logger.exception("[bot] handle_text failed")
        _send_reply(message, "Ошибка. Попробуйте позже.")


if __name__ == "__main__":
    bot.infinity_polling()
