"""Telegram-бот: хендлеры и вызов агента."""
import telebot

from config import settings
from prompts import WELCOME_MESSAGE
from run_agent import run_agent

bot = telebot.TeleBot(settings.telegram_token)


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(message, WELCOME_MESSAGE)


@bot.message_handler(func=lambda m: True)
def handle_text(message):
    try:
        answer = run_agent(message.text, message.chat.id)
        bot.reply_to(message, answer)
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}. Попробуйте позже.")


if __name__ == "__main__":
    bot.infinity_polling()
