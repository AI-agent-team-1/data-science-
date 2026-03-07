import telebot
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    load_dotenv(".env.txt")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit(
        "Задай OPENROUTER_API_KEY в .env или .env.txt. Ключ берётся на openrouter.ai"
    )

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "z-ai/glm-4.5-air:free"

llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE,
    model=MODEL_NAME,
)

# Замените 'bot_token' на токен вашего бота, сохранённого в секрет колаба
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

def _format_api_error(err: Exception) -> str:
    """Понятное сообщение об ошибке API."""
    s = str(err).lower()
    if "401" in s or "user not found" in s or "unauthorized" in s:
        return (
            "Ошибка доступа к OpenRouter (401). Проверь:\n"
            "• Ключ OPENROUTER_API_KEY в .env / .env.txt верный и без лишних пробелов\n"
            "• Аккаунт на openrouter.ai активен, ключ создан в разделе Keys"
        )
    if "429" in s or "rate limit" in s:
        return "Слишком много запросов. Подожди минуту и попробуй снова."
    return f"Ошибка: {err}. Попробуй позже."


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        response = llm.invoke(message.text).content
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, _format_api_error(e))

# Запуск бота
bot.polling()