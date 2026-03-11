# data-science-
# Telegram LLM Bot

Простой Telegram-бот, который отправляет сообщения пользователя в LLM через OpenRouter и возвращает ответ.

## Стек
- Python
- pyTelegramBotAPI
- LangChain
- OpenRouter API

## Установка

```bash
git clone https://github.com/yourusername/telegram-llm-bot.git
cd telegram-llm-bot
pip install -r requirements.txt

```
## Настройка

Создайте файл .env:
```bash
OPENROUTER_API_KEY=your_openrouter_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```
Ключ OpenRouter можно получить на https://openrouter.ai

## Запуск

python bot.py

## Модель

По умолчанию используется модель:
z-ai/glm-4.5-air:free

При необходимости можно изменить модель в bot.py.
