# Telegram LLM Bot для нефтегазовой отрасли

Telegram-бот на Python, который отвечает на вопросы в нефтегазовой тематике с использованием LLM через OpenRouter.

Бот умеет:
- отвечать на вопросы по локальной базе документов;
- искать информацию через RAG по проиндексированным PDF и DOCX;
- использовать веб-поиск как запасной источник, если локального контекста недостаточно;
- работать в Telegram в формате диалога.

## Возможности

- Telegram-интерфейс через `pyTelegramBotAPI`
- LLM через `OpenRouter`
- Локальная база знаний на `ChromaDB`
- Индексация документов из папки `docs/`
- Поддержка PDF и DOCX
- Fallback на веб-поиск через Tavily
- История диалога по каждому чату

## Стек

- Python
- pyTelegramBotAPI
- LangChain
- OpenRouter API
- ChromaDB
- sentence-transformers
- Tavily API
- pypdf
- python-docx


## Установка


