# Telegram-бот и веб-чат с RAG

**Консультант в нефтегазовой сфере** — технический ассистент, который опирается на локальную базу знаний (PDF/DOCX в `docs/`) и при необходимости на веб-поиск. Ответы ведутся через **LangGraph**: модель сама выбирает `rag_search`, `web_search` или прямой ответ.

Доступны два интерфейса: **Telegram** (`bot.py`) и **веб-чат** (FastAPI + статика в `web/static/`), оба используют один и тот же агент (`app.agent.run_agent`).

---

## Возможности

- История диалога: **SQLite** (`data/chat_history.db` или путь из `CHAT_HISTORY_DB`), TTL 24 часа, до 100 активных чатов (LRU), внутри чата лимит `MAX_HISTORY_MESSAGES`.
- **Telegram:** команды `/start`, `/help`, **`/reset`** — очистка истории в текущем чате. При старте проверяются обязательные переменные (в т.ч. `TELEGRAM_BOT_TOKEN`, `OPENROUTER_API_KEY`).
- **Веб:** `http://localhost:8000` после запуска uvicorn; API `POST /api/chat`, `POST /api/clear`, `GET /api/health`.
- **Инструменты агента:** `rag_search` (FAISS + эмбеддинги OpenRouter или keyword fallback), `web_search` (Tavily). Вызовы ограничены **`TOOL_TIMEOUT_SEC`** (по умолчанию 20 с).
- **RAG:** при старте приложения вызывается **`init_rag()`** — загрузка чанков из `docs/` и загрузка/построение FAISS в `rag_faiss_index/` (до первого пользовательского запроса).
- Опционально **Langfuse** — см. `LangFuse_observability.md` и `LANGFUSE_*` в `.env`.

---

## Стек

- Python **3.11+**
- **pyTelegramBotAPI**, **FastAPI**, **Uvicorn**
- **LangGraph** / **LangChain** — граф агента, tools, FAISS
- **OpenRouter** — чат и эмбеддинги
- **FAISS** (faiss-cpu), **Tavily**, **pypdf**, **python-docx**
- **SQLite** — персистентная история чатов

Версии прямых зависимостей зафиксированы в **`requirements.txt`**.

---

## Установка

1. Клонируйте репозиторий (подставьте свой URL):

   ```bash
   git clone https://github.com/AI-agent-team-1/data-science-.git
   cd data-science-
   ```

2. Виртуальное окружение и зависимости:

   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\pip.exe install -r requirements.txt
   # Linux/macOS:
   # source venv/bin/activate && pip install -r requirements.txt
   ```

3. Файл **`.env`** в корне (шаблон — `env.example`):

   | Переменная | Назначение |
   |------------|------------|
   | `OPENROUTER_API_KEY` | Обязательно для агента |
   | `TELEGRAM_BOT_TOKEN` | Обязательно для Telegram-бота |
   | `TAVILY_API_KEY` | Веб-поиск (без ключа поиск недоступен, ответ по RAG возможен) |
   | `MAX_HISTORY_MESSAGES` | Лимит сообщений в контексте (по умолчанию 20) |
   | `CHAT_HISTORY_DB` | Путь к файлу SQLite (необязательно; иначе `data/chat_history.db`) |
   | `TOOL_TIMEOUT_SEC` | Таймаут tools (по умолчанию 20) |

   Ключи: [OpenRouter](https://openrouter.ai), [@BotFather](https://t.me/BotFather), [Tavily](https://tavily.com).

4. Документы базы знаний — в папку **`docs/`** (`.pdf`, `.docx`).

---

## Запуск локально

**Только Telegram-бот:**

```bash
python bot.py
```

**Только веб** (из корня репозитория, с тем же `.env`):

```bash
python -m uvicorn web.api:app --reload --port 8000
```

Откройте `http://127.0.0.1:8000`.

На Windows удобно **`scripts\run_bot.ps1`** для бота.

---

## Docker (VPS)

Один сервис **`llm-bot`**: параллельно **Telegram-бот** и **uvicorn** на порту **8000** (см. `docker-compose.yml`).

Тома:

- `./docs` → `/app/docs`
- `./rag_faiss_index` → `/app/rag_faiss_index`
- `./data` → `/app/data` (SQLite истории; в compose задано `CHAT_HISTORY_DB=/app/data/chat_history.db`)

Сеть **`llm_shared`** должна существовать, если указана в compose (для Langfuse рядом с ботом):

```bash
docker network create llm_shared
```

Запуск:

```bash
mkdir -p docs rag_faiss_index data
docker compose up -d --build
```

Логи и остановка:

```bash
docker compose logs -f llm-bot
docker compose stop llm-bot
```

---

## Деплой на VPS

Секреты только в **`.env`** на сервере (не в git).

```bash
chmod +x scripts/deploy.sh   # при необходимости
./scripts/deploy.sh
```

Скрипт делает `git pull`, создаёт каталоги `docs`, `rag_faiss_index`, **`data`**, затем `docker compose up -d --build` и при наличии `curl` проверяет `/api/health`.

---

## Структура проекта

```
data-science-/
├── app/
│   ├── config.py
│   ├── web_search.py
│   ├── chat_history_sqlite.py  # История диалогов (SQLite)
│   ├── agent/
│   │   ├── prompts.py
│   │   ├── state.py
│   │   ├── tools.py            # rag_search, web_search, init_rag()
│   │   ├── graph.py
│   │   └── run_agent.py
│   ├── rag/
│   └── observability/          # Langfuse и др.
├── web/
│   ├── api.py                  # FastAPI + статика
│   └── static/index.html
├── deploy/langfuse/            # docker-compose Langfuse 3.x
├── scripts/
│   ├── run_bot.ps1
│   └── deploy.sh
├── bot.py
├── rag.py                      # Реэкспорт app.rag (совместимость со старыми импортами)
├── docs/
├── data/                       # SQLite (в .gitignore)
├── rag_faiss_index/            # FAISS (в .gitignore)
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── env.example
├── LangFuse_observability.md
├── CODE_REVIEW_FOLLOWUP.md     # Что сделано по замечаниям ревью
└── .env                        # не коммитить
```

---

## Как устроен агент

- **Граф:** узлы `agent` (LLM + tools) и `tools` (ToolNode); `tools_condition` решает, идти ли в tools или завершить.
- **История:** загружается из SQLite, в граф передаётся список сообщений; после ответа цепочка сохраняется обратно (с обрезкой по `MAX_HISTORY_MESSAGES`).
- **RAG:** FAISS загружается с диска с `allow_dangerous_deserialization=True` (локальный индекс своего проекта); при ошибках — keyword fallback.
- Ответы в Telegram проходят мягкую очистку Markdown-символов (`cleanup_markdown` в `bot.py`).

---

## См. также

- **`LangFuse_observability.md`** — версии SDK/сервера, переменные, VPS.
- **`CODE_REVIEW_FOLLOWUP.md`** — ответы по пунктам внутреннего код-ревью.
