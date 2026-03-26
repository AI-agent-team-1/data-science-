# 🤖 Telegram-бот с RAG

**Консультант в нефтегазовой сфере** — технический ассистент, который отвечает только на вопросы в области нефтегазовой тематики. Использует официальную информацию из локальной базы знаний (инструкции, регламенты, СТО и т.п.), не выдумывает факты; при недостатке данных честно сообщает об этом.

Реализован на Python с **RAG** и **агентской логикой на LangGraph**: модель сама решает, когда искать в базе знаний (`rag_search`), когда искать в интернете (`web_search`) и когда дать финальный ответ. Цикл «думай → вызов инструмента → снова думай» повторяется до итогового ответа пользователю.

---

## ✨ Возможности

- 💬 Чат с учётом истории диалога (TTL 24 часа + LRU до 100 chat_id; внутри диалога лимит `MAX_HISTORY_MESSAGES`).
- **Агент на LangGraph** с двумя инструментами (tools):
  - **`rag_search`** — поиск в локальной базе знаний (PDF, DOCX из папки `docs`): семантический поиск (FAISS + эмбеддинги OpenRouter) или fallback по ключевым словам.
  - **`web_search`** — веб-поиск через Tavily (до 3 результатов на запрос) для актуальной информации.
- Модель сама выбирает: вызвать RAG, веб-поиск, оба по очереди или сразу ответить; цикл выполняется, пока не будет дан финальный ответ без вызовов инструментов.
- 📄 База знаний: PDF и DOCX из папки `docs` (FAISS-индекс создаётся/подхватывается при первом вызове `rag_search` в `rag_faiss_index/`).
- Опционально: **Langfuse** — трейсы запросов (LLM + tools), см. `LangFuse_observability.md` и переменные `LANGFUSE_*` в `.env`.

---

## 🛠 Стек

- **Python 3.11+**
- **pyTelegramBotAPI** — Telegram Bot API
- **LangGraph** — граф агента (узлы agent / tools, условные переходы по tool calls)
- **LangChain** (langchain-openai, langchain-community) — LLM и векторное хранилище
- **OpenRouter** — чат (модель по умолчанию: `z-ai/glm-4.5-air:free`) и эмбеддинги (`openai/text-embedding-3-small`)
- **FAISS** (faiss-cpu) — векторный поиск по базе знаний
- **Tavily** (tavily-python) — веб-поиск для актуальной информации
- **pypdf**, **python-docx** — чтение документов

---

## 📦 Установка

1. **Клонируйте репозиторий** и перейдите в папку проекта:

   ```bash
   git clone https://github.com/AI-agent-team-1/data-science-.git
   cd data-science-
   ```

2. **Создайте виртуальное окружение** и установите зависимости (обязательно вызывайте `pip` из venv проекта):

   ```bash
   python -m venv venv
   .\venv\Scripts\pip.exe install -r requirements.txt
   ```

3. **Создайте файл `.env`** в корне проекта:

   ```
   OPENROUTER_API_KEY=ваш_ключ_openrouter
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   MODEL_NAME=z-ai/glm-4.5-air:free
   TELEGRAM_BOT_TOKEN=токен_бота_от_BotFather
   MAX_HISTORY_MESSAGES=20
   TAVILY_API_KEY=ваш_ключ_tavily
   ```

   🔑 Ключ OpenRouter: [openrouter.ai](https://openrouter.ai)  
   🤖 Токен бота: [@BotFather](https://t.me/BotFather) в Telegram.  
   🌐 Ключ Tavily (для веб-поиска): [tavily.com](https://tavily.com) — 1000 бесплатных запросов в месяц.

   Опционально в `.env` можно задать `TOOL_TIMEOUT_SEC` (таймаут вызовов tools, по умолчанию 20). Модель должна поддерживать **tool calling** (вызов инструментов).

4. **Положите документы** базы знаний в папку **`docs`** (внутри проекта). Поддерживаются форматы **.pdf** и **.docx**.

---

## 🚀 Запуск

Единственная точка входа — **bot.py** (настройки в `app/config.py`, агент и промпты — в `app/agent/`):

```bash
.\venv\Scripts\python.exe bot.py
```

На Windows можно также запустить **`scripts\run_bot.ps1`** (поднимает venv и `bot.py`) или активировать venv (`.\venv\Scripts\Activate.ps1`) и выполнить `python bot.py`.

RAG инициализируется **лениво**: индексация/загрузка FAISS начинается при первом вызове `rag_search` (а не при старте модуля).
FAISS индекс хранится в `rag_faiss_index/` и подхватывается при повторных запусках.
При проблемах загрузки индекса используется fallback по ключевым словам.
FAISS индекс загружается **без** `allow_dangerous_deserialization`.
Перед отправкой ответ проходит мягкую очистку `**bold**` / `__bold__` и обратных кавычек `` ` ``.

---

## 🐳 Docker (VPS / проще запускать)

`Dockerfile` собирает образ с зависимостями и запускает `bot.py`.
`docker-compose.yml` монтирует:
- `./docs` в `/app/docs`
- `./rag_faiss_index` в `/app/rag_faiss_index`

Запуск:

```bash
docker compose up -d --build
```

Логи:

```bash
docker compose logs -f llm-bot
```

Остановить:

```bash
docker compose stop llm-bot
```

---

## 🚀 Деплой на VPS

Секреты в `.env` **не коммитятся** (лежит на сервере).

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

---

## 📁 Структура проекта

```
LLM_bot/
├── app/
│   ├── __init__.py
│   ├── config.py      # Настройки из .env
│   ├── web_search.py  # Веб-поиск через Tavily
│   ├── agent/         # Агент LangGraph
│   │   ├── prompts.py # Системный промпт и приветствие
│   │   ├── state.py   # AgentState (messages)
│   │   ├── tools.py   # rag_search, web_search (@tool), инициализация RAG
│   │   ├── graph.py   # Граф: agent → tools_condition → tools / конец
│   │   └── run_agent.py  # Вызов графа, история диалога
│   └── rag/           # RAG: предобработка, индекс, поиск
├── deploy/
│   └── langfuse/      # Официальный docker-compose Langfuse 3.x (self-host)
├── scripts/
│   ├── run_bot.ps1    # Запуск бота на Windows (venv + bot.py)
│   └── deploy.sh      # Деплой на VPS (git pull, docker compose)
├── bot.py             # Точка входа: Telegram-бот
├── LangFuse_observability.md  # Langfuse: версии, .env, VPS
├── rag.py             # Реэкспорт app.rag (совместимость)
├── docs/              # База знаний (PDF, DOCX)
├── requirements.txt
├── .env               # Секреты (не коммитить)
└── rag_faiss_index/   # Индекс FAISS (в .gitignore)
```

---

## 📖 Как устроен агент

- **Граф (LangGraph):** два узла — `agent` (LLM с привязанными tools) и `tools` (ToolNode). После каждого ответа модели проверяется `tools_condition`: если есть вызовы инструментов — переход в `tools`, иначе — конец; после выполнения инструментов управление снова передаётся в `agent`.
- **Инструменты:** `rag_search(query)` и `web_search(query)` реализованы как LangChain tools; модель сама решает, с каким запросом и когда их вызывать.
- **Надёжность:** tool-вызовы ограничены таймаутом (`TOOL_TIMEOUT_SEC`), а исключения не раскрываются пользователю (логируются в консоль).
- **RAG:** при чтении из `docs` PDF/DOCX режутся на фрагменты; при наличии FAISS-индекса используется семантический поиск (OpenRouter), иначе — поиск по ключевым словам с fallback. Индексация/загрузка FAISS выполняется при первом вызове `rag_search`.
- **История:** в state графа передаётся история диалога (и новое сообщение пользователя); после выполнения графа история обновляется и при необходимости обрезается до `MAX_HISTORY_MESSAGES`.
