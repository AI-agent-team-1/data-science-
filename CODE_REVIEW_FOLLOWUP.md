# Ответы на замечания код-ревью

Файл описывает, что сделано по каждому пункту ревью команды.

---

## 1. Версии библиотек в `requirements.txt`

**Сделано:** у прямых зависимостей указаны фиксированные версии (`==`), подобранные по рабочему окружению (LangChain 1.2.x, LangGraph, FastAPI, Uvicorn, FAISS, Langfuse 3.x, Tavily и др.).

**Зачем:** воспроизводимые установки и меньше сюрпризов при обновлении окружения или сборке Docker.

---

## 2. SQLite вместо хранения истории только в RAM

**Сделано:** модуль `app/chat_history_sqlite.py` — одна таблица `chat_history` (`chat_id`, `last_access`, `messages_json`). Сообщения сериализуются через `messages_to_dict` / `messages_from_dict` из LangChain.

**Интеграция:** `app/agent/run_agent.py` читает и пишет историю через SQLite; TTL (24 ч) и LRU (до 100 чатов) сохранены. Путь к файлу: переменная `CHAT_HISTORY_DB` или по умолчанию `data/chat_history.db` в корне проекта.

**Docker / деплой:** в `docker-compose.yml` том `./data:/app/data` и `CHAT_HISTORY_DB=/app/data/chat_history.db`; в `scripts/deploy.sh` добавлено `mkdir -p data`; в `env.example` — комментарий; каталог `data/` в `.gitignore`.

---

## 3. Слишком широкие `Exception`

**Сделано:** в слоях ближе к данным и I/O сужены перехваты:
- `app/rag/ingest.py` — `PdfReadError`, `OSError` для PDF; `OSError`, `ValueError`, `KeyError` для DOCX и FAISS;
- `app/rag/retriever.py` — `OSError`, `ValueError`, `RuntimeError` при сбое FAISS;
- `app/rag/embeddings.py` — `ImportError`, `ValueError`, `TypeError`;
- `app/web_search.py` — сначала сеть/данные, затем запасной `Exception` с логом;
- `app/agent/tools.py` — в `_invoke_with_timeout` после таймаута: `OSError`, `ValueError`, `RuntimeError`.

**Границы приложения** (HTTP-эндпоинт чата, верхний обработчик Telegram) по-прежнему ловят `Exception`, чтобы не отдавать пользователю необработанный трейс — с полным `logger.exception`.

---

## 4. Инициализация RAG не при импорте `tools.py`

**Сделано:** тяжёлая загрузка вынесена в `init_rag()` в `app/agent/tools.py`; глобальные `_knowledge_chunks` / `_faiss_store` заполняются только при вызове.

**Вызов до первого запроса:**
- `bot.py` — после `validate_required_for_telegram_bot()`, `init_schema()`: `init_rag()`;
- `web/api.py` — в `lifespan` FastAPI: `validate_required_for_agent()`, `init_schema()`, `init_rag()` до `yield`.

Инструмент `rag_search` при необходимости вызывает `init_rag()` (идемпотентно), если модуль использовали без явного старта.

---

## 5. Логирование вместо `print`

**Сделано:** `logging.getLogger(__name__)` и уровни `info` / `warning` / `error` / `exception` в `run_agent`, `tools`, `ingest`, `retriever`, `embeddings`, `web_search`, `langfuse_tracing`, `bot.py`, `web/api.py`. В `bot.py` и при старте веба настроен `basicConfig` с форматом строки.

---

## 6. Очистка `_chat_locks` при eviction

**Сделано:** при `prune_stale_and_lru` из SQLite возвращаются удалённые `chat_id`; для каждого вызывается `_chat_locks.pop(cid, None)` под `_hist_lock`. Аналогично после второго prune в конце `run_agent`. В `clear_chat_history` — `pop` блокировки для очищаемого чата.

---

## 7. Таймаут для `rag_search`

**Сделано:** как для `web_search`, используется общий пул `ThreadPoolExecutor` и `future.result(timeout=TOOL_TIMEOUT_SEC)` (по умолчанию 20 с, переменная `TOOL_TIMEOUT_SEC`). Обёртка `_invoke_with_timeout` вызывается и для `rag_search`.

---

## 8. Где используется `rag.py`

**Сделано:** в корневом `rag.py` добавлен явный докстринг: файл — **реэкспорт** API из `app.rag` для старых скриптов вида `from rag import ...`; основной код идёт через `app.rag` / `app.agent.tools`. Отдельного дублирования логики нет.

---

## 9. Таблицы в DOCX (`_load_docx`)

**Сделано:** в `app/rag/ingest.py` после параграфов обходятся `document.tables`, строки и ячейки склеиваются через `" | "`.

---

## 10. Порядок операций в `preprocess_text`

**Сделано:** в `app/rag/preprocess.py` обновлён пайплайн: HTML → единые `\n` → склейка переносов с дефисом (`-\n` + непробел) → нормализация пробелов **по строкам** → сжатие множественных пустых строк. Убрана глобальная `normalize_whitespace` до переносов, которая могла ухудшать структуру.

---

## 11. Валидация конфига и `/reset` в Telegram

**Сделано:**
- `app/config.py` — `validate_required_for_agent()` (обязательный `OPENROUTER_API_KEY`) и `validate_required_for_telegram_bot()` (+ обязательный `TELEGRAM_BOT_TOKEN`);
- `bot.py` — проверка перед созданием `TeleBot`;
- команда `/reset` — вызывает `clear_chat_history(message.chat.id)` и короткое подтверждение пользователю;
- `HELP_MESSAGE` в `app/agent/prompts.py` обновлён: описаны `/start`, `/help`, `/reset`.

---

*Дата фиксации изменений: по состоянию репозитория после внедрения перечисленных правок.*
