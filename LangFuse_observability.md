# Observability (Langfuse)

## Версии

В проекте зафиксирован **Python SDK `langfuse` 3.x** (`requirements.txt`). Он ожидает **сервер Langfuse 3.x** (self-hosted или облако).  
Сервер **Langfuse 2.x** с SDK 3.x даёт несовместимые ответы API (ошибки валидации, пустые traces).

## Включение

1. Поднимите Langfuse (локально или на VPS). Готовый `docker-compose` для v3 лежит в репозитории: **`deploy/langfuse/`**. Затем создайте проект в UI.
2. Скопируйте **Public** и **Secret** ключи из Settings → API Keys.
3. В `.env`:

   ```env
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=http://localhost:3000
   ```

   Для Docker на VPS укажите URL, **доступный из контейнера бота** (не `localhost` хоста, если Langfuse в другом контейнере). Рекомендуется **общая сеть** `llm_shared`: один раз `docker network create llm_shared`, в корневом `docker-compose.yml` бота и в `deploy/langfuse/docker-compose.yml` сеть уже описана — после `docker compose up` у обоих стеков используйте `LANGFUSE_HOST=http://langfuse-web:3000`. Не используйте публичный IP ВМ из контейнера бота — часто даёт таймаут (hairpin). Либо `http://172.17.0.1:3000` к хосту, если общая сеть не настроена.

4. Перезапустите бота.

## Поведение в коде

- Один запрос пользователя → один `CallbackHandler` → трейс с шагами LLM и tools в LangGraph.
- `session_id` / `user_id` = `telegram chat_id` (строка).
- После завершения вызова графа вызывается `flush()` буфера Langfuse.

## Если traces не видны

- Проверьте `LANGFUSE_ENABLED=true` и ключи.
- Убедитесь, что `LANGFUSE_HOST` смотрит на живой сервер (из контейнера бота — `curl` / браузер недоступны из IDE, но логи должны быть без ошибок Langfuse).
- Подождите несколько секунд после запроса; при необходимости обновите UI.

## Отключение

Установите `LANGFUSE_ENABLED=false` или удалите ключи — тогда Langfuse не импортируется для трейсинга (кроме проверки флага).
