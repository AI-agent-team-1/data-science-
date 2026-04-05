# 1️⃣ Базовый образ с Python
FROM python:3.11-slim

# 2️⃣ Системные пакеты: git/wget + toolchain для колёс pip без готовых бинарников (faiss, cryptography и т.д.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ Создаём рабочую папку в контейнере
WORKDIR /app

# 4️⃣ Копируем все файлы проекта внутрь контейнера
COPY . /app

# 5️⃣ Устанавливаем Python-зависимости (wheel ускоряет сборку)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# 6️⃣ Команда по умолчанию — только Telegram-бот (в docker-compose для VPS переопределяется: бот + веб)
ENV PYTHONUNBUFFERED=1

# -u: unbuffered output, чтобы print/ошибки попадали в `docker logs`
CMD ["python", "-u", "bot.py"]
