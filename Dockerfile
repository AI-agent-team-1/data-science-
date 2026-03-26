# 1️⃣ Базовый образ с Python
FROM python:3.11-slim

# 2️⃣ Обновляем pip и устанавливаем git/wget (нужно для моделей)
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# 3️⃣ Создаём рабочую папку в контейнере
WORKDIR /app

# 4️⃣ Копируем все файлы проекта внутрь контейнера
COPY . /app

# 5️⃣ Устанавливаем Python-зависимости
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Команда запуска при старте контейнера
ENV PYTHONUNBUFFERED=1

# -u: unbuffered output, чтобы print/ошибки попадали в `docker logs`
CMD ["python", "-u", "bot.py"]