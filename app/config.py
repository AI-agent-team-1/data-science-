"""Настройки из .env — один источник для всего проекта."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Чтение переменных окружения (.env): ключи API, модель, лимиты."""
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model_name: str = os.getenv("MODEL_NAME", "z-ai/glm-4.5-air:free")
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")


# Единственный экземпляр настроек для всего приложения
settings = Settings()
