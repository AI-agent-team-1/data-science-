from __future__ import annotations

import hashlib
import re
from typing import Any

# Patterns for basic PII redaction in trace metadata.
# Note: мы санитизируем строки только перед отправкой "ручных" payload'ов
# (metadata/атрибуты). Input/output самих сообщений модели сейчас не
# перехватывается этим слоем без рефакторинга.

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Простой best-effort шаблон для телефонов (форматы с +7/8, пробелами и дефисами).
# Важно: без невалидных скобок, чтобы regex компилировался.
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,3}\)?[\s-]?)?\d{3}[\s-]?\d{2}[\s-]?\d{2}"
)

# Common token-like strings (best-effort).
SK_PREFIX_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_-]+)\b")
PK_PREFIX_PATTERN = re.compile(r"\b(pk-[A-Za-z0-9_-]+)\b")


def hash_user_id(user_id: str) -> str:
    """
    Стабильный хэш идентификатора пользователя.
    Возвращает вид: u_<16hex>.
    """
    h = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
    return f"u_{h}"


def sanitize_text(text: str) -> str:
    """Маскирует базовые PII/секреты в строке."""
    try:
        s = text
        s = EMAIL_PATTERN.sub("<email>", s)
        s = PHONE_PATTERN.sub("<phone>", s)
        s = SK_PREFIX_PATTERN.sub("<secret_key>", s)
        s = PK_PREFIX_PATTERN.sub("<public_key>", s)
        return s
    except Exception:
        # Никогда не ломаем трейсинг из-за санитизации.
        return "<masked>"


def sanitize_payload(payload: Any, *, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    """
    Рекурсивная санитизация payload:
    - str: sanitize_text
    - dict/list: рекурсивно по значениям
    - прочее: без изменений
    """
    try:
        if payload is None:
            return None

        # защита от циклов/слишком глубокой вложенности
        if _seen is None:
            _seen = set()
        if _depth > 8:
            return "<truncated>"
        obj_id = id(payload)
        if obj_id in _seen:
            return "<cycle>"

        if isinstance(payload, str):
            return sanitize_text(payload)
        if isinstance(payload, bytes):
            return sanitize_text(payload.decode("utf-8", errors="ignore"))

        if isinstance(payload, dict):
            _seen.add(obj_id)
            return {
                sanitize_text(str(k)): sanitize_payload(v, _depth=_depth + 1, _seen=_seen)
                for k, v in payload.items()
            }
        if isinstance(payload, list):
            _seen.add(obj_id)
            return [sanitize_payload(v, _depth=_depth + 1, _seen=_seen) for v in payload]
        if isinstance(payload, tuple):
            _seen.add(obj_id)
            return [sanitize_payload(v, _depth=_depth + 1, _seen=_seen) for v in payload]

        # Для нестандартных объектов (например, LangChain сообщения) делаем безопасную строку.
        # Это предотвращает исключения/циклы и даёт хоть какую-то диагностическую ценность.
        return sanitize_text(str(payload))
    except Exception:
        return "<masked>"


def langfuse_mask(*, data: Any) -> Any:
    """
    Совместимость с Langfuse SDK 3.x: mask-функция вызывается как mask(data=...).
    """
    return sanitize_payload(data)

