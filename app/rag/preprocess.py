"""Предобработка текста перед индексацией: очистка HTML, пробелов, переносов."""
from __future__ import annotations

import re


def strip_html(text: str) -> str:
    """Удаляет HTML-теги из текста."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    """Заменяет множественные пробелы/табы на один пробел, убирает ведущие/конечные."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_line_breaks(text: str) -> str:
    """Приводит переносы строк к единому виду: \\n, убирает лишние пустые строки."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """
    Полная предобработка: HTML → пробелы, нормализация пробелов и переносов.
    Применять к сырому тексту из PDF/DOCX перед чанкированием.
    """
    if not text:
        return ""
    text = strip_html(text)
    text = normalize_line_breaks(text)
    text = normalize_whitespace(text)
    return text
