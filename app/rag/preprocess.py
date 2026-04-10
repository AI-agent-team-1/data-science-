"""Предобработка текста перед индексацией: очистка HTML, пробелов, переносов."""
from __future__ import annotations

import re


def strip_html(text: str) -> str:
    """Удаляет HTML-теги из текста."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace_line(text: str) -> str:
    """Пробелы и табы внутри одной строки → один пробел, trim."""
    if not text:
        return ""
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_line_breaks(text: str) -> str:
    """Приводит переносы строк к \\n, сжимает пустые строки."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """
    Полная предобработка для чанкирования.

    Порядок: HTML → единые переносы → переносы слов с дефисом →
    нормализация пробелов *по строкам*, затем снова сжатие пустых строк.
    Так меньше риска склеить слова через перенос и «сломать» структуру абзацев.
    """
    if not text:
        return ""
    text = strip_html(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Слитые переносы с дефисом в конце строки (word-\ncont → wordcont)
    text = re.sub(r"-\n(?=\S)", "", text)
    lines = text.split("\n")
    lines = [normalize_whitespace_line(line) for line in lines]
    text = "\n".join(lines)
    text = normalize_line_breaks(text)
    return text
