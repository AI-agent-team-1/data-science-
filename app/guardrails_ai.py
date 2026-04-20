"""
Guardrails AI интеграция (Практика 6):
проверка входного запроса (input) и ответа (output) вокруг run_agent().

По умолчанию модуль "мягкий": если зависимости не установлены/валидатор не доступен,
он не валит приложение, а пропускает запрос.

ENV:
  GUARDRAILS_ENABLED=1|0
  GUARDRAILS_ON_FAIL=input_filter|input_exception|output_filter|output_exception (по умолчанию filter)
  GUARDRAILS_BANNED_PHRASES="ignore previous instructions;act as dan;do anything now"
  GUARDRAILS_JAILBREAK_THRESHOLD=0.8
  GUARDRAILS_REFUSAL_TEXT="..."
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _split_phrases(value: str) -> list[str]:
    parts = [p.strip() for p in value.replace(",", ";").split(";")]
    return [p for p in parts if p]


@dataclass(frozen=True)
class GuardrailsResult:
    allowed: bool
    text: str | None = None  # override/refusal text, if any
    reason: str | None = None


DEFAULT_REFUSAL = "Я не могу помочь с этим запросом по соображениям безопасности."


def _load_guardrails():
    """
    Возвращает (input_guard, output_guard) или (None, None) если Guardrails недоступен.
    """
    if not _env_bool("GUARDRAILS_ENABLED", False):
        return None, None

    try:
        from guardrails import Guard
        from guardrails.hub import BanList, DetectJailbreak, DetectPII, ToxicLanguage
    except Exception:
        # Guardrails не установлен/сломался — не блокируем работу бота.
        return None, None

    banned = _split_phrases(
        os.getenv(
            "GUARDRAILS_BANNED_PHRASES",
            "ignore previous instructions;act as dan;do anything now;system prompt;reveal your prompt",
        )
    )
    try:
        threshold = float(os.getenv("GUARDRAILS_JAILBREAK_THRESHOLD", "0.8"))
    except ValueError:
        threshold = 0.8

    # Input: jailbreak + banlist
    input_guard = (
        Guard()
        .use(DetectJailbreak(on_fail="filter", threshold=threshold))
        .use(BanList(on_fail="filter", banned_phrases=banned))
    )

    # Output: PII + токсичность
    output_guard = Guard().use(DetectPII(on_fail="filter")).use(ToxicLanguage(on_fail="filter"))

    return input_guard, output_guard


_INPUT_GUARD, _OUTPUT_GUARD = _load_guardrails()


def guard_input(user_text: str) -> GuardrailsResult:
    """Проверка входного сообщения."""
    if _INPUT_GUARD is None:
        return GuardrailsResult(allowed=True)

    refusal = os.getenv("GUARDRAILS_REFUSAL_TEXT", DEFAULT_REFUSAL)
    mode = os.getenv("GUARDRAILS_ON_FAIL", "filter").strip().lower()

    try:
        outcome = _INPUT_GUARD.validate(user_text)
        passed = getattr(outcome, "validation_passed", True)
    except Exception as e:
        if "exception" in mode:
            return GuardrailsResult(allowed=False, text=refusal, reason=str(e))
        return GuardrailsResult(allowed=True)

    if passed:
        return GuardrailsResult(allowed=True)

    # filter-mode: block with refusal
    return GuardrailsResult(allowed=False, text=refusal, reason="input blocked by guardrails")


def guard_output(bot_text: str) -> GuardrailsResult:
    """Проверка ответа бота."""
    if _OUTPUT_GUARD is None:
        return GuardrailsResult(allowed=True)

    refusal = os.getenv("GUARDRAILS_REFUSAL_TEXT", DEFAULT_REFUSAL)
    mode = os.getenv("GUARDRAILS_ON_FAIL", "filter").strip().lower()

    try:
        outcome = _OUTPUT_GUARD.validate(bot_text)
        passed = getattr(outcome, "validation_passed", True)
    except Exception as e:
        if "exception" in mode:
            return GuardrailsResult(allowed=False, text=refusal, reason=str(e))
        return GuardrailsResult(allowed=True)

    if passed:
        return GuardrailsResult(allowed=True)

    return GuardrailsResult(allowed=False, text=refusal, reason="output blocked by guardrails")

