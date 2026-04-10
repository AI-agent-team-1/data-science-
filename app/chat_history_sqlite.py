"""История диалогов в SQLite (одна таблица), потокобезопасный доступ."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any, List

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

logger = logging.getLogger(__name__)

_sqlite_lock = threading.Lock()


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_db_path() -> str:
    custom = (os.getenv("CHAT_HISTORY_DB") or "").strip()
    if custom:
        path = os.path.abspath(custom)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return path
    data_dir = os.path.join(_project_root(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "chat_history.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=60.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_schema() -> None:
    with _sqlite_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    chat_id INTEGER PRIMARY KEY NOT NULL,
                    last_access REAL NOT NULL,
                    messages_json TEXT NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()
    logger.info("SQLite chat history: schema OK (%s)", get_db_path())


def prune_stale_and_lru(
    now: float,
    ttl_seconds: float,
    max_sessions: int,
) -> List[int]:
    """Удаляет просроченные сессии и лишние по LRU (самые старые по last_access). Возвращает удалённые chat_id."""
    removed: List[int] = []
    with _sqlite_lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "SELECT chat_id FROM chat_history WHERE (? - last_access) > ?",
                (now, ttl_seconds),
            )
            expired = [row[0] for row in cur.fetchall()]
            for cid in expired:
                conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (cid,))
                removed.append(cid)

            cur = conn.execute("SELECT COUNT(*) FROM chat_history")
            count = int(cur.fetchone()[0])
            while count > max_sessions:
                cur = conn.execute(
                    "SELECT chat_id FROM chat_history ORDER BY last_access ASC LIMIT 1"
                )
                row = cur.fetchone()
                if not row:
                    break
                oid = int(row[0])
                conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (oid,))
                removed.append(oid)
                count -= 1
            conn.commit()
        finally:
            conn.close()
    if removed:
        logger.debug("chat_history prune removed chat_ids: %s", removed)
    return removed


def load_messages(chat_id: int, now: float, ttl_seconds: float) -> List[BaseMessage]:
    with _sqlite_lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "SELECT last_access, messages_json FROM chat_history WHERE chat_id = ?",
                (chat_id,),
            )
            row = cur.fetchone()
            if not row:
                return []
            last_access, raw = float(row[0]), row[1]
            if (now - last_access) > ttl_seconds:
                conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
                conn.commit()
                return []
            data = json.loads(raw)
            return list(messages_from_dict(data))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Corrupt history for chat_id=%s: %s", chat_id, e, exc_info=True)
            conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
            conn.commit()
            return []
        finally:
            conn.close()


def save_messages(chat_id: int, now: float, messages: List[BaseMessage]) -> None:
    payload = json.dumps(messages_to_dict(messages), ensure_ascii=False)
    with _sqlite_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO chat_history (chat_id, last_access, messages_json)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    last_access = excluded.last_access,
                    messages_json = excluded.messages_json;
                """,
                (chat_id, now, payload),
            )
            conn.commit()
        finally:
            conn.close()


def delete_session(chat_id: int) -> None:
    with _sqlite_lock:
        conn = _connect()
        try:
            conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
            conn.commit()
        finally:
            conn.close()
