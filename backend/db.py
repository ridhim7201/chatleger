"""SQLite storage layer for ChatLedger."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from schema import ActionItem, Decision, OpenQuestion

DB_PATH = Path(__file__).parent / "chatledger.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    sender TEXT NOT NULL,
    message_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL,
    task TEXT NOT NULL,
    deadline TEXT,
    source_message_id TEXT NOT NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages(message_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision TEXT NOT NULL,
    made_by TEXT NOT NULL,
    source_message_id TEXT NOT NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages(message_id)
);

CREATE TABLE IF NOT EXISTS open_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asker TEXT NOT NULL,
    question TEXT NOT NULL,
    answered INTEGER NOT NULL DEFAULT 0,
    source_message_id TEXT NOT NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages(message_id)
);

CREATE TABLE IF NOT EXISTS chunk_cache (
    chunk_hash TEXT PRIMARY KEY,
    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    error TEXT NOT NULL
);
"""


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def reset_db(db_path: Path = DB_PATH) -> None:
    """Clear all rows (used when re-processing a new upload), keep cache."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM action_items")
        conn.execute("DELETE FROM decisions")
        conn.execute("DELETE FROM open_questions")
        conn.execute("DELETE FROM extraction_failures")


def insert_messages(messages: list[dict], db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO messages (message_id, timestamp, sender, message_text) "
            "VALUES (:message_id, :timestamp, :sender, :message_text)",
            messages,
        )


def insert_action_items(items: list[ActionItem], db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO action_items (owner, task, deadline, source_message_id) "
            "VALUES (?, ?, ?, ?)",
            [(i.owner, i.task, i.deadline, i.source_message_id) for i in items],
        )


def insert_decisions(items: list[Decision], db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO decisions (decision, made_by, source_message_id) VALUES (?, ?, ?)",
            [(i.decision, i.made_by, i.source_message_id) for i in items],
        )


def insert_open_questions(items: list[OpenQuestion], db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO open_questions (asker, question, answered, source_message_id) "
            "VALUES (?, ?, ?, ?)",
            [(i.asker, i.question, int(i.answered), i.source_message_id) for i in items],
        )


def log_extraction_failure(
    chunk_id: str, chunk_hash: str, error: str, db_path: Path = DB_PATH
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO extraction_failures (chunk_id, chunk_hash, error) VALUES (?, ?, ?)",
            (chunk_id, chunk_hash, error),
        )


def get_cached_result(chunk_hash: str, db_path: Path = DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM chunk_cache WHERE chunk_hash = ?", (chunk_hash,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["result_json"])


def set_cached_result(chunk_hash: str, result: dict, db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chunk_cache (chunk_hash, result_json) VALUES (?, ?)",
            (chunk_hash, json.dumps(result)),
        )


_ALLOWED_TABLES: dict[str, str] = {
    "messages": "SELECT * FROM messages",
    "action_items": "SELECT * FROM action_items",
    "decisions": "SELECT * FROM decisions",
    "open_questions": "SELECT * FROM open_questions",
}


def fetch_all(table: str, db_path: Path = DB_PATH) -> list[dict]:
    sql = _ALLOWED_TABLES.get(table)
    if sql is None:
        raise ValueError(f"unknown table: {table}")
    with get_connection(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
