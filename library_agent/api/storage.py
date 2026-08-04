"""Small SQLite-backed store for Library Agent application metadata."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LibraryStorage:
    """Persist books, threads and rendered chat messages for the local app."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS books (
                    book_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    meta TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    chunks INTEGER NOT NULL DEFAULT 0,
                    indexed INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
                    book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS chat_messages_thread_idx
                    ON chat_messages (thread_id, created_at);
                """
            )

    def upsert_book(
        self,
        book_id: str,
        title: str,
        meta: str,
        filename: str,
        chunks: int = 0,
    ) -> None:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO books (book_id, title, meta, filename, chunks, indexed, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(book_id) DO UPDATE SET
                    title = excluded.title,
                    meta = excluded.meta,
                    filename = excluded.filename,
                    chunks = excluded.chunks,
                    indexed = 1,
                    updated_at = excluded.updated_at
                """,
                (book_id, title, meta, filename, chunks, timestamp, timestamp),
            )

    def list_books(self) -> list[dict]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT book_id AS id, title, meta, indexed
                FROM books
                ORDER BY updated_at DESC, title COLLATE NOCASE ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def update_book_meta(self, book_id: str, meta: str) -> None:
        """Refresh display metadata without changing the book identity."""
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE books SET meta = ?, updated_at = ? WHERE book_id = ?",
                (meta, timestamp, book_id),
            )

    def ensure_thread(self, thread_id: str, book_id: str) -> None:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO threads (thread_id, book_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (thread_id, book_id, timestamp, timestamp),
            )

    def add_message(self, thread_id: str, book_id: str, role: str, content: str) -> str:
        message_id = uuid4().hex
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (id, thread_id, book_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, thread_id, book_id, role, content, timestamp),
            )
            connection.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                (timestamp, thread_id),
            )
        return message_id

    def list_messages(self, book_id: str, thread_id: str) -> list[dict]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE book_id = ? AND thread_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (book_id, thread_id),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
            }
            for row in rows
        ]
