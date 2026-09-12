import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from .settings import AppSettings


@dataclass
class Session:
    id: str
    title: str
    created_at: str
    updated_at: str
    model: str
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    request_count: int
    last_model: str | None


class SessionStore:
    def __init__(self, settings: AppSettings):
        self._db_path = settings.db_path
        self._conn = self._connect()
        self._create_tables()

    def _connect(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS session (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model TEXT NOT NULL,
                total_input_tokens INTEGER NOT NULL DEFAULT 0,
                total_output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                request_count INTEGER NOT NULL DEFAULT 0,
                last_model TEXT
            )
        """)
        self._conn.commit()

    def _from_row(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            model=row["model"],
            total_input_tokens=row["total_input_tokens"],
            total_output_tokens=row["total_output_tokens"],
            total_tokens=row["total_tokens"],
            request_count=row["request_count"],
            last_model=row["last_model"],
        )

    def create(self, title: str, model: str) -> Session:
        session_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO session (
                id, title, created_at, updated_at, model,
                total_input_tokens, total_output_tokens, total_tokens,
                request_count, last_model
            )
            VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0, NULL)
            """,
            (session_id, title, now, now, model),
        )
        self._conn.commit()
        return self._require(session_id)

    def list(self) -> list[Session]:
        rows = self._conn.execute(
            "SELECT * FROM session ORDER BY updated_at DESC"
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def get(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT * FROM session WHERE id = ?", (session_id,)
        ).fetchone()
        return None if row is None else self._from_row(row)

    def add_usage(
        self,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        last_model: str | None,
    ) -> Session:
        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            """
            UPDATE session
            SET
                updated_at = ?,
                total_input_tokens = total_input_tokens + ?,
                total_output_tokens = total_output_tokens + ?,
                total_tokens = total_tokens + ?,
                request_count = request_count + 1,
                last_model = ?
            WHERE id = ?
            """,
            (now, input_tokens, output_tokens, total_tokens, last_model, session_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Session not found: {session_id}")
        self._conn.commit()
        return self._require(session_id)

    def delete(self, session_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM session WHERE id = ?", (session_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def _require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        return session
