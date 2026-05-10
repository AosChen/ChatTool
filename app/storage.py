from __future__ import annotations

import base64
import hashlib
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException

from app.models import ChatMessage, PersistedSession, SessionMeta, UserPublic

_ENC_PREFIX = "ENC:"


def _derive_aes_key(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()


def _encrypt(plaintext: str, key: bytes) -> str:
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return _ENC_PREFIX + base64.b64encode(nonce + ciphertext).decode("ascii")


def _decrypt(stored: str, key: bytes | None) -> str:
    if not stored.startswith(_ENC_PREFIX):
        return stored
    if key is None:
        return stored
    raw = base64.b64decode(stored[len(_ENC_PREFIX):])
    nonce, ciphertext = raw[:12], raw[12:]
    return AESGCM(key).decrypt(nonce, ciphertext, None).decode("utf-8")


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat()


class ChatStorage:
    def __init__(self, database_path: str, encryption_key: str | None = None) -> None:
        self.database_path = Path(database_path)
        self._aes_key = _derive_aes_key(encryption_key) if encryption_key else None

    @contextmanager
    def connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tools_enabled INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id
                    ON auth_sessions(user_id);

                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                    ON chat_sessions(user_id, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                    ON chat_messages(session_id, id ASC);
                """
            )

            existing_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(chat_sessions)").fetchall()
            }
            if "tools_enabled" not in existing_columns:
                connection.execute(
                    "ALTER TABLE chat_sessions ADD COLUMN tools_enabled INTEGER NOT NULL DEFAULT 1"
                )

            connection.commit()

    def _normalize_username(self, username: str) -> str:
        normalized = username.strip().lower()
        if not normalized:
            raise HTTPException(status_code=400, detail="Username is required")
        return normalized

    def _hash_password(self, password: str, salt_hex: str) -> str:
        hashed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            200_000,
        )
        return hashed.hex()

    def _session_from_row(self, session_row: sqlite3.Row, message_rows: list[sqlite3.Row]) -> PersistedSession:
        return PersistedSession(
            id=session_row["id"],
            title=session_row["title"],
            model=session_row["model"],
            created_at=session_row["created_at"],
            updated_at=session_row["updated_at"],
            tools_enabled=bool(session_row["tools_enabled"]),
            messages=[
                ChatMessage(role=row["role"], content=_decrypt(row["content"], self._aes_key))
                for row in message_rows
            ],
        )

    def create_user(self, username: str, password: str) -> UserPublic:
        user_id = str(uuid.uuid4())
        normalized_username = self._normalize_username(username)
        salt_hex = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt_hex)
        created_at = utc_now_iso()

        with self.connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO users (id, username, password_hash, password_salt, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, normalized_username, password_hash, salt_hex, created_at),
                )
            except sqlite3.IntegrityError as exc:
                raise HTTPException(status_code=409, detail="Username already exists") from exc
            connection.commit()

        return UserPublic(id=user_id, username=normalized_username)

    def authenticate_user(self, username: str, password: str) -> tuple[UserPublic | None, str | None]:
        normalized_username = self._normalize_username(username)
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, username, password_hash, password_salt
                FROM users
                WHERE username = ?
                """,
                (normalized_username,),
            ).fetchone()

        if row is None:
            return None, "user_not_found"

        expected_hash = self._hash_password(password, row["password_salt"])
        if not secrets.compare_digest(expected_hash, row["password_hash"]):
            return None, "wrong_password"
        return UserPublic(id=row["id"], username=row["username"]), None

    def create_auth_session(self, user_id: str, max_age_days: int) -> str:
        session_id = secrets.token_urlsafe(32)
        created_at = utc_now()
        expires_at = created_at + timedelta(days=max_age_days)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions (id, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, user_id, created_at.isoformat(), expires_at.isoformat()),
            )
            connection.commit()
        return session_id

    def get_user_by_auth_session(self, session_id: str) -> UserPublic | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT users.id, users.username, auth_sessions.expires_at
                FROM auth_sessions
                JOIN users ON users.id = auth_sessions.user_id
                WHERE auth_sessions.id = ?
                """,
                (session_id,),
            ).fetchone()

            if row is None:
                return None

            expires_at = datetime.fromisoformat(row["expires_at"])
            if expires_at <= utc_now():
                connection.execute("DELETE FROM auth_sessions WHERE id = ?", (session_id,))
                connection.commit()
                return None

        return UserPublic(id=row["id"], username=row["username"])

    def delete_auth_session(self, session_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE id = ?", (session_id,))
            connection.commit()

    def list_sessions(self, user_id: str) -> list[SessionMeta]:
        with self.connect() as connection:
            session_rows = connection.execute(
                """
                SELECT id, title, model, created_at, updated_at, tools_enabled
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (user_id,),
            ).fetchall()

        return [
            SessionMeta(
                id=row["id"],
                title=row["title"],
                model=row["model"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                tools_enabled=bool(row["tools_enabled"]),
            )
            for row in session_rows
        ]

    def list_messages(self, user_id: str, session_id: str) -> list[ChatMessage]:
        with self.connect() as connection:
            owner_row = connection.execute(
                "SELECT 1 FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if owner_row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            message_rows = connection.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ChatMessage(role=row["role"], content=_decrypt(row["content"], self._aes_key))
            for row in message_rows
        ]

    def get_session(self, user_id: str, session_id: str) -> PersistedSession:
        with self.connect() as connection:
            session_row = connection.execute(
                """
                SELECT id, title, model, created_at, updated_at, tools_enabled
                FROM chat_sessions
                WHERE id = ? AND user_id = ?
                """,
                (session_id, user_id),
            ).fetchone()
            if session_row is None:
                raise HTTPException(status_code=404, detail="Session not found")

            message_rows = connection.execute(
                """
                SELECT role, content, id
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return self._session_from_row(session_row, message_rows)

    def create_session(self, user_id: str, title: str, model: str, tools_enabled: bool = True) -> PersistedSession:
        session_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (id, user_id, title, model, created_at, updated_at, tools_enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, title, model, now, now, 1 if tools_enabled else 0),
            )
            connection.commit()

        return PersistedSession(
            id=session_id,
            title=title,
            model=model,
            created_at=now,
            updated_at=now,
            tools_enabled=tools_enabled,
            messages=[],
        )

    def update_session(
        self,
        user_id: str,
        session_id: str,
        *,
        title: str | None = None,
        model: str | None = None,
        tools_enabled: bool | None = None,
    ) -> PersistedSession:
        current = self.get_session(user_id, session_id)
        next_title = current.title if title is None else title
        next_model = current.model if model is None else model
        next_tools_enabled = current.tools_enabled if tools_enabled is None else tools_enabled
        now = utc_now_iso()

        with self.connect() as connection:
            result = connection.execute(
                """
                UPDATE chat_sessions
                SET title = ?, model = ?, tools_enabled = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (next_title, next_model, 1 if next_tools_enabled else 0, now, session_id, user_id),
            )
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Session not found")
            connection.commit()

        return self.get_session(user_id, session_id)

    def delete_session(self, user_id: str, session_id: str) -> None:
        with self.connect() as connection:
            result = connection.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Session not found")
            connection.commit()

    def append_message(self, user_id: str, session_id: str, role: str, content: str) -> PersistedSession:
        now = utc_now_iso()
        stored_content = _encrypt(content, self._aes_key) if self._aes_key else content
        with self.connect() as connection:
            session_row = connection.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if session_row is None:
                raise HTTPException(status_code=404, detail="Session not found")

            connection.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, stored_content, now),
            )
            connection.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (now, session_id, user_id),
            )
            connection.commit()

        return self.get_session(user_id, session_id)


storage: ChatStorage | None = None


def get_storage(database_path: str | None = None, encryption_key: str | None = None) -> ChatStorage:
    global storage
    if storage is None:
        if database_path is None:
            raise RuntimeError("Storage is not initialized")
        storage = ChatStorage(database_path, encryption_key)
        storage.initialize()
    return storage
