"""SQLite database manager with schema migrations, restrictive permissions, and optional Fernet encryption."""

import os
import sqlite3
from pathlib import Path
from typing import Any
from cryptography.fernet import Fernet, InvalidToken

CURRENT_SCHEMA_VERSION = 5


class MemoryDB:
    """Manages SQLite connection, schema versioning, and encrypted field storage."""

    def __init__(self, db_path: str = "data/nino.db", encryption_key: str | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cipher: Fernet | None = None
        if encryption_key and encryption_key.strip():
            try:
                self.cipher = Fernet(encryption_key.strip().encode("utf-8"))
            except Exception as e:
                raise ValueError(f"Invalid MEMORY_ENCRYPTION_KEY provided: {e}")

        self._init_db()
        self._apply_file_permissions()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection with foreign keys and WAL enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _apply_file_permissions(self) -> None:
        """Enforce owner-only read/write permissions on the database file."""
        if self.db_path.exists():
            try:
                os.chmod(self.db_path, 0o600)
            except Exception:
                # Windows might not support POSIX octal mode, but standard ACL applies
                pass

    def encrypt_text(self, text: str) -> str:
        """Encrypt plain text if encryption is enabled, otherwise return unchanged."""
        if self.cipher:
            encrypted_bytes = self.cipher.encrypt(text.encode("utf-8"))
            return "enc:" + encrypted_bytes.decode("utf-8")
        return text

    def decrypt_text(self, text: str) -> str:
        """Decrypt text if prefixed with enc: and cipher is available."""
        if text.startswith("enc:"):
            if not self.cipher:
                raise ValueError("Encountered encrypted data but MEMORY_ENCRYPTION_KEY is not configured.")
            raw_token = text[4:].encode("utf-8")
            try:
                return self.cipher.decrypt(raw_token).decode("utf-8")
            except InvalidToken as e:
                raise ValueError(f"Failed to decrypt memory data: invalid key or corrupted payload: {e}")
        return text

    def _init_db(self) -> None:
        """Run schema migrations up to CURRENT_SCHEMA_VERSION."""
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor = conn.execute("SELECT MAX(version) as max_version FROM schema_migrations")
            row = cursor.fetchone()
            current_version = row["max_version"] if row and row["max_version"] is not None else 0

            if current_version < 1:
                self._migrate_v1(conn)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (1)")
            if current_version < 2:
                self._migrate_v2(conn)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (2)")
            if current_version < 3:
                self._migrate_v3(conn)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (3)")
            if current_version < 4:
                self._migrate_v4(conn)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (4)")
            if current_version < 5:
                self._migrate_v5(conn)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (5)")

            # Ensure system admin account is present in users table
            conn.execute(
                """
                INSERT OR IGNORE INTO users (id, username, password_hash, salt)
                VALUES ('admin', 'admin', 'system_admin', 'system_admin')
                """
            )

    def _migrate_v5(self, conn: sqlite3.Connection) -> None:
        """Fifth schema migration: multi-user authentication tables and user-scoped conversation sessions."""
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            INSERT OR IGNORE INTO users (id, username, password_hash, salt)
            VALUES ('admin', 'admin', 'system_admin', 'system_admin');

            CREATE TABLE IF NOT EXISTS user_sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(token);
            CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            """
        )
        for table in ("sessions", "facts"):
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT DEFAULT 'admin'")
            except sqlite3.OperationalError:
                pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_user_id ON facts(user_id)")

    def _migrate_v4(self, conn: sqlite3.Connection) -> None:
        """Fourth schema migration: add origin column to facts, sessions, and messages for channel privacy."""
        for table in ("facts", "sessions", "messages"):
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN origin TEXT DEFAULT 'pc'")
            except sqlite3.OperationalError:
                pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_origin ON facts(origin)")

    def _migrate_v3(self, conn: sqlite3.Connection) -> None:
        """Third schema migration: reminders, automations, proactive_log, proactive_preferences tables."""

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                trigger_time TEXT NOT NULL,
                urgent INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                delivered_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS automations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                job_type TEXT NOT NULL,
                cron_expr TEXT,
                interval_seconds INTEGER,
                next_run_time TEXT,
                kwargs_json TEXT DEFAULT '{}',
                enabled INTEGER DEFAULT 1,
                snoozed_until TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS proactive_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                channel TEXT NOT NULL,
                delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS proactive_preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_reminders_trigger ON reminders(trigger_time, status);
            CREATE INDEX IF NOT EXISTS idx_automations_next_run ON automations(next_run_time, enabled);
            CREATE INDEX IF NOT EXISTS idx_proactive_log_time ON proactive_log(delivered_at);
            """
        )

    def _migrate_v2(self, conn: sqlite3.Connection) -> None:
        """Second schema migration: notes, tasks, calendar_events, and confirmations tables."""
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                due_date TEXT,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                location TEXT DEFAULT '',
                description TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS confirmations (
                id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                arguments_hash TEXT NOT NULL,
                summary TEXT NOT NULL,
                tier TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_notes_title ON notes(title);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_calendar_start ON calendar_events(start_time);
            CREATE INDEX IF NOT EXISTS idx_confirmations_status ON confirmations(status);
            """
        )

    def _migrate_v1(self, conn: sqlite3.Connection) -> None:
        """Initial schema migration: sessions, messages, facts, documents, chunks, vectors."""
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER DEFAULT 0,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                source_message_id TEXT,
                is_current INTEGER DEFAULT 1,
                superseded_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                file_hash TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chunk_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS vector_entries (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                vector BLOB NOT NULL,
                metadata TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_facts_current ON facts(is_current, category);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_vector_type ON vector_entries(entity_type);
            """
        )
