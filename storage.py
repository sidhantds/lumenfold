import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from rag import EMBEDDING_MODEL_NAME

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "chat_history.sqlite3"
# Vectors from different embedding models can't be compared, so each model
# keeps its indexes in its own folder. Indexes built by an older model are
# then simply "missing" and get rebuilt instead of returning wrong results.
VECTOR_DIR = DATA_DIR / "vectors" / EMBEDDING_MODEL_NAME.replace("/", "__")


def initialize_storage() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    vector_file TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    document_hash TEXT,
                    low_text_warning INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(chats)")
            }
            if "page_count" not in columns:
                connection.execute(
                    "ALTER TABLE chats ADD COLUMN page_count INTEGER NOT NULL DEFAULT 0"
                )
            if "document_hash" not in columns:
                connection.execute("ALTER TABLE chats ADD COLUMN document_hash TEXT")
            if "low_text_warning" not in columns:
                connection.execute(
                    "ALTER TABLE chats "
                    "ADD COLUMN low_text_warning INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def create_chat(
    document_name: str,
    vector_db: InMemoryVectorStore,
    chunk_count: int,
    page_count: int,
    document_hash: str | None = None,
    low_text_warning: bool = False,
    reuse_vector_file: str | None = None,
) -> str:
    chat_id = uuid.uuid4().hex
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    if reuse_vector_file is None:
        vector_file = f"{chat_id}.json"
        vector_db.dump(str(VECTOR_DIR / vector_file))
    else:
        vector_file = Path(reuse_vector_file).name

    with closing(sqlite3.connect(DB_PATH)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO chats (
                    id, title, document_name, vector_file, chunk_count, page_count,
                    document_hash, low_text_warning
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    Path(document_name).stem,
                    document_name,
                    vector_file,
                    chunk_count,
                    page_count,
                    document_hash,
                    low_text_warning,
                ),
            )
    return chat_id


def list_chats() -> list[dict]:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id, title, document_name, chunk_count, page_count,
                document_hash, low_text_warning, updated_at
            FROM chats
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_chat(chat_id: str) -> dict | None:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT
                id, title, document_name, vector_file, chunk_count, page_count,
                document_hash, low_text_warning
            FROM chats
            WHERE id = ?
            """,
            (chat_id,),
        ).fetchone()
    return dict(row) if row else None


def get_chat_by_hash(document_hash: str) -> dict | None:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT
                id, title, document_name, vector_file, chunk_count, page_count,
                document_hash, low_text_warning
            FROM chats
            WHERE document_hash = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (document_hash,),
        ).fetchone()
    return dict(row) if row else None


def vector_index_exists(chat: dict) -> bool:
    return (VECTOR_DIR / Path(chat["vector_file"]).name).is_file()


def load_vector_store(chat: dict, embeddings: Embeddings) -> InMemoryVectorStore:
    vector_path = VECTOR_DIR / Path(chat["vector_file"]).name
    if not vector_path.is_file():
        raise FileNotFoundError(f"Saved index not found: {vector_path}")
    return InMemoryVectorStore.load(str(vector_path), embeddings)


def load_messages(chat_id: str) -> list[dict]:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        rows = connection.execute(
            """
            SELECT role, content, sources
            FROM messages
            WHERE chat_id = ?
            ORDER BY id
            """,
            (chat_id,),
        ).fetchall()
    return [
        {"role": role, "content": content, "sources": json.loads(sources)}
        for role, content, sources in rows
    ]


def save_message(chat_id: str, message: dict) -> None:
    sources = json.dumps(message.get("sources", []))
    with closing(sqlite3.connect(DB_PATH)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO messages (chat_id, role, content, sources)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, message["role"], message["content"], sources),
            )
            if message["role"] == "user":
                has_user_message = connection.execute(
                    """
                    SELECT 1
                    FROM messages
                    WHERE chat_id = ? AND role = 'user'
                    LIMIT 1 OFFSET 1
                    """,
                    (chat_id,),
                ).fetchone()
                if not has_user_message:
                    title = message["content"].strip().replace("\n", " ")
                    if len(title) > 48:
                        title = title[:45].rstrip() + "..."
                    connection.execute(
                        """
                        UPDATE chats
                        SET title = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (title, chat_id),
                    )
                else:
                    connection.execute(
                        "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (chat_id,),
                    )
            else:
                connection.execute(
                    "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (chat_id,),
                )


def clear_conversation(chat_id: str, document_name: str) -> None:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        with connection:
            connection.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            connection.execute(
                """
                UPDATE chats
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (Path(document_name).stem, chat_id),
            )
