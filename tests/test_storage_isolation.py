import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import storage


class StorageIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        data_dir = Path(self.temp_dir.name)
        self.db_path_patch = patch.object(storage, "DB_PATH", data_dir / "chats.db")
        self.vector_dir_patch = patch.object(storage, "VECTOR_DIR", data_dir / "vectors")
        self.db_path_patch.start()
        self.vector_dir_patch.start()
        self.addCleanup(self.db_path_patch.stop)
        self.addCleanup(self.vector_dir_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

        storage.initialize_storage()
        with closing(sqlite3.connect(storage.DB_PATH)) as connection:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO chats (
                        id, owner_id, title, document_name, vector_file,
                        chunk_count, page_count, document_hash, low_text_warning
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ("chat-a", "browser-a", "A", "a.pdf", "a", 1, 1, "hash-a", 0),
                        ("chat-b", "browser-b", "B", "b.pdf", "b", 1, 1, "hash-b", 0),
                        ("legacy", None, "Legacy", "old.pdf", "old", 1, 1, None, 0),
                    ],
                )
                connection.executemany(
                    """
                    INSERT INTO messages (chat_id, role, content)
                    VALUES (?, ?, ?)
                    """,
                    [
                        ("chat-a", "user", "private to A"),
                        ("chat-b", "user", "private to B"),
                    ],
                )

    def test_chat_lists_and_messages_are_scoped_to_owner(self) -> None:
        self.assertEqual(
            [chat["id"] for chat in storage.list_chats(owner_id="browser-a")],
            ["chat-a"],
        )
        self.assertIsNone(storage.get_chat("chat-b", owner_id="browser-a"))
        self.assertIsNone(
            storage.get_chat_by_hash("hash-b", owner_id="browser-a")
        )
        self.assertIsNone(storage.get_chat("legacy", owner_id="browser-a"))
        self.assertEqual(
            storage.load_messages("chat-a", owner_id="browser-a")[0]["content"],
            "private to A",
        )
        self.assertEqual(
            storage.load_messages("chat-b", owner_id="browser-a"),
            [],
        )

    def test_owner_cannot_write_or_clear_another_sessions_chat(self) -> None:
        with self.assertRaises(ValueError):
            storage.save_message(
                "chat-b",
                {"role": "user", "content": "unauthorized"},
                owner_id="browser-a",
            )
        with self.assertRaises(ValueError):
            storage.clear_conversation(
                "chat-b",
                "b.pdf",
                owner_id="browser-a",
            )

        self.assertEqual(
            storage.load_messages("chat-b", owner_id="browser-b")[0]["content"],
            "private to B",
        )


if __name__ == "__main__":
    unittest.main()
