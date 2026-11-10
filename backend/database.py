import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATABASE_DIR / "careerintel.db"


def get_connection():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


import sqlite3
from contextlib import contextmanager
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parent / "careerintel.db"


@contextmanager
def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
    finally:
        connection.close()


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS career_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                full_name TEXT NOT NULL DEFAULT '',
                target_role TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                education TEXT NOT NULL DEFAULT '',
                experience_level TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                skills TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            INSERT OR IGNORE INTO career_profile (id)
            VALUES (1)
        """)

        connection.commit()

        connection.execute("""
            INSERT OR IGNORE INTO career_profile (id)
            VALUES (1)
        """)

        connection.commit()