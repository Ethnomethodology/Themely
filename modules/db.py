import sqlite3
from pathlib import Path

# Path for the SQLite database storing project names and file paths
db_file = Path.home() / ".themely_projects.db"

def get_connection():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(db_file, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database and ensure the projects table exists."""
    conn = get_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT    UNIQUE NOT NULL,
                path         TEXT    NOT NULL,
                last_opened  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
    return conn

def load_project_index():
    """Return a list of dicts with keys 'name' and 'path' for all saved projects."""
    conn = init_db()
    cursor = conn.execute("SELECT name, path FROM projects ORDER BY last_opened DESC;")
    return [{"name": row["name"], "path": row["path"]} for row in cursor]

def save_project_index(name: str, path: str):
    """Insert a new project name and path into the database (upsert and refresh last_opened)."""
    conn = init_db()
    with conn:
        conn.execute(
            """
            INSERT INTO projects(name, path, last_opened)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET
                path = excluded.path,
                last_opened = CURRENT_TIMESTAMP;
            """,
            (name, path)
        )


def delete_project_index(name: str):
    """Remove a project entry from the database by name."""
    conn = init_db()
    with conn:
        conn.execute(
            "DELETE FROM projects WHERE name = ?;",
            (name,)
        )
