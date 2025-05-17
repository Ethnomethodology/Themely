# modules/db.py
import sqlite3
from pathlib import Path
import streamlit as st # Added for st.cache_resource

# Path for the SQLite database storing project names and file paths
DB_FILE_PATH = Path.home() / ".themely_projects.db"

@st.cache_resource # Cache the connection
def get_connection():
    """Return a connection to the SQLite database."""
    # Using a global variable to store the connection might be problematic with Streamlit's execution model.
    # It's better to establish the connection when needed or manage it carefully.
    # For simplicity in this example, we establish it per call or cache it.
    conn = sqlite3.connect(DB_FILE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database and ensure all required tables exist."""
    conn = get_connection()
    with conn:
        # Projects table
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
        # Settings table (for global settings like model cache path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
    # conn.close() # Do not close if using st.cache_resource on get_connection
    return conn # Returning for chaining if needed, though not strictly used now

def load_project_index():
    """Return a list of dicts with keys 'name' and 'path' for all saved projects."""
    # init_db() # Ensure DB exists - called by app.py already
    conn = get_connection()
    cursor = conn.execute("SELECT name, path FROM projects ORDER BY last_opened DESC;")
    projects = [{"name": row["name"], "path": row["path"]} for row in cursor.fetchall()]
    # conn.close()
    return projects

def save_project_index(name: str, path: str):
    """Insert a new project name and path into the database (upsert and refresh last_opened)."""
    # init_db()
    conn = get_connection()
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
    # conn.close()

def delete_project_index(name: str):
    """Remove a project entry from the database by name."""
    # init_db()
    conn = get_connection()
    with conn:
        conn.execute(
            "DELETE FROM projects WHERE name = ?;",
            (name,)
        )
    # conn.close()

def save_setting(key: str, value: str):
    """Save a global setting to the settings table."""
    # init_db()
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value;
            """,
            (key, value)
        )
    # conn.close()

def load_setting(key: str) -> str | None:
    """Load a global setting from the settings table."""
    # init_db()
    conn = get_connection()
    cursor = conn.execute("SELECT value FROM settings WHERE key = ?;", (key,))
    row = cursor.fetchone()
    # conn.close()
    return row["value"] if row else None