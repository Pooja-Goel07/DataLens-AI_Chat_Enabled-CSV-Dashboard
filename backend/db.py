# db.py - Shared database engine + per-user active-table state (no global session).
#
# Replaces the old process-global `current_session` dict in upload.py. "Which table is
# active" is now per-user and persisted in the database, so it is correct across concurrent
# users and across multiple uvicorn workers.
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from fastapi import HTTPException

load_dotenv()

MYSQL_URI = os.getenv("MYSQL_URI")
if not MYSQL_URI:
    raise ValueError("MYSQL_URI not found in environment variables.")

_engine = None


def get_engine():
    """Return a single shared SQLAlchemy engine (created lazily)."""
    global _engine
    if _engine is None:
        _engine = create_engine(MYSQL_URI, pool_pre_ping=True)
    return _engine


def init_db():
    """Create the metadata tables that are not managed by SQLModel.

    `user_tables` tracks table ownership; `user_active_table` remembers each user's
    last-selected dataset. Both are created if absent so the app bootstraps cleanly.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_tables (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                table_name VARCHAR(64) NOT NULL,
                original_name VARCHAR(255),
                file_name VARCHAR(255),
                rows_count INT,
                columns_count INT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_user_table (username, table_name)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_active_table (
                username VARCHAR(255) PRIMARY KEY,
                table_name VARCHAR(64),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """))


def get_last_active_table(conn, username):
    """Return the user's most recently selected table name, or None."""
    row = conn.execute(
        text("SELECT table_name FROM user_active_table WHERE username=:u"),
        {"u": username},
    ).fetchone()
    return row[0] if row and row[0] else None


def set_active_table(username, table_name):
    """Persist the user's active table (upsert)."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_active_table (username, table_name)
            VALUES (:u, :t)
            ON DUPLICATE KEY UPDATE table_name = :t
        """), {"u": username, "t": table_name})


def clear_active_table(username, table_name=None):
    """Clear the user's active table. If table_name is given, only clear it when it matches."""
    engine = get_engine()
    with engine.begin() as conn:
        if table_name is None:
            conn.execute(
                text("DELETE FROM user_active_table WHERE username=:u"),
                {"u": username},
            )
        else:
            conn.execute(
                text("DELETE FROM user_active_table WHERE username=:u AND table_name=:t"),
                {"u": username, "t": table_name},
            )


def resolve_active_table(username, table_name=None):
    """Resolve and authorize the active table for a request. No shared state.

    - If `table_name` is provided, verify the user owns it.
    - If not, fall back to the user's persisted last-active table.
    Returns {"active_table": str|None, "table_info": {...}, "upload_time": str|None}.
    Raises 403 if the requested table is not owned by the user.
    """
    engine = get_engine()
    with engine.connect() as conn:
        if table_name is None:
            table_name = get_last_active_table(conn, username)
        if table_name is None:
            return {"active_table": None, "table_info": {}, "upload_time": None}

        row = conn.execute(text("""
            SELECT original_name, file_name, rows_count, columns_count, upload_time
            FROM user_tables
            WHERE username = :u AND table_name = :t
        """), {"u": username, "t": table_name}).fetchone()

        if not row:
            raise HTTPException(status_code=403, detail="Table not found or access denied")

        # Column list + types come from the live table schema.
        desc = conn.execute(text(f"DESCRIBE `{table_name}`")).fetchall()
        columns = [r[0] for r in desc]
        data_types = {r[0]: r[1] for r in desc}
        upload_time = row[4].isoformat() if row[4] else None

        return {
            "active_table": table_name,
            "table_info": {
                "original_name": row[0],
                "safe_name": table_name,
                "file_name": row[1],
                "columns": columns,
                "rows": row[2],
                "data_types": data_types,
                "upload_time": upload_time,
            },
            "upload_time": upload_time,
        }
