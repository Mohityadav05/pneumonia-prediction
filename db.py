"""
db.py — lightweight SQLite logging for the pneumonia-prediction app.

Drop this file next to main.py, then wire it in as shown at the bottom
of this file (see the "How to integrate" comment block).
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "app.db")


def init_db():
    """Call this once at app startup (e.g. right before app.run())."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                gate_passed INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT NOT NULL,        -- 'user' or 'assistant'
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def log_prediction(filename: str, predicted_label: str, confidence: float, gate_passed: bool = True):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions (filename, predicted_label, confidence, gate_passed, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (filename, predicted_label, confidence, int(gate_passed), datetime.utcnow().isoformat()),
        )
        conn.commit()


def log_chat_message(session_id: str, role: str, message: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, message, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, message, datetime.utcnow().isoformat()),
        )
        conn.commit()


def get_recent_predictions(limit: int = 20):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_chat_history(session_id: str, limit: int = 50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# How to integrate into main.py
# ---------------------------------------------------------------------------
# 1. At the top of main.py:
#        from db import init_db, log_prediction, log_chat_message
#
# 2. Right before `app.run(...)` (or at module load if you use gunicorn):
#        init_db()
#
# 3. Inside your /predict route, after you get a prediction:
#        log_prediction(
#            filename=uploaded_file.filename,
#            predicted_label=predicted_label,   # e.g. "PNEUMONIA" / "NORMAL"
#            confidence=float(confidence_score),
#            gate_passed=gate_result,           # True/False from the MobileNetV2 gate check
#        )
#
# 4. Inside your /chat route, after you get the user message and the LLM reply:
#        log_chat_message(session_id, "user", user_message)
#        log_chat_message(session_id, "assistant", llm_reply)
#
# That's it — app.db will be created automatically on first run.