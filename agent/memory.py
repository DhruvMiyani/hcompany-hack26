import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "agent_memory.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bets (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT,
                market        TEXT,
                direction     TEXT,
                amount        REAL,
                odds          REAL,
                status        TEXT DEFAULT 'pending',
                profit_loss   REAL,
                raw_answer    TEXT,
                created_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS lessons (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                lesson        TEXT NOT NULL,
                confidence    REAL DEFAULT 0.5,
                applies_when  TEXT,
                times_applied INTEGER DEFAULT 0,
                times_correct INTEGER DEFAULT 0,
                active        INTEGER DEFAULT 1,
                created_at    TEXT,
                updated_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS strategy (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                rules         TEXT NOT NULL,
                created_at    TEXT
            );
        """)


def store_bet(
    session_id: str,
    market: str,
    direction: str,
    amount: float,
    odds: Optional[float],
    raw_answer: str,
) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO bets (session_id, market, direction, amount, odds, raw_answer, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, market, direction, amount, odds, raw_answer, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def update_bet_outcome(bet_id: int, status: str, profit_loss: float) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE bets SET status=?, profit_loss=? WHERE id=?",
            (status, profit_loss, bet_id),
        )


def get_pending_bets() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM bets WHERE status='pending'").fetchall()
        return [dict(r) for r in rows]


def get_resolved_bets(limit: int = 20) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bets WHERE status IN ('won','lost') ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_bets(limit: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bets ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def store_lesson(lesson: str, confidence: float, applies_when: str) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO lessons (lesson, confidence, applies_when, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (lesson, confidence, applies_when, now, now),
        )


def get_active_lessons() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM lessons WHERE active=1 ORDER BY confidence DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def save_strategy(rules: list[str]) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO strategy (rules, created_at) VALUES (?, ?)",
            (json.dumps(rules), datetime.utcnow().isoformat()),
        )


def get_latest_strategy() -> list[str]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT rules FROM strategy ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row:
            return json.loads(row["rules"])
        return _default_strategy()


def _default_strategy() -> list[str]:
    return [
        "Focus on FIFA markets with clear favorite/underdog dynamics.",
        "Prefer markets where the price is significantly mispriced vs real-world probability.",
        "Start with small bets ($1-$5) to gather data before scaling.",
        "Avoid betting on very low-liquidity markets (less than $500 total volume).",
        "Track win rate: if below 50%, pause and re-evaluate strategy.",
    ]


def get_performance_summary() -> dict:
    with _conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending,
                SUM(amount) as total_wagered,
                SUM(COALESCE(profit_loss, 0)) as total_profit_loss
            FROM bets
        """).fetchone()
        return dict(row) if row else {}
