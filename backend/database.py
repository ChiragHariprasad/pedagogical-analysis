"""
SQLite database layer for the Pedagogical Intelligence System.

Uses the stdlib ``sqlite3`` module – zero external dependencies, file-based,
and sufficient for a class of ~140 students.

DB file: ``c:/projects/nlp/backend/survey_data.db``
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH: str = os.environ.get(
    "DATABASE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey_data.db")
)

DATABASE_URL: str | None = os.environ.get("DATABASE_URL")
IS_POSTGRES: bool = bool(DATABASE_URL and (DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")))

if IS_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor


class DBConnection:
    """A dual-database connection wrapper supporting SQLite and PostgreSQL.

    Encapsulates differences in placeholders (?, %s), schema keywords, and cursors.
    """

    def __init__(self) -> None:
        self.is_postgres = IS_POSTGRES
        if self.is_postgres:
            url = DATABASE_URL
            if url and url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            self.conn = psycopg2.connect(url)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        else:
            self.conn = sqlite3.connect(DB_PATH)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.row_factory = sqlite3.Row
            self.cursor = None

    def execute(self, query: str, params: tuple = ()) -> Any:
        if self.is_postgres:
            query = query.replace("?", "%s")
            self.cursor.execute(query, params)
            return self.cursor
        else:
            return self.conn.execute(query, params)

    def executescript(self, script: str) -> None:
        if self.is_postgres:
            self.cursor.execute(script)
        else:
            self.conn.executescript(script)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        if self.is_postgres and self.cursor:
            self.cursor.close()
        self.conn.close()


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def _get_conn() -> DBConnection:
    """Return a new DBConnection (either SQLite or PostgreSQL)."""
    return DBConnection()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create tables if they do not already exist."""
    conn = _get_conn()

    responses_schema = """
    CREATE TABLE IF NOT EXISTS responses (
        id              SERIAL PRIMARY KEY,
        survey_id       TEXT    NOT NULL,
        pedagogy_id     TEXT    NOT NULL,
        pedagogy_name   TEXT    NOT NULL,
        effectiveness   INTEGER NOT NULL CHECK (effectiveness BETWEEN 0 AND 5),
        engagement      INTEGER NOT NULL CHECK (engagement    BETWEEN 0 AND 5),
        clarity         INTEGER NOT NULL CHECK (clarity       BETWEEN 0 AND 5),
        relevance       INTEGER NOT NULL CHECK (relevance     BETWEEN 0 AND 5),
        feedback        TEXT    NOT NULL,
        absa_result_json TEXT   NOT NULL,
        FOREIGN KEY (survey_id) REFERENCES surveys (id)
    );
    """ if IS_POSTGRES else """
    CREATE TABLE IF NOT EXISTS responses (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        survey_id       TEXT    NOT NULL,
        pedagogy_id     TEXT    NOT NULL,
        pedagogy_name   TEXT    NOT NULL,
        effectiveness   INTEGER NOT NULL CHECK (effectiveness BETWEEN 0 AND 5),
        engagement      INTEGER NOT NULL CHECK (engagement    BETWEEN 0 AND 5),
        clarity         INTEGER NOT NULL CHECK (clarity       BETWEEN 0 AND 5),
        relevance       INTEGER NOT NULL CHECK (relevance     BETWEEN 0 AND 5),
        feedback        TEXT    NOT NULL,
        absa_result_json TEXT   NOT NULL,
        FOREIGN KEY (survey_id) REFERENCES surveys (id)
    );
    """

    try:
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS surveys (
                id          TEXT PRIMARY KEY,
                submitted_at TEXT NOT NULL,
                student_email TEXT UNIQUE
            );
            {responses_schema}
            """
        )
        conn.commit()
        if IS_POSTGRES:
            logger.info("PostgreSQL database initialised.")
        else:
            logger.info("Database initialised at %s", DB_PATH)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def insert_survey(
    survey_id: str,
    submitted_at: str,
    responses_with_absa: List[Dict[str, Any]],
    student_email: Optional[str] = None,
) -> None:
    """Insert a survey header and all associated responses in one transaction.

    Each entry in *responses_with_absa* must contain::

        {
            "pedagogy_id": str,
            "pedagogy_name": str,
            "effectiveness": int,
            "engagement": int,
            "clarity": int,
            "relevance": int,
            "feedback": str,
            "absa_result": dict,   # the ABSAResult dict
        }
    """
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO surveys (id, submitted_at, student_email) VALUES (?, ?, ?)",
            (survey_id, submitted_at, student_email),
        )
        for r in responses_with_absa:
            conn.execute(
                """
                INSERT INTO responses
                    (survey_id, pedagogy_id, pedagogy_name,
                     effectiveness, engagement, clarity, relevance,
                     feedback, absa_result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    survey_id,
                    r["pedagogy_id"],
                    r["pedagogy_name"],
                    r["effectiveness"],
                    r["engagement"],
                    r["clarity"],
                    r["relevance"],
                    r["feedback"],
                    json.dumps(r["absa_result"]),
                ),
            )
        conn.commit()
        logger.info(
            "Inserted survey %s with %d responses.", survey_id, len(responses_with_absa)
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def _row_to_dict(row: sqlite3.Row | dict[str, Any]) -> Dict[str, Any]:
    """Convert a ``sqlite3.Row`` or dict to a plain dict with parsed ABSA JSON."""
    d = dict(row)
    if "absa_result_json" in d:
        d["absa_result"] = json.loads(d["absa_result_json"])
        del d["absa_result_json"]
    return d


def get_all_responses() -> List[Dict[str, Any]]:
    """Return every response row with ABSA JSON parsed."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT r.*, s.submitted_at
            FROM responses r
            JOIN surveys s ON r.survey_id = s.id
            ORDER BY r.id DESC
            """
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_responses_by_pedagogy(pedagogy_id: str) -> List[Dict[str, Any]]:
    """Return responses filtered to a single pedagogy type."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT r.*, s.submitted_at
            FROM responses r
            JOIN surveys s ON r.survey_id = s.id
            WHERE r.pedagogy_id = ?
            ORDER BY r.id DESC
            """,
            (pedagogy_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_response_count() -> int:
    """Return total number of individual response rows."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM responses").fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def get_analytics(pedagogy_id: Optional[str] = None) -> Dict[str, Any]:
    """Return aggregated analytics, optionally filtered by pedagogy.

    Returns::

        {
            "total_responses": int,
            "total_surveys": int,
            "pedagogy_analytics": [
                {
                    "pedagogy_id": str,
                    "pedagogy_name": str,
                    "count": int,
                    "avg_effectiveness": float,
                    "avg_engagement": float,
                    "avg_clarity": float,
                    "avg_relevance": float,
                    "sentiment_distribution": {"Positive": int, "Negative": int, "Neutral": int},
                    "top_aspects": [{"aspect": str, "count": int}, …],
                },
                …
            ]
        }
    """
    conn = _get_conn()
    try:
        # Total counts
        if pedagogy_id:
            total_responses = conn.execute(
                "SELECT COUNT(*) AS cnt FROM responses WHERE pedagogy_id = ?",
                (pedagogy_id,),
            ).fetchone()["cnt"]
            total_surveys = conn.execute(
                "SELECT COUNT(DISTINCT survey_id) AS cnt FROM responses WHERE pedagogy_id = ?",
                (pedagogy_id,),
            ).fetchone()["cnt"]
        else:
            total_responses = conn.execute(
                "SELECT COUNT(*) AS cnt FROM responses"
            ).fetchone()["cnt"]
            total_surveys = conn.execute(
                "SELECT COUNT(*) AS cnt FROM surveys"
            ).fetchone()["cnt"]

        # Per-pedagogy aggregation
        where = "WHERE r.pedagogy_id = ?" if pedagogy_id else ""
        params: tuple = (pedagogy_id,) if pedagogy_id else ()
        agg_rows = conn.execute(
            f"""
            SELECT
                r.pedagogy_id,
                r.pedagogy_name,
                COUNT(*)                    AS cnt,
                AVG(r.effectiveness)        AS avg_effectiveness,
                AVG(r.engagement)           AS avg_engagement,
                AVG(r.clarity)              AS avg_clarity,
                AVG(r.relevance)            AS avg_relevance
            FROM responses r
            {where}
            GROUP BY r.pedagogy_id
            ORDER BY avg_effectiveness DESC
            """,
            params,
        ).fetchall()

        pedagogy_analytics: List[Dict[str, Any]] = []
        for agg in agg_rows:
            pid = agg["pedagogy_id"]

            # Fetch all ABSA results for this pedagogy to compute sentiment
            # distribution & top aspects
            detail_rows = conn.execute(
                "SELECT absa_result_json FROM responses WHERE pedagogy_id = ?",
                (pid,),
            ).fetchall()

            sentiment_dist: Dict[str, int] = {
                "Positive": 0,
                "Negative": 0,
                "Neutral": 0,
            }
            aspect_counter: Dict[str, int] = {}
            for dr in detail_rows:
                absa = json.loads(dr["absa_result_json"])
                for aspect_obj in absa.get("aspects", []):
                    sent = aspect_obj.get("sentiment", "Neutral")
                    sentiment_dist[sent] = sentiment_dist.get(sent, 0) + 1
                    asp = aspect_obj.get("aspect", "unknown")
                    aspect_counter[asp] = aspect_counter.get(asp, 0) + 1

            top_aspects = sorted(
                [{"aspect": k, "count": v} for k, v in aspect_counter.items()],
                key=lambda x: x["count"],
                reverse=True,
            )[:10]

            pedagogy_analytics.append(
                {
                    "pedagogy_id": pid,
                    "pedagogy_name": agg["pedagogy_name"],
                    "count": agg["cnt"],
                    "avg_effectiveness": round(agg["avg_effectiveness"], 2),
                    "avg_engagement": round(agg["avg_engagement"], 2),
                    "avg_clarity": round(agg["avg_clarity"], 2),
                    "avg_relevance": round(agg["avg_relevance"], 2),
                    "sentiment_distribution": sentiment_dist,
                    "top_aspects": top_aspects,
                }
            )

        return {
            "total_responses": total_responses,
            "total_surveys": total_surveys,
            "pedagogy_analytics": pedagogy_analytics,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Email duplicate check
# ---------------------------------------------------------------------------


def check_email_submitted(email: str) -> bool:
    """Return True if a survey with this student_email already exists."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM surveys WHERE student_email = ?",
            (email,),
        ).fetchone()
        return row["cnt"] > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data wipe (for re-seeding)
# ---------------------------------------------------------------------------


def wipe_all_data() -> None:
    """Delete all rows from responses and surveys tables."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM responses")
        conn.execute("DELETE FROM surveys")
        conn.commit()
        logger.info("Wiped all data from responses and surveys tables.")
    finally:
        conn.close()
