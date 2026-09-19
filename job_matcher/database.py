"""SQLite persistence layer.

Two tables:
    profile - the baseline resume (text + cached embedding). Single row.
    jobs    - one row per job posting with all score components.

Embeddings are stored as raw float32 bytes for compactness. All access
goes through the ``Database`` class so the schema is defined in one place.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config


# --------------------------------------------------------------------------
# Row containers
# --------------------------------------------------------------------------
@dataclass
class Job:
    id: int
    filename: str
    title: str
    company: str
    raw_text: str
    tfidf_score: float
    embedding_score: float
    keyword_score: float
    final_score: float
    top_keywords: str          # comma-separated, human readable
    status: str                # 'ok' | 'empty' | 'error'
    note: str                  # extraction warning / message
    date_added: float


@dataclass
class Profile:
    resume_text: str
    embedding: Optional[np.ndarray]
    filename: str
    date_added: float


def _embedding_to_blob(vec: Optional[np.ndarray]) -> Optional[bytes]:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class Database:
    """Thin, well-typed wrapper around a SQLite connection."""

    def __init__(self, path: str = config.DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    # -- schema ------------------------------------------------------------
    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS profile (
                id           INTEGER PRIMARY KEY CHECK (id = 1),
                filename     TEXT,
                resume_text  TEXT NOT NULL,
                embedding    BLOB,
                date_added   REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT NOT NULL UNIQUE,
                title           TEXT,
                company         TEXT,
                raw_text        TEXT,
                embedding       BLOB,
                tfidf_score     REAL DEFAULT 0,
                embedding_score REAL DEFAULT 0,
                keyword_score   REAL DEFAULT 0,
                final_score     REAL DEFAULT 0,
                top_keywords    TEXT DEFAULT '',
                status          TEXT DEFAULT 'ok',
                note            TEXT DEFAULT '',
                date_added      REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    # -- profile -----------------------------------------------------------
    def save_profile(self, resume_text: str, embedding: Optional[np.ndarray],
                     filename: str = "") -> None:
        self.conn.execute(
            """
            INSERT INTO profile (id, filename, resume_text, embedding, date_added)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename    = excluded.filename,
                resume_text = excluded.resume_text,
                embedding   = excluded.embedding,
                date_added  = excluded.date_added
            """,
            (filename, resume_text, _embedding_to_blob(embedding), time.time()),
        )
        self.conn.commit()

    def get_profile(self) -> Optional[Profile]:
        row = self.conn.execute("SELECT * FROM profile WHERE id = 1").fetchone()
        if row is None:
            return None
        return Profile(
            resume_text=row["resume_text"],
            embedding=_blob_to_embedding(row["embedding"]),
            filename=row["filename"] or "",
            date_added=row["date_added"],
        )

    def has_profile(self) -> bool:
        return self.get_profile() is not None

    # -- jobs --------------------------------------------------------------
    def job_exists(self, filename: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM jobs WHERE filename = ?", (filename,)
        ).fetchone()
        return row is not None

    def upsert_job(self, *, filename: str, title: str, company: str,
                   raw_text: str, embedding: Optional[np.ndarray],
                   tfidf_score: float, embedding_score: float,
                   keyword_score: float, final_score: float,
                   top_keywords: str, status: str, note: str) -> int:
        """Insert a job, or update it in place if the filename already exists.

        This is what powers duplicate detection: re-adding the same file
        updates the existing row rather than creating a second entry.
        """
        self.conn.execute(
            """
            INSERT INTO jobs (filename, title, company, raw_text, embedding,
                              tfidf_score, embedding_score, keyword_score,
                              final_score, top_keywords, status, note, date_added)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                title           = excluded.title,
                company         = excluded.company,
                raw_text        = excluded.raw_text,
                embedding       = excluded.embedding,
                tfidf_score     = excluded.tfidf_score,
                embedding_score = excluded.embedding_score,
                keyword_score   = excluded.keyword_score,
                final_score     = excluded.final_score,
                top_keywords    = excluded.top_keywords,
                status          = excluded.status,
                note            = excluded.note,
                date_added      = excluded.date_added
            """,
            (filename, title, company, raw_text, _embedding_to_blob(embedding),
             tfidf_score, embedding_score, keyword_score, final_score,
             top_keywords, status, note, time.time()),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM jobs WHERE filename = ?", (filename,)
        ).fetchone()
        return int(row["id"])

    def update_job_scores(self, job_id: int, *, tfidf_score: float,
                          embedding_score: float, keyword_score: float,
                          final_score: float, top_keywords: str) -> None:
        self.conn.execute(
            """
            UPDATE jobs SET tfidf_score = ?, embedding_score = ?,
                            keyword_score = ?, final_score = ?, top_keywords = ?
            WHERE id = ?
            """,
            (tfidf_score, embedding_score, keyword_score, final_score,
             top_keywords, job_id),
        )
        self.conn.commit()

    def get_job_embedding(self, job_id: int) -> Optional[np.ndarray]:
        row = self.conn.execute(
            "SELECT embedding FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return _blob_to_embedding(row["embedding"])

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            filename=row["filename"],
            title=row["title"] or "",
            company=row["company"] or "",
            raw_text=row["raw_text"] or "",
            tfidf_score=row["tfidf_score"],
            embedding_score=row["embedding_score"],
            keyword_score=row["keyword_score"],
            final_score=row["final_score"],
            top_keywords=row["top_keywords"] or "",
            status=row["status"] or "ok",
            note=row["note"] or "",
            date_added=row["date_added"],
        )

    def get_all_jobs(self, ranked: bool = True) -> list[Job]:
        order = "final_score DESC, date_added DESC" if ranked else "date_added DESC"
        rows = self.conn.execute(f"SELECT * FROM jobs ORDER BY {order}").fetchall()
        return [self._row_to_job(r) for r in rows]

    def get_job(self, job_id: int) -> Optional[Job]:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return self._row_to_job(row) if row else None

    def delete_job(self, job_id: int) -> None:
        self.conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self.conn.commit()

    def clear_jobs(self) -> None:
        self.conn.execute("DELETE FROM jobs")
        self.conn.commit()

    def count_jobs(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
