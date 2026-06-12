"""
SQLite-backed JobStore: zero-dependency, crash-safe (WAL), single-file.

Concurrency model: a single connection shared across the event loop. Every
operation runs in a worker thread via ``asyncio.to_thread`` and is
serialized behind an ``asyncio.Lock``, which sidesteps cross-thread cursor
issues while keeping the public API fully async.
"""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

from pyagentic.api.jobs._models import (
    JobRecord,
    JobStatus,
    JobUpdate,
)
from pyagentic.api.jobs.store._base import JobStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    session_id    TEXT,
    status        TEXT NOT NULL,
    request_json  TEXT NOT NULL,
    construct_json TEXT,
    result_json   TEXT,
    error         TEXT,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);

CREATE TABLE IF NOT EXISTS job_updates (
    job_id       TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    event        TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (job_id, seq)
);
"""


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    """Convert a jobs table row into a JobRecord."""
    return JobRecord(
        job_id=row["job_id"],
        session_id=row["session_id"],
        status=JobStatus(row["status"]),
        request=json.loads(row["request_json"]),
        construct_payload=(
            json.loads(row["construct_json"]) if row["construct_json"] else None
        ),
        result_json=row["result_json"],
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


class SQLiteJobStore(JobStore):
    """JobStore implementation backed by a local SQLite database file.

    Attributes:
        path (Path): Location of the database file.
    """

    def __init__(self, path: str) -> None:
        """Create a store pointing at the given database file.

        Args:
            path (str): Database file path; parent directories are created
                on :meth:`initialize`.
        """
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def _run(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run a blocking DB operation in a worker thread, serialized."""
        if self._conn is None:
            raise RuntimeError("SQLiteJobStore is not initialized; call initialize().")
        async with self._lock:
            return await asyncio.to_thread(fn, self._conn)

    async def initialize(self) -> None:
        """Create the database file, schema, and pragmas. Idempotent.

        Raises:
            sqlite3.Error: If the database cannot be created or opened.
        """
        if self._conn is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def _open() -> sqlite3.Connection:
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            conn.commit()
            return conn

        async with self._lock:
            self._conn = await asyncio.to_thread(_open)

    async def close(self) -> None:
        """Close the underlying connection."""
        if self._conn is None:
            return
        async with self._lock:
            conn, self._conn = self._conn, None
            await asyncio.to_thread(conn.close)

    async def create_job(self, job: JobRecord) -> None:
        """Persist a new job record.

        Args:
            job (JobRecord): The record to persist.
        """

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO jobs (job_id, session_id, status, request_json, "
                "construct_json, result_json, error, created_at, started_at, finished_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    job.session_id,
                    job.status.value,
                    json.dumps(job.request, default=str),
                    (
                        json.dumps(job.construct_payload, default=str)
                        if job.construct_payload is not None
                        else None
                    ),
                    job.result_json,
                    job.error,
                    job.created_at,
                    job.started_at,
                    job.finished_at,
                ),
            )
            conn.commit()

        await self._run(_insert)

    async def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Fetch a job record by ID.

        Args:
            job_id (str): The job identifier.

        Returns:
            Optional[JobRecord]: The record, or None if unknown.
        """

        def _get(conn: sqlite3.Connection) -> Optional[JobRecord]:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return _row_to_record(row) if row else None

        return await self._run(_get)

    async def list_jobs(
        self,
        *,
        status: Optional[JobStatus] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JobRecord]:
        """List job records, newest first, with optional filters.

        Args:
            status (Optional[JobStatus]): Filter by status.
            session_id (Optional[str]): Filter by session.
            limit (int): Maximum number of records returned.
            offset (int): Number of records to skip.

        Returns:
            list[JobRecord]: Matching records ordered by created_at descending.
        """

        def _list(conn: sqlite3.Connection) -> list[JobRecord]:
            clauses, params = [], []
            if status is not None:
                clauses.append("status = ?")
                params.append(status.value)
            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(session_id)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC "
                f"LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            return [_row_to_record(r) for r in rows]

        return await self._run(_list)

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        result_json: Optional[str] = None,
        error: Optional[str] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
    ) -> None:
        """Update a job's status and optional terminal fields.

        Args:
            job_id (str): The job identifier.
            status (JobStatus): New status.
            result_json (Optional[str]): Final result JSON when succeeding.
            error (Optional[str]): Failure description when failing.
            started_at (Optional[float]): Execution start timestamp.
            finished_at (Optional[float]): Terminal transition timestamp.
        """

        def _update(conn: sqlite3.Connection) -> None:
            sets = ["status = ?"]
            params: list = [status.value]
            for col, val in (
                ("result_json", result_json),
                ("error", error),
                ("started_at", started_at),
                ("finished_at", finished_at),
            ):
                if val is not None:
                    sets.append(f"{col} = ?")
                    params.append(val)
            params.append(job_id)
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", params)
            conn.commit()

        await self._run(_update)

    async def append_update(
        self, job_id: str, seq: int, event: str, payload_json: str
    ) -> bool:
        """Append one update, idempotently on ``(job_id, seq)``.

        Args:
            job_id (str): The job identifier.
            seq (int): Monotonic per-job sequence number.
            event (str): SSE event name.
            payload_json (str): Exact JSON payload string.

        Returns:
            bool: True if newly inserted, False if a duplicate was ignored.
        """

        def _append(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "INSERT OR IGNORE INTO job_updates "
                "(job_id, seq, event, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, seq, event, payload_json, time.time()),
            )
            conn.commit()
            return cur.rowcount > 0

        return await self._run(_append)

    async def load_updates(
        self, job_id: str, since: int = -1, limit: Optional[int] = None
    ) -> list[JobUpdate]:
        """Load a job's updates with seq greater than the given cursor.

        Args:
            job_id (str): The job identifier.
            since (int): Exclusive seq cursor; -1 loads from the beginning.
            limit (Optional[int]): Maximum number of updates returned.

        Returns:
            list[JobUpdate]: Updates ordered by seq ascending.
        """

        def _load(conn: sqlite3.Connection) -> list[JobUpdate]:
            sql = (
                "SELECT * FROM job_updates WHERE job_id = ? AND seq > ? "
                "ORDER BY seq ASC"
            )
            params: tuple = (job_id, since)
            if limit is not None:
                sql += " LIMIT ?"
                params = (*params, limit)
            rows = conn.execute(sql, params).fetchall()
            return [
                JobUpdate(
                    job_id=r["job_id"],
                    seq=r["seq"],
                    event=r["event"],
                    payload_json=r["payload_json"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

        return await self._run(_load)

    async def max_seq(self, job_id: str) -> int:
        """Return the highest persisted seq for a job, or -1 if none.

        Args:
            job_id (str): The job identifier.

        Returns:
            int: Highest seq, or -1 when the job has no updates.
        """

        def _max(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                "SELECT MAX(seq) AS m FROM job_updates WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return row["m"] if row["m"] is not None else -1

        return await self._run(_max)

    async def jobs_by_status(self, status: JobStatus) -> list[JobRecord]:
        """Return all jobs currently in the given status.

        Args:
            status (JobStatus): Status to match.

        Returns:
            list[JobRecord]: Matching records.
        """

        def _by_status(conn: sqlite3.Connection) -> list[JobRecord]:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC",
                (status.value,),
            ).fetchall()
            return [_row_to_record(r) for r in rows]

        return await self._run(_by_status)

    async def purge_expired(self, ttl_seconds: int) -> int:
        """Delete terminal jobs that finished more than ttl_seconds ago.

        Args:
            ttl_seconds (int): Retention period in seconds.

        Returns:
            int: Number of job records deleted.
        """

        def _purge(conn: sqlite3.Connection) -> int:
            cutoff = time.time() - ttl_seconds
            cur = conn.execute(
                "DELETE FROM jobs WHERE finished_at IS NOT NULL "
                "AND finished_at < ? AND status IN (?, ?, ?)",
                (
                    cutoff,
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                ),
            )
            conn.commit()
            return cur.rowcount

        return await self._run(_purge)
