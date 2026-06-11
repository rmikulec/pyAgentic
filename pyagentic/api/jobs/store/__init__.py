"""
Job store implementations and the factory.
"""

from pyagentic.api.jobs.store._base import JobStore
from pyagentic.api.jobs.store._sqlite import SQLiteJobStore

__all__ = ["JobStore", "SQLiteJobStore", "build_store"]


def build_store(path: str) -> JobStore:
    """Build a durable JobStore.

    Args:
        path (str): SQLite database path (``":memory:"`` for an ephemeral
            in-process store).

    Returns:
        JobStore: A SQLite-backed store.
    """
    return SQLiteJobStore(path)
