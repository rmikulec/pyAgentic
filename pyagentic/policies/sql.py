# pyagentic/policies/sql.py
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Type

from sqlalchemy import (
    Table,
    Column,
    MetaData,
    Integer,
    String,
    Float,
    DateTime,
    JSON,
    Boolean,
    create_engine,
    select,
    update,
    insert,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text

from pyagentic._base._policy import Policy
from pyagentic._base._events import Event, EventKind


class SQLPolicy:
    """
    Automatically persists a Pydantic state model to a SQL table.

    Features:
    - Creates a SQL table matching the Pydantic model (if not exists)
    - Inserts or updates rows whenever the state changes
    - Can auto-load the most recent row on INIT
    """

    def __init__(
        self,
        model_type: Type[Any],
        connection_string: str = "sqlite:///agent_state.db",
        table_name: str | None = None,
        pk_field: str = "id",
        auto_create: bool = True,
        auto_load: bool = True,
        auto_commit: bool = True,
    ):
        self.model_type = model_type
        self.connection_string = connection_string
        self.engine: Engine = create_engine(connection_string, future=True)
        self.metadata = MetaData()
        self.table_name = table_name or model_type.__name__.lower()
        self.pk_field = pk_field
        self.auto_create = auto_create
        self.auto_load = auto_load
        self.auto_commit = auto_commit
        self._table: Table | None = None

        if self.auto_create:
            self._table = self._ensure_table_exists()

    # ----------------------------------------------------------
    # Utility: create table schema from Pydantic model
    # ----------------------------------------------------------
    def _ensure_table_exists(self) -> Table:
        """Create table if it doesn't already exist."""
        metadata = self.metadata
        if self.table_name in metadata.tables:
            return metadata.tables[self.table_name]

        columns = []
        for name, field in self.model_type.model_fields.items():
            t = field.annotation
            col_type = self._sql_type_for(t)
            is_pk = name == self.pk_field
            columns.append(Column(name, col_type, primary_key=is_pk))
        columns.append(Column("last_updated", DateTime, default=datetime.utcnow))
        table = Table(self.table_name, metadata, *columns)
        metadata.create_all(self.engine)
        return table

    # ----------------------------------------------------------
    # Type mapping
    # ----------------------------------------------------------
    def _sql_type_for(self, t):
        """Map Python/Pydantic types to SQL types."""
        if t in (int,):
            return Integer
        if t in (float,):
            return Float
        if t in (bool,):
            return Boolean
        if t in (str,):
            return String
        return JSON  # fallback for dicts, lists, etc.

    # ----------------------------------------------------------
    # Core Event Handler
    # ----------------------------------------------------------
    async def handle_event(self, event: Event):
        """
        On SET: upsert the Pydantic model into SQL.
        On INIT: optionally load latest data from SQL.
        """
        if event.kind == EventKind.INIT and self.auto_load:
            await self._load_state(event)
        elif event.kind == EventKind.SET:
            await self._sync_state(event)

    # ----------------------------------------------------------
    # Sync to DB
    # ----------------------------------------------------------
    async def _sync_state(self, event: Event):
        """Insert or update row based on the new Pydantic state value."""
        state_value = event.new_value
        if not isinstance(state_value, self.model_type):
            return  # skip non-model updates

        model_dict = state_value.model_dump()
        model_dict["last_updated"] = datetime.utcnow()

        if not self._table:
            self._table = self._ensure_table_exists()

        try:
            async with asyncio.to_thread(self.engine.begin)() as conn:
                # try update first
                pk_value = model_dict.get(self.pk_field)
                if pk_value is not None:
                    stmt = (
                        update(self._table)
                        .where(self._table.c[self.pk_field] == pk_value)
                        .values(**model_dict)
                    )
                    result = await asyncio.to_thread(conn.execute, stmt)
                    if result.rowcount == 0:
                        stmt = insert(self._table).values(**model_dict)
                        await asyncio.to_thread(conn.execute, stmt)
                else:
                    stmt = insert(self._table).values(**model_dict)
                    await asyncio.to_thread(conn.execute, stmt)

                if self.auto_commit:
                    await asyncio.to_thread(conn.commit)
        except SQLAlchemyError as e:
            print(f"[SQLPolicy] Sync failed for {event.name}: {e}")

    # ----------------------------------------------------------
    # Load from DB
    # ----------------------------------------------------------
    async def _load_state(self, event: Event):
        """Load the latest row from the SQL table and set it to state."""
        if not self._table:
            self._table = self._ensure_table_exists()

        try:
            async with asyncio.to_thread(self.engine.connect)() as conn:
                query = select(self._table).order_by(self._table.c.last_updated.desc()).limit(1)
                result = await asyncio.to_thread(conn.execute, query)
                row = result.fetchone()
                if row:
                    data = dict(row._mapping)
                    data.pop("last_updated", None)
                    model_instance = self.model_type.model_validate(data)
                    state = event.context.get("state")
                    if state:
                        state.set(event.name, model_instance)
        except SQLAlchemyError as e:
            print(f"[SQLPolicy] Load failed for {event.name}: {e}")
