# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite connector implementing the :class:`~auto_ontology.connectors.base.SQLDatabase` ABC."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import pandas as pd

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

logger = logging.getLogger(__name__)

# VM instructions between deadline checks while a capped statement runs: often
# enough to stop within milliseconds, rare enough to cost nothing measurable.
_PROGRESS_INTERVAL = 10_000

# SQLite matches identifiers case-insensitively for ASCII letters only, so "Ä"
# and "ä" stay distinct tables; ``str.casefold`` would merge them.
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _fold_identifier(name: str) -> str:
    return name.translate(_ASCII_LOWER)


def _sqlite_path_from_connection_string(connection_string: str) -> Path:
    """Resolve a ``sqlite:///`` URI or bare filesystem path to a ``.sqlite`` file."""
    if connection_string.startswith("sqlite:"):
        parsed = urlparse(connection_string)
        path = unquote(parsed.path)
        if not path:
            raise ValueError(f"Invalid SQLite connection string: {connection_string!r}")
        if len(path) > 2 and path[0] == "/" and path[1].isalpha() and path[2] == ":":
            path = path[1:]
        return Path(path)
    return Path(connection_string)


class SQLiteDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by the stdlib ``sqlite3`` module.

    Parameters
    ----------
    connection_string:
        A SQLite URI or filesystem path, e.g.::

            sqlite:////absolute/path/to/db.sqlite
            /absolute/path/to/db.sqlite
    """

    supports_statement_timeout = True

    def __init__(self, connection_string: str) -> None:
        db_path = _sqlite_path_from_connection_string(connection_string)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        self._db_path = db_path
        self._connection_string = connection_string
        self._database_name = db_path.stem
        # sqlite3 connections are bound to the thread that created them, but the
        # semantic-compilation pipeline profiles tables across a ThreadPoolExecutor.
        # Give each thread its own connection (mirroring the Postgres pool) so a
        # connection is never shared across threads.
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        # Open eagerly on the constructing thread to surface connection errors early.
        self._conn
        logger.debug(
            "SQLite connected (database=%r, path=%s).", self._database_name, db_path
        )

    @property
    def _conn(self) -> sqlite3.Connection:
        """Return this thread's SQLite connection, creating it on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._conns_lock:
                self._all_conns.append(conn)
        return conn

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "sqlite"

    @property
    def database_name(self) -> str:
        return self._database_name

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Run *sql*; with *timeout_s*, abort it and raise :class:`StatementTimeout`.

        SQLite has no statement timeout, so a progress handler checks the clock
        every few thousand VM instructions and aborts once the deadline passes.
        Connections are per thread, so the handler only ever sees this call.
        """
        if timeout_s is None:
            return self._execute(sql, parameters)

        conn = self._conn
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(
            lambda: int(time.monotonic() > deadline), _PROGRESS_INTERVAL
        )
        try:
            return self._execute(sql, parameters)
        except sqlite3.OperationalError as exc:
            if time.monotonic() > deadline and "interrupted" in str(exc):
                raise StatementTimeout(timeout_s) from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)

    def _execute(self, sql: str, parameters: Optional[list]) -> pd.DataFrame:
        cur = self._conn.execute(sql, parameters or [])
        if cur.description is None:
            return pd.DataFrame()
        rows = cur.fetchall()
        return pd.DataFrame(
            [dict(row) for row in rows], columns=[col[0] for col in cur.description]
        )

    def get_tables(self) -> pd.DataFrame:
        view_type = TableTypes.VIEW
        base_table_type = TableTypes.BASE_TABLE
        return self.execute(
            f"""
            SELECT
                'main' AS table_schema,
                name   AS table_name,
                CASE type
                    WHEN 'view' THEN '{view_type}'
                    ELSE '{base_table_type}'
                END AS table_type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )

    def get_columns(self) -> pd.DataFrame:
        tables = self.get_tables()
        rows: list[dict[str, object]] = []
        for table_name in tables["table_name"]:
            info = self._conn.execute(f'PRAGMA table_xinfo("{table_name}")').fetchall()
            position = 1
            for _cid, name, data_type, notnull, _default, _pk, hidden in info:
                if hidden == 1:
                    continue
                rows.append(
                    {
                        "table_schema": "main",
                        "table_name": table_name,
                        "column_name": name,
                        "data_type": data_type or "TEXT",
                        "is_nullable": not notnull,
                        "ordinal_position": position,
                    }
                )
                position += 1
        return pd.DataFrame(rows)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                'main' AS table_schema,
                name   AS table_name,
                sql    AS view_definition
            FROM sqlite_master
            WHERE type = 'view'
            ORDER BY name
            """
        )

    def get_pks(self) -> pd.DataFrame:
        tables = self.get_tables()
        rows: list[dict[str, object]] = []
        for table_name in tables["table_name"]:
            info = self._conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            for cid, name, _type, _notnull, _default, pk in info:
                if pk > 0:
                    rows.append(
                        {
                            "table_schema": "main",
                            "table_name": table_name,
                            "column_name": name,
                            "ordinal_position": pk,
                        }
                    )
        return pd.DataFrame(rows)

    def get_fks(self) -> pd.DataFrame:
        """Foreign keys, naming each column as the catalog stores it.

        ``PRAGMA foreign_key_list`` reports the constraint as it was written:
        ``REFERENCES parent`` (no column list, meaning the parent's primary key)
        leaves the referenced column empty, and names keep whatever case the
        DDL used, which SQLite matches case-insensitively. Both are resolved
        against the declared tables here, so ``add_fks`` can find every column
        instead of silently dropping the key.
        """
        tables = self.get_tables()
        declared = {_fold_identifier(name): name for name in tables["table_name"]}
        table_info: dict[str, list] = {}

        def columns_of(table_name: str) -> list:
            if table_name not in table_info:
                info = self._conn.execute(
                    f'PRAGMA table_xinfo("{table_name}")'
                ).fetchall()
                table_info[table_name] = [c for c in info if c[6] != 1]
            return table_info[table_name]

        def declared_column(table_name: str, column_name: str) -> str:
            for column in columns_of(table_name):
                if _fold_identifier(column[1]) == _fold_identifier(column_name):
                    return column[1]
            return column_name

        rows: list[dict[str, object]] = []
        for table_name in tables["table_name"]:
            fks = self._conn.execute(
                f'PRAGMA foreign_key_list("{table_name}")'
            ).fetchall()
            for _id, seq, ref_table, from_col, ref_col, *_rest in fks:
                parent = declared.get(_fold_identifier(ref_table))
                if parent is not None:
                    if ref_col is None:
                        key = sorted(
                            (c for c in columns_of(parent) if c[5] > 0),
                            key=lambda c: c[5],
                        )
                        ref_col = key[seq][1] if seq < len(key) else None
                    else:
                        ref_col = declared_column(parent, ref_col)
                rows.append(
                    {
                        "table_schema": "main",
                        "table_name": table_name,
                        "column_name": declared_column(table_name, from_col),
                        "referenced_schema": "main",
                        "referenced_table": parent or ref_table,
                        "referenced_column": ref_col,
                    }
                )
        return pd.DataFrame(rows)

    def close(self) -> None:
        with self._conns_lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:
                    logger.exception("Failed to close SQLite connection")
            self._all_conns.clear()
        self._local = threading.local()
