# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL Server connector backed by pymssql."""

from __future__ import annotations

import math
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional
from urllib.parse import unquote, urlparse

import pandas as pd
import pymssql

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

logger = logging.getLogger(__name__)

# pymssql sets DB-Library query/login timeouts through process-global state.
# Keep every connection setup and operation using this module serialized so one
# statement cannot change another statement's timeout. Reset DB-Library to its
# uncapped value before releasing the lock; every operation also sets its own
# value explicitly, including zero for unbounded queries.
_MSSQL_OPERATION_LOCK = threading.RLock()


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Convert an `mssql://` URL into `pymssql.connect` arguments."""
    parsed = urlparse(connection_string)
    scheme, _, driver = parsed.scheme.lower().partition("+")
    if scheme not in {"mssql", "sqlserver"} or driver not in {"", "pymssql"}:
        raise ValueError("Invalid SQL Server connection string scheme")
    if not parsed.hostname:
        raise ValueError("SQL Server connection string is missing a host")

    database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ValueError("SQL Server connection string is missing a database")
    try:
        port = parsed.port or 1433
    except ValueError as exc:
        raise ValueError("SQL Server connection string has an invalid port") from exc

    return {
        "server": unquote(parsed.hostname),
        "port": port,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": database,
    }


class MSSQLDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by `pymssql` and FreeTDS.

    Connection URLs use the form::

        mssql://USER:PASSWORD@HOST:1433/DBNAME

    URL components must percent-encode reserved characters in credentials and
    database names. Connections request read-only intent and are opened per
    operation, so a connector instance holds no server session between calls.
    """

    supports_statement_timeout = True

    def __init__(self, connection_string: str) -> None:
        self._connect_kwargs = _parse_connection_string(connection_string)
        self._database_name = str(self._connect_kwargs["database"])
        # Fail early on unreachable hosts and make sure the named database is
        # actually selected, without including credentials in the exception.
        self.ping()

    @property
    def dialect(self) -> str:
        """Return sqlglot's SQL Server dialect name."""
        return "tsql"

    @property
    def database_name(self) -> str:
        return self._database_name

    def _connect(self) -> pymssql.Connection:
        return pymssql.connect(
            **self._connect_kwargs,
            autocommit=True,
            read_only=True,
            login_timeout=15,
            encryption="require",
        )

    @contextmanager
    def _connection(
        self, timeout_s: float | None = None
    ) -> Iterator[pymssql.Connection]:
        """Own one globally serialized DB-Library timeout scope."""
        with _MSSQL_OPERATION_LOCK:
            connection = self._connect()
            try:
                # pymssql's connect(timeout=...) only sets dbsettime for
                # nonzero values. Set the underlying property every time so
                # uncapped operations cannot inherit a previous statement cap.
                connection._conn.query_timeout = (
                    max(1, math.ceil(timeout_s)) if timeout_s is not None else 0
                )
                yield connection
            finally:
                try:
                    connection._conn.query_timeout = 0
                except Exception:
                    logger.warning(
                        "Failed to reset the process-wide SQL Server query timeout",
                        exc_info=True,
                    )
                connection.close()

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Run *sql* and return its result as a DataFrame.

        `pymssql` uses `%s` placeholders for positional parameters. Its
        query timeout is configured on the connection before each statement.
        """
        with self._connection(timeout_s) as connection:
            cursor = connection.cursor(as_dict=False)
            started = time.monotonic()
            try:
                cursor.execute(
                    sql, tuple(parameters) if parameters is not None else None
                )
                if cursor.description is None:
                    return pd.DataFrame()
                return pd.DataFrame(
                    cursor.fetchall(), columns=[d[0] for d in cursor.description]
                )
            except pymssql.Error as exc:
                if timeout_s is not None and _is_query_timeout(exc):
                    raise StatementTimeout.since(timeout_s, started) from exc
                raise
            finally:
                cursor.close()

    def get_tables(self) -> pd.DataFrame:
        """Return visible base tables and views."""
        view_type = TableTypes.VIEW
        base_table_type = TableTypes.BASE_TABLE
        return self.execute(
            f"""
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME AS table_name,
                CASE TABLE_TYPE
                    WHEN 'VIEW' THEN '{view_type}'
                    ELSE '{base_table_type}'
                END AS table_type
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE IN ('BASE TABLE', 'VIEW')
            ORDER BY TABLE_SCHEMA, TABLE_NAME
            """
        )

    def get_columns(self) -> pd.DataFrame:
        """Return visible columns with boolean nullability and ordinal order."""
        return self.execute(
            """
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME AS table_name,
                COLUMN_NAME AS column_name,
                DATA_TYPE AS data_type,
                CAST(CASE WHEN IS_NULLABLE = 'YES' THEN 1 ELSE 0 END AS BIT)
                    AS is_nullable,
                ORDINAL_POSITION AS ordinal_position
            FROM INFORMATION_SCHEMA.COLUMNS
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
            """
        )

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """SQL Server query history requires elevated permissions; omit it."""
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        """Return view definitions that the current login can see."""
        return self.execute(
            """
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME AS table_name,
                VIEW_DEFINITION AS view_definition
            FROM INFORMATION_SCHEMA.VIEWS
            ORDER BY TABLE_SCHEMA, TABLE_NAME
            """
        )

    def get_pks(self) -> pd.DataFrame:
        """Return primary-key columns in key order."""
        return self.execute(
            """
            SELECT
                kcu.TABLE_SCHEMA AS table_schema,
                kcu.TABLE_NAME AS table_name,
                kcu.COLUMN_NAME AS column_name,
                kcu.ORDINAL_POSITION AS ordinal_position
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS kcu
              ON tc.CONSTRAINT_CATALOG = kcu.CONSTRAINT_CATALOG
             AND tc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA
             AND tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
             AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
             AND tc.TABLE_NAME = kcu.TABLE_NAME
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            ORDER BY kcu.TABLE_SCHEMA, kcu.TABLE_NAME, kcu.ORDINAL_POSITION
            """
        )

    def get_fks(self) -> pd.DataFrame:
        """Return foreign-key columns, including references to unique indexes."""
        return self.execute(
            """
            SELECT
                source_schema.name AS table_schema,
                source_table.name AS table_name,
                source_column.name AS column_name,
                target_schema.name AS referenced_schema,
                target_table.name AS referenced_table,
                target_column.name AS referenced_column
            FROM sys.foreign_key_columns AS fk
            JOIN sys.tables AS source_table
              ON fk.parent_object_id = source_table.object_id
            JOIN sys.schemas AS source_schema
              ON source_table.schema_id = source_schema.schema_id
            JOIN sys.columns AS source_column
              ON fk.parent_object_id = source_column.object_id
             AND fk.parent_column_id = source_column.column_id
            JOIN sys.tables AS target_table
              ON fk.referenced_object_id = target_table.object_id
            JOIN sys.schemas AS target_schema
              ON target_table.schema_id = target_schema.schema_id
            JOIN sys.columns AS target_column
              ON fk.referenced_object_id = target_column.object_id
             AND fk.referenced_column_id = target_column.column_id
            ORDER BY source_schema.name, source_table.name,
                     fk.constraint_object_id, fk.constraint_column_id
            """
        )

    def ping(self) -> None:
        """Verify the connection and selected database."""
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT DB_NAME()")
                row = cursor.fetchone()
                if not row or str(row[0]).casefold() != self._database_name.casefold():
                    raise RuntimeError(
                        "SQL Server did not select the requested database"
                    )
            finally:
                cursor.close()

    def close(self) -> None:
        """No-op because each operation owns and closes its connection."""


def _is_query_timeout(exc: pymssql.Error) -> bool:
    """Recognize FreeTDS query-timeout errors without masking other failures."""
    message = str(exc).casefold()
    return "timeout" in message or "timed out" in message
