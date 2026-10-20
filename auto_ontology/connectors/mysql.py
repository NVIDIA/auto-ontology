# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MySQL connector implementing Auto Ontology's SQLDatabase ABC."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

import mysql.connector
import pandas as pd

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

logger = logging.getLogger(__name__)

# The server stopping a statement at the session cap: MySQL's
# ER_QUERY_TIMEOUT (``max_execution_time``) and MariaDB's ER_STATEMENT_TIMEOUT
# (``max_statement_time``).
_TIMEOUT_ERRNOS = frozenset({3024, 1969})


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Convert a MySQL URI into ``mysql.connector.connect`` keyword arguments."""
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "mysql":
        raise ValueError(f"Invalid MySQL connection string: {connection_string!r}")
    if not parsed.hostname:
        raise ValueError("MySQL connection string is missing a host")
    database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ValueError("MySQL connection string is missing a database")

    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": database,
    }


# ER_UNKNOWN_SYSTEM_VARIABLE: this server does not have the variable we tried.
_UNKNOWN_VARIABLE_ERRNO = 1193

# How each server spells the cap, as (variable, value for a cap in seconds):
# MySQL 5.7.8+ takes milliseconds, MariaDB 10.1+ takes seconds. Tried in order.
_TIMEOUT_VARIABLES: tuple[tuple[str, Callable[[float], float]], ...] = (
    ("max_execution_time", lambda s: max(1, int(s * 1000))),
    ("max_statement_time", float),
)

# ``_timeout_variable`` before the first capped statement has probed the server.
_NOT_PROBED = "not probed"


class MySQLDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by ``mysql-connector-python``."""

    supports_statement_timeout = True

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        self._connect_kwargs = _parse_connection_string(connection_string)
        self._database_name = str(self._connect_kwargs["database"])
        # Which of _TIMEOUT_VARIABLES this server accepts, learned on the first
        # capped statement so later ones skip the variants it rejected. None
        # once neither is supported.
        self._timeout_variable: str | None = _NOT_PROBED
        self.ping()

    def _set_statement_timeout(self, cursor: Any, timeout_s: float) -> None:
        """Cap statements on *cursor*'s session at *timeout_s* seconds.

        A server that knows neither variable (MySQL < 5.7.8, MariaDB < 10.1,
        other MySQL-protocol engines) runs the statement uncapped, as it did
        before the cap existed, rather than failing every query.
        """
        for variable, to_value in _TIMEOUT_VARIABLES:
            if self._timeout_variable not in (_NOT_PROBED, variable):
                continue
            try:
                cursor.execute(f"SET SESSION {variable} = %s", (to_value(timeout_s),))
            except mysql.connector.Error as exc:
                if exc.errno != _UNKNOWN_VARIABLE_ERRNO:
                    raise
                continue
            self._timeout_variable = variable
            return
        if self._timeout_variable is not None:
            logger.warning(
                "MySQL server for %s supports no statement timeout variable; "
                "statements on it run uncapped",
                self._database_name,
            )
        self._timeout_variable = None

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "mysql"

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
        """Run *sql*; with *timeout_s*, the server stops it after that long.

        The cap is a session variable, and each call owns its connection, so it
        never outlives the statement. MySQL applies ``max_execution_time`` to
        read-only ``SELECT`` only, which is all the agent issues.

        Raises:
            StatementTimeout: the server stopped the statement at the cap.
        """
        connection = mysql.connector.connect(**self._connect_kwargs)
        try:
            cursor = connection.cursor(dictionary=True)
            started = time.monotonic()
            try:
                if timeout_s is not None:
                    self._set_statement_timeout(cursor, timeout_s)
                cursor.execute(sql, parameters)
                if cursor.description is None:
                    return pd.DataFrame()
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return pd.DataFrame(rows, columns=columns)
            except mysql.connector.Error as exc:
                if timeout_s is not None and exc.errno in _TIMEOUT_ERRNOS:
                    raise StatementTimeout.since(timeout_s, started) from exc
                raise
            finally:
                cursor.close()
        finally:
            connection.close()

    def get_tables(self) -> pd.DataFrame:
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
            FROM information_schema.tables
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_TYPE IN ('BASE TABLE', 'VIEW')
            ORDER BY TABLE_SCHEMA, TABLE_NAME
            """
        )

    def get_columns(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME AS table_name,
                COLUMN_NAME AS column_name,
                DATA_TYPE AS data_type,
                IS_NULLABLE AS is_nullable,
                ORDINAL_POSITION AS ordinal_position
            FROM information_schema.columns
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
            """
        )

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        # Query history requires optional Performance Schema consumers and
        # elevated privileges. It is not needed for schema ingestion.
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME AS table_name,
                VIEW_DEFINITION AS view_definition
            FROM information_schema.views
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_SCHEMA, TABLE_NAME
            """
        )

    def get_pks(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                kcu.TABLE_SCHEMA AS table_schema,
                kcu.TABLE_NAME AS table_name,
                kcu.COLUMN_NAME AS column_name,
                kcu.ORDINAL_POSITION AS ordinal_position
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA
             AND tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
             AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
             AND tc.TABLE_NAME = kcu.TABLE_NAME
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
              AND tc.TABLE_SCHEMA = DATABASE()
            ORDER BY kcu.TABLE_SCHEMA, kcu.TABLE_NAME, kcu.ORDINAL_POSITION
            """
        )

    def get_fks(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME AS table_name,
                COLUMN_NAME AS column_name,
                REFERENCED_TABLE_SCHEMA AS referenced_schema,
                REFERENCED_TABLE_NAME AS referenced_table,
                REFERENCED_COLUMN_NAME AS referenced_column
            FROM information_schema.key_column_usage
            WHERE TABLE_SCHEMA = DATABASE()
              AND REFERENCED_TABLE_NAME IS NOT NULL
            ORDER BY TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME
            """
        )

    def ping(self) -> None:
        connection = mysql.connector.connect(**self._connect_kwargs)
        try:
            connection.ping(reconnect=False)
        finally:
            connection.close()

    def close(self) -> None:
        """No-op because each operation owns and closes its connection."""
