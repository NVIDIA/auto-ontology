# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MySQL connector implementing Auto Ontology's SQLDatabase ABC."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import unquote, urlparse

import mysql.connector
import pandas as pd

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase


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


def _set_statement_timeout(cursor: Any, timeout_s: float) -> None:
    """Cap statements on *cursor*'s session at *timeout_s* seconds.

    MySQL spells the variable ``max_execution_time`` (milliseconds); MariaDB,
    which this connector also reaches, rejects that name and uses
    ``max_statement_time`` (seconds) instead.
    """
    try:
        cursor.execute(
            "SET SESSION max_execution_time = %s", (max(1, int(timeout_s * 1000)),)
        )
    except mysql.connector.Error:
        cursor.execute("SET SESSION max_statement_time = %s", (float(timeout_s),))


class MySQLDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by ``mysql-connector-python``."""

    supports_statement_timeout = True

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        self._connect_kwargs = _parse_connection_string(connection_string)
        self._database_name = str(self._connect_kwargs["database"])
        self.ping()

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
        """
        connection = mysql.connector.connect(**self._connect_kwargs)
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                if timeout_s is not None:
                    _set_statement_timeout(cursor, timeout_s)
                cursor.execute(sql, parameters)
                if cursor.description is None:
                    return pd.DataFrame()
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return pd.DataFrame(rows, columns=columns)
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
