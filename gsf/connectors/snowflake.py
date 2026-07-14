# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snowflake connector implementing the NeMo-Retriever SQLDatabase ABC."""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import snowflake.connector
from nemo_retriever.tabular_data.sql_database import SQLDatabase

logger = logging.getLogger(__name__)


def _quoted_identifier(name: str) -> str:
    """Return a Snowflake-quoted identifier (preserves case and special chars)."""
    return '"' + name.replace('"', '""') + '"'


def _parse_connection_string(
    connection_string: str,
) -> tuple[dict[str, Any], str, str]:
    """Parse a Snowflake URL into connector kwargs, warehouse, and database name.

    Required URL parts: ``user``, ``password``, ``account`` (host), ``warehouse``,
    and ``database`` query param.

    Expected format::

        snowflake://USER:PASSWORD@ACCOUNT?warehouse=WH&database=SF_DB
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "snowflake":
        raise ValueError(f"Not a Snowflake URL: {connection_string}")
    if not parsed.hostname:
        raise ValueError(
            f"Invalid Snowflake connection string (missing account): {connection_string}"
        )

    user = unquote(parsed.username or "")
    if not user:
        raise ValueError(
            "Snowflake connection string requires user in the URL, e.g. "
            "snowflake://user:pass@account?warehouse=COMPUTE_WH&database=MY_DB"
        )

    if parsed.password is None:
        raise ValueError(
            "Snowflake connection string requires password in the URL, e.g. "
            "snowflake://user:pass@account?warehouse=COMPUTE_WH&database=MY_DB"
        )
    password = unquote(parsed.password)
    if not password:
        raise ValueError("Snowflake connection string requires a non-empty password")

    query = parse_qs(parsed.query)
    warehouse = query.get("warehouse", [None])[0]
    if not warehouse:
        raise ValueError("Snowflake connection string requires ?warehouse=COMPUTE_WH")

    database = query.get("database", [None])[0]
    if database:
        database = unquote(database)
    else:
        database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ValueError(
            "Snowflake connection string requires ?database=SNOWFLAKE_DB, e.g. "
            "snowflake://user:pass@account?warehouse=COMPUTE_WH&database=MY_DB"
        )

    connect_kwargs: dict[str, Any] = {
        "user": user,
        "password": password,
        "account": parsed.hostname,
        "database": database,
        "warehouse": warehouse,
        "login_timeout": 10,
    }

    role = query.get("role", [None])[0]
    if role:
        connect_kwargs["role"] = role

    schema = query.get("schema", [None])[0]
    if schema:
        connect_kwargs["schema"] = schema

    return connect_kwargs, warehouse, database


class SnowflakeDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by ``snowflake-connector-python``.

    Parameters
    ----------
    connection_string:
        ``snowflake://user:password@account?warehouse=COMPUTE_WH&database=MY_DB``
    """

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        (
            self._connect_kwargs,
            self._warehouse,
            self._database_name,
        ) = _parse_connection_string(connection_string)
        # Optional ingestion allowlist: when set, schema introspection only
        # returns objects in these schemas. Empty/None means "all schemas".
        # Compared case-insensitively (Snowflake upper-cases unquoted names).
        self._schema_filter: set[str] | None = (
            {s.upper() for s in schemas if s and s.strip()} if schemas else None
        )
        if self._schema_filter:
            logger.info(
                "Snowflake ingestion restricted to schemas: %s",
                sorted(self._schema_filter),
            )

    def _filter_by_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict a schema-introspection frame to the configured allowlist.

        No-op when no filter is set or the frame lacks a ``table_schema`` column.
        """
        if self._schema_filter is None or df.empty or "table_schema" not in df.columns:
            return df
        return df[df["table_schema"].str.upper().isin(self._schema_filter)]

    @property
    def dialect(self) -> str:
        return "snowflake"

    @property
    def database_name(self) -> str:
        return self._database_name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        with snowflake.connector.connect(**self._connect_kwargs) as conn:
            with conn.cursor() as cur:
                cur.execute(f"USE WAREHOUSE {_quoted_identifier(self._warehouse)}")
                if parameters:
                    cur.execute(sql, parameters)
                else:
                    cur.execute(sql)
                if cur.description is None:
                    return pd.DataFrame()
                columns = [desc[0].lower() for desc in cur.description]
                return pd.DataFrame(cur.fetchall(), columns=columns)

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        """List the database's schemas (excluding ``INFORMATION_SCHEMA``).

        Used by the connection UI to let the user pick which schemas to ingest.
        """
        df = self.execute("""
            SELECT SCHEMA_NAME AS schema_name
            FROM INFORMATION_SCHEMA.SCHEMATA
            WHERE SCHEMA_NAME != 'INFORMATION_SCHEMA'
            ORDER BY SCHEMA_NAME
        """)
        if df.empty:
            return []
        return [str(name) for name in df["schema_name"].tolist()]

    def get_tables(self) -> pd.DataFrame:
        return self._filter_by_schema(
            self.execute("""
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME   AS table_name,
                TABLE_TYPE   AS table_type
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA != 'INFORMATION_SCHEMA'
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        )

    def get_columns(self) -> pd.DataFrame:
        df = self.execute("""
            SELECT
                TABLE_SCHEMA     AS table_schema,
                TABLE_NAME       AS table_name,
                COLUMN_NAME      AS column_name,
                DATA_TYPE        AS data_type,
                IS_NULLABLE      AS is_nullable,
                ORDINAL_POSITION AS ordinal_position
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA != 'INFORMATION_SCHEMA'
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
        """)
        if df.empty:
            return df

        # https://stackoverflow.com/questions/64667418/snowflake-sys-facade-in-information-schema-columns
        df["column_name"] = df["column_name"].str.removesuffix("$SYS_FACADE$0")
        df["column_name"] = df["column_name"].str.removesuffix("$SYS_FACADE$1")
        df = df.drop_duplicates(subset=["table_schema", "table_name", "column_name"])
        return self._filter_by_schema(df)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Return recent queries from ``INFORMATION_SCHEMA.QUERY_HISTORY``."""
        try:
            df = self.execute(f"""
                SELECT
                    END_TIME   AS end_time,
                    QUERY_TEXT AS query_text
                FROM TABLE(
                    INFORMATION_SCHEMA.QUERY_HISTORY(
                        DATEADD(hour, -{hours}, CURRENT_TIMESTAMP()),
                        CURRENT_TIMESTAMP(),
                        RESULT_LIMIT => 10000
                    )
                )
                WHERE QUERY_TYPE NOT IN (
                    'USE', 'SHOW', 'GRANT', 'CREATE_USER', 'CREATE_ROLE',
                    'DROP', 'COMMIT', 'ALTER_SESSION', 'CALL'
                )
                  AND EXECUTION_STATUS = 'SUCCESS'
                  AND LOWER(QUERY_TEXT) NOT LIKE '%information_schema%'
                ORDER BY END_TIME DESC
            """)
            return df[["end_time", "query_text"]]
        except snowflake.connector.errors.Error:
            logger.exception("Failed to fetch Snowflake query history")
            return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        db = _quoted_identifier(self._database_name)
        df = self.execute(f"SHOW VIEWS IN DATABASE {db}")
        if df.empty:
            return pd.DataFrame(
                columns=["table_schema", "table_name", "view_definition"]
            )

        df = df.loc[df["schema_name"] != "INFORMATION_SCHEMA"]
        df = df.rename(
            columns={
                "schema_name": "table_schema",
                "name": "table_name",
                "text": "view_definition",
            }
        )[["table_schema", "table_name", "view_definition"]]
        return self._filter_by_schema(df)

    def get_pks(self) -> pd.DataFrame:
        db = _quoted_identifier(self._database_name)
        df = self.execute(f"SHOW PRIMARY KEYS IN DATABASE {db}")
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "table_schema",
                    "table_name",
                    "column_name",
                    "ordinal_position",
                ]
            )

        df = df.rename(
            columns={
                "schema_name": "table_schema",
                "key_sequence": "ordinal_position",
            }
        )[["table_schema", "table_name", "column_name", "ordinal_position"]]
        return self._filter_by_schema(df)

    def get_fks(self) -> pd.DataFrame:
        db = _quoted_identifier(self._database_name)
        df = self.execute(f"SHOW IMPORTED KEYS IN DATABASE {db}")
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "table_schema",
                    "table_name",
                    "column_name",
                    "referenced_schema",
                    "referenced_table",
                    "referenced_column",
                ]
            )

        df = df.rename(
            columns={
                "fk_schema_name": "table_schema",
                "fk_table_name": "table_name",
                "fk_column_name": "column_name",
                "pk_schema_name": "referenced_schema",
                "pk_table_name": "referenced_table",
                "pk_column_name": "referenced_column",
            }
        )[
            [
                "table_schema",
                "table_name",
                "column_name",
                "referenced_schema",
                "referenced_table",
                "referenced_column",
            ]
        ]
        return self._filter_by_schema(df)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify credentials, the warehouse, and that schemas are visible."""
        with snowflake.connector.connect(**self._connect_kwargs) as conn:
            conn.execute_string(
                f"USE WAREHOUSE {_quoted_identifier(self._warehouse)}; SHOW SCHEMAS"
            )

    def close(self) -> None:
        """No persistent connection to close (connections are per-query)."""
