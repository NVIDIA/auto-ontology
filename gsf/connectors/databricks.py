# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Databricks connector implementing the NeMo-Retriever SQLDatabase ABC."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
from databricks import sql
from databricks.sql.client import Connection
from databricks.sql.exc import Error
from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes
from nemo_retriever.tabular_data.sql_database import SQLDatabase

logger = logging.getLogger(__name__)


def _quoted_identifier(name: str) -> str:
    """Return a Databricks-quoted identifier."""
    return f"`{name.replace('`', '``')}`"


def _parse_connection_string(
    connection_string: str,
) -> tuple[dict[str, Any], str]:
    """Parse a Databricks URL into connector kwargs and a catalog name.

    Expected format::

        databricks://token:ACCESS_TOKEN@HOST/CATALOG?http_path=SQL_HTTP_PATH
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "databricks":
        raise ValueError(f"Not a Databricks URL: {connection_string}")

    if not parsed.hostname:
        raise ValueError(
            "Databricks connection string requires a server hostname, e.g. "
            "databricks://token:access-token@workspace.cloud.databricks.com"
            "/main?http_path=%2Fsql%2F1.0%2Fwarehouses%2Fwarehouse-id"
        )

    if parsed.password is None:
        raise ValueError("Databricks connection string requires an access token")
    access_token = unquote(parsed.password)
    if not access_token:
        raise ValueError(
            "Databricks connection string requires a non-empty access token"
        )

    catalog = unquote(parsed.path.lstrip("/"))
    if not catalog:
        raise ValueError("Databricks connection string requires a catalog name")

    query = parse_qs(parsed.query)
    http_path = query.get("http_path", [None])[0]
    if not http_path:
        raise ValueError(
            "Databricks connection string requires ?http_path=/sql/1.0/warehouses/..."
        )

    return (
        {
            "server_hostname": parsed.hostname,
            "http_path": unquote(http_path),
            "access_token": access_token,
            "catalog": catalog,
        },
        catalog,
    )


class DatabricksDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by Databricks SQL."""

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        self._connect_kwargs, self._database_name = _parse_connection_string(
            connection_string
        )
        self._schema_filter: set[str] | None = (
            {schema.casefold() for schema in schemas if schema.strip()}
            if schemas
            else None
        )

    @property
    def dialect(self) -> str:
        return "databricks"

    @property
    def database_name(self) -> str:
        return self._database_name

    @contextmanager
    def _connect(self) -> Iterator[Connection]:
        connection = sql.connect(**self._connect_kwargs)
        try:
            yield connection
        finally:
            connection.close()

    def _filter_by_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (
            self._schema_filter is None
            or frame.empty
            or "table_schema" not in frame.columns
        ):
            return frame
        return frame[
            frame["table_schema"].astype(str).str.casefold().isin(self._schema_filter)
        ]

    def execute(self, sql_text: str, parameters: Optional[list] = None) -> pd.DataFrame:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_text, parameters)
                if cursor.description is None:
                    return pd.DataFrame()
                columns = [description[0].lower() for description in cursor.description]
                return pd.DataFrame(cursor.fetchall(), columns=columns)

    def get_schemas(self) -> list[str]:
        frame = self.execute(
            f"SHOW SCHEMAS IN {_quoted_identifier(self._database_name)}"
        )
        if frame.empty:
            return []
        return [str(name) for name in frame.iloc[:, 0].tolist()]

    def get_tables(self) -> pd.DataFrame:
        base_table = TableTypes.BASE_TABLE
        view = TableTypes.VIEW
        materialized_view = TableTypes.MATERIALIZED_VIEW
        catalog = _quoted_identifier(self._database_name)
        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    table_schema,
                    table_name,
                    CASE table_type
                        WHEN 'VIEW' THEN '{view}'
                        WHEN 'MATERIALIZED_VIEW' THEN '{materialized_view}'
                        ELSE '{base_table}'
                    END AS table_type
                FROM {catalog}.information_schema.tables
                WHERE table_schema != 'information_schema'
                ORDER BY table_schema, table_name
            """)
        )

    def get_columns(self) -> pd.DataFrame:
        catalog = _quoted_identifier(self._database_name)
        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    table_schema,
                    table_name,
                    column_name,
                    full_data_type AS data_type,
                    is_nullable,
                    ordinal_position
                FROM {catalog}.information_schema.columns
                WHERE table_schema != 'information_schema'
                ORDER BY table_schema, table_name, ordinal_position
            """)
        )

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        try:
            return self.execute(f"""
                SELECT
                    end_time,
                    statement_text AS query_text
                FROM system.query.history
                WHERE end_time >= current_timestamp() - INTERVAL {int(hours)} HOURS
                  AND execution_status = 'FINISHED'
                  AND NULLIF(TRIM(statement_text), '') IS NOT NULL
                ORDER BY end_time DESC
                LIMIT 10000
            """)
        except Error:
            logger.exception("Failed to fetch Databricks query history")
            return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        catalog = _quoted_identifier(self._database_name)
        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    table_schema,
                    table_name,
                    view_definition
                FROM {catalog}.information_schema.views
                WHERE table_schema != 'information_schema'
                ORDER BY table_schema, table_name
            """)
        )

    def get_pks(self) -> pd.DataFrame:
        catalog = _quoted_identifier(self._database_name)
        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    keys.table_schema,
                    keys.table_name,
                    keys.column_name,
                    keys.ordinal_position
                FROM {catalog}.information_schema.key_column_usage AS keys
                JOIN {catalog}.information_schema.table_constraints AS constraints
                  ON keys.constraint_catalog = constraints.constraint_catalog
                 AND keys.constraint_schema = constraints.constraint_schema
                 AND keys.constraint_name = constraints.constraint_name
                WHERE constraints.constraint_type = 'PRIMARY KEY'
                ORDER BY
                    keys.table_schema,
                    keys.table_name,
                    keys.ordinal_position
            """)
        )

    def get_fks(self) -> pd.DataFrame:
        catalog = _quoted_identifier(self._database_name)
        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    foreign_keys.table_schema,
                    foreign_keys.table_name,
                    foreign_keys.column_name,
                    referenced_keys.table_schema AS referenced_schema,
                    referenced_keys.table_name AS referenced_table,
                    referenced_keys.column_name AS referenced_column
                FROM {catalog}.information_schema.key_column_usage AS foreign_keys
                JOIN {catalog}.information_schema.table_constraints AS constraints
                  ON foreign_keys.constraint_catalog = constraints.constraint_catalog
                 AND foreign_keys.constraint_schema = constraints.constraint_schema
                 AND foreign_keys.constraint_name = constraints.constraint_name
                JOIN {catalog}.information_schema.referential_constraints AS refs
                  ON foreign_keys.constraint_catalog = refs.constraint_catalog
                 AND foreign_keys.constraint_schema = refs.constraint_schema
                 AND foreign_keys.constraint_name = refs.constraint_name
                JOIN {catalog}.information_schema.key_column_usage AS referenced_keys
                  ON refs.unique_constraint_catalog =
                     referenced_keys.constraint_catalog
                 AND refs.unique_constraint_schema =
                     referenced_keys.constraint_schema
                 AND refs.unique_constraint_name = referenced_keys.constraint_name
                 AND foreign_keys.position_in_unique_constraint =
                     referenced_keys.ordinal_position
                WHERE constraints.constraint_type = 'FOREIGN KEY'
                ORDER BY
                    foreign_keys.table_schema,
                    foreign_keys.table_name,
                    foreign_keys.ordinal_position
            """)
        )

    def ping(self) -> None:
        self.execute("SELECT current_catalog()")

    def close(self) -> None:
        """No persistent connection to close (connections are per-operation)."""
