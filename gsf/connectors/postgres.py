# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL connector implementing the NeMo-Retriever SQLDatabase ABC."""

from __future__ import annotations

import weakref
from typing import Optional

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes
from nemo_retriever.tabular_data.sql_database import SQLDatabase


class PostgresDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by ``psycopg`` (v3).

    Parameters
    ----------
    connection_string:
        A ``libpq``-style connection URI.

        Expected format::

            postgresql://USER:PASSWORD@HOST:5432/DBNAME
    """

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        # Pool transparently replaces connections killed by server timeouts or
        # network middleboxes. `check_connection` runs a quick liveness probe
        # before handing a connection out; `max_idle`/`max_lifetime` cap how
        # long a connection can live, so stale ones are recycled before any
        # plausible firewall idle timeout fires.
        self._pool: ConnectionPool = ConnectionPool(
            connection_string,
            kwargs={"autocommit": True},
            min_size=1,
            max_size=4,
            max_idle=300.0,
            max_lifetime=1800.0,
            check=ConnectionPool.check_connection,
            open=True,
        )
        # Close the pool at GC or interpreter exit (whichever comes first) so its
        # worker/scheduler threads stop gracefully. Without this, psycopg_pool's
        # own finalizer warns ("couldn't stop thread ... within 5.0 seconds") and
        # stalls shutdown ~5s per thread when a caller forgets to close(). Binding
        # ``self._pool.close`` (not ``self``) keeps the connector GC-eligible.
        self._finalizer = weakref.finalize(self, self._pool.close)
        self._database_name: str = self.execute("SELECT current_database()").iloc[0, 0]

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "postgres"

    @property
    def database_name(self) -> str:
        return self._database_name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, parameters)
                if cur.description is None:
                    return pd.DataFrame()
                rows = cur.fetchall()
            return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_tables(self) -> pd.DataFrame:
        # Filter tables that are part of partitioned tables.
        # relkind distinguishes materialized views (m) from ordinary tables (r).
        view_type = TableTypes.VIEW
        materialized_view_type = TableTypes.MATERIALIZED_VIEW
        base_table_type = TableTypes.BASE_TABLE
        return self.execute(f"""
            SELECT
                t.table_schema AS table_schema,
                t.table_name   AS table_name,
                CASE c.relkind
                    WHEN 'v' THEN '{view_type}'
                    WHEN 'm' THEN '{materialized_view_type}'
                    ELSE '{base_table_type}'
                END AS table_type
            FROM information_schema.tables t
            JOIN pg_namespace n ON n.nspname = t.table_schema
            JOIN pg_class c ON c.relname = t.table_name AND c.relnamespace = n.oid
            WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
              AND c.relispartition = false
              AND c.relkind IN ('r', 'v', 'm', 'f')
            ORDER BY t.table_schema, t.table_name
        """)

    def get_columns(self) -> pd.DataFrame:
        return self.execute("""
            SELECT
                c.table_schema       AS table_schema,
                c.table_name         AS table_name,
                c.column_name        AS column_name,
                c.data_type          AS data_type,
                c.is_nullable        AS is_nullable,
                c.ordinal_position   AS ordinal_position
            FROM information_schema.columns c
            JOIN pg_namespace n ON n.nspname = c.table_schema
            JOIN pg_class pc ON pc.relname = c.table_name AND pc.relnamespace = n.oid
            WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
              AND pc.relispartition = false
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Return recent queries from ``pg_stat_activity``.

        Uses ``state_change`` as ``end_time`` and filters to the last ``hours``
        hours. Excludes the current backend and rows with no recorded query.
        """
        try:
            # Todo: Add filter of exclude information schema and pg_catalog tables
            return self.execute(
                """
                SELECT
                    state_change AS end_time,
                    query        AS query_text
                FROM pg_stat_activity
                WHERE pid != pg_backend_pid()
                  AND query IS NOT NULL
                  AND query <> ''
                  AND state_change >= now() - make_interval(hours => %s)
                ORDER BY state_change DESC
                """,
                [hours],
            )
        except psycopg.Error:
            return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        return self.execute("""
            SELECT
                v.table_schema     AS table_schema,
                v.table_name       AS table_name,
                v.view_definition  AS view_definition
            FROM information_schema.views v
            WHERE v.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY v.table_schema, v.table_name
        """)

    def get_pks(self) -> pd.DataFrame:
        return self.execute("""
            SELECT
                kcu.table_schema         AS table_schema,
                kcu.table_name           AS table_name,
                kcu.column_name          AS column_name,
                kcu.ordinal_position     AS ordinal_position
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY kcu.table_schema, kcu.table_name, kcu.ordinal_position
        """)

    def get_fks(self) -> pd.DataFrame:
        return self.execute("""
            SELECT
                kcu.table_schema         AS table_schema,
                kcu.table_name           AS table_name,
                kcu.column_name          AS column_name,
                ccu.table_schema         AS referenced_schema,
                ccu.table_name           AS referenced_table,
                ccu.column_name          AS referenced_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
             AND tc.table_schema    = ccu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY kcu.table_schema, kcu.table_name, kcu.column_name
        """)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify connectivity and that the catalog is readable."""
        with psycopg.connect(self._connection_string) as conn:
            conn.execute("SELECT schema_name FROM information_schema.schemata")

    def close(self) -> None:
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer.detach()
        if self._pool and not self._pool.closed:
            self._pool.close()
