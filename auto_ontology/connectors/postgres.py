# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL connector implementing the :class:`~auto_ontology.connectors.base.SQLDatabase` ABC."""

from __future__ import annotations

import time
import weakref
from typing import Optional

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout


def _fetch_frame(
    conn: psycopg.Connection, sql: str, parameters: Optional[list]
) -> pd.DataFrame:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, parameters)
        if cur.description is None:
            return pd.DataFrame()
        rows = cur.fetchall()
    return pd.DataFrame(rows)


class PostgresDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by ``psycopg`` (v3).

    Parameters
    ----------
    connection_string:
        A ``libpq``-style connection URI.

        Expected format::

            postgresql://USER:PASSWORD@HOST:5432/DBNAME
    """

    supports_statement_timeout = True

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

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Run *sql*; with *timeout_s*, the server cancels it after that long.

        The cap is ``SET LOCAL statement_timeout`` inside a transaction, so it
        ends with the statement and never leaks onto the pooled connection.

        Raises:
            StatementTimeout: the server cancelled the statement at the cap.
        """
        with self._pool.connection() as conn:
            if timeout_s is None:
                return _fetch_frame(conn, sql, parameters)
            started = time.monotonic()
            try:
                with conn.transaction():
                    conn.execute(
                        "SELECT set_config('statement_timeout', %s, true)",
                        (f"{max(1, int(timeout_s * 1000))}ms",),
                    )
                    return _fetch_frame(conn, sql, parameters)
            except psycopg.errors.QueryCanceled as exc:
                # QueryCanceled also covers pg_cancel_backend(); only the
                # timeout we set is ours to relabel.
                if "statement timeout" not in str(exc):
                    raise
                raise StatementTimeout.since(timeout_s, started) from exc

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_tables(self) -> pd.DataFrame:
        """Every relation a user can query, one row each.

        Driven from ``pg_class`` rather than ``information_schema.tables``,
        because that view omits materialized views entirely — so a matview could
        never reach the catalog no matter what ``relkind`` was allowed through,
        and :data:`TableTypes.MATERIALIZED_VIEW` was unreachable.

        ``relkind`` cases, all of which are queryable and belong in the catalog:
          r  ordinary table          p  partitioned table (the parent)
          v  view                    m  materialized view
          f  foreign table

        ``relispartition = false`` drops partition *children*: they are an
        implementation detail of their parent, and listing both would show the
        same rows twice under names no one writes queries against. The parent
        ``p`` must be kept for exactly that reason — excluding both, as this did
        before, made partitioned tables invisible.
        """
        view_type = TableTypes.VIEW
        materialized_view_type = TableTypes.MATERIALIZED_VIEW
        base_table_type = TableTypes.BASE_TABLE
        return self.execute(f"""
            SELECT
                n.nspname   AS table_schema,
                c.relname   AS table_name,
                CASE c.relkind
                    WHEN 'v' THEN '{view_type}'
                    WHEN 'm' THEN '{materialized_view_type}'
                    ELSE '{base_table_type}'
                END AS table_type
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
              AND n.nspname NOT LIKE 'pg\\_%'
              AND c.relispartition = false
              AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
              -- Restores what information_schema filtered for free: a relation
              -- the ingesting role cannot read must not be catalogued, or
              -- text-to-SQL will offer it and every query dies with
              -- "permission denied".
              AND has_table_privilege(c.oid, 'SELECT')
            ORDER BY n.nspname, c.relname
        """)

    def get_columns(self) -> pd.DataFrame:
        """Columns for every relation :meth:`get_tables` returns.

        Also driven from ``pg_catalog``, and for the same reason: matview
        columns are absent from ``information_schema.columns``, so a matview
        that reached the catalog would arrive with no columns at all.

        ``format_type(atttypid, NULL)`` gives the unqualified type name without
        precision (``character varying``, not ``character varying(255)``). It is
        **not** identical to ``information_schema.columns.data_type``, which
        flattens a domain to its base type and reports ``ARRAY`` /
        ``USER-DEFINED`` where this reports ``text[]`` / ``mpaa_rating``. The
        more specific name is deliberate — it is what the text-to-SQL agent
        needs to write a valid predicate against an enum or an array column.

        ``ordinal_position`` is a row number, not ``attnum``. ``attnum`` keeps
        the slots of dropped columns, so a table that has ever had a
        ``DROP COLUMN`` would number its columns 1, 3, 4 and disagree with both
        ``information_schema`` and the model-interchange export.

        ``has_table_privilege`` restores the filtering ``information_schema``
        applied for free: without it, a role with rights on one schema still
        catalogues, embeds and offers every table it cannot read, and every
        generated query against them dies with ``permission denied``.
        """
        return self.execute("""
            SELECT
                n.nspname                        AS table_schema,
                c.relname                        AS table_name,
                a.attname                        AS column_name,
                format_type(a.atttypid, NULL)    AS data_type,
                NOT a.attnotnull                 AS is_nullable,
                row_number() OVER (
                    PARTITION BY c.oid ORDER BY a.attnum
                )                                AS ordinal_position
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
              AND n.nspname NOT LIKE 'pg\\_%'
              AND c.relispartition = false
              AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
              AND has_table_privilege(c.oid, 'SELECT')
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY n.nspname, c.relname, a.attnum
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
        """Primary-key columns, in key order.

        Driven from ``pg_constraint`` for the same reason as :meth:`get_fks`:
        ``information_schema.table_constraints`` admits a table only for its
        owner or a role with a privilege *other than* ``SELECT``, so a read-only
        ingesting role saw no primary keys at all. The filters match
        :meth:`get_tables`.
        """
        return self.execute("""
            SELECT
                n.nspname   AS table_schema,
                c.relname   AS table_name,
                a.attname   AS column_name,
                k.ord       AS ordinal_position
            FROM pg_constraint con
            CROSS JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord)
            JOIN pg_class c     ON c.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = k.attnum
            WHERE con.contype = 'p'
              AND c.relispartition = false
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
              AND n.nspname NOT LIKE 'pg\\_%'
              AND has_table_privilege(c.oid, 'SELECT')
            ORDER BY n.nspname, c.relname, k.ord
        """)

    def get_fks(self) -> pd.DataFrame:
        """One row per foreign-key column, paired with the column it references.

        Driven from ``pg_constraint`` rather than ``information_schema``, which
        got four things wrong here:

        - ``table_constraints`` and ``constraint_column_usage`` admit a table
          only for its owner (or, for the former, a privilege other than
          ``SELECT``), so a read-only ingesting role saw no foreign keys at all.
        - The views are matched by constraint *name*, which is unique per
          table, not per schema: two tables with a same-named constraint were
          each given the other's referenced table.
        - ``constraint_column_usage`` has no column order, so a composite key
          came back as every combination of its columns rather than the pairs.
          ``conkey`` and ``confkey`` are parallel arrays; unnesting them
          together keeps the pairs, and ``ord`` keeps key order.
        - Its ``table_schema`` is the *referenced* table's schema, so joining it
          to the constraint's own schema dropped every key that crosses a
          schema boundary.

        ``relispartition`` on both ends drops what :meth:`get_tables` hides:
        the constraint copies Postgres clones onto each partition of a
        partitioned table (and onto each partition of a referenced one), and
        any key a schema declares directly on a partition. The key on the
        partitioned parent itself is kept. ``has_table_privilege`` on both ends
        matches :meth:`get_tables`: a key into a table the role cannot read
        leads nowhere it can query.

        Each row is one column pair; a composite key is several rows with no
        constraint name, so it is only as composite downstream as the catalog
        makes it.
        """
        return self.execute("""
            SELECT
                n.nspname   AS table_schema,
                c.relname   AS table_name,
                a.attname   AS column_name,
                rn.nspname  AS referenced_schema,
                rc.relname  AS referenced_table,
                ra.attname  AS referenced_column
            FROM pg_constraint con
            CROSS JOIN LATERAL unnest(con.conkey, con.confkey)
                WITH ORDINALITY AS k(attnum, ref_attnum, ord)
            JOIN pg_class c      ON c.oid  = con.conrelid
            JOIN pg_namespace n  ON n.oid  = c.relnamespace
            JOIN pg_attribute a  ON a.attrelid = c.oid AND a.attnum = k.attnum
            JOIN pg_class rc     ON rc.oid = con.confrelid
            JOIN pg_namespace rn ON rn.oid = rc.relnamespace
            JOIN pg_attribute ra ON ra.attrelid = rc.oid AND ra.attnum = k.ref_attnum
            WHERE con.contype = 'f'
              AND c.relispartition = false
              AND rc.relispartition = false
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
              AND n.nspname NOT LIKE 'pg\\_%'
              AND has_table_privilege(c.oid, 'SELECT')
              AND has_table_privilege(rc.oid, 'SELECT')
            ORDER BY n.nspname, c.relname, con.conname, k.ord
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
