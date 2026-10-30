# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB connector for in-process SQL execution.

Wraps ``duckdb.connect()`` with helpers to register pandas DataFrames or
scan CSV/Parquet/JSON files directly from the filesystem.  No server or Docker
service is required — DuckDB runs fully in-process.

This is the reference implementation of
:class:`~auto_ontology.connectors.base.SQLDatabase`.

Example
-------
::

    from duckdb_connector import DuckDB  # run from tabular-dev-tools/

    conn = DuckDB("./spider2.duckdb")
    rows = conn.execute("SELECT * FROM Airlines.flights LIMIT 5")
    # rows -> [{"flight_id": 1, ...}]
"""

from __future__ import annotations

import itertools
import logging
import threading
from datetime import datetime
from pathlib import Path
import duckdb
import pandas as pd
from typing import Optional

from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

logger = logging.getLogger(__name__)

_EPHEMERAL_COUNTER = itertools.count()


def _execute(
    conn: duckdb.DuckDBPyConnection, sql: str, parameters: Optional[list]
) -> pd.DataFrame:
    if parameters:
        rel = conn.execute(sql, parameters)
    else:
        rel = conn.execute(sql)
    return rel.df()


class DuckDBDatabase(SQLDatabase):
    """In-process DuckDB connection with convenience helpers.

    Parameters
    ----------
    database:
        Path to a persistent DuckDB database file, or ``None`` / ``":memory:"``
        for an ephemeral in-memory database (default: in-memory).
    read_only:
        Open the database in read-only mode (default: True).  Multiple
        processes can hold a read-only connection simultaneously; set to
        ``False`` only when you need to write to the file.
    """

    supports_statement_timeout = True

    def __init__(self, connection_string: str, *, read_only: bool = True) -> None:
        db_path = connection_string
        if db_path.startswith("duckdb://"):
            db_path = db_path[len("duckdb://") :]
        self.conn = duckdb.connect(database=db_path, read_only=read_only)
        self._connection_string: str = (
            f":memory:{next(_EPHEMERAL_COUNTER)}"
            if db_path in ("", ":memory:")
            else str(Path(db_path).resolve())
        )
        self._database_name: str = self.execute("SELECT current_database()").iloc[0, 0]
        logger.debug(
            "DuckDB connected (database=%r, read_only=%s).",
            self._database_name,
            read_only,
        )

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "duckdb"

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
        """Execute a SQL statement and return a pandas DataFrame.

        Parameters
        ----------
        sql:
            SQL query to execute.
        parameters:
            Optional positional parameters.
        timeout_s:
            Interrupt the statement after this many seconds and raise
            :class:`StatementTimeout`. DuckDB has no server-side cap, so a
            timer thread calls ``interrupt()``.
        """
        logger.debug("DuckDB executing (→ DataFrame): %s", sql[:200])
        if timeout_s is None:
            return _execute(self.conn, sql, parameters)

        # ``interrupt()`` stops whatever is running on the connection it is
        # called on, and profiling runs tables in parallel. A cursor of its own
        # (same database, separate client context) means the timer can only
        # ever stop this statement.
        cursor = self.conn.cursor()
        fired = threading.Event()

        def _interrupt() -> None:
            fired.set()
            cursor.interrupt()

        timer = threading.Timer(timeout_s, _interrupt)
        timer.daemon = True
        timer.start()
        try:
            return _execute(cursor, sql, parameters)
        except duckdb.InterruptException as exc:
            if fired.is_set():
                raise StatementTimeout(timeout_s) from exc
            raise
        finally:
            timer.cancel()
            timer.join()
            cursor.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_tables(self) -> pd.DataFrame:
        """Return all tables from information_schema as a DataFrame."""
        return self.execute(
            """
            SELECT
                table_schema,
                table_name
            FROM information_schema.tables
            ORDER BY table_schema, table_name
        """
        )

    def get_columns(self) -> pd.DataFrame:
        """Return all columns from information_schema as a DataFrame."""
        return self.execute(
            """
            SELECT
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable = 'YES' AS is_nullable,
                ordinal_position
            FROM information_schema.columns
            ORDER BY table_schema, table_name, ordinal_position
        """
        )

    def get_queries(self) -> pd.DataFrame:
        """DuckDB has no built-in query history — loads sample queries from a CSV."""
        csv_path = (
            Path(__file__).parent
            / "benchmarks"
            / self._database_name
            / "sample_queries.csv"
        )
        if not csv_path.exists():
            logger.warning(
                "No sample queries CSV found at %s; returning empty DataFrame.",
                csv_path,
            )
            return pd.DataFrame(columns=["query_text", "end_time"])
        df = pd.read_csv(csv_path)
        df["end_time"] = datetime.today()
        return df

    def get_views(self) -> pd.DataFrame:
        """Return all views from information_schema."""
        return self.execute(
            """
            SELECT
                table_schema,
                table_name,
                view_definition
            FROM information_schema.views
            ORDER BY table_catalog, table_schema, table_name
        """
        )

    def get_pks(self) -> pd.DataFrame:
        """Return declared primary-key columns, one row per key column.

        ``duckdb_constraints()`` lists a key's columns in declared order, so
        ``ordinal_position`` is each column's place in the key -- not in the
        table -- and a composite key declared ``(line_no, order_id)`` keeps
        that order. Scoped to the connected database: an attached catalog's
        keys would otherwise land on same-named tables in this one.
        """
        return self.execute(
            """
            SELECT
                schema_name AS table_schema,
                table_name,
                UNNEST(constraint_column_names) AS column_name,
                UNNEST(range(1, len(constraint_column_names) + 1)) AS ordinal_position
            FROM duckdb_constraints()
            WHERE constraint_type = 'PRIMARY KEY'
              AND database_name = current_database()
            ORDER BY table_schema, table_name, ordinal_position
        """
        )

    def get_fks(self) -> pd.DataFrame:
        """Return declared foreign-key columns, one row per column pair.

        Two ``UNNEST`` calls in one select list are zipped by position, which
        pairs each column of a composite key with the column it references.
        DuckDB refuses foreign keys across schemas, so the referenced schema
        is always the referencing table's own.
        """
        return self.execute(
            """
            SELECT
                schema_name AS table_schema,
                table_name,
                UNNEST(constraint_column_names) AS column_name,
                schema_name AS referenced_schema,
                referenced_table,
                UNNEST(referenced_column_names) AS referenced_column
            FROM duckdb_constraints()
            WHERE constraint_type = 'FOREIGN KEY'
              AND database_name = current_database()
            ORDER BY table_schema, table_name, column_name
        """
        )

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
