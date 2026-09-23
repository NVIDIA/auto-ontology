# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for SQL database connectors.

Every connector used by ingestion or the text-to-SQL agent implements this
interface; the concrete ones live alongside this module in
:mod:`auto_ontology.connectors` and are discovered through
:func:`auto_ontology.connectors.registry.get_connectors`.

The contract is expressed entirely in terms of DataFrames with named
columns rather than typed row objects, because the ingestion pipeline feeds
them straight into pandas. Each method's docstring lists the columns the
caller relies on — a connector that omits a required column fails at
ingestion time, not at import, so keep the shapes exact.

Example
-------
::

    from auto_ontology.connectors.base import SQLDatabase

    class MyConnector(SQLDatabase):
        def __init__(self, connection_string: str) -> None:
            ...

        def execute(self, sql, parameters=None):
            ...

        # implement remaining abstract methods ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from auto_ontology.utils.sql_identifiers import qualified_name


class SQLDatabase(ABC):
    """Abstract SQL database connector.

    Subclasses must implement all abstract methods. The context-manager
    protocol (``__enter__`` / ``__exit__``) is provided by this base class
    and delegates to :meth:`close`.

    Parameters
    ----------
    connection_string:
        A driver-specific connection string or database path.
    dialect:
        SQL dialect used by this connector (e.g. ``"duckdb"``, ``"snowflake"``).
        Used by the text-to-SQL agent to emit dialect-appropriate SQL.
    """

    #: Whether :meth:`execute` takes a ``timeout_s`` cap that is free to apply.
    #: Callers running a query that should be cheap — profiling samples — pass
    #: a cap when this is set, so an engine that has stopped answering costs
    #: seconds instead of the driver's own timeout (900s on Kyuubi).
    #:
    #: Having a ``timeout_s`` parameter is not enough to set this. Databricks
    #: has one, but honouring it there means opening a fresh session per
    #: statement, which during ingestion costs far more than the cap saves.
    supports_statement_timeout: bool = False

    @abstractmethod
    def __init__(self, connection_string: str) -> None: ...

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        """Execute a SQL statement and return the result as a DataFrame.

        Parameters
        ----------
        sql:
            SQL query to execute.
        parameters:
            Optional positional parameters for parameterised queries.
        """

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    @abstractmethod
    def get_tables(self) -> pd.DataFrame:
        """Return all tables.

        Required columns: ``table_schema``, ``table_name``.

        Optional column: ``table_type`` — stored on the catalog table.
        Canonical values are defined in
        :class:`auto_ontology.catalog.constants.TableTypes` (``view``,
        ``materialized view``, ``base table``). Connectors that do not
        provide it default to ``base table`` during ingestion.
        """

    @abstractmethod
    def get_columns(self) -> pd.DataFrame:
        """Return all columns.

        Expected columns: ``table_schema``, ``table_name``,
        ``column_name``, ``data_type``, ``is_nullable``.

        ``is_nullable`` must be a **boolean**, not the ``'YES'``/``'NO'``
        text that ``information_schema`` reports — convert at the query
        (``is_nullable = 'YES'``) or in the driver loop. Both strings are
        truthy, so a connector that leaks them makes every column read as
        nullable. A missing or null value means "could not determine", which
        the catalog stores as NULL and callers treat as nullable.
        """

    @abstractmethod
    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Return recent / historical queries if the backend supports it.

        Expected columns: ``end_time``, ``query_text``.
        Connectors without query history should return an empty DataFrame
        with those two columns.

        Parameters
        ----------
        hours:
            Only return queries whose ``end_time`` falls within the last
            ``hours`` hours. Defaults to 24.
        """

    @abstractmethod
    def get_views(self) -> pd.DataFrame:
        """Return all views.

        Expected columns: ``table_schema``, ``table_name``,
        ``view_definition``.
        """

    @abstractmethod
    def get_pks(self) -> pd.DataFrame:
        """Return primary key columns.

        Expected columns: ``table_schema``, ``table_name``,
        ``column_name``, ``ordinal_position``.
        """

    @abstractmethod
    def get_fks(self) -> pd.DataFrame:
        """Return foreign key columns.

        Expected columns: ``table_schema``, ``table_name``,
        ``column_name``, ``referenced_schema``, ``referenced_table``,
        ``referenced_column``.
        """

    @property
    @abstractmethod
    def dialect(self) -> str:
        """Return the dialect of the database.

        Returns:
            A sqlglot-compatible dialect name for this engine
            (e.g. ``"duckdb"``, ``"snowflake"``).
        """

    @property
    @abstractmethod
    def database_name(self) -> str:
        """Return the name of the connected database.

        Returns:
            The database name as reported by the backend (e.g. ``"mydb"``).
        """

    def qualify(self, schema: Optional[str], table: str) -> str:
        """Return a quoted, fully-qualified reference to *table*.

        Deliberately concrete rather than abstract: the two-level
        ``schema.table`` default is right for every engine whose connection
        binds a database, so connectors only override when it is not.

        Engines with a three-level namespace (``catalog.schema.table`` --
        Kyuubi, Trino, Databricks) MUST override this to prepend the bound
        catalog. Without it a bare ``schema.table`` resolves against the
        session's default catalog, where the tables do not exist, and every
        profiling probe fails with TABLE_OR_VIEW_NOT_FOUND -- silently, since
        the caller treats a failed probe as "no profile available".

        Do not reach for :attr:`database_name` to build this at the call site:
        it means the catalog on those three connectors but the database on the
        rest, where ``mydb.public.orders`` is invalid. Qualification is the
        connector's rule, not the caller's.

        Args:
            schema: Owning schema, or ``None`` for an unqualified table.
            table: Table name.

        Returns:
            A reference safe to interpolate into SQL for this dialect.
        """
        return qualified_name(schema, table, dialect=self.dialect)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this connector."""

    def __enter__(self) -> "SQLDatabase":
        return self

    def __exit__(self, *args) -> None:
        self.close()
