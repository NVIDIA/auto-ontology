# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle Database connector implementing Auto Ontology's SQLDatabase ABC.

Uses ``python-oracledb`` in thin mode, so no Oracle Instant Client is needed.
Thin mode talks to Oracle Database 12.1 and later, which is also the first
release with the ``FETCH FIRST n ROWS ONLY`` row-limiting clause this
connector rewrites ``LIMIT`` into.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import oracledb
import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

logger = logging.getLogger(__name__)

DEFAULT_PORT = 1521

# The driver giving up on a round trip at ``call_timeout``: DPY-4024 in thin
# mode, ORA-03156 (and the older ORA-03136) in thick mode.
_TIMEOUT_CODES = frozenset({"DPY-4024", "ORA-03156", "ORA-03136"})

# What Oracle answers to syntax it stops understanding part-way through a
# statement -- ``LIMIT``, or ``AS`` before a table alias. Up to 21c that is
# ORA-00933 "SQL command not properly ended", or ORA-00907 "missing right
# parenthesis" inside a subquery; 23ai names the offending word instead, as
# ORA-03048 "SQL reserved word ... is not syntactically valid".
_REWRITABLE_SYNTAX_CODES = frozenset({"ORA-00933", "ORA-00907", "ORA-03048"})

_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)

_TABLE_COLUMNS = ["table_schema", "table_name", "table_type"]
_COLUMN_COLUMNS = [
    "table_schema",
    "table_name",
    "column_name",
    "data_type",
    "is_nullable",
    "ordinal_position",
]
_VIEW_COLUMNS = ["table_schema", "table_name", "view_definition"]
_PK_COLUMNS = ["table_schema", "table_name", "column_name", "ordinal_position"]
_FK_COLUMNS = [
    "table_schema",
    "table_name",
    "column_name",
    "referenced_schema",
    "referenced_table",
    "referenced_column",
]


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Convert an Oracle URI into ``oracledb.connect`` keyword arguments.

    The path is the **service name**, not a SID: it is what a pluggable
    database is addressed by, and the only form Oracle Cloud accepts.
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "oracle":
        raise ValueError("Invalid Oracle connection string: scheme must be oracle://")
    if not parsed.hostname:
        raise ValueError("Oracle connection string is missing a host")
    service_name = unquote(parsed.path.lstrip("/"))
    if not service_name:
        raise ValueError("Oracle connection string is missing a service name")

    return {
        "host": parsed.hostname,
        "port": parsed.port or DEFAULT_PORT,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "service_name": service_name,
    }


def _quoted_literal(value: str) -> str:
    """Return *value* as a single-quoted SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def _error_code(exc: BaseException) -> str:
    """Return the ``ORA-``/``DPY-`` code of a driver error, or ``''``."""
    detail = exc.args[0] if exc.args else None
    return str(getattr(detail, "full_code", "") or "")


def to_oracle_sql(sql: str) -> str:
    """Rewrite generic SQL into syntax Oracle accepts.

    The profiling and probe queries the core issues are written once for every
    engine, with ``LIMIT n`` and ``(subquery) AS alias``. Oracle has neither:
    it spells the first ``FETCH FIRST n ROWS ONLY`` and rejects ``AS`` before a
    table alias. sqlglot's Oracle generator emits both correctly.

    SQL that sqlglot cannot parse is returned unchanged, so Oracle reports its
    own error for it rather than this function inventing one.
    """
    try:
        statements = sqlglot.transpile(sql, read="oracle", write="oracle")
    except SqlglotError:
        return sql
    if len(statements) != 1 or not statements[0]:
        return sql
    return statements[0]


def drop_database_qualifier(sql: str, database_name: str) -> str:
    """Rewrite ``database.schema.table`` references to ``schema.table``.

    Everywhere else a table is known by three names, and SQL written against
    the catalog carries all three. Oracle has no such level: a session is
    already inside one database, and ``FREEPDB1.HR.EMPLOYEES`` is read as a
    column reference and rejected. Only a qualifier equal to *database_name*
    is removed, so anything else still fails loudly.

    SQL that sqlglot cannot parse is returned unchanged.
    """
    try:
        tree = sqlglot.parse_one(sql, read="oracle")
    except SqlglotError:
        return sql
    target = database_name.casefold()
    changed = False
    for node in tree.find_all(exp.Table, exp.Column):
        catalog = node.args.get("catalog")
        if catalog is not None and catalog.name.casefold() == target:
            node.set("catalog", None)
            changed = True
    if not changed:
        return sql
    return tree.sql(dialect="oracle")


def _output_type_handler(cursor: Any, metadata: Any) -> Any:
    """Fetch LOB columns as plain ``str``/``bytes`` instead of LOB locators.

    A locator is only readable while its connection is open, and every call
    here closes its connection before the DataFrame is returned.
    """
    if metadata.type_code is oracledb.DB_TYPE_CLOB:
        return cursor.var(oracledb.DB_TYPE_LONG, arraysize=cursor.arraysize)
    if metadata.type_code is oracledb.DB_TYPE_NCLOB:
        return cursor.var(oracledb.DB_TYPE_LONG_NVARCHAR, arraysize=cursor.arraysize)
    if metadata.type_code is oracledb.DB_TYPE_BLOB:
        return cursor.var(oracledb.DB_TYPE_LONG_RAW, arraysize=cursor.arraysize)
    return None


class OracleDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by ``python-oracledb`` (thin mode).

    Parameters
    ----------
    connection_string:
        ``oracle://user:password@host:1521/service_name``
    schemas:
        Optional ingestion allowlist. Empty or ``None`` means only the schema
        of the connecting user -- an Oracle account can usually *see* far more
        schemas than it is meant to model, so "everything visible" is not a
        usable default the way it is on other engines.
    """

    supports_statement_timeout = True

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        self._connect_kwargs = _parse_connection_string(connection_string)
        self._database_name = str(self._connect_kwargs["service_name"])
        # Cheap test for whether a statement names the database at all, so the
        # common case skips parsing. See ``drop_database_qualifier``.
        self._database_qualifier_re = re.compile(
            rf'(?<![\w$#"]){re.escape(self._database_name)}"?\s*\.',
            re.IGNORECASE,
        )
        # Compared upper-cased: Oracle folds unquoted identifiers to upper case,
        # so a user typing "hr" must still match the schema named "HR".
        self._schema_names: list[str] = sorted(
            {s.strip().upper() for s in (schemas or []) if s and s.strip()}
        )
        if self._schema_names:
            logger.info(
                "Oracle ingestion restricted to schemas: %s", self._schema_names
            )
        self.ping()

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "oracle"

    @property
    def database_name(self) -> str:
        return self._database_name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _connect(self) -> Any:
        connection = oracledb.connect(**self._connect_kwargs)
        connection.outputtypehandler = _output_type_handler
        return connection

    def _run(
        self,
        connection: Any,
        sql: str,
        parameters: Optional[list],
    ) -> pd.DataFrame:
        cursor = connection.cursor()
        try:
            cursor.execute(sql, parameters or [])
            if cursor.description is None:
                return pd.DataFrame()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)
        finally:
            cursor.close()

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Run *sql*; with *timeout_s*, the driver abandons it after that long.

        A ``database.schema.table`` reference loses its database part (see
        :func:`drop_database_qualifier`), and a statement containing ``LIMIT``
        is rewritten up front (see
        :func:`to_oracle_sql`). Any other statement is sent as written, and
        rewritten only if Oracle rejects its syntax (see
        ``_REWRITABLE_SYNTAX_CODES``) -- valid Oracle SQL is therefore never put
        through a second parser it does not need.

        Positional *parameters* bind to ``:1``, ``:2``, ...

        Raises:
            StatementTimeout: the statement outlived *timeout_s*.
        """
        # Oracle rejects a trailing semicolon on a single statement.
        statement = sql.strip().rstrip(";").strip()
        if self._database_qualifier_re.search(statement):
            statement = drop_database_qualifier(statement, self._database_name)
        if _LIMIT_RE.search(statement):
            statement = to_oracle_sql(statement)

        connection = self._connect()
        started = time.monotonic()
        try:
            if timeout_s is not None:
                connection.call_timeout = max(1, int(timeout_s * 1000))
            try:
                return self._run(connection, statement, parameters)
            except oracledb.Error as exc:
                if _error_code(exc) not in _REWRITABLE_SYNTAX_CODES:
                    raise
                rewritten = to_oracle_sql(statement)
                if rewritten == statement:
                    raise
                return self._run(connection, rewritten, parameters)
        except oracledb.Error as exc:
            if timeout_s is not None and _error_code(exc) in _TIMEOUT_CODES:
                raise StatementTimeout.since(timeout_s, started) from exc
            raise
        finally:
            try:
                connection.close()
            except oracledb.Error:
                # A call timeout leaves the connection already dead.
                logger.debug("Oracle connection was already closed", exc_info=True)

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def _owner_condition(self, column: str = "OWNER") -> str:
        """The ``WHERE`` condition selecting which schemas to introspect."""
        if not self._schema_names:
            return f"{column} = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')"
        names = ", ".join(_quoted_literal(name) for name in self._schema_names)
        return f"UPPER({column}) IN ({names})"

    def get_schemas(self) -> list[str]:
        """List the schemas the connecting user could ingest.

        Used by the connection UI to let the user pick which schemas to ingest,
        and by the connection test — it runs a real query, so it proves the
        endpoint, the credentials and the service name all work.

        Schemas Oracle itself maintains (``SYS``, ``SYSTEM``, ``XDB``, ...) are
        left out; they hold the data dictionary, not anything to model.
        """
        frame = self.execute(
            """
            SELECT USERNAME
            FROM ALL_USERS
            WHERE ORACLE_MAINTAINED = 'N'
            ORDER BY USERNAME
            """
        )
        if frame.empty:
            return []
        return [str(name) for name in frame.iloc[:, 0].tolist()]

    def get_tables(self) -> pd.DataFrame:
        # ALL_TABLES also lists the container table behind each materialized
        # view, under the view's own name; those rows are reported once, as the
        # materialized view. Nested-table storage, index-organized overflow
        # segments and the recycle bin are storage details, not relations.
        frame = self.execute(
            f"""
            SELECT
                t.OWNER AS "table_schema",
                t.TABLE_NAME AS "table_name",
                '{TableTypes.BASE_TABLE}' AS "table_type"
            FROM ALL_TABLES t
            WHERE {self._owner_condition("t.OWNER")}
              AND t.NESTED = 'NO'
              AND t.SECONDARY = 'N'
              AND t.DROPPED = 'NO'
              AND t.IOT_TYPE IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM ALL_MVIEWS m
                  WHERE m.OWNER = t.OWNER AND m.MVIEW_NAME = t.TABLE_NAME
              )
            UNION ALL
            SELECT OWNER, VIEW_NAME, '{TableTypes.VIEW}'
            FROM ALL_VIEWS
            WHERE {self._owner_condition()}
            UNION ALL
            SELECT OWNER, MVIEW_NAME, '{TableTypes.MATERIALIZED_VIEW}'
            FROM ALL_MVIEWS
            WHERE {self._owner_condition()}
            ORDER BY 1, 2
            """
        )
        if frame.empty:
            return pd.DataFrame(columns=_TABLE_COLUMNS)
        return frame

    def get_columns(self) -> pd.DataFrame:
        # NULLABLE is 'Y'/'N' text; both are truthy, so it is converted to a
        # number in SQL and to a real boolean below.
        frame = self.execute(
            f"""
            SELECT
                OWNER AS "table_schema",
                TABLE_NAME AS "table_name",
                COLUMN_NAME AS "column_name",
                DATA_TYPE AS "data_type",
                CASE WHEN NULLABLE = 'Y' THEN 1 ELSE 0 END AS "is_nullable",
                COLUMN_ID AS "ordinal_position"
            FROM ALL_TAB_COLUMNS
            WHERE {self._owner_condition()}
              AND TABLE_NAME NOT LIKE 'BIN$%'
            ORDER BY OWNER, TABLE_NAME, COLUMN_ID
            """
        )
        if frame.empty:
            return pd.DataFrame(columns=_COLUMN_COLUMNS)
        frame["is_nullable"] = frame["is_nullable"].astype(int).astype(bool)
        return frame

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        # Query history lives in V$SQL, which needs a grant ordinary accounts
        # do not have. It is not needed for schema ingestion.
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        # Two statements, not a UNION: both definitions are LONG columns, and
        # Oracle does not allow a LONG in a set operation.
        views = self.execute(
            f"""
            SELECT
                OWNER AS "table_schema",
                VIEW_NAME AS "table_name",
                TEXT AS "view_definition"
            FROM ALL_VIEWS
            WHERE {self._owner_condition()}
            ORDER BY OWNER, VIEW_NAME
            """
        )
        materialized = self.execute(
            f"""
            SELECT
                OWNER AS "table_schema",
                MVIEW_NAME AS "table_name",
                QUERY AS "view_definition"
            FROM ALL_MVIEWS
            WHERE {self._owner_condition()}
            ORDER BY OWNER, MVIEW_NAME
            """
        )
        frames = [frame for frame in (views, materialized) if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=_VIEW_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def get_pks(self) -> pd.DataFrame:
        frame = self.execute(
            f"""
            SELECT
                c.OWNER AS "table_schema",
                c.TABLE_NAME AS "table_name",
                cc.COLUMN_NAME AS "column_name",
                cc.POSITION AS "ordinal_position"
            FROM ALL_CONSTRAINTS c
            JOIN ALL_CONS_COLUMNS cc
              ON cc.OWNER = c.OWNER
             AND cc.CONSTRAINT_NAME = c.CONSTRAINT_NAME
            WHERE c.CONSTRAINT_TYPE = 'P'
              AND {self._owner_condition("c.OWNER")}
              AND c.TABLE_NAME NOT LIKE 'BIN$%'
            ORDER BY c.OWNER, c.TABLE_NAME, cc.POSITION
            """
        )
        if frame.empty:
            return pd.DataFrame(columns=_PK_COLUMNS)
        return frame

    def get_fks(self) -> pd.DataFrame:
        frame = self.execute(
            f"""
            SELECT
                c.OWNER AS "table_schema",
                c.TABLE_NAME AS "table_name",
                cc.COLUMN_NAME AS "column_name",
                rc.OWNER AS "referenced_schema",
                rc.TABLE_NAME AS "referenced_table",
                rc.COLUMN_NAME AS "referenced_column"
            FROM ALL_CONSTRAINTS c
            JOIN ALL_CONS_COLUMNS cc
              ON cc.OWNER = c.OWNER
             AND cc.CONSTRAINT_NAME = c.CONSTRAINT_NAME
            JOIN ALL_CONS_COLUMNS rc
              ON rc.OWNER = c.R_OWNER
             AND rc.CONSTRAINT_NAME = c.R_CONSTRAINT_NAME
             AND rc.POSITION = cc.POSITION
            WHERE c.CONSTRAINT_TYPE = 'R'
              AND {self._owner_condition("c.OWNER")}
              AND c.TABLE_NAME NOT LIKE 'BIN$%'
            ORDER BY c.OWNER, c.TABLE_NAME, cc.POSITION
            """
        )
        if frame.empty:
            return pd.DataFrame(columns=_FK_COLUMNS)
        return frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        connection = self._connect()
        try:
            connection.ping()
        finally:
            connection.close()

    def close(self) -> None:
        """No-op because each operation owns and closes its connection."""
