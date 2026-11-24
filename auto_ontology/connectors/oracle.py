# SPDX-FileCopyrightText: Copyright (c) 2026 Mario Nascimento
# SPDX-License-Identifier: Apache-2.0

"""Oracle Database connector implementing Auto Ontology's SQLDatabase ABC.

Uses ``python-oracledb`` in thin mode, so no Oracle Instant Client is needed.
Thin mode talks to Oracle Database 12.1 and later, which is also the first
release with ``FETCH FIRST n ROWS ONLY``, the row limit sqlglot's Oracle
dialect writes for the core's profiling queries
(:func:`auto_ontology.utils.sql_limits.with_row_limit`).

Statements run as written. Callers write Oracle SQL because :attr:`dialect`
says to, and SQL the agent writes reaches Oracle as the user sees it, so a
mistake comes back as Oracle's own error rather than as a silent rewrite.
"""

from __future__ import annotations

import base64
import binascii
import logging
import math
import ssl
import time
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import oracledb
import pandas as pd

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

logger = logging.getLogger(__name__)

DEFAULT_PORT = 1521

# Query-string options the connection URL accepts; see _parse_connection_string.
_CONNECTION_OPTIONS = frozenset(
    {"protocol", "ssl_ca", "ssl_ca_data", "ssl_server_cert_dn"}
)

# The driver giving up on a round trip at ``call_timeout``: DPY-4024 in thin
# mode, ORA-03156 (and the older ORA-03136) in thick mode.
_TIMEOUT_CODES = frozenset({"DPY-4024", "ORA-03156", "ORA-03136"})

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

    ``?protocol=tcps`` connects over TLS and verifies the server: its
    certificate must chain to a trusted CA, and must name the host, or match
    ``ssl_server_cert_dn`` when that is given. The CA is the system's trust
    store unless ``ssl_ca`` (a PEM file path) or ``ssl_ca_data`` (the
    base64-encoded PEM or DER bytes, as the connection form uploads them)
    names one. There is no option to skip verification, and a TCPS connection
    that fails verification fails; it never falls back to TCP.

    An option this function does not know is an error rather than ignored, so
    a misspelt ``protocol`` cannot leave the connection unencrypted.
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "oracle":
        raise ValueError("Invalid Oracle connection string: scheme must be oracle://")
    if not parsed.hostname:
        raise ValueError("Oracle connection string is missing a host")
    service_name = unquote(parsed.path.lstrip("/"))
    if not service_name:
        raise ValueError("Oracle connection string is missing a service name")

    kwargs: dict[str, Any] = {
        "host": parsed.hostname,
        "port": parsed.port or DEFAULT_PORT,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "service_name": service_name,
    }
    kwargs.update(_tls_kwargs(parse_qs(parsed.query, keep_blank_values=True)))
    return kwargs


def _tls_kwargs(query: dict[str, list[str]]) -> dict[str, Any]:
    """The ``oracledb.connect`` arguments for the URL's TLS options."""
    unknown = sorted(set(query) - _CONNECTION_OPTIONS)
    if unknown:
        raise ValueError(
            f"Unsupported Oracle connection option(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(sorted(_CONNECTION_OPTIONS))}"
        )
    repeated = sorted(key for key, values in query.items() if len(values) > 1)
    if repeated:
        raise ValueError(f"Oracle connection option given twice: {', '.join(repeated)}")
    options = {key: values[0].strip() for key, values in query.items()}

    protocol = (options.pop("protocol", "") or "tcp").lower()
    if protocol not in ("tcp", "tcps"):
        raise ValueError(
            f"Unsupported Oracle protocol {protocol!r}: use tcp, or tcps for TLS"
        )
    tls_options = sorted(key for key, value in options.items() if value)
    if protocol == "tcp":
        if tls_options:
            raise ValueError(
                f"{', '.join(tls_options)} only apply to TLS connections: "
                "add protocol=tcps"
            )
        return {}
    if options.get("ssl_ca") and options.get("ssl_ca_data"):
        raise ValueError("Give the Oracle CA as ssl_ca or ssl_ca_data, not both")

    context = ssl.create_default_context()
    try:
        if options.get("ssl_ca"):
            context = ssl.create_default_context(cafile=options["ssl_ca"])
        elif options.get("ssl_ca_data"):
            context = ssl.create_default_context(
                cadata=_decode_ca_data(options["ssl_ca_data"])
            )
    except (OSError, ssl.SSLError) as exc:
        raise ValueError(f"Oracle CA certificate could not be loaded: {exc}") from exc

    kwargs: dict[str, Any] = {
        "protocol": "tcps",
        "ssl_context": context,
        "ssl_server_dn_match": True,
    }
    if options.get("ssl_server_cert_dn"):
        # The driver matches the certificate's subject against this DN instead
        # of the host name, so the context's own host name check would reject
        # exactly the servers this option exists for. The chain is still
        # verified (CERT_REQUIRED stays on).
        context.check_hostname = False
        kwargs["ssl_server_cert_dn"] = options["ssl_server_cert_dn"]
    return kwargs


def _decode_ca_data(encoded: str) -> str | bytes:
    """Decode ``ssl_ca_data``: base64 of a PEM bundle or a DER certificate."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Oracle ssl_ca_data is not valid base64") from exc
    if not raw:
        raise ValueError("Oracle ssl_ca_data is empty")
    # ``cadata`` takes PEM as text and DER as bytes.
    if raw.lstrip().startswith(b"-----BEGIN"):
        return raw.decode("ascii", errors="strict")
    return raw


def _quoted_literal(value: str) -> str:
    """Return *value* as a single-quoted SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


# Oracle allows at most 1,000 expressions in one IN list (ORA-01795).
_MAX_IN_LIST = 1000


def _in_list(column: str, values: list[str]) -> str:
    """``column IN (...)`` for *values*, split into lists Oracle accepts.

    Each list holds at most :data:`_MAX_IN_LIST` literals, and the lists are
    joined with ``OR``, so an allowlist of any size stays one condition.
    """
    groups = [
        ", ".join(_quoted_literal(value) for value in values[i : i + _MAX_IN_LIST])
        for i in range(0, len(values), _MAX_IN_LIST)
    ]
    if len(groups) == 1:
        return f"{column} IN ({groups[0]})"
    return "(" + " OR ".join(f"{column} IN ({group})" for group in groups) + ")"


def _error_code(exc: BaseException) -> str:
    """Return the ``ORA-``/``DPY-`` code of a driver error, or ``''``."""
    detail = exc.args[0] if exc.args else None
    return str(getattr(detail, "full_code", "") or "")


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
        ``oracle://user:password@host:1521/service_name``, or with
        ``?protocol=tcps`` for TLS (see :func:`_parse_connection_string`).
    schemas:
        Optional ingestion allowlist. Empty or ``None`` means only the schema
        of the connecting user -- an Oracle account can usually *see* far more
        schemas than it is meant to model, so "everything visible" is not a
        usable default the way it is on other engines.
    """

    supports_statement_timeout = True

    # ``TEXT`` is ``CLOB`` in Oracle, and a CLOB cannot be compared with ``=``
    # (ORA-22849 on 23ai, ORA-00932 on 19c), so the semantic FK join check's
    # cast-to-text retry could never succeed. 4000 is the VARCHAR2 ceiling unless the database runs with
    # MAX_STRING_SIZE=EXTENDED; a longer value fails the retry, which leaves
    # that join unconfirmed, as before.
    text_cast_type = "VARCHAR2(4000)"

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        self._connect_kwargs = _parse_connection_string(connection_string)
        self._database_name = str(self._connect_kwargs["service_name"])
        # As given; matched to real schema names on first use, by
        # ``_resolved_schema_names``.
        self._schema_names: list[str] = sorted(
            {s.strip() for s in (schemas or []) if s and s.strip()}
        )
        self._resolved_schemas: list[str] | None = None
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

    @staticmethod
    def _spend(connection: Any, timeout_s: float | None, started: float) -> None:
        """Give the next round trip whatever is left of the statement's budget.

        ``call_timeout`` bounds one round trip, and a statement takes one to
        execute plus one per batch of rows, so each gets only the time the
        earlier ones left over.

        Raises:
            StatementTimeout: the budget is already spent.
        """
        if timeout_s is None:
            return
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise StatementTimeout.since(timeout_s, started)
        # Never 0: that would switch the driver's timeout off.
        connection.call_timeout = max(1, math.ceil(remaining * 1000))

    def _run(
        self,
        connection: Any,
        sql: str,
        parameters: Optional[list],
        timeout_s: float | None,
        started: float,
    ) -> pd.DataFrame:
        cursor = connection.cursor()
        try:
            self._spend(connection, timeout_s, started)
            cursor.execute(sql, parameters or [])
            if cursor.description is None:
                return pd.DataFrame()
            columns = [description[0] for description in cursor.description]
            rows: list[Any] = []
            while True:
                self._spend(connection, timeout_s, started)
                batch = cursor.fetchmany()
                if not batch:
                    break
                rows.extend(batch)
            return pd.DataFrame(rows, columns=columns)
        finally:
            cursor.close()

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Run *sql*; with *timeout_s*, it is abandoned after that long.

        Positional *parameters* bind to ``:1``, ``:2``, ...

        Raises:
            StatementTimeout: executing the statement and fetching its rows
                together outlived *timeout_s*.
        """
        # Oracle rejects a trailing semicolon on a single statement. Dropping it
        # changes nothing about what the statement means.
        statement = sql.strip().rstrip(";").strip()

        connection = self._connect()
        started = time.monotonic()
        try:
            return self._run(connection, statement, parameters, timeout_s, started)
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
        return _in_list(column, self._resolved_schema_names())

    def _resolved_schema_names(self) -> list[str]:
        """The allowlist as the exact schema names Oracle stores.

        Schema names are case-sensitive: ``HR`` and a schema created as
        ``"hr"`` are different schemas, so the filter compares exactly. A name
        picked from ``get_schemas()`` is already exact. A typed name is read the
        way Oracle reads an identifier: ``"hr"`` in double quotes is exact, and
        a bare ``hr`` that is not itself a schema name means ``HR``.
        """
        if self._resolved_schemas is not None:
            return self._resolved_schemas
        quoted: set[str] = set()
        bare: list[str] = []
        for name in self._schema_names:
            if len(name) > 2 and name.startswith('"') and name.endswith('"'):
                quoted.add(name[1:-1])
            else:
                bare.append(name)
        candidates = sorted(quoted | set(bare) | {name.upper() for name in bare})
        frame = self.execute(
            f"SELECT USERNAME FROM ALL_USERS WHERE {_in_list('USERNAME', candidates)}"
        )
        existing = set() if frame.empty else {str(n) for n in frame.iloc[:, 0]}
        resolved = set(quoted)
        for name in bare:
            resolved.add(
                name
                if name in existing or name.upper() not in existing
                else name.upper()
            )
        self._resolved_schemas = sorted(resolved)
        return self._resolved_schemas

    def get_schemas(self) -> list[str]:
        """List the schemas the connecting user could ingest.

        Used by the connection UI to let the user pick which schemas to ingest,
        and by the connection test — it runs a real query, so it proves the
        endpoint, the credentials and the service name all work.

        Schemas Oracle itself maintains (``SYS``, ``SYSTEM``, ``XDB``, ...) are
        left out; they hold the data dictionary, not anything to model. So are
        schemas with no table or view this account can read: in Oracle every
        database user is a schema, and a database that gives each person a
        login has far more users than schemas worth offering.
        """
        frame = self.execute(
            """
            SELECT u.USERNAME
            FROM ALL_USERS u
            WHERE u.ORACLE_MAINTAINED = 'N'
              AND (
                  EXISTS (SELECT 1 FROM ALL_TABLES t WHERE t.OWNER = u.USERNAME)
                  OR EXISTS (SELECT 1 FROM ALL_VIEWS v WHERE v.OWNER = u.USERNAME)
                  OR EXISTS (SELECT 1 FROM ALL_MVIEWS m WHERE m.OWNER = u.USERNAME)
              )
            ORDER BY u.USERNAME
            """
        )
        if frame.empty:
            return []
        return [str(name) for name in frame.iloc[:, 0].tolist()]

    def get_tables(self) -> pd.DataFrame:
        # ALL_TABLES also lists the container table behind each materialized
        # view, under the view's own name; those rows are reported once, as the
        # materialized view. Nested-table storage, the overflow and mapping
        # tables behind an index-organized table and the recycle bin are
        # storage details, not relations; the index-organized table itself
        # (IOT_TYPE = 'IOT') is an ordinary table.
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
              AND (t.IOT_TYPE IS NULL OR t.IOT_TYPE = 'IOT')
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
