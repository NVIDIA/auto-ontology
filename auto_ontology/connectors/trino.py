# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trino connector implementing Auto Ontology's SQLDatabase ABC.

Trino is a distributed query engine that federates other systems -- Hive, Iceberg,
PostgreSQL, Kafka -- behind one ANSI SQL surface. It speaks HTTP rather than a
wire protocol, so it is reached with the official ``trino`` Python client
(``trino.dbapi``) instead of a socket driver.

Three properties of the engine shape this connector:

* **Three-level naming.** Objects are ``catalog.schema.table``. A connection
  binds one catalog, which becomes :attr:`~TrinoDatabase.database_name` -- NeMo
  routes SQL by that name, so one connection per catalog is the unit here, the
  same model :mod:`auto_ontology.connectors.kyuubi` uses.
* **A real ``information_schema``.** Unlike Spark/Kyuubi, every Trino catalog
  exposes ``information_schema.{schemata,tables,columns,views}``, so metadata is
  read with ordinary queries rather than a ``DESCRIBE`` per table.
* **No constraints.** Trino declares neither primary nor foreign keys -- the
  underlying connectors do not surface them. :meth:`TrinoDatabase.get_pks` and
  :meth:`TrinoDatabase.get_fks` therefore return correctly shaped *empty*
  frames rather than raising, so ingestion just records no keys.

Authentication is optional. A development cluster typically runs unauthenticated
over plain HTTP, where the username is only an identity label for the query
history; a hardened cluster uses HTTP basic auth over TLS. The transport is
picked from the URL:

``http_scheme=...``
    Explicit, and always wins.
``https``
    Chosen when a password is supplied (the Trino client refuses to send basic
    credentials over cleartext HTTP) or the port is 443 / 8443.
``http``
    Everything else, including the default port 8080.

Example
-------
::

    CONNECTION_STRINGS=trino://analyst@trino.example.com:8080/hive?schema=default
    CONNECTION_STRINGS=trino://analyst:SECRET@trino.example.com:443/hive
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

if TYPE_CHECKING:
    from trino.dbapi import Connection

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8080
DEFAULT_SOURCE = "auto-ontology"

# Ports that are TLS by convention. 8080 (the Trino default) and everything else
# is assumed to be cleartext unless the URL says otherwise.
_TLS_PORTS = frozenset({443, 8443})

# Trino's own metadata schema, present in every catalog and never worth ingesting.
_SYSTEM_SCHEMA = "information_schema"

_TABLE_SCHEMA = ["table_schema", "table_name", "table_type"]

_COLUMN_SCHEMA = [
    "table_schema",
    "table_name",
    "column_name",
    "data_type",
    "is_nullable",
    "ordinal_position",
]

_VIEW_SCHEMA = ["table_schema", "table_name", "view_definition"]

_PK_SCHEMA = ["table_schema", "table_name", "column_name", "ordinal_position"]

_FK_SCHEMA = [
    "table_schema",
    "table_name",
    "column_name",
    "referenced_schema",
    "referenced_table",
    "referenced_column",
]


def _quoted_identifier(name: str) -> str:
    """Return an ANSI double-quoted identifier.

    Trino folds unquoted identifiers to lower case and rejects any that contain a
    space or a reserved word, so every generated statement quotes its identifiers
    -- a catalog named ``my catalog`` or a table named ``order`` is otherwise a
    syntax error rather than a lookup failure.
    """
    return '"' + name.replace('"', '""') + '"'


def _qualified(*parts: str) -> str:
    """Return a double-quoted dotted reference, e.g. ``"hive"."default"."t"``."""
    return ".".join(_quoted_identifier(part) for part in parts)


def _quoted_literal(value: str) -> str:
    """Return a single-quoted SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def _first(query: dict[str, list[str]], key: str) -> str | None:
    """Return the first value for *key*, URL-decoded, or ``None``."""
    value = query.get(key, [None])[0]
    return unquote(value) if value else None


def _is_true(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Parse a Trino URL into the settings this connector needs.

    Expected format::

        trino://USER[:PASSWORD]@HOST[:8080]/CATALOG[?schema=...&http_scheme=...&source=...]

    The password is optional: an unauthenticated cluster takes the username as a
    plain identity label, so ``trino://analyst@host:8080/hive`` is valid.
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "trino":
        raise ValueError(f"Not a Trino URL: {connection_string}")
    if not parsed.hostname:
        raise ValueError(
            f"Invalid Trino connection string (missing host): {connection_string}"
        )

    username = unquote(parsed.username or "")
    if not username:
        raise ValueError(
            "Trino connection string requires a username, e.g. "
            "trino://analyst@trino.example.com:8080/hive"
        )

    # Distinguish "no password" (unauthenticated) from an empty one: urlparse
    # reports ``None`` for ``user@host`` and ``""`` for ``user:@host``, and both
    # mean the client must not attach a BasicAuthentication handler.
    password = unquote(parsed.password) if parsed.password else None

    query = parse_qs(parsed.query)
    catalog = unquote(parsed.path.lstrip("/")) or _first(query, "catalog") or ""
    if not catalog:
        raise ValueError(
            "Trino connection string requires a catalog, e.g. "
            "trino://analyst@trino.example.com:8080/hive"
        )

    port = parsed.port or DEFAULT_PORT

    http_scheme = (_first(query, "http_scheme") or "").strip().lower()
    if http_scheme and http_scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported Trino http_scheme: {http_scheme!r} (use http or https)"
        )
    if not http_scheme:
        # The Trino client refuses to send basic credentials over cleartext, so a
        # password implies TLS; otherwise only the conventional TLS ports do.
        http_scheme = "https" if (password or port in _TLS_PORTS) else "http"

    return {
        "host": parsed.hostname,
        "port": port,
        "catalog": catalog,
        "username": username,
        "password": password,
        "schema": _first(query, "schema"),
        "http_scheme": http_scheme,
        "source": _first(query, "source") or DEFAULT_SOURCE,
        "verify_ssl": _is_true(_first(query, "verify_ssl"), default=True),
    }


def _run(cursor: Any, sql: str, parameters: Optional[list]) -> pd.DataFrame:
    """Execute *sql* on *cursor* and collect the result as a DataFrame."""
    if parameters:
        cursor.execute(sql, parameters)
    else:
        cursor.execute(sql)
    rows = cursor.fetchall()
    if cursor.description is None:
        return pd.DataFrame()
    columns = [str(desc[0]) for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


class TrinoDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by Trino over its HTTP protocol.

    Parameters
    ----------
    connection_string:
        ``trino://user[:password]@host:8080/catalog[?schema=...]``
    schemas:
        Optional ingestion allowlist. Empty or ``None`` means every schema in the
        catalog except ``information_schema``. A ``?schema=`` in the URL seeds
        the same allowlist, so a connection pinned to one schema needs no
        separate filter.
    """

    supports_statement_timeout = True

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        self._settings = _parse_connection_string(connection_string)
        self._catalog = str(self._settings["catalog"])

        allowlist = [s for s in (schemas or []) if s and s.strip()]
        pinned = self._settings["schema"]
        if not allowlist and pinned:
            allowlist = [str(pinned)]
        # Compared case-insensitively: Trino lower-cases unquoted identifiers, so
        # a user typing "Public" must still match the schema named "public".
        self._schema_names: list[str] = sorted({s.strip() for s in allowlist})
        self._schema_filter: set[str] | None = (
            {s.casefold() for s in self._schema_names} if self._schema_names else None
        )
        if self._schema_filter:
            logger.info(
                "Trino ingestion restricted to schemas: %s", sorted(self._schema_filter)
            )

        self._lock = threading.RLock()
        self._connection: "Connection | None" = None

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _open(self) -> "Connection":
        """Open a Trino DBAPI connection.

        The driver is imported lazily so importing this module stays cheap, and
        so a deployment that never configures a Trino connection does not need
        the client installed.
        """
        from trino.auth import BasicAuthentication
        from trino.dbapi import connect

        password = self._settings["password"]
        connection = connect(
            host=str(self._settings["host"]),
            port=int(self._settings["port"]),
            user=str(self._settings["username"]),
            catalog=self._catalog,
            # Bind the pinned schema when there is exactly one, so an unqualified
            # table reference in generated SQL resolves the way the user expects.
            schema=self._schema_names[0] if len(self._schema_names) == 1 else None,
            http_scheme=str(self._settings["http_scheme"]),
            auth=(
                BasicAuthentication(str(self._settings["username"]), str(password))
                if password
                else None
            ),
            source=str(self._settings["source"]),
            verify=bool(self._settings["verify_ssl"]),
        )
        logger.info(
            "Opened Trino connection to %s://%s:%s/%s",
            self._settings["http_scheme"],
            self._settings["host"],
            self._settings["port"],
            self._catalog,
        )
        return connection

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        """Yield a cursor on the shared connection.

        Access is serialised: ``trino.dbapi.Connection`` keeps a single HTTP
        session and mutates per-request state (session properties, the
        transaction id) on it, so concurrent cursors would interleave.
        """
        with self._lock:
            if self._connection is None:
                self._connection = self._open()
            cursor = self._connection.cursor()
            try:
                yield cursor
            finally:
                try:
                    cursor.close()
                except Exception:
                    logger.debug("Failed to close Trino cursor", exc_info=True)

    def _reset_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                logger.debug("Failed to close Trino connection", exc_info=True)
            self._connection = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "trino"

    @property
    def database_name(self) -> str:
        return self._catalog

    def qualify(self, schema: Optional[str], table: str) -> str:
        """Prepend the bound catalog: Trino names are ``catalog.schema.table``.

        A connection binds one catalog, and a bare ``schema.table`` resolves
        against the session default instead -- so the base two-level default
        would point profiling probes at the wrong catalog.

        Raises:
            ValueError: *schema* is missing. A two-part name is read as
                ``schema.table``, so emitting ``catalog.table`` would put the
                catalog in the schema position and quietly resolve elsewhere.
        """
        if not schema:
            raise ValueError(
                f"Trino requires a schema to qualify {table!r} "
                f"in catalog {self._catalog!r}"
            )
        return _qualified(self._catalog, schema, table)

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
        """Run *sql* and return the result as a DataFrame.

        A statement that returns no rows (DDL, ``SET SESSION``) yields an empty
        frame rather than raising.

        With *timeout_s*, the coordinator fails the query once it has run that
        long, queueing included (``query_max_run_time``). The connection is
        shared, so the property is reset afterwards while the lock is still
        held, and no other statement ever runs under it.
        """
        with self._cursor() as cursor:
            if timeout_s is None:
                return _run(cursor, sql, parameters)
            _run(
                cursor,
                f"SET SESSION query_max_run_time = '{max(1, int(timeout_s))}s'",
                None,
            )
            try:
                return _run(cursor, sql, parameters)
            except Exception as exc:
                # ``TrinoQueryError.error_name``; read by attribute so the
                # driver stays a lazy import.
                if getattr(exc, "error_name", None) == "EXCEEDED_TIME_LIMIT":
                    raise StatementTimeout(timeout_s) from exc
                raise
            finally:
                try:
                    _run(cursor, "RESET SESSION query_max_run_time", None)
                except Exception:
                    # Drop the connection rather than let the cap leak onto
                    # ingestion's metadata scans; the next call reopens it.
                    logger.warning(
                        "Failed to reset Trino query_max_run_time; reconnecting",
                        exc_info=True,
                    )
                    self._reset_connection()

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def _information_schema(self, relation: str) -> str:
        """Return the quoted reference to one ``information_schema`` relation."""
        return _qualified(self._catalog, _SYSTEM_SCHEMA, relation)

    def _schema_condition(self, column: str = "table_schema") -> str:
        """The ``WHERE`` condition selecting which schemas to introspect.

        The allowlist is pushed to the server: ``information_schema`` on a
        federated catalog is a live scan of the underlying system's metadata, so
        filtering after the fact would still pay for every schema.
        """
        if not self._schema_names:
            return f"{column} <> {_quoted_literal(_SYSTEM_SCHEMA)}"
        names = ", ".join(_quoted_literal(name) for name in self._schema_names)
        return f"{column} IN ({names})"

    def _filter_by_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Drop rows outside the allowlist, matching schema names case-insensitively.

        The server-side predicate matches exactly; this second pass catches an
        allowlist entry whose casing differs from the catalog's.
        """
        if (
            self._schema_filter is None
            or frame.empty
            or "table_schema" not in frame.columns
        ):
            return frame
        keep = (
            frame["table_schema"].astype(str).str.casefold().isin(self._schema_filter)
        )
        return frame.loc[keep].reset_index(drop=True)

    def get_schemas(self) -> list[str]:
        """List the catalog's schemas.

        Used by the connection UI to let the user pick which schemas to ingest,
        and by the connection test — it runs a real query, so it proves the
        endpoint, the credentials and the catalog all work.
        """
        frame = self.execute(
            f"SELECT schema_name FROM {self._information_schema('schemata')} "
            f"WHERE schema_name <> {_quoted_literal(_SYSTEM_SCHEMA)} "
            "ORDER BY schema_name"
        )
        if frame.empty:
            return []
        return [str(name) for name in frame.iloc[:, 0].tolist()]

    def get_tables(self) -> pd.DataFrame:
        frame = self.execute(f"""
            SELECT
                table_schema,
                table_name,
                CASE table_type
                    WHEN 'VIEW' THEN '{TableTypes.VIEW}'
                    ELSE '{TableTypes.BASE_TABLE}'
                END AS table_type
            FROM {self._information_schema("tables")}
            WHERE {self._schema_condition()}
            ORDER BY table_schema, table_name
        """)
        if frame.empty:
            return pd.DataFrame(columns=_TABLE_SCHEMA)
        return self._filter_by_schema(frame)

    def get_columns(self) -> pd.DataFrame:
        frame = self.execute(f"""
            SELECT
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable,
                ordinal_position
            FROM {self._information_schema("columns")}
            WHERE {self._schema_condition()}
            ORDER BY table_schema, table_name, ordinal_position
        """)
        if frame.empty:
            return pd.DataFrame(columns=_COLUMN_SCHEMA)
        return self._filter_by_schema(frame)

    def get_views(self) -> pd.DataFrame:
        """Return view definitions, or nothing when the catalog has no view support.

        ``information_schema.views`` exists in every catalog, but a connector
        that cannot store views (Kafka, JMX, most JDBC-backed ones) fails the
        query instead of returning an empty result. That is not an ingestion
        error — the catalog simply has no views.
        """
        try:
            frame = self.execute(f"""
                SELECT
                    table_schema,
                    table_name,
                    view_definition
                FROM {self._information_schema("views")}
                WHERE {self._schema_condition()}
                ORDER BY table_schema, table_name
            """)
        except Exception:
            logger.debug(
                "Trino catalog %s does not expose information_schema.views",
                self._catalog,
                exc_info=True,
            )
            return pd.DataFrame(columns=_VIEW_SCHEMA)
        if frame.empty:
            return pd.DataFrame(columns=_VIEW_SCHEMA)
        return self._filter_by_schema(frame)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Trino's query history lives in the coordinator, not in SQL, so this is empty.

        ``system.runtime.queries`` holds only what is still resident on the
        coordinator (minutes, not hours) and is unreadable without cluster-admin
        privileges, so it is not a usable history source.
        """
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_pks(self) -> pd.DataFrame:
        """Trino declares no primary keys, so this is always empty."""
        return pd.DataFrame(columns=_PK_SCHEMA)

    def get_fks(self) -> pd.DataFrame:
        """Trino declares no foreign keys, so this is always empty."""
        return pd.DataFrame(columns=_FK_SCHEMA)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify the endpoint, the credentials, and that the catalog is readable."""
        self.execute(f"SELECT 1 FROM {self._information_schema('schemata')} LIMIT 1")

    def close(self) -> None:
        with self._lock:
            self._reset_connection()
