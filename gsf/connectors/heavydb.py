# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HeavyDB connector implementing the NeMo-Retriever SQLDatabase ABC.

HeavyDB (formerly OmniSciDB / MapD) is a GPU-accelerated, columnar analytics
database. It speaks a Calcite-based SQL dialect over a Thrift wire protocol.
Connectivity is provided by the ``pyheavydb`` DBAPI driver (imported as
``heavydb``).

Unlike PostgreSQL or Snowflake, HeavyDB does not expose a portable
``information_schema`` for column metadata (the system tables live in a
separate ``information_schema`` database whose layout varies by server
version). Instead, schema introspection here uses the Thrift client's
metadata RPCs (``get_tables_meta`` / ``get_table_details``) — the same calls
the official client and Heavy Immerse rely on — which are stable across
server versions and scoped to the connected database.

HeavyDB has no schemas, primary keys, or foreign keys, so ``table_schema`` is
reported as the database name and ``get_pks`` / ``get_fks`` return empty
frames with the expected columns.

Example
-------
::

    CONNECTION_STRINGS=heavydb://admin:HyperInteractive@localhost:6274/heavyai?protocol=binary
"""

from __future__ import annotations

import logging
import re
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Optional
from urllib.parse import parse_qs, unquote, urlparse

import heavydb
import pandas as pd
from heavydb.common.ttypes import TDatumType
from gsf.connectors.base import SQLDatabase

if TYPE_CHECKING:
    from heavydb.connection import Connection

logger = logging.getLogger(__name__)

# HeavyDB auto-creates an implicit physical ``rowid`` column per table that is
# not part of the user-facing schema; exclude it from column introspection.
_IMPLICIT_COLUMNS = frozenset({"rowid"})

# ``heavydb.connect`` performs the login RPC inline and exposes no timeout. A
# misconfigured endpoint (classically a binary Thrift handshake against an
# HTTP-only port) leaves the driver blocked on a socket read that never
# returns. Bound it so connection attempts fail fast with a clear message
# instead of hanging the request and surfacing as an opaque 500.
_CONNECT_TIMEOUT_SECONDS = 15


def _connect_with_timeout(
    connect_kwargs: dict[str, Any], timeout: float
) -> "Connection":
    """Call ``heavydb.connect`` but give up after *timeout* seconds.

    Runs the (potentially blocking) connect on a daemon thread so a hung
    handshake never blocks the caller; the orphaned thread cannot keep the
    process alive.
    """
    result: dict[str, Any] = {}

    def _run() -> None:
        try:
            result["conn"] = heavydb.connect(**connect_kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            result["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        host = connect_kwargs.get("host")
        port = connect_kwargs.get("port")
        protocol = connect_kwargs.get("protocol")
        raise TimeoutError(
            f"Timed out after {timeout:g}s connecting to HeavyDB at "
            f"{host}:{port} using protocol {protocol!r}. Verify the host and "
            "port and that the protocol matches the server "
            "(binary vs http/https)."
        )
    if "error" in result:
        raise result["error"]
    return result["conn"]


_COLUMN_SCHEMA = [
    "table_schema",
    "table_name",
    "column_name",
    "data_type",
    "is_nullable",
    "ordinal_position",
]

# HeavyDB has a flat namespace: it accepts only bare table names, never
# ``schema.table`` or ``database.schema.table``. The text-to-SQL agent treats
# HeavyDB as PostgreSQL (see ``dialect``) and therefore qualifies tables with a
# schema. Two flavours show up:
#   * ``public.flights`` — a literal ``public`` schema, rejected with
#     "Object 'PUBLIC' not found".
#   * ``flights.flights`` / ``flights.flights.flights`` — because this connector
#     reports ``table_schema`` (and the implied catalog) as the *database name*,
#     the agent qualifies with the database name, rejected with
#     "Object '<table>' not found".
# Strip both so the query resolves to the default namespace.
#
# The negative lookbehind keeps us from touching string literals
# (``'public.html'``), dotted identifiers, or table aliases (``f.col``). The
# pattern removes one qualifier segment per pass; applied as a fixpoint (see
# ``_strip_schema_qualifiers``) a multi-part chain like ``flights.flights.flights``
# collapses to the bare ``flights`` over successive passes.


def _build_schema_qualifier_re(database_name: str) -> "re.Pattern[str]":
    """Compile the schema/catalog-qualifier strip regex for *database_name*."""
    names = "|".join(re.escape(n) for n in ("public", database_name))
    return re.compile(
        rf"""(?<![\w."'.])("?)(?:{names})\1\s*\.\s*(?=["\w])""",
        re.IGNORECASE,
    )


def _strip_schema_qualifiers(sql: str, pattern: "re.Pattern[str]") -> str:
    """Remove unsupported ``public.`` / ``<database>.`` qualifiers from *sql*.

    Runs *pattern* to a fixpoint so multi-part qualifier chains (e.g.
    ``flights.flights.flights``) are fully stripped, not just their first
    segment.
    """
    previous = ""
    while previous != sql:
        previous = sql
        sql = pattern.sub("", sql)
    return sql


def _type_name(type_info: Any) -> str:
    """Render a ``TTypeInfo`` as a SQL type name (e.g. ``BIGINT``, ``STR[]``)."""
    base = TDatumType._VALUES_TO_NAMES.get(type_info.type, "UNKNOWN")
    return f"{base}[]" if getattr(type_info, "is_array", False) else base


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Parse a HeavyDB URL into ``heavydb.connect`` keyword arguments.

    Expected format::

        heavydb://user:password@host:6274/dbname?protocol=binary

    Optional query params: ``protocol`` (``binary``/``http``/``https``,
    default ``binary``), ``bin_cert_validate``, and ``bin_ca_certs``
    (binary-encrypted connections only).
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "heavydb":
        raise ValueError(f"Not a HeavyDB URL: {connection_string}")

    host = parsed.hostname
    if not host:
        raise ValueError(
            "HeavyDB connection string requires a host, e.g. "
            "heavydb://user:pass@host:6274/heavyai?protocol=binary"
        )

    user = unquote(parsed.username or "")
    if not user:
        raise ValueError("HeavyDB connection string requires a user")

    if parsed.password is None:
        raise ValueError("HeavyDB connection string requires a password")
    password = unquote(parsed.password)

    dbname = unquote(parsed.path.lstrip("/"))
    if not dbname:
        raise ValueError(
            "HeavyDB connection string requires a database name in the path, "
            "e.g. heavydb://user:pass@host:6274/heavyai"
        )

    query = parse_qs(parsed.query)
    connect_kwargs: dict[str, Any] = {
        "user": user,
        "password": password,
        "host": host,
        "port": parsed.port or 6274,
        "dbname": dbname,
        "protocol": query.get("protocol", ["binary"])[0],
    }

    bin_ca_certs = query.get("bin_ca_certs", [None])[0]
    if bin_ca_certs:
        connect_kwargs["bin_ca_certs"] = bin_ca_certs

    bin_cert_validate = query.get("bin_cert_validate", [None])[0]
    if bin_cert_validate is not None:
        connect_kwargs["bin_cert_validate"] = bin_cert_validate.lower() in (
            "1",
            "true",
            "yes",
        )

    return connect_kwargs


class HeavyDBDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by ``pyheavydb`` (imported as ``heavydb``).

    Parameters
    ----------
    connection_string:
        ``heavydb://user:password@host:6274/dbname?protocol=binary``
    """

    def __init__(self, connection_string: str) -> None:
        self._connect_kwargs = _parse_connection_string(connection_string)
        self._database_name: str = self._connect_kwargs["dbname"]
        # HeavyDB is flat-namespaced; strip schema/catalog qualifiers the
        # text-to-SQL agent emits (``public.`` and ``<database_name>.``).
        self._schema_qualifier_re = _build_schema_qualifier_re(self._database_name)

    @property
    def dialect(self) -> str:
        # sqlglot has no dedicated HeavyDB dialect. HeavyDB's SQL is a
        # Calcite-based, largely PostgreSQL-compatible dialect, so "postgres"
        # is the closest fit for the text-to-SQL agent's SQL generation.
        return "postgres"

    @property
    def database_name(self) -> str:
        return self._database_name

    @contextmanager
    def _connect(self) -> Iterator["Connection"]:
        """Open a short-lived connection, closing it when the block exits.

        Per-operation connections (rather than a long-lived pool) keep the
        connector resilient to server-side session timeouts.
        """
        conn = _connect_with_timeout(self._connect_kwargs, _CONNECT_TIMEOUT_SECONDS)
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        """Execute a SQL statement and return the result as a DataFrame.

        Note: HeavyDB uses *named* parameter style (``:name``), so
        ``parameters`` should be a mapping when supplied.
        """
        sql = _strip_schema_qualifiers(sql, self._schema_qualifier_re)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, parameters)
                if cur.description is None:
                    return pd.DataFrame()
                columns = [desc[0] for desc in cur.description]
                return pd.DataFrame(cur.fetchall(), columns=columns)

    # ------------------------------------------------------------------
    # Schema introspection (via Thrift metadata RPCs)
    # ------------------------------------------------------------------

    def get_tables(self) -> pd.DataFrame:
        rows = []
        with self._connect() as conn:
            for meta in conn._client.get_tables_meta(conn._session):
                rows.append(
                    {
                        "table_schema": self._database_name,
                        "table_name": meta.table_name,
                        "table_type": "VIEW" if meta.is_view else "TABLE",
                    }
                )
        return pd.DataFrame(rows, columns=["table_schema", "table_name", "table_type"])

    def get_columns(self) -> pd.DataFrame:
        rows = []
        with self._connect() as conn:
            for meta in conn._client.get_tables_meta(conn._session):
                position = 0
                for name, type_info in zip(meta.col_names, meta.col_types):
                    if name in _IMPLICIT_COLUMNS:
                        continue
                    position += 1
                    rows.append(
                        {
                            "table_schema": self._database_name,
                            "table_name": meta.table_name,
                            "column_name": name,
                            "data_type": _type_name(type_info),
                            "is_nullable": "YES" if type_info.nullable else "NO",
                            "ordinal_position": position,
                        }
                    )
        return pd.DataFrame(rows, columns=_COLUMN_SCHEMA)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """HeavyDB exposes no portable query history.

        Query history is only available via the optional, disabled-by-default
        ``request_logs`` log-based system table, so this returns an empty frame
        with the expected columns.
        """
        logger.debug("HeavyDB has no portable query history; returning empty frame.")
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        rows = []
        with self._connect() as conn:
            for meta in conn._client.get_tables_meta(conn._session):
                if not meta.is_view:
                    continue
                details = conn._client.get_table_details(conn._session, meta.table_name)
                rows.append(
                    {
                        "table_schema": self._database_name,
                        "table_name": meta.table_name,
                        "view_definition": details.view_sql or "",
                    }
                )
        return pd.DataFrame(
            rows, columns=["table_schema", "table_name", "view_definition"]
        )

    def get_pks(self) -> pd.DataFrame:
        # HeavyDB does not support primary key constraints.
        return pd.DataFrame(
            columns=["table_schema", "table_name", "column_name", "ordinal_position"]
        )

    def get_fks(self) -> pd.DataFrame:
        # HeavyDB does not support foreign key constraints.
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify connectivity, credentials, and that the catalog is readable.

        Uses the Thrift ``get_tables_meta`` metadata RPC rather than a SQL probe
        like ``SELECT 1``: HeavyDB's Calcite dialect rejects FROM-less selects,
        whereas the metadata RPC validates the session and catalog access
        without depending on any table existing.
        """
        with self._connect() as conn:
            conn._client.get_tables_meta(conn._session)

    def close(self) -> None:
        """No persistent connection to close (connections are per-operation)."""
