# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse connector implementing GSF's SQLDatabase ABC.

ClickHouse is a column-oriented OLAP engine. It is reached here over its HTTP
interface with ``httpx`` rather than through ``clickhouse-connect`` or the
native TCP protocol: the whole surface used by GSF is "POST a statement, read
back JSON", the HTTP endpoint is enabled on every deployment including
ClickHouse Cloud, and it keeps the dependency out of the image.

Four properties of the engine shape this connector:

* **Two-level naming.** Objects are ``database.table``; there is no catalog
  above the database and no schema below it, which puts ClickHouse in the same
  shape as MySQL. The connection binds one database, that database is both
  :attr:`~ClickHouseDatabase.database_name` and the ``table_schema`` every
  introspection method reports, and only its tables are ingested. Reporting
  another database's tables under this connection would make
  ``gsf.retrieval.text_to_sql.formatters_util.qualify_table`` emit a
  three-part ``bound.other.table``, which ClickHouse cannot parse -- so a
  second database is a second connection.
* **``system`` tables, not ``information_schema``.** ClickHouse ships an
  ``information_schema`` compatibility layer, but it has changed shape across
  versions and cannot express what is wanted here. ``system.tables`` and
  ``system.columns`` are stable and carry more: the engine name (which is how a
  materialized view is told from a plain one) and ``is_in_primary_key``.
* **A sorting key, not a primary key constraint.** MergeTree's "primary key" is
  the sparse index prefix of the sorting key. It is *not* a uniqueness
  constraint -- ``ReplacingMergeTree`` deduplicates only on merge, and plain
  ``MergeTree`` never does. :meth:`~ClickHouseDatabase.get_pks` reports it
  anyway, because it is the engine's own statement of what identifies a row and
  is what join inference has to work from; treat it as a strong hint rather than
  a guarantee.
* **No foreign keys.** ClickHouse declares none, so
  :meth:`~ClickHouseDatabase.get_fks` returns a correctly shaped *empty* frame
  rather than raising, and ingestion simply records no keys.

Authentication is HTTP basic. The transport is picked from the URL:

``http_scheme=...``
    Explicit, and always wins (a scheme pasted into the host counts).
``https``
    Chosen on the conventional TLS ports, 443 and 8443 -- the latter is what
    ClickHouse Cloud listens on.
``http``
    Everything else, including the default port 8123. Note a password does
    *not* imply TLS here: self-managed ClickHouse very commonly serves
    authenticated traffic on cleartext 8123 behind a private network, so
    inferring HTTPS from a credential -- the way :mod:`gsf.connectors.trino`
    does, because Trino refuses to send one in cleartext -- would break the
    common case rather than secure it.

Example
-------
::

    CONNECTION_STRINGS=clickhouse://default:SECRET@ch.example.com:8123/analytics
    CONNECTION_STRINGS=clickhouse://user:SECRET@abc.clickhouse.cloud:8443/default
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import pandas as pd

from gsf.catalog.constants import TableTypes
from gsf.connectors.base import SQLDatabase

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8123
DEFAULT_HTTP_TIMEOUT_S = 60.0

# A query with no LIMIT can stream a whole table back. The cap bounds how much
# of one can land in a single DataFrame; ClickHouse applies it server-side, so
# unlike a client-side truncation it also bounds the work the server does.
DEFAULT_MAX_ROWS = 50_000

# Ports that are TLS by convention: 443, and 8443 which is ClickHouse Cloud's.
# 8123 (the HTTP default) and everything else is cleartext unless the URL says
# otherwise.
_TLS_PORTS = frozenset({443, 8443})

# Extra wall-clock allowed on top of a caller's statement timeout, so the
# server's own ``max_execution_time`` fires first and reports which query it
# killed rather than the client giving up on an answer that was coming.
_TIMEOUT_GRACE_S = 10.0

# ClickHouse's own bookkeeping databases. Never worth ingesting, and named here
# so that binding one is rejected rather than silently cataloguing the engine's
# internals.
_SYSTEM_DATABASES = frozenset({"system", "information_schema", "INFORMATION_SCHEMA"})

# ``system.tables.engine`` values that mean "this is a view". ClickHouse has
# four kinds; all are queried like a table, and only the materialized one holds
# its own data.
_VIEW_ENGINES = frozenset({"View", "LiveView", "WindowView"})
_MATERIALIZED_VIEW_ENGINE = "MaterializedView"


class ClickHouseError(RuntimeError):
    """An error reported by ClickHouse, or reaching it.

    The message carries the server's own ``Code: N. DB::Exception: ...`` text
    verbatim so that :mod:`gsf.connectors.db_errors` can tell "your SQL was
    wrong" apart from "the server is unreachable" the same way it does for
    every other engine.
    """


def _quoted_literal(value: str) -> str:
    """Return a single-quoted SQL string literal.

    Backslash is an escape character inside ClickHouse string literals (unlike
    ANSI SQL), so it has to be doubled before the quote is.
    """
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


# How ClickHouse reports a column that does not exist. The numeric codes are
# the stable part -- 47 is UNKNOWN_IDENTIFIER, 16 is NO_SUCH_COLUMN_IN_TABLE --
# but the prose is matched too, since the code is absent from some proxied
# error envelopes.
_UNKNOWN_COLUMN_MARKERS = (
    "code: 47.",
    "code: 16.",
    "unknown identifier",
    "missing column",
    "there is no column",
)


def _is_unknown_column(error: Exception) -> bool:
    """Whether *error* is ClickHouse rejecting a column name it does not know."""
    text = str(error).lower()
    return any(marker in text for marker in _UNKNOWN_COLUMN_MARKERS)


def _first(query: dict[str, list[str]], key: str) -> str | None:
    """Return the first value for *key*, URL-decoded, or ``None``."""
    value = query.get(key, [None])[0]
    return unquote(value) if value else None


def _is_true(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _positive_int(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"Expected a positive integer, got {value!r}") from error
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer, got {value!r}")
    return parsed


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Parse a ClickHouse URL into the settings this connector needs.

    Expected format::

        clickhouse://[USER[:PASSWORD]@]HOST[:8123]/DATABASE[?http_scheme=...]

    Credentials are optional: a development server running as the default user
    with an empty password takes none.
    """
    parsed = urlparse(connection_string)
    scheme = parsed.scheme.split("+", 1)[0].lower()
    if scheme != "clickhouse":
        raise ValueError(f"Not a ClickHouse URL: {connection_string}")
    if not parsed.hostname:
        raise ValueError(
            f"Invalid ClickHouse connection string (missing host): {connection_string}"
        )

    database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ValueError(
            "ClickHouse connection string requires a database, e.g. "
            "clickhouse://default@ch.example.com:8123/analytics"
        )
    if database in _SYSTEM_DATABASES:
        raise ValueError(
            f"Refusing to bind ClickHouse's {database!r} database: it holds the "
            "engine's own bookkeeping tables, not user data"
        )

    query = parse_qs(parsed.query)
    port = parsed.port or DEFAULT_PORT

    http_scheme = (_first(query, "http_scheme") or "").strip().lower()
    if http_scheme and http_scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported ClickHouse http_scheme: {http_scheme!r} (use http or https)"
        )
    if not http_scheme:
        http_scheme = "https" if port in _TLS_PORTS else "http"

    return {
        "host": parsed.hostname,
        "port": port,
        "database": database,
        # An absent username means the server's default user; an absent
        # password means an empty one, which is what a stock install ships.
        "username": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password) if parsed.password else None,
        "http_scheme": http_scheme,
        "verify_ssl": _is_true(_first(query, "verify_ssl"), default=True),
        "max_rows": _positive_int(_first(query, "max_rows"), DEFAULT_MAX_ROWS),
    }


def _error_message(response: httpx.Response) -> str:
    """Render an error response as the server's own diagnosis.

    ClickHouse answers a failed statement with ``text/plain`` rather than JSON:
    ``Code: 60. DB::Exception: Table analytics.nope does not exist.``. That
    line is the whole diagnosis, so it is passed through rather than parsed.
    """
    detail = (response.text or "").strip()[:500] or response.reason_phrase
    return f"ClickHouse request failed (HTTP {response.status_code}): {detail}"


# ClickHouse reports its types in ``system.columns`` and in a result's ``meta``
# the same way, wrapping the nullable ones. ``LowCardinality`` may wrap
# ``Nullable`` but never the other way round, so those two prefixes cover it.
#
# Deliberately not a substring test: ``Array(Nullable(String))`` is a
# *non*-nullable column of arrays that happen to hold nulls, and treating it as
# nullable would mark most of a typical schema wrong.
_NULLABLE_PREFIXES = ("Nullable(", "LowCardinality(Nullable(")

# Wrappers that do not change what the column *is*, only how it is stored or
# whether it admits null. Peeled off before a type is classified.
_TRANSPARENT_WRAPPERS = ("Nullable(", "LowCardinality(")


def _inner_type(data_type: str) -> str:
    """Return *data_type* with its transparent wrappers peeled off.

    ``LowCardinality(Nullable(String))`` -> ``String``. Unwrapping is what lets
    the classifiers below anchor at the start of the string instead of scanning
    it; see :func:`_is_datetime`.
    """
    inner = str(data_type).strip()
    while inner.startswith(_TRANSPARENT_WRAPPERS) and inner.endswith(")"):
        inner = inner[inner.index("(") + 1 : -1].strip()
    return inner


def _is_nullable(data_type: str) -> bool:
    """Whether a ClickHouse type annotation describes a nullable column."""
    return str(data_type).startswith(_NULLABLE_PREFIXES)


def _is_datetime(data_type: str) -> bool:
    """Whether *data_type* is a date/time column: ``Date``, ``Date32``,
    ``DateTime``, ``DateTime64(3)``, ``DateTime('UTC')``, or any of those
    wrapped in ``Nullable``/``LowCardinality``.

    Anchored at the start of the *unwrapped* type rather than searched for as a
    substring. A substring test reads ``Enum8('Date' = 1, 'Time' = 2)`` as a
    date column, and the coercion that follows is not harmless there: it rewrites
    every value to ``NaT``, so the column survives as an all-null timestamp
    rather than failing loudly. ``Array(DateTime)`` is excluded for the same
    reason it is not nullable -- it is an array, not a timestamp.
    """
    return _inner_type(data_type).startswith("Date")


def _coerce_datetimes(
    frame: pd.DataFrame, columns: list[str], types: list[str]
) -> pd.DataFrame:
    """Parse date-typed columns into real timestamps.

    JSONCompact renders every ``Date``/``DateTime`` as a string, so without this
    a min/max profile of a timestamp column compares text -- which happens to
    order correctly for ``YYYY-MM-DD`` and then silently does not for anything
    downstream that does date arithmetic.
    """
    for column, column_type in zip(columns, types):
        if not _is_datetime(column_type):
            continue
        try:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
        except (TypeError, ValueError):
            logger.debug("Could not parse %r as a datetime column", column)
    return frame


class ClickHouseDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by the ClickHouse HTTP interface.

    Parameters
    ----------
    connection_string:
        ``clickhouse://[user[:password]@]host:8123/database``
    """

    #: ClickHouse takes ``max_execution_time`` as a per-request setting, and
    #: applying it costs nothing extra -- there is no session to rebuild --
    #: so profiling probes get to cap a server that has stopped answering.
    supports_statement_timeout: bool = True

    def __init__(self, connection_string: str) -> None:
        self._settings = _parse_connection_string(connection_string)
        self._database_name = str(self._settings["database"])
        self._max_rows = int(self._settings["max_rows"])
        self._base_url = (
            f"{self._settings['http_scheme']}://"
            f"{self._settings['host']}:{self._settings['port']}"
        )

        self._lock = threading.RLock()
        self._client: httpx.Client | None = None

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.Client:
        """Return the shared HTTP client, creating it on first use.

        One client per connector keeps the connection pool (and the TLS
        handshake) alive across the hundreds of small statements that ingestion
        and profiling issue. ``httpx.Client`` is thread-safe, so unlike the
        socket-based connectors this needs no serialisation around requests --
        only around creating the client itself.
        """
        with self._lock:
            if self._client is None:
                username = self._settings["username"]
                password = self._settings["password"]
                self._client = httpx.Client(
                    base_url=self._base_url,
                    # Sent even when only one of the two is set: ClickHouse
                    # reads an empty password as "no password", which is what
                    # a stock ``default`` user has.
                    auth=(
                        httpx.BasicAuth(username or "default", password or "")
                        if (username or password)
                        else None
                    ),
                    verify=bool(self._settings["verify_ssl"]),
                    timeout=DEFAULT_HTTP_TIMEOUT_S,
                )
                logger.info(
                    "Opened ClickHouse connection to %s/%s",
                    self._base_url,
                    self._database_name,
                )
            return self._client

    def _post(
        self,
        sql: str,
        timeout_s: float | None = None,
        cap_rows: bool = True,
    ) -> dict[str, Any]:
        """Run *sql* over the HTTP interface and return the decoded JSON body.

        The statement travels as the request body and every knob as a query
        parameter, which is the shape ClickHouse's HTTP interface expects.
        ``default_format`` rather than an appended ``FORMAT JSONCompact`` so
        that a statement which names its own format still wins, and so DDL --
        which returns no result set at all -- comes back as an empty body
        instead of a parse error.

        Returns:
            The decoded body, or ``{}`` for a statement that produced no result.

        Raises:
            ClickHouseError: the server could not be reached, or it rejected
                the statement.
        """
        client = self._ensure_client()
        params: dict[str, str] = {
            "database": self._database_name,
            "default_format": "JSONCompact",
        }
        if cap_rows:
            # Applied server-side, so a runaway query is bounded by work done
            # rather than by rows kept.
            params["max_result_rows"] = str(self._max_rows)
            params["result_overflow_mode"] = "break"
        if timeout_s is not None:
            params["max_execution_time"] = str(max(1, int(timeout_s)))

        timeout = (
            DEFAULT_HTTP_TIMEOUT_S
            if timeout_s is None
            else timeout_s + _TIMEOUT_GRACE_S
        )
        try:
            response = client.post(
                "/", content=sql.encode("utf-8"), params=params, timeout=timeout
            )
        except httpx.HTTPError as error:
            # The wording matters: ``gsf.connectors.db_errors`` classifies by
            # message text, and "could not connect" is one of the phrases it
            # looks for. Leaving it to the wrapped httpx message is not enough --
            # a timeout stringifies to "" and a multi-address failure to "All
            # connection attempts failed", neither of which matches a marker, so
            # the text-to-SQL graph would treat an unreachable server as bad SQL
            # and burn retries rewriting a query that was never the problem.
            raise ClickHouseError(
                f"Could not connect to ClickHouse at {self._base_url}: {error!r}"
            ) from error
        if response.status_code >= 400:
            raise ClickHouseError(_error_message(response))

        body = response.text.strip()
        if not body:
            return {}
        try:
            return dict(response.json())
        except ValueError as error:
            raise ClickHouseError(
                f"ClickHouse returned a non-JSON response: {body[:200]}"
            ) from error

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "clickhouse"

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
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Run *sql* and return the result as a DataFrame.

        A statement that returns no rows (DDL, ``SET``) yields an empty frame
        rather than raising.

        Parameters
        ----------
        sql:
            The statement.
        parameters:
            Not supported. ClickHouse's HTTP interface binds parameters by
            *name* (``{id:UInt64}`` fed from a ``param_id`` query argument),
            which carries a declared type per placeholder; a positional list
            has neither names nor types to supply, and inventing them would
            mean guessing a type per value -- binding a numeric id as a string
            turns an index lookup into a full scan, or a type error. Nothing in
            GSF passes parameters to a connector, so this raises rather than
            silently interpolating them into the statement.
        timeout_s:
            Optional cap on how long the server may spend on the statement.

        Raises:
            ValueError: *parameters* was supplied.
            ClickHouseError: the server could not be reached, or it rejected
                the statement.
        """
        if parameters:
            raise ValueError(
                "ClickHouse's HTTP interface has no positional parameter binding; "
                "inline the values into the statement instead"
            )
        return self._query(sql, timeout_s=timeout_s, cap_rows=True)

    def _query(
        self,
        sql: str,
        timeout_s: float | None = None,
        cap_rows: bool = True,
    ) -> pd.DataFrame:
        """Run *sql* and decode the JSONCompact body into a DataFrame.

        *cap_rows* applies ``max_result_rows``. Callers running a user's query
        want it; the introspection methods below do not. Their result size is
        bounded by the schema rather than the data, and a warehouse with enough
        columns to reach the cap would have ``get_columns`` silently truncated --
        the catalog would then be ingested with whole tables missing their
        columns, behind nothing louder than a warning.
        """
        body = self._post(sql, timeout_s=timeout_s, cap_rows=cap_rows)
        meta = body.get("meta") or []
        if not meta:
            return pd.DataFrame()
        columns = [str(descriptor.get("name") or "") for descriptor in meta]
        types = [str(descriptor.get("type") or "") for descriptor in meta]
        rows: list[list[Any]] = list(body.get("data") or [])

        # ClickHouse reports the break as a flag on the response rather than as
        # an error, so without this a truncated answer is indistinguishable from
        # a complete one.
        if cap_rows and len(rows) >= self._max_rows:
            logger.warning(
                "ClickHouse result truncated at max_rows=%s; add a LIMIT or raise "
                "max_rows on the connection. Statement: %s",
                self._max_rows,
                sql,
            )

        return _coerce_datetimes(pd.DataFrame(rows, columns=columns), columns, types)

    def _introspect(self, sql: str) -> pd.DataFrame:
        """Run a metadata query against the ``system`` tables, uncapped."""
        return self._query(sql, cap_rows=False)

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def _database_predicate(self, column: str = "database") -> str:
        """The ``WHERE`` condition restricting introspection to the bound database."""
        return f"{column} = {_quoted_literal(self._database_name)}"

    def get_tables(self) -> pd.DataFrame:
        """Return the bound database's tables, with views classified by engine.

        ``system.tables`` names the engine rather than a table type, which is
        the only place the distinction between a plain and a materialized view
        survives -- ``information_schema.tables`` collapses both to ``VIEW``.
        Temporary tables are excluded: they belong to a session that is not
        this one, so they are gone by the time anything queries them.
        """
        view_engines = ", ".join(
            _quoted_literal(name) for name in sorted(_VIEW_ENGINES)
        )
        frame = self._introspect(f"""
            SELECT
                database AS table_schema,
                name AS table_name,
                multiIf(
                    engine = {_quoted_literal(_MATERIALIZED_VIEW_ENGINE)},
                        {_quoted_literal(str(TableTypes.MATERIALIZED_VIEW))},
                    engine IN ({view_engines}),
                        {_quoted_literal(str(TableTypes.VIEW))},
                    {_quoted_literal(str(TableTypes.BASE_TABLE))}
                ) AS table_type
            FROM system.tables
            WHERE {self._database_predicate()}
              AND NOT is_temporary
            ORDER BY table_schema, table_name
        """)
        if frame.empty:
            return pd.DataFrame(columns=["table_schema", "table_name", "table_type"])
        return frame

    def get_columns(self) -> pd.DataFrame:
        """Return the bound database's columns.

        Nullability is derived from the type annotation rather than selected:
        ClickHouse encodes it in the type (``Nullable(String)``) and its
        ``information_schema`` shim has reported ``is_nullable`` as both a
        ``UInt8`` and a ``'YES'``/``'NO'`` string across versions, so reading
        the type is both stabler and the only source that survives a
        ``LowCardinality`` wrapper.
        """
        frame = self._introspect(f"""
            SELECT
                database AS table_schema,
                table AS table_name,
                name AS column_name,
                type AS data_type,
                position AS ordinal_position
            FROM system.columns
            WHERE {self._database_predicate()}
            ORDER BY table_schema, table_name, ordinal_position
        """)
        columns = [
            "table_schema",
            "table_name",
            "column_name",
            "data_type",
            "is_nullable",
            "ordinal_position",
        ]
        if frame.empty:
            return pd.DataFrame(columns=columns)
        # Built with ``astype(bool)`` so the column is a real bool dtype rather
        # than an object column of Python bools; see ``test_nullability.py``.
        frame["is_nullable"] = frame["data_type"].map(_is_nullable).astype(bool)
        # Reindexed rather than returned as-is: ``is_nullable`` was appended
        # above, so without this it trails ``ordinal_position``.
        return frame[columns]

    def get_views(self) -> pd.DataFrame:
        """Return view definitions from ``system.tables``.

        ``as_select`` holds the bare ``SELECT`` and is what a caller wants, but
        it was only added in 23.x; older servers reject the column outright, so
        the whole statement is retried against ``create_table_query`` -- the
        full ``CREATE VIEW ... AS SELECT ...``, which is more than asked for but
        strictly better than no definition at all.
        """
        view_columns = ["table_schema", "table_name", "view_definition"]
        engines = ", ".join(
            _quoted_literal(name)
            for name in sorted(_VIEW_ENGINES | {_MATERIALIZED_VIEW_ENGINE})
        )
        for definition_column in ("as_select", "create_table_query"):
            try:
                frame = self._introspect(f"""
                    SELECT
                        database AS table_schema,
                        name AS table_name,
                        {definition_column} AS view_definition
                    FROM system.tables
                    WHERE {self._database_predicate()}
                      AND engine IN ({engines})
                    ORDER BY table_schema, table_name
                """)
            except ClickHouseError as error:
                # Only an absent column is a reason to retry. Catching every
                # ClickHouseError here means a server that goes away mid-pass
                # fails both attempts and returns "no views" for a database
                # full of them, behind nothing louder than a debug log.
                if not _is_unknown_column(error):
                    raise
                logger.debug(
                    "ClickHouse does not expose system.tables.%s; falling back",
                    definition_column,
                    exc_info=True,
                )
                continue
            if frame.empty:
                return pd.DataFrame(columns=view_columns)
            return frame
        return pd.DataFrame(columns=view_columns)

    def get_pks(self) -> pd.DataFrame:
        """Return each table's MergeTree primary-key columns, in key order.

        This is the engine's sparse-index key, not a uniqueness constraint --
        see this module's docstring.

        ``ordinal_position`` is the column's position *within the key*, which is
        what the contract asks for and what makes a compound key usable. That
        ordering exists only in ``system.tables.primary_key``, a comma-separated
        expression listed in key order; ``system.columns`` flags which columns
        participate (``is_in_primary_key``) but carries no key ordinal, so
        ordering its rows by ``position`` yields the *table's* column order. For
        ``ORDER BY (day, id)`` on a table declared ``(id, day)`` that reports the
        key as ``(id, day)`` -- reversed, and reversed silently.

        The split parts are joined back against ``system.columns`` so that a key
        built on an expression (``ORDER BY (toYYYYMM(day), id)``) contributes
        only its real columns: splitting on commas would otherwise tear a
        multi-argument function into fragments that are not column names at all.

        A table with no primary key (``Log``, ``Memory``, a plain view)
        contributes no rows, which is the correct "no keys declared".
        """
        frame = self._introspect(f"""
            SELECT
                keys.table_schema AS table_schema,
                keys.table_name AS table_name,
                keys.column_name AS column_name,
                row_number() OVER (
                    PARTITION BY keys.table_schema, keys.table_name
                    ORDER BY keys.key_position
                ) AS ordinal_position
            FROM (
                SELECT
                    table_schema,
                    table_name,
                    trimBoth(parts[part_index]) AS column_name,
                    part_index AS key_position
                FROM (
                    SELECT
                        database AS table_schema,
                        name AS table_name,
                        splitByChar(',', primary_key) AS parts
                    FROM system.tables
                    WHERE {self._database_predicate()}
                      AND primary_key != ''
                )
                ARRAY JOIN arrayEnumerate(parts) AS part_index
            ) AS keys
            INNER JOIN (
                SELECT database, table, name
                FROM system.columns
                WHERE {self._database_predicate()}
                  AND is_in_primary_key
            ) AS cols
              ON cols.database = keys.table_schema
             AND cols.table = keys.table_name
             AND cols.name = keys.column_name
            ORDER BY table_schema, table_name, ordinal_position
        """)
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "table_schema",
                    "table_name",
                    "column_name",
                    "ordinal_position",
                ]
            )
        return frame

    def get_fks(self) -> pd.DataFrame:
        """ClickHouse declares no foreign keys, so this is always empty."""
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

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Return recent successful queries from ``system.query_log``.

        The log is on by default but is genuinely optional -- it is switched
        off on write-heavy clusters, and a restricted user cannot read it -- so
        a failure here is reported as "no history" rather than as an ingestion
        error.

        Only ``QueryFinish`` rows are returned: a ``QueryStart`` row is the
        same statement logged twice, and the exception rows are queries that
        never produced a result, so neither is evidence of how the data is
        actually used.
        """
        try:
            frame = self._introspect(f"""
                SELECT
                    event_time AS end_time,
                    query AS query_text
                FROM system.query_log
                WHERE type = 'QueryFinish'
                  AND event_time >= now() - INTERVAL {int(hours)} HOUR
                  AND has(databases, {_quoted_literal(self._database_name)})
                ORDER BY end_time DESC
            """)
        except ClickHouseError:
            logger.debug(
                "ClickHouse query history is unavailable on %s",
                self._base_url,
                exc_info=True,
            )
            return pd.DataFrame(columns=["end_time", "query_text"])
        if frame.empty:
            return pd.DataFrame(columns=["end_time", "query_text"])
        return frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify the endpoint, the credentials, and that the database exists.

        ``SELECT 1`` is a complete check here even though it names no table: the
        bound database travels as the request's ``database`` parameter, and the
        server resolves that before it runs anything, answering
        ``UNKNOWN_DATABASE`` for one that does not exist.

        Deliberately *not* a lookup in ``system.databases``. That view is
        filtered by the caller's grants, so a read-only user sees only the
        databases it was granted -- checking it there reports a database that
        plainly exists as missing, which is both wrong and unactionable
        (the fix is a grant, not a name).
        """
        self.execute("SELECT 1")

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    logger.debug("Failed to close ClickHouse client", exc_info=True)
                self._client = None
