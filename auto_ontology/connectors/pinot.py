# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apache Pinot connector implementing Auto Ontology's SQLDatabase ABC.

Pinot is a real-time OLAP store. It is reached here over its REST APIs with
``httpx`` rather than through ``pinotdb``: the whole surface used by Auto
Ontology is "POST a statement, read back JSON" plus two metadata endpoints, and
keeping it on ``httpx`` keeps the dependency out of the image -- the same
choice :mod:`auto_ontology.connectors.clickhouse` makes.

A Pinot cluster splits the two jobs a connector needs across two services:

* **The broker runs queries** (``POST /query/sql``, port 8099 by default). It is
  the host in the connection URL.
* **The controller owns metadata** (``GET /tables`` and
  ``GET /tables/{table}/schema``, port 9000 by default). Pinot has no
  ``information_schema``, so introspection goes here. It defaults to the
  broker's host on port 9000, and is set explicitly with ``controller=`` when
  it lives elsewhere -- the usual shape of a managed or Kubernetes deployment.

Properties of the engine that shape this connector:

* **Two-level naming.** Pinot 1.1 added logical databases; everything before
  that, and every table created without one, lives in ``default``. The
  connection binds one database, which is both
  :attr:`~PinotDatabase.database_name` and the ``table_schema`` every
  introspection method reports. The broker accepts ``default.orders`` and
  ``analytics.orders`` alike, so the base class's ``schema.table`` qualification
  is correct as-is. The database also travels in the ``database`` header, which
  is how Pinot resolves an unqualified name and scopes ``GET /tables``.
* **Two query engines.** The original single-stage engine cannot join; the
  multi-stage engine can, and also drops the single-stage engine's implicit
  ``LIMIT 10`` on a query that names none. Agent-written SQL joins, so the
  multi-stage engine is on by default and ``multistage=false`` turns it off for
  a cluster that does not run it.
* **Errors arrive inside a successful response.** A broker that loses a server
  mid-query still answers HTTP 200, with whatever rows it got and the failure in
  an ``exceptions`` list. Returning those rows would hand the agent a partial
  answer it has no way to recognise as partial, so any reported exception fails
  the statement.
* **Keys are upsert keys.** ``primaryKeyColumns`` is declared only on upsert
  and dedup tables, where it is the engine's own statement of row identity, and
  is reported by :meth:`~PinotDatabase.get_pks`. There are no foreign keys, no
  views and no query-history API, so those methods return correctly shaped
  empty frames.
* **Nulls are opt-in.** Without column-based null handling a Pinot column
  stores a default value (``"null"``, ``0``, ``Long.MIN_VALUE``) in place of a
  null, so a field's ``notNull`` flag only says something when the schema turns
  null handling on. Otherwise nullability is reported as unknown.

Authentication is HTTP basic from the URL's credentials, or a bearer token from
``token=``; the same credential is sent to the broker and the controller. A
connection saved in the UI carries the token as a password with no user, which
:func:`~auto_ontology.connectors.connection_string_factory.build_connection_string`
turns into ``token=``.

Example
-------
::

    CONNECTION_STRINGS=pinot://broker.example.com:8099/default
    CONNECTION_STRINGS=pinot://admin:SECRET@broker:8099/default?controller=http://controller:9000
    CONNECTION_STRINGS=pinot://broker.example.com:443/analytics?token=SECRET
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional
from urllib.parse import quote, unquote, urlparse

import httpx
import pandas as pd
import sqlglot
import sqlglot.errors

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

logger = logging.getLogger(__name__)

DEFAULT_BROKER_PORT = 8099
DEFAULT_CONTROLLER_PORT = 9000
DEFAULT_DATABASE = "default"
DEFAULT_HTTP_TIMEOUT_S = 60.0

# Ports that are TLS by convention. Everything else, including the default
# broker and controller ports, is cleartext unless the URL says otherwise.
_TLS_PORTS = frozenset({443, 8443})

# Extra wall-clock allowed on top of a caller's statement timeout, so the
# broker's own ``timeoutMs`` fires first and reports which query it stopped
# rather than the client giving up on an answer that was coming.
_TIMEOUT_GRACE_S = 10.0

# Pinot's ``QueryErrorCode`` values that mean "stopped at the time limit":
# QUERY_SCHEDULING_TIMEOUT, EXECUTION_TIMEOUT and BROKER_TIMEOUT.
_TIMEOUT_ERROR_CODES = frozenset({240, 250, 400})

# Codes for a cluster that could not serve the query at all -- a server shutting
# down or out of capacity, segments with no live replica, a broker that could
# not reach a server. No rewrite of the SQL can fix these, so they are reported
# in words :mod:`auto_ontology.connectors.db_errors` reads as infrastructure,
# which stops the text-to-SQL graph from retrying them.
_UNAVAILABLE_ERROR_CODES = frozenset({210, 211, 305, 420, 425, 427, 510})

# The field-spec lists of a Pinot schema, in the order columns are reported.
_FIELD_SPEC_KEYS = (
    "dimensionFieldSpecs",
    "metricFieldSpecs",
    "dateTimeFieldSpecs",
    "complexFieldSpecs",
)

_TABLE_COLUMNS = ["table_schema", "table_name", "table_type"]
_COLUMN_COLUMNS = [
    "table_schema",
    "table_name",
    "column_name",
    "data_type",
    "is_nullable",
    "ordinal_position",
]
_PK_COLUMNS = ["table_schema", "table_name", "column_name", "ordinal_position"]
_FK_COLUMNS = [
    "table_schema",
    "table_name",
    "column_name",
    "referenced_schema",
    "referenced_table",
    "referenced_column",
]
_VIEW_COLUMNS = ["table_schema", "table_name", "view_definition"]


class PinotError(RuntimeError):
    """An error reported by Pinot, or reaching it.

    The message carries Pinot's own error code and text so that
    :mod:`auto_ontology.connectors.db_errors` can tell "your SQL was wrong"
    apart from "the cluster is unreachable" the same way it does for every
    other engine.
    """


def _query_params(query: str) -> dict[str, str]:
    """Decode a URL query string into its first value per key.

    Each value is percent-decoded exactly once, and ``+`` is kept as a literal
    plus. ``parse_qs`` reads ``+`` as a space, which corrupts a base64-style
    bearer token pasted into ``token=``, and decoding its output again would
    turn an escaped ``%25`` back into a live escape.
    """
    params: dict[str, str] = {}
    for part in query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        params.setdefault(unquote(key), unquote(value))
    return params


def _param(params: dict[str, str], key: str) -> str | None:
    """Return *key*'s value, or ``None`` when it is absent or empty."""
    return params.get(key) or None


def _is_true(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _http_scheme(value: str | None, port: int) -> str:
    scheme = (value or "").strip().lower()
    if scheme and scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported Pinot http_scheme: {scheme!r} (use http or https)"
        )
    return scheme or ("https" if port in _TLS_PORTS else "http")


def _controller_url(raw: str | None, broker_host: str, http_scheme: str) -> str:
    """Resolve the controller's base URL.

    Absent, it is the broker's host on the controller's default port, which is
    what a single-node or quickstart cluster looks like. A bare ``host[:port]``
    takes the broker's scheme and, without a port, the controller's default
    port. A full URL without a port means its scheme's default port (443 or 80),
    as it would anywhere else: that is how a controller behind an ingress is
    addressed.
    """
    if not raw:
        return f"{http_scheme}://{broker_host}:{DEFAULT_CONTROLLER_PORT}"
    value = raw.strip().rstrip("/")
    has_scheme = "://" in value
    if not has_scheme:
        value = f"{http_scheme}://{value}"
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid Pinot controller URL: {raw!r}")
    if parsed.path or parsed.query:
        raise ValueError(
            f"Pinot controller URL must be a host and port only, got {raw!r}"
        )
    port = parsed.port
    if port is None:
        if has_scheme:
            port = 443 if parsed.scheme == "https" else 80
        else:
            port = DEFAULT_CONTROLLER_PORT
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Parse a Pinot URL into the settings this connector needs.

    Expected format::

        pinot://[USER[:PASSWORD]@]BROKER[:8099][/DATABASE][?controller=...&token=...]

    The database defaults to ``default``, which is where every table lives on a
    cluster that predates Pinot's logical databases.
    """
    parsed = urlparse(connection_string)
    scheme = parsed.scheme.split("+", 1)[0].lower()
    if scheme != "pinot":
        raise ValueError(f"Not a Pinot URL: {connection_string}")
    if not parsed.hostname:
        raise ValueError(
            f"Invalid Pinot connection string (missing broker host): {connection_string}"
        )

    query = _query_params(parsed.query)
    port = parsed.port or DEFAULT_BROKER_PORT
    http_scheme = _http_scheme(_param(query, "http_scheme"), port)

    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None
    token = _param(query, "token")
    if token and (username or password):
        raise ValueError(
            "A Pinot connection takes either a username and password or a token, "
            "not both"
        )

    return {
        "broker_url": f"{http_scheme}://{parsed.hostname}:{port}",
        "controller_url": _controller_url(
            _param(query, "controller"), parsed.hostname, http_scheme
        ),
        "database": unquote(parsed.path.strip("/")) or DEFAULT_DATABASE,
        "username": username,
        "password": password,
        "token": token,
        "verify_ssl": _is_true(_param(query, "verify_ssl"), default=True),
        "multistage": _is_true(_param(query, "multistage"), default=True),
    }


# The single-stage engine's row cap for a query that names no ``LIMIT``.
_SINGLE_STAGE_IMPLICIT_LIMIT = 10


def _has_outer_limit(sql: str) -> bool:
    """Whether *sql*'s outermost query names a ``LIMIT``.

    Unparseable SQL counts as limited: this only decides whether to warn, and a
    false warning on SQL sqlglot cannot read is noise.
    """
    try:
        statement = sqlglot.parse_one(sql, read="trino")
    except sqlglot.errors.ParseError:
        return True
    return statement.args.get("limit") is not None


def _exceptions(body: Any) -> list[dict[str, Any]]:
    """The ``exceptions`` a broker response reports, normalised to dicts."""
    if not isinstance(body, dict):
        return []
    raw = body.get("exceptions") or []
    return [item for item in raw if isinstance(item, dict)]


def _error_code(exception: dict[str, Any]) -> int | None:
    try:
        return int(exception.get("errorCode"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _describe(exceptions: list[dict[str, Any]]) -> str:
    """Render broker exceptions as Pinot's own diagnosis, first error first."""
    parts = []
    for exception in exceptions:
        message = str(exception.get("message") or "").strip()
        # Server-side messages carry a full Java stack trace after the first
        # line; the first line is the diagnosis and the rest is noise to an LLM.
        message = message.splitlines()[0] if message else "(no message)"
        parts.append(f"[{exception.get('errorCode')}] {message}"[:500])
    return "; ".join(parts)


def _coerce_timestamps(
    frame: pd.DataFrame, columns: list[str], types: list[str]
) -> pd.DataFrame:
    """Parse ``TIMESTAMP`` columns, which the broker renders as strings."""
    for column, column_type in zip(columns, types):
        if column_type != "TIMESTAMP":
            continue
        try:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
        except (TypeError, ValueError):
            logger.debug("Could not parse %r as a timestamp column", column)
    return frame


def _column_type(field: dict[str, Any]) -> str:
    """A field spec's type as Pinot's query results name it.

    A multi-value dimension reads back as a list, and the broker types it
    ``STRING_ARRAY``, ``INT_ARRAY`` and so on. Reporting the scalar type instead
    would have profiling treat a column of lists as a column of strings.
    """
    data_type = str(field.get("dataType") or "").upper() or "UNKNOWN"
    if field.get("singleValueField") is False:
        return f"{data_type}_ARRAY"
    return data_type


class PinotDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by the Pinot broker and controller.

    Parameters
    ----------
    connection_string:
        ``pinot://[user[:password]@]broker:8099/database[?controller=host:9000]``
    """

    #: Pinot takes ``timeoutMs`` as a per-query option, so applying a cap costs
    #: nothing extra and profiling probes get to bound a stalled cluster.
    supports_statement_timeout: bool = True

    def __init__(self, connection_string: str) -> None:
        self._settings = _parse_connection_string(connection_string)
        self._database_name = str(self._settings["database"])
        self._lock = threading.RLock()
        self._client: httpx.Client | None = None
        self._introspection: list[tuple[str, dict[str, Any]]] | None = None

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.Client:
        """Return the shared HTTP client, creating it on first use.

        One client serves both services: requests name an absolute URL, so it
        holds a connection pool per host. ``httpx.Client`` is thread-safe, so
        only creating it needs the lock.
        """
        with self._lock:
            if self._client is None:
                headers: dict[str, str] = {}
                # Sent only for a named database: a cluster older than Pinot 1.1
                # has no databases, and ``default`` is what it resolves anyway.
                if self._database_name != DEFAULT_DATABASE:
                    headers["database"] = self._database_name
                auth: httpx.Auth | None = None
                if self._settings["token"]:
                    headers["Authorization"] = f"Bearer {self._settings['token']}"
                elif self._settings["username"] or self._settings["password"]:
                    auth = httpx.BasicAuth(
                        self._settings["username"] or "",
                        self._settings["password"] or "",
                    )
                self._client = httpx.Client(
                    headers=headers,
                    auth=auth,
                    verify=bool(self._settings["verify_ssl"]),
                    timeout=DEFAULT_HTTP_TIMEOUT_S,
                )
                logger.info(
                    "Opened Pinot connection to broker %s, controller %s, database %s",
                    self._settings["broker_url"],
                    self._settings["controller_url"],
                    self._database_name,
                )
            return self._client

    def _query_options(self, timeout_s: float | None) -> str:
        # Sent either way: a broker can be configured to default to either
        # engine, so leaving the option out would not turn multi-stage off.
        engine = "true" if self._settings["multistage"] else "false"
        options = [f"useMultistageEngine={engine}"]
        if timeout_s is not None:
            options.append(f"timeoutMs={max(1, int(timeout_s * 1000))}")
        return ";".join(options)

    def _post_query(self, sql: str, timeout_s: float | None) -> dict[str, Any]:
        """Send *sql* to the broker and return the decoded response.

        Raises:
            StatementTimeout: the broker stopped the statement at *timeout_s*,
                or no answer arrived within it plus a grace period.
            PinotError: the broker could not be reached, or it reported any
                exception -- including alongside rows, see the module docstring.
        """
        client = self._ensure_client()
        broker_url = self._settings["broker_url"]
        payload: dict[str, str] = {"sql": sql}
        options = self._query_options(timeout_s)
        if options:
            payload["queryOptions"] = options

        timeout = (
            DEFAULT_HTTP_TIMEOUT_S
            if timeout_s is None
            else timeout_s + _TIMEOUT_GRACE_S
        )
        started = time.monotonic()
        try:
            response = client.post(
                f"{broker_url}/query/sql", json=payload, timeout=timeout
            )
        except httpx.ReadTimeout as error:
            # Connected and sent, then no answer: a slow statement, not an
            # unreachable broker. Without a cap the client's own deadline is
            # the limit that fired, so it is reported as that.
            raise StatementTimeout.since(
                DEFAULT_HTTP_TIMEOUT_S if timeout_s is None else timeout_s, started
            ) from error
        except httpx.HTTPError as error:
            # "could not connect" is a phrase ``db_errors`` classifies as
            # infrastructure; httpx's own message often matches nothing.
            raise PinotError(
                f"Could not connect to Pinot broker at {broker_url}: {error!r}"
            ) from error

        try:
            body: Any = response.json()
        except ValueError:
            body = None

        exceptions = _exceptions(body)
        if exceptions:
            codes = {_error_code(exception) for exception in exceptions}
            message = f"Pinot query failed: {_describe(exceptions)}"
            if timeout_s is not None and codes & _TIMEOUT_ERROR_CODES:
                raise StatementTimeout.since(timeout_s, started) from PinotError(
                    message
                )
            if codes & _UNAVAILABLE_ERROR_CODES:
                raise PinotError(
                    f"Pinot could not connect to the servers for this query: "
                    f"{_describe(exceptions)}"
                )
            raise PinotError(message)

        if response.status_code >= 400 or not isinstance(body, dict):
            detail = (response.text or "").strip()[:500] or response.reason_phrase
            raise PinotError(
                f"Pinot broker request failed (HTTP {response.status_code}): {detail}"
            )

        # The multi-stage engine sets this when an operator hit a row limit
        # configured to truncate rather than fail; the rows are then a prefix of
        # the real answer.
        if body.get("partialResult") is True:
            raise PinotError(
                "Pinot returned a partial result (a join or aggregation hit a "
                "configured row limit). Narrow the query or add filters."
            )
        return body

    def _get_controller(self, path: str) -> Any:
        """GET *path* from the controller and return the decoded JSON.

        Returns:
            The decoded body, or ``None`` for a 404.
        """
        client = self._ensure_client()
        controller_url = self._settings["controller_url"]
        try:
            response = client.get(f"{controller_url}{path}")
        except httpx.HTTPError as error:
            raise PinotError(
                f"Could not connect to Pinot controller at {controller_url}: {error!r}"
            ) from error
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            detail = (response.text or "").strip()[:500] or response.reason_phrase
            raise PinotError(
                f"Pinot controller request {path} failed "
                f"(HTTP {response.status_code}): {detail}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise PinotError(
                f"Pinot controller returned a non-JSON response for {path}: "
                f"{response.text[:200]}"
            ) from error

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        sqlglot has no Pinot dialect. Pinot's multi-stage engine is built on
        Apache Calcite -- ANSI-style, double-quoted identifiers, ``LIMIT`` --
        and Trino is the closest dialect sqlglot does have: it writes
        ``LOWER(x) LIKE`` where Postgres would write ``ILIKE``, which Pinot
        lacks. Must be a member of ``sqlglot.dialects.DIALECTS``; see
        ``CONNECTOR_REGISTRY``.
        """
        return "trino"

    @property
    def database_name(self) -> str:
        return self._database_name

    def qualify(self, schema: Optional[str], table: str) -> str:
        """Return a quoted reference to *table*, prefixed only when it must be.

        A table in ``default`` is named bare: that is the only form a cluster
        older than Pinot 1.1 is sure to resolve, and on newer ones the broker
        resolves a bare name against ``default`` anyway. A named database is
        kept as the prefix, which every version with databases accepts.
        """
        if schema == DEFAULT_DATABASE:
            schema = None
        return super().qualify(schema, table)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Run *sql* on the broker and return the result as a DataFrame.

        Parameters
        ----------
        sql:
            The statement.
        parameters:
            Not supported: the broker's SQL endpoint has no parameter binding.
            Nothing in Auto Ontology passes parameters to a connector, so this
            raises rather than interpolating them into the statement.
        timeout_s:
            Optional cap on how long the broker may spend on the statement.

        Raises:
            ValueError: *parameters* was supplied.
            StatementTimeout: the statement outlived *timeout_s*.
            PinotError: the broker could not be reached, or it rejected the
                statement.
        """
        if parameters:
            raise ValueError(
                "Pinot's SQL endpoint has no parameter binding; inline the values "
                "into the statement instead"
            )
        body = self._post_query(sql, timeout_s)
        result = body.get("resultTable") or {}
        schema = result.get("dataSchema") or {}
        columns = [str(name) for name in schema.get("columnNames") or []]
        if not columns:
            return pd.DataFrame()
        types = [str(kind) for kind in schema.get("columnDataTypes") or []]
        rows = list(result.get("rows") or [])
        # The single-stage engine silently caps a query with no LIMIT at ten
        # rows, and says nothing in the response. Pinot has no option to turn
        # that off, so the most that can be done is to say so.
        if (
            not self._settings["multistage"]
            and len(rows) == _SINGLE_STAGE_IMPLICIT_LIMIT
            and not _has_outer_limit(sql)
        ):
            logger.warning(
                "Pinot single-stage engine may have truncated this result to its "
                "implicit LIMIT %d; add an explicit LIMIT, or enable the "
                "multi-stage engine on the connection. Statement: %s",
                _SINGLE_STAGE_IMPLICIT_LIMIT,
                sql,
            )
        return _coerce_timestamps(pd.DataFrame(rows, columns=columns), columns, types)

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def _table_names(self) -> list[str]:
        """The bound database's logical table names, unqualified and sorted.

        A logical name covers a hybrid table's ``_OFFLINE`` and ``_REALTIME``
        halves, which is what a query names. In a named database the
        controller lists names with their ``database.`` prefix; it is stripped
        so ``table_name`` is what goes after ``table_schema``.
        """
        body = self._get_controller("/tables")
        names = body.get("tables") if isinstance(body, dict) else None
        if not isinstance(names, list):
            raise PinotError(
                f"Unexpected response from Pinot controller /tables: {body!r}"
            )
        prefix = f"{self._database_name}."
        return sorted(
            str(name)[len(prefix) :] if str(name).startswith(prefix) else str(name)
            for name in names
        )

    def _introspect(self) -> list[tuple[str, dict[str, Any]]]:
        """Each describable table and its Pinot schema, in table-name order.

        A table whose schema cannot be read (never uploaded, or dropped between
        the two calls) is left out of everything this connector reports, not
        just its columns: listing a table with no columns trips the coverage
        check in :func:`auto_ontology.catalog.extract.create_dataframe` and
        fails the whole ingestion over one table. If tables were listed and not
        one could be described, that is raised instead -- an empty catalog
        would look exactly like a database with nothing in it.

        Cached for the connector's lifetime, as
        :meth:`~auto_ontology.connectors.kyuubi.KyuubiDatabase._introspect` is:
        tables, columns and keys are all derived from one listing, so they
        always agree, and each schema is one controller round trip.
        """
        with self._lock:
            if self._introspection is not None:
                return self._introspection
            names = self._table_names()
            described = []
            for table in names:
                schema = self._get_controller(f"/tables/{quote(table, safe='')}/schema")
                if not isinstance(schema, dict):
                    logger.warning(
                        "Pinot table %s.%s has no schema; skipping it",
                        self._database_name,
                        table,
                    )
                    continue
                described.append((table, schema))
            if names and not described:
                raise PinotError(
                    f"Pinot controller listed {len(names)} tables in "
                    f"{self._database_name!r} but returned a schema for none"
                )
            self._introspection = described
            return described

    def get_tables(self) -> pd.DataFrame:
        rows = [
            (self._database_name, table, str(TableTypes.BASE_TABLE))
            for table, _ in self._introspect()
        ]
        return pd.DataFrame(rows, columns=_TABLE_COLUMNS)

    def get_columns(self) -> pd.DataFrame:
        """Return every column, read from each table's Pinot schema.

        ``is_nullable`` is the field's ``notNull`` flag inverted when the schema
        enables column-based null handling, and ``None`` ("unknown") when it
        does not -- see the module docstring.
        """
        rows = []
        for table, schema in self._introspect():
            null_handling = schema.get("enableColumnBasedNullHandling") is True
            position = 0
            for key in _FIELD_SPEC_KEYS:
                for field in schema.get(key) or []:
                    if not isinstance(field, dict) or not field.get("name"):
                        continue
                    position += 1
                    rows.append(
                        (
                            self._database_name,
                            table,
                            str(field["name"]),
                            _column_type(field),
                            (field.get("notNull") is not True)
                            if null_handling
                            else None,
                            position,
                        )
                    )
        return pd.DataFrame(rows, columns=_COLUMN_COLUMNS)

    def get_pks(self) -> pd.DataFrame:
        """Return upsert/dedup tables' ``primaryKeyColumns``, in key order."""
        rows = [
            (self._database_name, table, str(column), position)
            for table, schema in self._introspect()
            for position, column in enumerate(
                schema.get("primaryKeyColumns") or [], start=1
            )
        ]
        return pd.DataFrame(rows, columns=_PK_COLUMNS)

    def get_fks(self) -> pd.DataFrame:
        """Pinot declares no foreign keys, so this is always empty."""
        return pd.DataFrame(columns=_FK_COLUMNS)

    def get_views(self) -> pd.DataFrame:
        """Pinot has no views, so this is always empty."""
        return pd.DataFrame(columns=_VIEW_COLUMNS)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Pinot keeps no queryable history, so this is always empty."""
        return pd.DataFrame(columns=["end_time", "query_text"])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify both services answer with these credentials.

        The controller is checked as well as the broker because ingestion
        depends on it: a connection whose controller is unreachable would
        otherwise test green and then ingest nothing.

        The broker probe is a table-free ``SELECT 1`` on the multi-stage
        engine. The single-stage engine only accepts that from Pinot 1.5 on --
        1.4 rejects it as multi-stage-only -- so with ``multistage=false`` the
        probe counts a real table instead, which the single-stage engine
        answers from segment metadata. With no table to count, nothing can
        prove the broker serves queries, so only its health check is read.
        """
        tables = self._table_names()
        if self._settings["multistage"]:
            self.execute("SELECT 1")
        elif tables:
            self.execute(
                f"SELECT COUNT(*) FROM {self.qualify(self._database_name, tables[0])}"
            )
        else:
            self._get_health(self._settings["broker_url"])

    def _get_health(self, base_url: str) -> None:
        """Raise unless the service at *base_url* reports itself healthy."""
        client = self._ensure_client()
        try:
            response = client.get(f"{base_url}/health")
        except httpx.HTTPError as error:
            raise PinotError(
                f"Could not connect to Pinot broker at {base_url}: {error!r}"
            ) from error
        if response.status_code >= 400:
            raise PinotError(
                f"Pinot broker at {base_url} is unhealthy "
                f"(HTTP {response.status_code}): {response.text[:200]}"
            )

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    logger.debug("Failed to close Pinot client", exc_info=True)
                self._client = None
