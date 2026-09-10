# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apache Kyuubi connector implementing GSF's SQLDatabase ABC.

Kyuubi fronts a Spark SQL engine over the HiveServer2 Thrift protocol, so it is
reached with a Hive client rather than a REST driver. NVIDIA's ProcR / Cloud
Data Platform deployment exposes it at ``hive.<region>.data.nvidia.com:10000``
over TLS, with Iceberg catalogs (``nvdp``, ``kratos``, ...) underneath.

Authentication is SASL PLAIN. Two credential modes are supported:

``auth=ssa`` (default)
    The username is a Starfleet SSA client id and the password is a token minted
    from that client's ``client_credentials`` grant. Tokens expire in an hour, so
    they are minted on demand and refreshed here rather than baked into the
    connection string -- a cached connector would otherwise start failing an hour
    after it was built.

``auth=token``
    The username is an account name and the password is a JWT supplied verbatim
    (an SSO ``id_token``, or a token copied from the data platform's profile
    page). Nothing is refreshed; the connection breaks when the token expires.

Two behaviours of the stack drive the implementation:

* ``pyhive`` builds a plaintext ``TSocket`` even when the endpoint is TLS-only,
  which makes the connection hang forever with no error rather than fail. The
  SASL transport is therefore assembled here over an explicit ``TSSLSocket`` and
  passed via ``thrift_transport``.
* A cold engine start costs ~90s while Kyuubi launches a Spark driver on
  Kubernetes. The session is held open and reused (a warm reconnect is ~3s)
  instead of being rebuilt per query.

Example
-------
::

    CONNECTION_STRINGS=kyuubi://nvssa-prd-abc:SECRET@hive.pdx-aws.data.nvidia.com:10000/nvdp?ssa_url=https://svc.ssa.nvidia.com&truststore=/etc/ssl/global_ca.jks&truststore_password=nvidia.com
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import ssl
import tempfile
import threading
import time
import urllib.request
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
from gsf.catalog.extract import IncompleteCatalogExtractionError
from gsf.connectors.base import SQLDatabase

from gsf.connectors.db_errors import is_session_lost

if TYPE_CHECKING:
    from pyhive.hive import Connection

logger = logging.getLogger(__name__)

DEFAULT_PORT = 10000
DEFAULT_SSA_SCOPE = "public"

# Tokens live an hour; refresh early so a query never starts with a credential
# that expires mid-flight.
_TOKEN_REFRESH_MARGIN_SECONDS = 300
_TOKEN_REQUEST_TIMEOUT_SECONDS = 30

# A cold Kyuubi session provisions a Spark driver on Kubernetes, which routinely
# takes 60-120s. The socket timeout has to clear that or the client gives up
# while the engine is still starting.
_SOCKET_TIMEOUT_SECONDS = 900

# Polling cadence for an async statement: short at first so a fast query is not
# held up, backed off so a long scan does not spin on the socket.
_POLL_INTERVAL_START_SECONDS = 0.2
_POLL_INTERVAL_MAX_SECONDS = 5.0
# A statement quicker than this logs only its summary line; past it, every
# operation-state change is reported as it happens.
_SLOW_QUERY_SECONDS = 5.0
# How often a statement that is still pending repeats its state.
_PROGRESS_LOG_INTERVAL_SECONDS = 60.0


class KyuubiQueryTimeout(TimeoutError):
    """A statement outlived the caller's ``timeout_s`` and was cancelled.

    Deliberately not a transport error: the socket is fine and the server is
    answering, so reopening the session and retrying would only burn the same
    time again. Callers that set a timeout want a fast, honest failure.
    """


# ``DESCRIBE TABLE`` appends sections (``# Partition Information``, ``# Detailed
# Table Information``) after the column list, separated by a blank or ``#`` row.
# Only the leading block describes columns.
_DESCRIBE_TERMINATORS = ("#", "")

_COLUMN_SCHEMA = [
    "table_schema",
    "table_name",
    "column_name",
    "data_type",
    "is_nullable",
    "ordinal_position",
]

_PK_SCHEMA = ["table_schema", "table_name", "column_name", "ordinal_position"]

_FK_SCHEMA = [
    "table_schema",
    "table_name",
    "column_name",
    "referenced_schema",
    "referenced_table",
    "referenced_column",
]


def _transport_errors() -> tuple[type[BaseException], ...]:
    """Exception types meaning "the socket died", not "the server said no".

    Thrift is imported lazily so importing this module stays cheap, and so a
    deployment that never configures a Kyuubi connection does not need the
    driver installed.
    """
    from thrift.transport.TTransport import TTransportException

    return (
        BrokenPipeError,
        ConnectionResetError,
        ConnectionAbortedError,
        EOFError,
        TTransportException,
    )


def _quoted_identifier(name: str) -> str:
    """Return a Spark SQL backtick-quoted identifier."""
    return "`" + name.replace("`", "``") + "`"


def _qualified(*parts: str) -> str:
    """Return a backtick-quoted dotted reference, e.g. ``nvdp``.``raw``.``t``."""
    return ".".join(_quoted_identifier(p) for p in parts)


def _first(query: dict[str, list[str]], key: str) -> str | None:
    """Return the first value for *key*, URL-decoded, or ``None``."""
    value = query.get(key, [None])[0]
    return unquote(value) if value else None


def _is_true(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Parse a Kyuubi URL into the settings this connector needs.

    Expected formats::

        kyuubi://CLIENT_ID:CLIENT_SECRET@HOST:10000/CATALOG?ssa_url=https://SVC.ssa.nvidia.com
        kyuubi://USER:JWT@HOST:10000/CATALOG?auth=token
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "kyuubi":
        raise ValueError(f"Not a Kyuubi URL: {connection_string}")
    if not parsed.hostname:
        raise ValueError(
            f"Invalid Kyuubi connection string (missing host): {connection_string}"
        )

    username = unquote(parsed.username or "")
    if not username:
        raise ValueError(
            "Kyuubi connection string requires a username, e.g. "
            "kyuubi://client_id:secret@hive.pdx-aws.data.nvidia.com:10000/nvdp"
        )

    secret = unquote(parsed.password) if parsed.password is not None else ""
    if not secret:
        raise ValueError(
            "Kyuubi connection string requires a password (SSA client secret, "
            "or a JWT when auth=token)"
        )

    catalog = unquote(parsed.path.lstrip("/"))
    query = parse_qs(parsed.query)
    if not catalog:
        catalog = _first(query, "catalog") or ""
    if not catalog:
        raise ValueError(
            "Kyuubi connection string requires a catalog, e.g. "
            "kyuubi://client_id:secret@host:10000/nvdp"
        )

    auth = (_first(query, "auth") or "ssa").lower()
    if auth not in ("ssa", "token"):
        raise ValueError(f"Unsupported Kyuubi auth mode: {auth!r} (use ssa or token)")

    ssa_url = _first(query, "ssa_url")
    if auth == "ssa" and not ssa_url:
        raise ValueError(
            "Kyuubi connection string with auth=ssa requires "
            "?ssa_url=https://<service_id>.ssa.nvidia.com"
        )

    return {
        "host": parsed.hostname,
        "port": parsed.port or DEFAULT_PORT,
        "catalog": catalog,
        "username": username,
        "secret": secret,
        "auth": auth,
        "ssa_url": (ssa_url or "").rstrip("/"),
        "ssa_scope": _first(query, "ssa_scope") or DEFAULT_SSA_SCOPE,
        "use_ssl": _is_true(_first(query, "ssl"), default=True),
        "verify_ssl": _is_true(_first(query, "verify_ssl"), default=True),
        "ca_bundle": _first(query, "ca_bundle"),
        "truststore": _first(query, "truststore"),
        "truststore_data": _first(query, "truststore_data"),
        "truststore_password": _first(query, "truststore_password"),
    }


def _write_temp_jks(encoded: str) -> str:
    """Decode a base64 keystore to a temp file and return its path."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Kyuubi truststore_data is not valid base64") from exc
    if not raw:
        raise ValueError("Kyuubi truststore_data is empty")

    handle = tempfile.NamedTemporaryFile(
        suffix=".jks", prefix="kyuubi-truststore-", delete=False
    )
    with handle:
        handle.write(raw)
    return handle.name


def _jks_to_pem(jks_path: str, password: str | None) -> str:
    """Convert a Java truststore to a PEM bundle and return the new file's path.

    NVIDIA distributes its internal CAs as ``global_ca.jks``, which Python's ssl
    module cannot read. ``pyjks`` parses it in-process, so no ``keytool`` (and no
    JDK) has to be present on the host.
    """
    import jks

    store = jks.KeyStore.load(jks_path, password or "")
    chunks = []
    for alias, cert in store.certs.items():
        body = base64.encodebytes(cert.cert).decode().strip()
        chunks.append(
            f"# {alias}\n-----BEGIN CERTIFICATE-----\n{body}\n-----END CERTIFICATE-----\n"
        )
    if not chunks:
        raise ValueError(f"Truststore contains no trusted certificates: {jks_path}")

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pem", prefix="kyuubi-ca-", delete=False
    )
    with handle:
        handle.write("\n".join(chunks))
    logger.info("Converted %d CA certs from %s", len(chunks), jks_path)
    return handle.name


class KyuubiDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by Apache Kyuubi over HiveServer2.

    Parameters
    ----------
    connection_string:
        ``kyuubi://client_id:secret@host:10000/catalog?ssa_url=https://svc.ssa.nvidia.com``
    schemas:
        Optional ingestion allowlist. Empty or ``None`` means every schema in the
        catalog.
    """

    # The cap is enforced by polling the statement already in flight, so it
    # costs nothing to ask for one -- no extra session, no extra round trip.
    supports_statement_timeout = True

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        self._settings = _parse_connection_string(connection_string)
        self._catalog = str(self._settings["catalog"])

        # Compared case-insensitively: Spark lower-cases unquoted identifiers.
        self._schema_filter: set[str] | None = (
            {s.lower() for s in schemas if s and s.strip()} if schemas else None
        )
        if self._schema_filter:
            logger.info(
                "Kyuubi ingestion restricted to schemas: %s",
                sorted(self._schema_filter),
            )

        self._lock = threading.RLock()
        self._connection: "Connection | None" = None
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._ca_pem: str | None = None
        self._ca_pem_is_temporary = False
        self._introspection: (
            tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]] | None
        ) = None

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------

    def _mint_ssa_token(self) -> str:
        """Exchange the SSA client credentials for a short-lived access token."""
        url = (
            f"{self._settings['ssa_url']}/token"
            f"?scope={self._settings['ssa_scope']}&grant_type=client_credentials"
        )
        basic = base64.b64encode(
            f"{self._settings['username']}:{self._settings['secret']}".encode()
        ).decode()
        request = urllib.request.Request(
            url,
            data=b"",
            method="POST",
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(
            request, timeout=_TOKEN_REQUEST_TIMEOUT_SECONDS
        ) as response:
            payload = json.load(response)

        token = payload.get("access_token")
        if not token:
            raise ValueError("SSA token endpoint returned no access_token")

        expires_in = float(payload.get("expires_in") or 3600)
        self._token_expires_at = time.monotonic() + expires_in
        logger.info(
            "Minted SSA token for %s (expires in %.0fs)",
            self._settings["username"],
            expires_in,
        )
        return str(token)

    def _password(self) -> str:
        """Return the SASL PLAIN password, refreshing an SSA token when stale."""
        if self._settings["auth"] == "token":
            return str(self._settings["secret"])

        expiring = (
            time.monotonic() >= self._token_expires_at - _TOKEN_REFRESH_MARGIN_SECONDS
        )
        if self._token is None or expiring:
            self._token = self._mint_ssa_token()
        return self._token

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _ca_certs(self) -> str | None:
        """Return a PEM bundle path for TLS verification, converting a JKS once."""
        if self._ca_pem is not None:
            return self._ca_pem

        bundle = self._settings["ca_bundle"]
        truststore = self._settings["truststore"]
        truststore_data = self._settings["truststore_data"]
        password = self._settings["truststore_password"]
        if bundle:
            self._ca_pem = str(bundle)
        elif truststore_data:
            # Uploaded through the UI: the keystore travels with the connection,
            # so write it out before pyjks reads it. A path on the server would
            # have to exist in every pod that runs a query.
            jks_path = _write_temp_jks(str(truststore_data))
            try:
                self._ca_pem = _jks_to_pem(jks_path, password)
            finally:
                try:
                    os.unlink(jks_path)
                except OSError:
                    logger.debug("Failed to remove %s", jks_path, exc_info=True)
            self._ca_pem_is_temporary = True
        elif truststore:
            self._ca_pem = _jks_to_pem(str(truststore), password)
            self._ca_pem_is_temporary = True
        return self._ca_pem

    def _build_transport(self, password: str) -> Any:
        """Assemble a SASL PLAIN transport, over TLS unless explicitly disabled."""
        import thrift_sasl
        from pyhive.hive import get_installed_sasl
        from thrift.transport.TSocket import TSocket
        from thrift.transport.TSSLSocket import TSSLSocket

        host = str(self._settings["host"])
        port = int(self._settings["port"])

        if not self._settings["use_ssl"]:
            socket: Any = TSocket(host, port)
        else:
            # Build the context here rather than passing ``ca_certs`` to
            # TSSLSocket: that path loads *only* the given file, so omitting a
            # truststore would leave no trust anchors at all and fail every
            # handshake. ``cafile=None`` falls back to the system CA store,
            # which is what a host that already trusts the issuer needs.
            context = ssl.create_default_context(cafile=self._ca_certs())
            if not self._settings["verify_ssl"]:
                # Explicit opt-out for deployments without the internal CA.
                logger.warning(
                    "Kyuubi TLS certificate verification disabled for %s:%s",
                    host,
                    port,
                )
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            socket = TSSLSocket(
                host=host,
                port=port,
                ssl_context=context,
                server_hostname=host,
            )
        socket.setTimeout(_SOCKET_TIMEOUT_SECONDS * 1000)

        username = str(self._settings["username"])
        return thrift_sasl.TSaslClientTransport(
            lambda: get_installed_sasl(
                host=host,
                sasl_auth="PLAIN",
                service=None,
                username=username,
                password=password,
            ),
            "PLAIN",
            socket,
        )

    def _open(self) -> "Connection":
        """Open a Kyuubi session, waiting out the Spark engine's cold start."""
        from pyhive import hive

        password = self._password()
        transport = self._build_transport(password)
        started = time.monotonic()
        connection = hive.Connection(
            thrift_transport=transport,
            username=str(self._settings["username"]),
        )
        logger.info(
            "Opened Kyuubi session on %s:%s in %.1fs",
            self._settings["host"],
            self._settings["port"],
            time.monotonic() - started,
        )
        return connection

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        """Yield a cursor on the shared session, reconnecting once if it is dead.

        The session is serialised behind a lock: a Kyuubi/Thrift connection
        multiplexes a single socket, so concurrent cursors would interleave
        frames and corrupt the stream.
        """
        with self._lock:
            if self._connection is None:
                self._connection = self._open()
            try:
                cursor = self._connection.cursor()
            except Exception:
                # A session dropped by an idle-timed-out engine only surfaces
                # when it is next used, so rebuild once before giving up.
                logger.info("Kyuubi session unusable; reconnecting")
                self._reset_connection()
                self._connection = self._open()
                cursor = self._connection.cursor()
            try:
                yield cursor
            finally:
                try:
                    cursor.close()
                except Exception:
                    logger.debug("Failed to close Kyuubi cursor", exc_info=True)

    def _reset_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                logger.debug("Failed to close Kyuubi session", exc_info=True)
            self._connection = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.

        Kyuubi executes Spark SQL, so the agent should generate Spark SQL.
        """
        return "spark"

    @property
    def database_name(self) -> str:
        return self._catalog

    def qualify(self, schema: Optional[str], table: str) -> str:
        """Prepend the bound catalog: Spark names are ``catalog.schema.table``.

        The base two-level default would resolve against ``spark_catalog``, not
        the Iceberg catalog this connection binds, so every probe would raise
        TABLE_OR_VIEW_NOT_FOUND. Matches how :meth:`get_tables` and the
        ``DESCRIBE TABLE`` pass already qualify.

        Raises:
            ValueError: *schema* is missing. Spark reads a two-part name as
                ``schema.table``, so emitting ``catalog.table`` would put the
                catalog in the schema position and quietly resolve to
                something else -- every table here lives in a schema, so an
                absent one is a bug worth surfacing.
        """
        if not schema:
            raise ValueError(
                f"Kyuubi requires a schema to qualify {table!r} "
                f"in catalog {self._catalog!r}"
            )
        return _qualified(self._catalog, schema, table)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @staticmethod
    def _cancel(cursor: Any) -> None:
        """Ask the server to stop a statement we have given up on."""
        try:
            cursor.cancel()
        except Exception:
            logger.debug("Failed to cancel Kyuubi statement", exc_info=True)

    def _await_completion(
        self, cursor: Any, label: str, started: float, timeout_s: int | None
    ) -> None:
        """Poll an async statement until it leaves the pending states.

        Statements run with ``async_=True`` so the client is never parked in
        one blocking Thrift call. A synchronous ``ExecuteStatement`` holds the
        socket open for the whole query, which makes the 900s socket timeout
        the only backstop and leaves no way to tell a statement queued for a
        Spark engine from one that is scanning: both are silence. Polling costs
        a cheap round trip and reports the operation state, so ``PENDING`` and
        ``RUNNING`` show up in the log as they happen.

        Raises:
            KyuubiQueryTimeout: *timeout_s* elapsed; the statement is cancelled.
            RuntimeError: the server ended the statement in a failed state.
        """
        from TCLIService.ttypes import TOperationState

        pending = {
            TOperationState.INITIALIZED_STATE,
            TOperationState.PENDING_STATE,
            TOperationState.RUNNING_STATE,
        }
        names = TOperationState._VALUES_TO_NAMES
        interval = _POLL_INTERVAL_START_SECONDS
        last_state: int | None = None
        last_logged = 0.0
        while True:
            status = cursor.poll()
            state = status.operationState
            elapsed = time.perf_counter() - started
            # A fast statement says everything it needs to in its summary line;
            # only a slow one is worth narrating while it is still going.
            if elapsed >= _SLOW_QUERY_SECONDS and (
                state != last_state
                or elapsed - last_logged >= _PROGRESS_LOG_INTERVAL_SECONDS
            ):
                logger.info(
                    "kyuubi: %s after %.1fs — %s",
                    names.get(state, state),
                    elapsed,
                    label,
                )
                last_logged = elapsed
            last_state = state
            if state == TOperationState.FINISHED_STATE:
                return
            if state not in pending:
                raise RuntimeError(
                    f"Kyuubi statement ended in {names.get(state, state)}: "
                    f"{getattr(status, 'errorMessage', None) or label}"
                )
            if timeout_s is not None and elapsed >= timeout_s:
                self._cancel(cursor)
                raise KyuubiQueryTimeout(
                    f"Kyuubi statement exceeded {timeout_s}s in state "
                    f"{names.get(state, state)} and was cancelled: {label}"
                )
            if timeout_s is not None:
                interval = min(interval, max(timeout_s - elapsed, 0.0))
            time.sleep(interval)
            interval = min(
                max(interval, _POLL_INTERVAL_START_SECONDS) * 2,
                _POLL_INTERVAL_MAX_SECONDS,
            )

    def _execute_once(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        wait_seconds: float = 0.0,
        timeout_s: int | None = None,
    ) -> pd.DataFrame:
        # A statement is silent until it returns, and a cold Spark engine or a
        # wide scan can hold the socket for the full 900s timeout -- which in
        # the logs is indistinguishable from a hang. Timing each phase turns
        # "ingestion is stuck" into "it is 900s into SELECT * FROM raw.elk_log".
        # ``wait`` is time spent queued behind another statement: the session is
        # serialised behind a lock, so one slow query stalls every caller behind
        # it, and that queueing is otherwise invisible.
        label = " ".join(sql.split())[:110]
        started = time.perf_counter()
        with self._cursor() as cursor:
            connected = time.perf_counter()
            if parameters:
                cursor.execute(sql, parameters, async_=True)
            else:
                cursor.execute(sql, async_=True)
            self._await_completion(cursor, label, connected, timeout_s)
            executed = time.perf_counter()
            if cursor.description is None:
                logger.info(
                    "kyuubi: %.2fs (wait %.2fs, connect %.2fs, execute %.2fs) "
                    "— no rows — %s",
                    wait_seconds + executed - started,
                    wait_seconds,
                    connected - started,
                    executed - connected,
                    label,
                )
                return pd.DataFrame()
            # pyhive reports names as ``table.column``; keep only the column so
            # result keys match what the caller's SQL asked for.
            columns = [str(desc[0]).split(".")[-1] for desc in cursor.description]
            rows = cursor.fetchall()
            logger.info(
                "kyuubi: %.2fs (wait %.2fs, connect %.2fs, execute %.2fs, "
                "fetch %.2fs) — %d row(s) — %s",
                wait_seconds + time.perf_counter() - started,
                wait_seconds,
                connected - started,
                executed - connected,
                time.perf_counter() - executed,
                len(rows),
                label,
            )
            return pd.DataFrame(rows, columns=columns)

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: int | None = None,
    ) -> pd.DataFrame:
        """Run *sql*, reopening the session once if it has gone away.

        *timeout_s* caps how long the statement may stay pending before it is
        cancelled and :class:`KyuubiQueryTimeout` is raised. It is off by
        default: a metadata scan over a large catalog legitimately takes
        minutes. Callers whose query should be cheap — profiling samples, say —
        pass a cap so a warehouse that is not answering costs seconds rather
        than the full socket timeout. A timeout is never retried.

        The retry lives here rather than around ``cursor()`` because
        ``hive.Connection.cursor()`` only constructs a local object -- it never
        touches the socket, so a session the server has already discarded looks
        healthy until a statement is sent. Kyuubi retires an idle Spark engine
        after a few hours, so any connector cached longer than that hits this on
        its next query.

        A retired session surfaces in one of two shapes, and both have to be
        handled:

        * the socket is gone, and the write fails with ``BrokenPipeError`` or a
          ``TTransportException``; or
        * the Kyuubi server is still up and answers normally, reporting
          ``Invalid SessionHandle`` for the handle it no longer knows. That
          arrives as an ordinary driver error on a perfectly healthy
          connection, so it is recognised by message rather than by type.

        Missing the second shape is not a degraded retry but a permanent
        failure: nothing else clears ``self._connection``, so a connector cached
        in a long-lived worker would keep replaying the dead handle for every
        subsequent query until the process restarted.

        A statement the server actually rejected is a real SQL error and is
        raised unchanged. Retrying is safe because the failed statement never
        reached an engine, so re-running it cannot double-apply anything.
        """
        queued = time.perf_counter()
        with self._lock:
            wait_seconds = time.perf_counter() - queued
            # A failed statement is the one whose duration matters most -- a
            # read timeout costs the full socket timeout before it raises -- so
            # every exit path reports how long it burned.
            started = time.perf_counter()
            try:
                return self._execute_once(
                    sql, parameters, wait_seconds=wait_seconds, timeout_s=timeout_s
                )
            except KyuubiQueryTimeout:
                # The caller asked for a bound and got it. Reopening the
                # session would not make the engine any faster.
                raise
            except _transport_errors() as exc:
                logger.info(
                    "Kyuubi session lost after %.2fs (%s); reopening and retrying once",
                    time.perf_counter() - started,
                    type(exc).__name__,
                )
            except Exception as exc:
                if not is_session_lost(exc, sql):
                    logger.info(
                        "kyuubi: failed after %.2fs (%s) — %s",
                        time.perf_counter() - started,
                        type(exc).__name__,
                        " ".join(sql.split())[:110],
                    )
                    raise
                logger.info(
                    "Kyuubi rejected the session handle after %.2fs (%s); "
                    "reopening and retrying once",
                    time.perf_counter() - started,
                    type(exc).__name__,
                )
            self._reset_connection()
            return self._execute_once(
                sql, parameters, wait_seconds=wait_seconds, timeout_s=timeout_s
            )

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def _keep_schema(self, schema: str) -> bool:
        return self._schema_filter is None or schema.lower() in self._schema_filter

    def get_schemas(self) -> list[str]:
        """List the catalog's databases.

        Used by the connection UI to let the user pick which schemas to ingest,
        and by the connection test -- it runs a real query, so it proves the
        credentials, the TLS chain, and the engine all work.
        """
        frame = self.execute(f"SHOW DATABASES IN {_quoted_identifier(self._catalog)}")
        if frame.empty:
            return []
        return [str(name) for name in frame.iloc[:, 0].tolist()]

    def _ingested_schemas(self) -> list[str]:
        return [s for s in self.get_schemas() if self._keep_schema(s)]

    def _table_names(self, schema: str) -> list[str]:
        """Return the table names in *schema*, or an empty list if unreadable."""
        try:
            frame = self.execute(f"SHOW TABLES IN {_qualified(self._catalog, schema)}")
        except Exception:
            logger.exception("Failed to list Kyuubi tables in %s", schema)
            return []
        if frame.empty:
            return []
        # SHOW TABLES yields (namespace, tableName, isTemporary).
        column = "tableName" if "tableName" in frame.columns else frame.columns[1]
        return [str(name) for name in frame[column].tolist()]

    def _view_names(self, schema: str) -> set[str]:
        try:
            frame = self.execute(f"SHOW VIEWS IN {_qualified(self._catalog, schema)}")
        except Exception:
            # Not every catalog implements SHOW VIEWS; absence is not an error.
            logger.debug("SHOW VIEWS unsupported in %s", schema, exc_info=True)
            return set()
        if frame.empty:
            return set()
        column = "viewName" if "viewName" in frame.columns else frame.columns[1]
        return {str(name) for name in frame[column].tolist()}

    def _introspect(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
        """Describe every ingested table once, returning the tables and columns.

        Tables that cannot be described are dropped from *both* frames rather
        than reported without columns. Ingestion groups columns by the schemas it
        finds in the tables frame, so a schema whose tables all fail to describe
        would otherwise reach the graph writer with an empty column set and
        abort the whole run -- one unreadable table taking every other schema
        down with it. ``nvdp.system.table_creation_locks`` does exactly that:
        it is listed by ``SHOW TABLES`` but ``DESCRIBE`` returns Nessie 403.

        This is the same contract
        :func:`gsf.catalog.extract._validate_relation_column_coverage` enforces
        for every connector: never report a relation you have no columns for.
        Dropping is right for *one* unreadable table, but it cannot be right for
        all of them -- a dead session or a revoked catalog grant fails every
        ``DESCRIBE`` alike, and silently dropping the lot yields an empty catalog
        that looks exactly like a database with nothing in it. So if tables were
        listed and not one could be described, that is raised rather than
        dropped.

        Results are cached for the connector's lifetime: introspection is a read
        of slow-moving metadata, and each DESCRIBE is a separate Spark round
        trip.
        """
        with self._lock:
            if self._introspection is not None:
                return self._introspection

            table_rows: list[dict[str, str]] = []
            column_rows: list[dict[str, Any]] = []
            views_by_schema: dict[str, set[str]] = {}
            skipped: list[str] = []

            for schema in self._ingested_schemas():
                views = self._view_names(schema)
                kept_views: set[str] = set()
                for table in self._table_names(schema):
                    columns = self._describe(schema, table)
                    if not columns:
                        skipped.append(f"{schema}.{table}")
                        continue
                    if table in views:
                        kept_views.add(table)
                    table_rows.append(
                        {
                            "table_schema": schema,
                            "table_name": table,
                            "table_type": "VIEW" if table in views else "BASE TABLE",
                        }
                    )
                    for position, (name, data_type) in enumerate(columns, start=1):
                        column_rows.append(
                            {
                                "table_schema": schema,
                                "table_name": table,
                                "column_name": name,
                                "data_type": data_type,
                                # Spark reports nullability only in the table's
                                # Iceberg metadata, not in DESCRIBE.
                                "is_nullable": "YES",
                                "ordinal_position": position,
                            }
                        )
                views_by_schema[schema] = kept_views

            if skipped and not table_rows:
                raise IncompleteCatalogExtractionError(
                    f"Kyuubi listed {len(skipped)} relation(s) and could not describe "
                    f"any of them: {', '.join(skipped[:10])}"
                    f"{'' if len(skipped) <= 10 else f' (+{len(skipped) - 10} more)'}. "
                    "Refusing to ingest an empty catalog."
                )
            if skipped:
                logger.warning(
                    "Skipped %d Kyuubi table(s) whose columns could not be read: %s",
                    len(skipped),
                    ", ".join(skipped),
                )

            tables = (
                pd.DataFrame(table_rows)
                if table_rows
                else pd.DataFrame(columns=["table_schema", "table_name", "table_type"])
            )
            columns_frame = (
                pd.DataFrame(column_rows)
                if column_rows
                else pd.DataFrame(columns=_COLUMN_SCHEMA)
            )
            self._introspection = (tables, columns_frame, views_by_schema)
            return self._introspection

    def get_tables(self) -> pd.DataFrame:
        return self._introspect()[0].copy()

    def _describe(self, schema: str, table: str) -> list[tuple[str, str]]:
        """Return ``(column_name, data_type)`` pairs for one table.

        Spark has no queryable ``information_schema``, so each table is described
        individually and the trailing metadata sections are discarded.
        """
        try:
            frame = self.execute(
                f"DESCRIBE TABLE {_qualified(self._catalog, schema, table)}"
            )
        except Exception as exc:  # noqa: BLE001 - one bad table must not stop the sweep
            # Kyuubi wraps the whole Spark/Nessie stack trace in the message, so
            # log the first line at warning and keep the rest for debug.
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            logger.warning(
                "Cannot describe Kyuubi table %s.%s, skipping it: %s",
                schema,
                table,
                reason[:300],
            )
            logger.debug(
                "DESCRIBE failure detail for %s.%s", schema, table, exc_info=True
            )
            return []
        if frame.empty:
            return []

        columns: list[tuple[str, str]] = []
        for record in frame.itertuples(index=False):
            values = list(record)
            name = str(values[0] or "").strip()
            if name.startswith(_DESCRIBE_TERMINATORS[0]) or not name:
                break
            data_type = str(values[1] or "").strip() if len(values) > 1 else ""
            columns.append((name, data_type))
        return columns

    def get_columns(self) -> pd.DataFrame:
        return self._introspect()[1].copy()

    def get_views(self) -> pd.DataFrame:
        # Only views that survived introspection: one the connector could not
        # describe is not reported as a table either, so listing it here would
        # reintroduce the mismatch that aborts ingestion.
        views_by_schema = self._introspect()[2]
        rows: list[dict[str, str]] = []
        for schema, views in views_by_schema.items():
            for view in sorted(views):
                definition = ""
                try:
                    frame = self.execute(
                        f"SHOW CREATE TABLE {_qualified(self._catalog, schema, view)}"
                    )
                    if not frame.empty:
                        definition = str(frame.iloc[0, 0])
                except Exception:
                    logger.exception(
                        "Failed to read Kyuubi view definition for %s.%s", schema, view
                    )
                rows.append(
                    {
                        "table_schema": schema,
                        "table_name": view,
                        "view_definition": definition,
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=["table_schema", "table_name", "view_definition"]
            )
        return pd.DataFrame(rows)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Kyuubi exposes no query history over SQL, so this is always empty."""
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_pks(self) -> pd.DataFrame:
        """Spark/Iceberg tables carry no primary keys."""
        return pd.DataFrame(columns=_PK_SCHEMA)

    def get_fks(self) -> pd.DataFrame:
        """Spark/Iceberg tables carry no foreign keys."""
        return pd.DataFrame(columns=_FK_SCHEMA)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify credentials, TLS, and that the catalog's databases are visible."""
        self.execute(f"SHOW DATABASES IN {_quoted_identifier(self._catalog)}")

    def close(self) -> None:
        with self._lock:
            self._introspection = None
            self._reset_connection()
            if self._ca_pem and self._ca_pem_is_temporary:
                try:
                    os.unlink(self._ca_pem)
                except OSError:
                    logger.debug("Failed to remove %s", self._ca_pem, exc_info=True)
                self._ca_pem = None
                self._ca_pem_is_temporary = False
