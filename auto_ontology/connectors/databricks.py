# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Databricks connector implementing the NeMo-Retriever SQLDatabase ABC."""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
from databricks import sql
from databricks.sql.client import Connection
from databricks.sql.exc import Error
from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

from auto_ontology.connectors.db_errors import is_session_lost

logger = logging.getLogger(__name__)

# The SQL connector logs several INFO lines per operation (session opened,
# session closed, "HTTP Response with status code 200" from both the thrift
# client and the retry policy). This connector opens a connection per operation,
# so at INFO that is 4-6 lines of noise per query with no diagnostic value.
# Warnings and errors — including retry/timeout warnings — still come through.
logging.getLogger("databricks.sql").setLevel(logging.WARNING)

# Which credential a connector is running under, logged with every statement.
# "SSO federation" means the caller's SSO token was exchanged for a Databricks
# token, so the query carries that user's own Unity Catalog grants.
AUTH_SSO_FEDERATION = "the signed-in user (SSO federation)"
AUTH_STORED_TOKEN = "the stored access token"


def _quoted_identifier(name: str) -> str:
    """Return a Databricks-quoted identifier."""
    return f"`{name.replace('`', '``')}`"


# Constraint definitions as DESCRIBE TABLE EXTENDED reports them, e.g.
#   PRIMARY KEY (`order_id`)
#   FOREIGN KEY (`customer_id`) REFERENCES `cat`.`sch`.`customers` (`id`)
_PRIMARY_KEY_RE = re.compile(r"PRIMARY\s+KEY\s*\(([^)]*)\)", re.IGNORECASE)
_FOREIGN_KEY_RE = re.compile(
    r"FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+([^\s(]+)\s*\(([^)]*)\)",
    re.IGNORECASE,
)
_SCOPE_PERMISSION_ERROR_RE = re.compile(
    r"(?:PERMISSION_DENIED|INSUFFICIENT_PERMISSIONS).*?"
    r"(?:USE\s+CATALOG|USE\s+SCHEMA|does\s+not\s+have\s+USE)",
    re.IGNORECASE | re.DOTALL,
)


def _is_scope_permission_error(error: BaseException) -> bool:
    """Whether one metadata failure proves the whole catalog/schema is unreadable."""
    return bool(_SCOPE_PERMISSION_ERROR_RE.search(str(error)))


def _is_statement_timeout(error: BaseException) -> bool:
    """Whether the warehouse cancelled a statement at its ``statement_timeout``.

    Databricks reports it as error class ``QUERY_EXECUTION_TIMEOUT_EXCEEDED``
    (SQLSTATE ``57KD0``) inside an ordinary ``ServerOperationError``.
    """
    text = str(error)
    return "QUERY_EXECUTION_TIMEOUT_EXCEEDED" in text or "57KD0" in text


def _identifier_list(raw: str) -> list[str]:
    """Split a parenthesised identifier list, stripping backticks and spaces."""
    return [part.strip().strip("`").strip() for part in raw.split(",") if part.strip()]


def _split_qualified(reference: str) -> tuple[str, str]:
    """Split ``catalog.schema.table`` (any qualification) into ``(schema, table)``."""
    parts = [part.strip().strip("`") for part in reference.split(".") if part.strip()]
    if not parts:
        return "", ""
    table = parts[-1]
    schema = parts[-2] if len(parts) > 1 else ""
    return schema, table


def _parse_connection_string(
    connection_string: str,
) -> tuple[dict[str, Any], str, str]:
    """Parse a Databricks URL into connector kwargs, catalog, and auth mode.

    Expected format::

        databricks://token:ACCESS_TOKEN@HOST/CATALOG?http_path=SQL_HTTP_PATH[&auth=sso]
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "databricks":
        raise ValueError(f"Not a Databricks URL: {connection_string}")

    if not parsed.hostname:
        raise ValueError(
            "Databricks connection string requires a server hostname, e.g. "
            "databricks://token:access-token@workspace.cloud.databricks.com"
            "/main?http_path=%2Fsql%2F1.0%2Fwarehouses%2Fwarehouse-id"
        )

    if parsed.password is None:
        raise ValueError("Databricks connection string requires an access token")
    access_token = unquote(parsed.password)
    if not access_token:
        raise ValueError(
            "Databricks connection string requires a non-empty access token"
        )

    catalog = unquote(parsed.path.lstrip("/"))
    if not catalog:
        raise ValueError("Databricks connection string requires a catalog name")

    query = parse_qs(parsed.query)
    http_path = query.get("http_path", [None])[0]
    if not http_path:
        raise ValueError(
            "Databricks connection string requires ?http_path=/sql/1.0/warehouses/..."
        )

    auth_mode = (
        AUTH_SSO_FEDERATION
        if query.get("auth", [""])[0].strip().lower() == "sso"
        else AUTH_STORED_TOKEN
    )

    return (
        {
            "server_hostname": parsed.hostname,
            "http_path": unquote(http_path),
            "access_token": access_token,
            "catalog": catalog,
            # The connector fetches a telemetry feature flag on every connect.
            # That endpoint is unreachable from some networks and stalls for
            # 30s per attempt with retries, so a connect can take 60s+. We do
            # not use the telemetry, and disabling it skips the fetch entirely.
            "enable_telemetry": False,
        },
        catalog,
        auth_mode,
    )


class DatabricksDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by Databricks SQL."""

    # Advertises that ``execute`` honours ``timeout_s``; callers that want a cap
    # check this rather than assuming every connector supports one.
    supports_statement_timeout = True

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        (
            self._connect_kwargs,
            self._database_name,
            self._auth_mode,
        ) = _parse_connection_string(connection_string)
        self._schema_filter: set[str] | None = (
            {schema.casefold() for schema in schemas if schema.strip()}
            if schemas
            else None
        )
        # Original casing, kept for the SQL predicate. ``_schema_filter`` is case-folded
        # for the client-side backstop, but a folded name can only be matched in SQL by
        # wrapping the column in lower(), which is exactly what must be avoided here —
        # see :meth:`_schema_predicate`.
        self._schema_names: list[str] = sorted(
            {schema.strip() for schema in schemas if schema.strip()}
            if schemas
            else set()
        )
        # Columns/PKs/FKs from one DESCRIBE EXTENDED pass; see :meth:`_describe_pass`.
        self._describe_cache: tuple[pd.DataFrame, ...] | None = None
        # Raw SHOW TABLES result from get_tables(); reused by _describe_pass to avoid
        # a second round-trip to the warehouse for the same metadata.
        self._show_tables_cache: pd.DataFrame | None = None
        # (schema, table) pairs _describe_pass came back with columns for. Anything
        # listed but absent from this set is withheld from get_tables()/get_views();
        # see :meth:`_describe_pass`.
        self._described_relations: set[tuple[str, str]] = set()
        # Connection held open across a batch of statements; see :meth:`reuse_connection`.
        self._shared_connection: Connection | None = None

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "databricks"

    @property
    def database_name(self) -> str:
        return self._database_name

    def qualify(self, schema: Optional[str], table: str) -> str:
        """Prepend the bound catalog: Unity Catalog names are three-level.

        ``_database_name`` is the catalog here (see ``SHOW TABLES IN
        <catalog>.<schema>`` in :meth:`get_tables`), so the base two-level
        default would drop it and probes would miss the catalog entirely.

        Raises:
            ValueError: *schema* is missing. A two-part name is read as
                ``schema.table``, so emitting ``catalog.table`` would put the
                catalog in the schema position and quietly resolve elsewhere.
        """
        if not schema:
            raise ValueError(
                f"Databricks requires a schema to qualify {table!r} "
                f"in catalog {self._database_name!r}"
            )
        parts = (self._database_name, schema, table)
        return ".".join(_quoted_identifier(p) for p in parts)

    @property
    def auth_mode(self) -> str:
        """Which credential this connector runs under, for logging.

        Either :data:`AUTH_SSO_FEDERATION` (the caller's SSO token was exchanged
        for a Databricks token) or :data:`AUTH_STORED_TOKEN`.
        """
        return self._auth_mode

    @contextmanager
    def reuse_connection(self) -> Iterator[None]:
        """Hold one connection open for every statement issued inside the block.

        Opening a connection is the single most expensive and least reliable part of
        talking to this warehouse: ~0.9s at best, and measured hanging for 333s — and
        sometimes not completing at all. Introspection issues one statement per table,
        so a connection per statement turns that risk into a near-certainty over a
        hundred-odd calls.

        Ingestion wraps its whole extraction in this, so the entire run pays for one
        connect. Statements that need their own server-side timeout still open a
        dedicated connection, since the cap is fixed per session.
        """
        if self._shared_connection is not None:
            # Already inside a block; the outermost one owns the connection.
            yield
            return
        self._shared_connection = sql.connect(**self._connect_kwargs)
        try:
            yield
        finally:
            # Whatever is held now, which is not necessarily what was opened
            # above: a statement that met a discarded session replaced it.
            held = self._shared_connection
            self._shared_connection = None
            if held is not None:
                try:
                    held.close()
                except Exception:  # noqa: BLE001 - a failed close must not fail the run
                    logger.debug(
                        "databricks: shared connection close failed", exc_info=True
                    )

    def _reopen_shared_connection(self) -> Connection:
        """Replace the connection held by :meth:`reuse_connection`, returning it.

        Stored back on the instance rather than handed out alone, because the
        statements after this one have to reach a live session too, and the
        enclosing block's ``finally`` is what eventually closes it.
        """
        stale = self._shared_connection
        self._shared_connection = None
        if stale is not None:
            try:
                stale.close()
            except Exception:  # noqa: BLE001 - a failed close must not fail the run
                logger.debug(
                    "databricks: stale shared connection close failed", exc_info=True
                )
        self._shared_connection = sql.connect(**self._connect_kwargs)
        return self._shared_connection

    @contextmanager
    def _connect(self, timeout_s: int | None = None) -> Iterator[Connection]:
        # Inside reuse_connection, statements share that connection and must not close
        # it. A per-statement cap needs its own session, so it opens one regardless.
        if self._shared_connection is not None and timeout_s is None:
            yield self._shared_connection
            return

        kwargs = dict(self._connect_kwargs)
        if timeout_s is not None:
            # Server-side cap: Databricks cancels the statement itself and
            # returns an error, so a runaway query cannot pin the caller.
            # It bounds execution only — connect and warehouse scheduling
            # happen outside it.
            kwargs["session_configuration"] = {"statement_timeout": int(timeout_s)}
        connection = sql.connect(**kwargs)
        try:
            yield connection
        finally:
            connection.close()

    def _schema_match(self, column: str) -> str | None:
        """Match *column* against the ingestion allowlist, or ``None`` when unset.

        The allowlist has to reach the server: ``information_schema`` on a large catalog
        (kdc_ca1 carries ~1857 schemas) is punishing to scan in full, and filtering only
        in pandas means the whole catalog is still read and shipped before all but a
        couple of schemas are discarded.

        The comparison uses the column **bare** — no ``lower()`` around it. Wrapping the
        column in a function makes it non-sargable, so Databricks cannot use metadata
        pruning and falls back to scanning every schema and filtering after the fact,
        which defeats the point of pushing the filter down. The cost is exact-case
        matching, which is fine: these names come from the connection's own schema list,
        captured from Databricks itself, and :meth:`_filter_by_schema` still runs
        case-insensitively afterwards as a backstop.

        One schema uses ``=`` rather than a one-element ``IN`` — the simplest form, and
        the one a planner is most likely to prune on.
        """
        if not self._schema_names:
            return None
        # Names come from the connection's own allowlist, not from request input; quotes
        # are escaped so a stray one cannot break out of the literal.
        quoted = ["'" + name.replace("'", "''") + "'" for name in self._schema_names]
        if len(quoted) == 1:
            return f"{column} = {quoted[0]}"
        return f"{column} IN ({', '.join(quoted)})"

    def _schema_condition(self, column: str = "table_schema") -> str:
        """The full ``WHERE`` condition selecting which schemas to introspect.

        With an allowlist this is *only* the allowlist match: excluding
        ``information_schema`` as well would be redundant (an allowlist never names it)
        and leaves the planner a second predicate to reason about for no benefit. Without
        one, the scan is catalog-wide and Databricks' own metadata schema is excluded.
        """
        match = self._schema_match(column)
        return match if match is not None else f"{column} != 'information_schema'"

    def _schema_predicate(self, column: str = "table_schema") -> str:
        """``AND``-prefixed allowlist match for queries that already have a ``WHERE``.

        Used by the constraint queries, which filter on ``constraint_type`` and have no
        ``information_schema`` exclusion to fold into. Empty when no allowlist is set.
        """
        match = self._schema_match(column)
        return f"AND {match}" if match is not None else ""

    def _filter_by_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (
            self._schema_filter is None
            or frame.empty
            or "table_schema" not in frame.columns
        ):
            return frame
        return frame[
            frame["table_schema"].astype(str).str.casefold().isin(self._schema_filter)
        ]

    def _run(
        self,
        connection: Connection,
        sql_text: str,
        parameters: Optional[list] = None,
        *,
        connect_seconds: float = 0.0,
    ) -> pd.DataFrame:
        """Run one statement on an already-open *connection*.

        Split out from :meth:`execute` so a batch of statements can share a single
        connection: opening one costs the best part of a second, and occasionally stalls
        for minutes, so paying that per statement dominates any per-table loop.
        """
        # Metadata scans are silent and can run for minutes, which is indistinguishable
        # from a hang in the logs. Timing each phase (connect / execute / fetch) turns
        # "ingestion is stuck" into "it is 90s into fetching information_schema.columns".
        label = " ".join(sql_text.split())[:110]
        started = time.perf_counter()
        with connection.cursor() as cursor:
            cursor.execute(sql_text, parameters)
            executed = time.perf_counter()
            if cursor.description is None:
                logger.info(
                    "databricks: %.2fs (connect %.2fs, execute %.2fs) — no rows — %s",
                    connect_seconds + executed - started,
                    connect_seconds,
                    executed - started,
                    label,
                )
                return pd.DataFrame()
            columns = [description[0].lower() for description in cursor.description]
            rows = cursor.fetchall()
            logger.info(
                "databricks: %.2fs (connect %.2fs, execute %.2fs, fetch %.2fs) — "
                "%d row(s) — %s",
                connect_seconds + time.perf_counter() - started,
                connect_seconds,
                executed - started,
                time.perf_counter() - executed,
                len(rows),
                label,
            )
            return pd.DataFrame(rows, columns=columns)

    def execute(
        self,
        sql_text: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: int | None = None,
    ) -> pd.DataFrame:
        """Run *sql_text* on its own connection, optionally capped at *timeout_s*.

        Ingestion leaves the cap off: metadata scans over a large
        ``information_schema`` legitimately take minutes.

        Inside :meth:`reuse_connection` the session outlives the statement, so
        the warehouse may have discarded it in the meantime. That comes back as
        an ordinary statement error on a healthy connection, so it is reopened
        and retried once rather than failing the whole batch.
        """
        started = time.perf_counter()
        with self._connect(timeout_s) as connection:
            connect_seconds = time.perf_counter() - started
            shared = connection is self._shared_connection
            try:
                return self._run(
                    connection, sql_text, parameters, connect_seconds=connect_seconds
                )
            except Exception as exc:
                if timeout_s is not None and _is_statement_timeout(exc):
                    raise StatementTimeout(timeout_s) from exc
                # Only a shared connection outlives its statement, so only it can
                # be holding a session the warehouse has since dropped. A
                # per-statement connection was opened moments ago, and reopening
                # it would just repeat whatever went wrong.
                if not shared or not is_session_lost(exc, sql_text):
                    raise
                logger.info(
                    "databricks: shared session was rejected; "
                    "reopening and retrying once"
                )
                return self._run(
                    self._reopen_shared_connection(),
                    sql_text,
                    parameters,
                    connect_seconds=connect_seconds,
                )

    def get_schemas(self) -> list[str]:
        frame = self.execute(
            f"SHOW SCHEMAS IN {_quoted_identifier(self._database_name)}"
        )
        if frame.empty:
            return []
        return [str(name) for name in frame.iloc[:, 0].tolist()]

    def _single_schema(self) -> str | None:
        """The schema being ingested, when the allowlist names exactly one.

        A single schema unlocks ``SHOW TABLES`` / ``SHOW COLUMNS``, which are metadata
        lookups scoped to that schema rather than queries against ``information_schema`` —
        a view spanning the entire catalog, and punishingly slow on a catalog with
        thousands of schemas.
        """
        return self._schema_names[0] if len(self._schema_names) == 1 else None

    def _tables_via_show(self, schema: str) -> pd.DataFrame:
        """Table list from ``SHOW TABLES IN <catalog>.<schema>``.

        ``SHOW TABLES`` returns both tables and views with no type column. We run
        ``SHOW VIEWS`` in the same schema to build a view-name set, then classify
        each row correctly. This avoids storing views as ``BASE_TABLE``,
        which would break any downstream consumer that discriminates on node type.
        """
        catalog = _quoted_identifier(self._database_name)
        schema_quoted = _quoted_identifier(schema)
        tables_raw = self.execute(f"SHOW TABLES IN {catalog}.{schema_quoted}")
        views_raw = self.execute(f"SHOW VIEWS IN {catalog}.{schema_quoted}")
        view_names: set[str] = set()
        if not views_raw.empty and "viewname" in views_raw.columns:
            view_names = set(views_raw["viewname"].astype(str))
        return self._tables_frame(tables_raw, schema, view_names)

    @staticmethod
    def _tables_frame(
        frame: pd.DataFrame, schema: str, view_names: set[str] | None = None
    ) -> pd.DataFrame:
        """Shape a ``SHOW TABLES`` result into the tables contract.

        *view_names*, when provided, is used to classify rows — those whose
        ``tableName`` appears in the set receive ``TableTypes.VIEW`` instead of
        ``TableTypes.BASE_TABLE``.
        """
        if frame.empty or "tablename" not in frame.columns:
            return pd.DataFrame(columns=["table_schema", "table_name", "table_type"])
        names = frame["tablename"].astype(str)
        if view_names:
            table_type = names.map(
                lambda n: TableTypes.VIEW if n in view_names else TableTypes.BASE_TABLE
            )
        else:
            table_type = TableTypes.BASE_TABLE
        return pd.DataFrame(
            {
                # SHOW TABLES reports the schema it listed; fall back to the requested one.
                "table_schema": frame.get("database", schema).astype(str),
                "table_name": names,
                "table_type": table_type,
            }
        )

    @staticmethod
    def _describe_columns_only(described: pd.DataFrame) -> pd.DataFrame:
        """Trim ``DESCRIBE`` output to the leading column rows.

        Databricks appends metadata sections (``# Partition Information``, ``# Detailed
        Table Information``, ``# Constraints``) after the columns, separated by a blank
        row.
        """
        if described.empty or "col_name" not in described.columns:
            return pd.DataFrame()
        names = described["col_name"].astype(str)
        trailing = names.str.strip().eq("") | names.str.startswith("#")
        if trailing.any():
            described = described.loc[: trailing.idxmax() - 1]
        return described

    @staticmethod
    def _describe_constraints(described: pd.DataFrame) -> list[tuple[str, str]]:
        """``(name, definition)`` pairs from the ``# Constraints`` section, if any.

        ``DESCRIBE TABLE EXTENDED`` lists declared constraints under a ``# Constraints``
        heading, with the constraint name in ``col_name`` and its definition in
        ``data_type``:

            pk_orders   PRIMARY KEY (`order_id`)
            fk_customer FOREIGN KEY (`customer_id`) REFERENCES `c`.`s`.`customers` (`id`)

        A table with no declared constraints has no such section, so this returns an
        empty list — which is the common case, since Unity Catalog constraints are
        informational and often not declared at all.
        """
        if described.empty or "col_name" not in described.columns:
            return []
        names = described["col_name"].astype(str).str.strip()
        marker = names.str.casefold().eq("# constraints")
        if not marker.any():
            return []

        start = int(marker.idxmax()) + 1
        section = described.loc[start:]
        pairs: list[tuple[str, str]] = []
        for name, definition in zip(
            section["col_name"].astype(str),
            section.get("data_type", pd.Series(dtype=str)).astype(str),
        ):
            name = name.strip()
            # A blank or new '#' heading ends the section.
            if not name or name.startswith("#"):
                break
            pairs.append((name, definition.strip()))
        return pairs

    _COLUMN_FIELDS = [
        "table_schema",
        "table_name",
        "column_name",
        "data_type",
        "ordinal_position",
    ]
    _PK_FIELDS = ["table_schema", "table_name", "column_name", "ordinal_position"]
    _FK_FIELDS = [
        "table_schema",
        "table_name",
        "column_name",
        "referenced_schema",
        "referenced_table",
        "referenced_column",
    ]

    def _describe_pass(self, schema: str) -> tuple[pd.DataFrame, ...]:
        """One ``DESCRIBE TABLE EXTENDED`` per table, yielding columns, PKs and FKs.

        The extended form carries both the column list and the declared constraints, so
        a single pass replaces three separate sources: two ``information_schema`` joins
        (``key_column_usage`` x ``table_constraints`` x ``referential_constraints``) that
        are the slowest statements in introspection, plus the column scan.

        Every statement shares ONE connection — opening one costs ~0.9s and has been
        measured stalling for minutes, so a connect per table would dominate the run.

        The result is cached for the run: NeMo-Retriever's extract operator calls
        ``get_columns``, ``get_pks`` and ``get_fks`` separately, and re-describing every
        table three times would triple the cost. :meth:`get_tables` clears it, and it
        runs first in every extraction, so each ingest sees fresh metadata.

        Relations this comes back empty-handed for are recorded in
        ``_described_relations`` so :meth:`get_tables` can withhold them. A relation
        listed as a table but carrying no columns is the failure
        :func:`auto_ontology.catalog.extract._validate_relation_column_coverage` rejects, and it
        rejects the *whole* ingest — so a single unreadable table must be dropped here
        rather than advertised and left for the coverage gate to trip over.
        """
        if self._describe_cache is not None:
            return self._describe_cache

        catalog = _quoted_identifier(self._database_name)
        column_frames: list[pd.DataFrame] = []
        pk_rows: list[dict[str, Any]] = []
        fk_rows: list[dict[str, Any]] = []
        described_relations: set[tuple[str, str]] = set()

        started = time.perf_counter()
        with self._connect() as connection:
            connect_seconds = time.perf_counter() - started
            if self._show_tables_cache is not None:
                # get_tables() already ran SHOW TABLES; reuse its result so we
                # don't issue a second round-trip for the same metadata.
                tables = self._show_tables_cache
            else:
                tables = self._tables_frame(
                    self._run(
                        connection,
                        f"SHOW TABLES IN {catalog}.{_quoted_identifier(schema)}",
                        connect_seconds=connect_seconds,
                    ),
                    schema,
                )

            # The schema is taken per row rather than from the allowlist entry: Unity
            # Catalog folds identifiers to lower case, so a connection configured for
            # "Sales" gets "sales" back from SHOW TABLES. Labelling columns with the
            # configured spelling while get_tables() labels tables with the reported
            # one leaves two frames that cannot be joined on (schema, table).
            for row_schema, table_name in zip(
                tables["table_schema"].astype(str), tables["table_name"].astype(str)
            ):
                qualified = (
                    f"{catalog}.{_quoted_identifier(row_schema)}"
                    f".{_quoted_identifier(table_name)}"
                )
                try:
                    described = self._run(
                        connection, f"DESCRIBE TABLE EXTENDED {qualified}"
                    )
                except Error as error:
                    # A missing USE CATALOG/SCHEMA grant invalidates every remaining
                    # relation. Stop after the first denial instead of issuing and
                    # swallowing one doomed DESCRIBE per table.
                    if _is_scope_permission_error(error):
                        logger.error(
                            "databricks: schema-scoped DESCRIBE denied for %s.%s "
                            "(%s); aborting extraction",
                            row_schema,
                            table_name,
                            type(error).__name__,
                        )
                        raise
                    # One table can still disappear mid-run or carry a narrower grant.
                    # It is dropped from the tables frame too (see the docstring), so
                    # the rest of the schema still ingests as a consistent catalog.
                    logger.warning(
                        "databricks: DESCRIBE failed for %s.%s (%s); "
                        "dropping it from the catalog",
                        row_schema,
                        table_name,
                        type(error).__name__,
                    )
                    continue

                columns = self._describe_columns_only(described)
                if columns.empty:
                    # DESCRIBE answered but named no columns. Reporting the relation
                    # anyway is the "non-empty table map, zero columns" catalog the
                    # coverage gate exists to reject.
                    logger.warning(
                        "databricks: DESCRIBE returned no columns for %s.%s; "
                        "dropping it from the catalog",
                        row_schema,
                        table_name,
                    )
                    continue

                frame = {
                    "table_schema": row_schema,
                    "table_name": table_name,
                    "column_name": columns["col_name"].astype(str),
                    "data_type": columns.get("data_type", pd.NA),
                    "ordinal_position": range(1, len(columns) + 1),
                }
                if "comment" in columns.columns:
                    # DESCRIBE reports the column comment, which normalize_columns
                    # already types as the column description.
                    frame["description"] = columns["comment"]
                column_frames.append(pd.DataFrame(frame))
                described_relations.add((row_schema, table_name))

                for _name, definition in self._describe_constraints(described):
                    self._collect_constraint(
                        definition, row_schema, table_name, pk_rows, fk_rows
                    )

        self._described_relations = described_relations
        self._describe_cache = (
            pd.concat(column_frames, ignore_index=True)
            if column_frames
            else pd.DataFrame(columns=self._COLUMN_FIELDS),
            pd.DataFrame(pk_rows, columns=self._PK_FIELDS),
            pd.DataFrame(fk_rows, columns=self._FK_FIELDS),
        )
        if pk_rows or fk_rows:
            logger.info(
                "databricks: %d primary-key and %d foreign-key column(s) declared in %s",
                len(pk_rows),
                len(fk_rows),
                schema,
            )
        return self._describe_cache

    @staticmethod
    def _collect_constraint(
        definition: str,
        schema: str,
        table_name: str,
        pk_rows: list[dict[str, Any]],
        fk_rows: list[dict[str, Any]],
    ) -> None:
        """Append the rows one constraint definition contributes."""
        primary = _PRIMARY_KEY_RE.search(definition)
        if primary:
            for position, column in enumerate(_identifier_list(primary.group(1)), 1):
                pk_rows.append(
                    {
                        "table_schema": schema,
                        "table_name": table_name,
                        "column_name": column,
                        "ordinal_position": position,
                    }
                )
            return

        foreign = _FOREIGN_KEY_RE.search(definition)
        if not foreign:
            return
        local_columns = _identifier_list(foreign.group(1))
        referenced_schema, referenced_table = _split_qualified(foreign.group(2))
        referenced_columns = _identifier_list(foreign.group(3))
        for position, column in enumerate(local_columns):
            # Composite keys pair positionally with the referenced columns.
            referenced = (
                referenced_columns[position]
                if position < len(referenced_columns)
                else ""
            )
            fk_rows.append(
                {
                    "table_schema": schema,
                    "table_name": table_name,
                    "column_name": column,
                    "referenced_schema": referenced_schema or schema,
                    "referenced_table": referenced_table,
                    "referenced_column": referenced,
                }
            )

    def _with_columns(self, listed: pd.DataFrame) -> pd.DataFrame:
        """*listed* minus the relations :meth:`_describe_pass` found no columns for."""
        if listed.empty:
            return listed
        keep = [
            (row_schema, table_name) in self._described_relations
            for row_schema, table_name in zip(
                listed["table_schema"].astype(str), listed["table_name"].astype(str)
            )
        ]
        return listed.loc[keep].reset_index(drop=True)

    def get_tables(self) -> pd.DataFrame:
        # First call of every extraction, so this is where the per-run caches reset.
        self._describe_cache = None
        self._show_tables_cache = None
        self._described_relations = set()
        schema = self._single_schema()
        if schema:
            logger.info("databricks: listing tables via SHOW TABLES IN %s", schema)
            listed = self._tables_via_show(schema)
            # Cache the raw table list so _describe_pass can reuse it without a
            # second SHOW TABLES round-trip.
            self._show_tables_cache = listed
            # Describe here rather than leaving it to get_columns(): the caller keeps
            # whatever this returns, so a relation that turns out to have no columns
            # has to be withheld now or not at all. The pass is cached, so get_columns()
            # pays nothing for it.
            self._describe_pass(schema)
            return self._with_columns(listed)

        base_table = TableTypes.BASE_TABLE
        view = TableTypes.VIEW
        materialized_view = TableTypes.MATERIALIZED_VIEW
        catalog = _quoted_identifier(self._database_name)
        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    table_schema,
                    table_name,
                    CASE table_type
                        WHEN 'VIEW' THEN '{view}'
                        WHEN 'MATERIALIZED_VIEW' THEN '{materialized_view}'
                        ELSE '{base_table}'
                    END AS table_type
                FROM {catalog}.information_schema.tables
                WHERE {self._schema_condition()}
                ORDER BY table_schema, table_name
            """)
        )

    def get_columns(self) -> pd.DataFrame:
        schema = self._single_schema()
        if schema:
            logger.info("databricks: describing columns per table in %s", schema)
            return self._describe_pass(schema)[0]

        catalog = _quoted_identifier(self._database_name)
        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    table_schema,
                    table_name,
                    column_name,
                    full_data_type AS data_type,
                    is_nullable = 'YES' AS is_nullable,
                    ordinal_position
                FROM {catalog}.information_schema.columns
                WHERE {self._schema_condition()}
                ORDER BY table_schema, table_name, ordinal_position
            """)
        )

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Query history — disabled, returns nothing.

        The ``system.query.history`` scan is another whole-workspace read that costs
        minutes on a busy workspace, and the history is only used to enrich ingestion,
        never to answer a question. Returning empty keeps the ingest moving; revisit
        with a bounded source when the rest of introspection is fast.
        """
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        catalog = _quoted_identifier(self._database_name)
        schema = self._single_schema()
        if schema:
            # SHOW VIEWS is a metadata lookup scoped to the schema, where
            # information_schema.views is a catalog-wide read that has been measured
            # taking minutes here. It reports no definition, which nothing consumes
            # today — recovering one would mean SHOW CREATE TABLE per view.
            frame = self.execute(
                f"SHOW VIEWS IN {catalog}.{_quoted_identifier(schema)}"
            )
            if frame.empty or "viewname" not in frame.columns:
                return pd.DataFrame(
                    columns=["table_schema", "table_name", "view_definition"]
                )
            views = pd.DataFrame(
                {
                    # SHOW VIEWS reports the namespace it listed; fall back to the
                    # requested schema.
                    "table_schema": frame.get("namespace", schema).astype(str),
                    "table_name": frame["viewname"].astype(str),
                    "view_definition": pd.NA,
                }
            )
            if self._describe_cache is None:
                # get_views() before get_tables(): nothing has been described, so
                # there is no coverage to filter against.
                return views
            # A view get_tables() withheld must not reappear here — the writer would
            # take it as a relation with no columns.
            return self._with_columns(views)

        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    table_schema,
                    table_name,
                    view_definition
                FROM {catalog}.information_schema.views
                WHERE {self._schema_condition()}
                ORDER BY table_schema, table_name
            """)
        )

    def get_pks(self) -> pd.DataFrame:
        schema = self._single_schema()
        if schema:
            return self._describe_pass(schema)[1]

        catalog = _quoted_identifier(self._database_name)
        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    keys.table_schema,
                    keys.table_name,
                    keys.column_name,
                    keys.ordinal_position
                FROM {catalog}.information_schema.key_column_usage AS keys
                JOIN {catalog}.information_schema.table_constraints AS constraints
                  ON keys.constraint_catalog = constraints.constraint_catalog
                 AND keys.constraint_schema = constraints.constraint_schema
                 AND keys.constraint_name = constraints.constraint_name
                WHERE constraints.constraint_type = 'PRIMARY KEY'
                {self._schema_predicate("keys.table_schema")}
                ORDER BY
                    keys.table_schema,
                    keys.table_name,
                    keys.ordinal_position
            """)
        )

    def get_fks(self) -> pd.DataFrame:
        schema = self._single_schema()
        if schema:
            return self._describe_pass(schema)[2]

        catalog = _quoted_identifier(self._database_name)
        return self._filter_by_schema(
            self.execute(f"""
                SELECT
                    foreign_keys.table_schema,
                    foreign_keys.table_name,
                    foreign_keys.column_name,
                    referenced_keys.table_schema AS referenced_schema,
                    referenced_keys.table_name AS referenced_table,
                    referenced_keys.column_name AS referenced_column
                FROM {catalog}.information_schema.key_column_usage AS foreign_keys
                JOIN {catalog}.information_schema.table_constraints AS constraints
                  ON foreign_keys.constraint_catalog = constraints.constraint_catalog
                 AND foreign_keys.constraint_schema = constraints.constraint_schema
                 AND foreign_keys.constraint_name = constraints.constraint_name
                JOIN {catalog}.information_schema.referential_constraints AS refs
                  ON foreign_keys.constraint_catalog = refs.constraint_catalog
                 AND foreign_keys.constraint_schema = refs.constraint_schema
                 AND foreign_keys.constraint_name = refs.constraint_name
                JOIN {catalog}.information_schema.key_column_usage AS referenced_keys
                  ON refs.unique_constraint_catalog =
                     referenced_keys.constraint_catalog
                 AND refs.unique_constraint_schema =
                     referenced_keys.constraint_schema
                 AND refs.unique_constraint_name = referenced_keys.constraint_name
                 AND foreign_keys.position_in_unique_constraint =
                     referenced_keys.ordinal_position
                WHERE constraints.constraint_type = 'FOREIGN KEY'
                {self._schema_predicate("foreign_keys.table_schema")}
                ORDER BY
                    foreign_keys.table_schema,
                    foreign_keys.table_name,
                    foreign_keys.ordinal_position
            """)
        )

    def ping(self) -> None:
        self.execute("SELECT current_catalog()")

    def close(self) -> None:
        """No persistent connection to close (connections are per-operation)."""
