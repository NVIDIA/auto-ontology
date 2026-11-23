# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Google BigQuery connector implementing Auto Ontology's SQLDatabase ABC.

BigQuery is reached through the official ``google-cloud-bigquery`` client, which
submits every statement as a query *job* over the REST API.

These properties of the engine shape this connector:

* **Three-level naming.** Objects are ``project.dataset.table``, and datasets
  play the role of schemas -- so the schema allowlist picks datasets, the same
  model :mod:`auto_ontology.connectors.trino` uses for a catalog. Jobs run, and
  are billed, in the URL's host project. A connection either spans that
  project's datasets, or names its datasets explicitly with repeated
  ``dataset=project.dataset`` parameters -- typically public datasets in
  ``bigquery-public-data``, which only another project can pay to query.
* **A named database.** :attr:`~BigQueryDatabase.database_name` is the URL path
  when present, else the host project. The path is what lets several
  connections bill the same project and still be routed apart.
* **Datasets resolve to their own project.** The catalog keys tables by
  dataset, so generated SQL says ``austin_311.requests``; BigQuery alone would
  look for that dataset in the billing project. Two-part names over a named
  dataset are therefore prefixed with its project before the job is sent.
* **Per-dataset ``INFORMATION_SCHEMA``.** Metadata views are scoped to one
  dataset (or to a region, which would force the user to know it). One job
  cannot read datasets in different locations, so introspection runs one query
  per dataset rather than a ``UNION`` across them.
* **Billing by bytes scanned.** ``LIMIT`` does not reduce what a query scans,
  so a profiling probe on a wide table can cost real money. Every job carries
  ``maximum_bytes_billed`` (10 GiB unless the URL overrides it); a query over
  the cap fails before it runs instead of being billed.
* **Date-sharded tables are one wildcard table.** ``events_20210101``,
  ``events_20210102``, ... are queried together as ``events_*``, so the catalog
  carries that one table, with the newest shard's columns and the
  ``_TABLE_SUFFIX`` pseudo-column. Hundreds of near-identical shards would
  otherwise crowd search, and a query naming the wildcard would match none of
  them. :meth:`~BigQueryDatabase.qualify` points the wildcard at its newest
  shard, so profiling reads one day rather than scanning every shard.
* **Unenforced keys.** BigQuery stores primary and foreign keys as metadata
  only. They are reported anyway, because they are the modeller's own statement
  of how tables join. Foreign keys are read from the table resource rather than
  ``INFORMATION_SCHEMA``: ``CONSTRAINT_COLUMN_USAGE`` does not name the
  referencing table, so the views cannot pair composite-key columns reliably.

Authentication is a service-account key carried in the URL as URL-safe base64
(``credentials=``), or a key file on disk (``credentials_file=``). With
neither, the client uses Application Default Credentials -- the workload
identity of the pod. The UI always supplies a key (see
:mod:`auto_ontology.connectors.connection_string_factory`), so falling back to
the pod's identity is reserved for operator-written ``CONNECTION_STRINGS``.

Example
-------
::

    CONNECTION_STRINGS=bigquery://my-project?location=US&credentials=<base64 key>
    CONNECTION_STRINGS=bigquery://my-project?schema=analytics&credentials_file=/secrets/sa.json
    CONNECTION_STRINGS=bigquery://my-project/austin?dataset=bigquery-public-data.austin_311
"""

from __future__ import annotations

import base64
import binascii
import concurrent.futures
import datetime as dt
import decimal
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import sqlglot
from sqlglot import exp

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout
from auto_ontology.utils.sql_identifiers import qualified_name

if TYPE_CHECKING:
    from google.cloud.bigquery import Client, QueryJobConfig

logger = logging.getLogger(__name__)

DEFAULT_MAXIMUM_BYTES_BILLED = 10 * 1024**3

# Attached to every job, so query history can skip the connector's own
# statements and billing exports can attribute them.
_JOB_LABELS = {"source": "auto-ontology"}

# Extra wall-clock allowed on top of a caller's statement timeout, so the
# server's ``job_timeout_ms`` fires first and the job is stopped rather than the
# client abandoning a job that keeps running (and billing).
_TIMEOUT_GRACE_S = 10.0

_QUERY_HISTORY_LIMIT = 10_000

_TABLE_TYPES = {
    "VIEW": TableTypes.VIEW,
    "MATERIALIZED VIEW": TableTypes.MATERIALIZED_VIEW,
}

# The clauses a first-rows read may have; anything else needs a real query.
_SAMPLE_CLAUSES = frozenset({"expressions", "from", "from_", "limit"})

_TABLE_SCHEMA = ["table_schema", "table_name", "table_type", "description"]

_COLUMN_SCHEMA = [
    "table_schema",
    "table_name",
    "column_name",
    "data_type",
    "is_nullable",
    "ordinal_position",
    "description",
]

_VIEW_SCHEMA = ["table_schema", "table_name", "view_definition"]

_QUERY_SCHEMA = ["end_time", "query_text"]

_PK_SCHEMA = ["table_schema", "table_name", "column_name", "ordinal_position"]

_FK_SCHEMA = [
    "table_schema",
    "table_name",
    "column_name",
    "referenced_schema",
    "referenced_table",
    "referenced_column",
]

# A shard ends in a 4- to 8-digit date (2017, 201708, 20170801) after a
# non-digit, so ``gsod1929`` and ``events_20201101`` qualify and
# ``icoads_core_1662_2000`` groups with nothing.
_SHARD_NAME = re.compile(r"^(?P<prefix>.*\D)(?P<suffix>\d{4,8})$")

# BigQuery's pseudo-column on a wildcard table. Listing it as a column is what
# lets the agent's catalog check accept ``WHERE _TABLE_SUFFIX BETWEEN ...``.
TABLE_SUFFIX_COLUMN = "_TABLE_SUFFIX"


def _fits(old: exp.DataType, new: exp.DataType) -> bool:
    if old.this != new.this:
        return False
    if old.this == exp.DataType.Type.STRUCT:
        fields = {f.name.casefold(): f.args.get("kind") for f in new.expressions}
        return all(
            f.args.get("kind") is not None
            and fields.get(f.name.casefold()) is not None
            and _fits(f.args["kind"], fields[f.name.casefold()])
            for f in old.expressions
        )
    if old.this == exp.DataType.Type.ARRAY and old.expressions and new.expressions:
        return _fits(old.expressions[0], new.expressions[0])
    return old.sql("bigquery") == new.sql("bigquery")


def type_fits(old: str, new: str | None) -> bool:
    """Whether a column of type *old* reads through a wildcard typed *new*.

    A ``STRUCT`` may gain fields in newer shards, which read as NULL in older
    ones; anything else must match.
    """
    if new is None:
        return False
    if old == new:
        return True
    try:
        return _fits(
            exp.DataType.build(old, dialect="bigquery"),
            exp.DataType.build(new, dialect="bigquery"),
        )
    except (sqlglot.errors.SqlglotError, ValueError):
        return False


@dataclass(frozen=True)
class ShardFamily:
    """Tables ``<prefix><date>`` that BigQuery reads together as ``<prefix>*``."""

    prefix: str
    shards: tuple[str, ...]
    #: Other tables ``<prefix>*`` also matches, which BigQuery unions in too.
    also_matches: tuple[str, ...] = ()

    @property
    def wildcard(self) -> str:
        return f"{self.prefix}*"

    @property
    def latest(self) -> str:
        """The newest shard, whose schema a wildcard query uses."""
        return self.shards[-1]

    def note(self) -> str:
        first, last = self.shards[0], self.shards[-1]
        start, end = first[len(self.prefix) :], last[len(self.prefix) :]
        note = (
            f"Wildcard table over {len(self.shards)} date-sharded tables {first} "
            f"... {last}, all with this schema. Select shards with the "
            f"{TABLE_SUFFIX_COLUMN} pseudo-column, e.g. WHERE {TABLE_SUFFIX_COLUMN} "
            f"BETWEEN '{start}' AND '{end}'."
        )
        if self.also_matches:
            note += f" The wildcard also matches {', '.join(self.also_matches)}."
        return note


def shard_families(
    names: list[str], columns: dict[str, dict[str, str]] | None = None
) -> list[ShardFamily]:
    """Group date-sharded table names into wildcard families of two shards or more.

    A wildcard query reads every shard with the newest one's schema: a column
    a shard lacks reads as NULL, but one only an older shard has cannot be
    selected, and a changed type fails the query. Given each table's
    *columns* (name to type), a shard with a column the newest does not have
    the same way stays a table of its own, listed in ``also_matches``.
    """
    unique = sorted(set(names))
    groups: dict[str, list[str]] = {}
    for name in unique:
        match = _SHARD_NAME.match(name)
        if match:
            groups.setdefault(match.group("prefix"), []).append(name)
    families = []
    for prefix, members in sorted(groups.items()):
        shards = sorted(members, key=lambda n: (len(n), n))
        if columns is not None:
            newest = {c.casefold(): t for c, t in columns.get(shards[-1], {}).items()}
            shards = [
                n
                for n in shards
                if all(
                    type_fits(t, newest.get(c.casefold()))
                    for c, t in columns.get(n, {}).items()
                )
            ]
        if len(shards) < 2:
            continue
        also = tuple(n for n in unique if n.startswith(prefix) and n not in shards)
        families.append(ShardFamily(prefix, tuple(shards), also))
    return families


def _first(query: dict[str, list[str]], key: str) -> str | None:
    """Return the first value for *key*, URL-decoded, or ``None``."""
    value = query.get(key, [None])[0]
    return unquote(value).strip() or None if value else None


def _decode_credentials(encoded: str) -> dict[str, Any]:
    """Decode a service-account key from the URL-safe base64 the URL carries.

    Raw JSON is accepted too, so a hand-written connection string does not have
    to be encoded first.
    """
    text = encoded.strip()
    if not text.startswith("{"):
        try:
            text = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode()
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError(
                "BigQuery credentials are not valid base64-encoded JSON"
            ) from exc
    try:
        info = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("BigQuery credentials are not a valid JSON key") from exc
    if not isinstance(info, dict) or info.get("type") != "service_account":
        raise ValueError("BigQuery credentials must be a service-account JSON key")
    # google-auth posts a signed assertion to the key's ``token_uri`` and puts
    # the reply in its error, so any other host would let whoever writes a
    # connection make the server call it.
    token_uri = str(info.get("token_uri") or _GOOGLE_TOKEN_URI).strip()
    if token_uri not in _GOOGLE_TOKEN_URIS:
        raise ValueError(
            "BigQuery credentials must use Google's token endpoint "
            f"({_GOOGLE_TOKEN_URI})"
        )
    return {**info, "token_uri": token_uri}


_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
_GOOGLE_TOKEN_URIS = frozenset(
    {_GOOGLE_TOKEN_URI, "https://accounts.google.com/o/oauth2/token"}
)


_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def _byte_size(text: str) -> int | None:
    """``"20GiB"`` -> bytes; ``"0"``, ``"none"`` and ``"unlimited"`` -> no cap."""
    value = text.strip().lower().replace(" ", "")
    if value in ("0", "none", "unlimited"):
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]*)", value)
    if not match or match.group(2) not in _BYTE_UNITS:
        raise ValueError(
            f"BigQuery maximum_bytes_billed must be a byte size such as "
            f"10737418240 or 20GiB, got {text!r}"
        )
    return int(float(match.group(1)) * _BYTE_UNITS[match.group(2)]) or None


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Parse a BigQuery URL into the settings this connector needs.

    Expected format::

        bigquery://PROJECT[/DATABASE][?dataset=PROJECT.DATASET&dataset=...
                           &location=...&schema=...&credentials=...
                           &credentials_file=...&maximum_bytes_billed=...]

    ``dataset`` repeats rather than taking a list because ``CONNECTION_STRINGS``
    is split on commas. A dataset given without a project lives in ``PROJECT``.
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "bigquery":
        raise ValueError("Not a BigQuery URL")
    project = unquote(parsed.netloc or "").strip()
    if not project:
        raise ValueError(
            "BigQuery connection string requires a project, e.g. bigquery://my-project"
        )
    database = unquote(parsed.path.strip("/")).strip() or project

    query = parse_qs(parsed.query)

    datasets: list[tuple[str, str]] = []
    owners: dict[str, str] = {}
    for raw in query.get("dataset", []):
        owner, _, dataset = unquote(raw).strip().rpartition(".")
        owner = owner or project
        if not dataset:
            raise ValueError(f"BigQuery dataset {raw!r} must be [project.]dataset")
        # A two-part name in SQL can only resolve to one of them.
        previous = owners.setdefault(dataset.casefold(), owner)
        if previous != owner:
            raise ValueError(
                f"BigQuery dataset {dataset!r} is named in both {previous!r} and "
                f"{owner!r}; one connection cannot hold two datasets of that name"
            )
        datasets.append((owner, dataset))

    raw_cap = _first(query, "maximum_bytes_billed")
    maximum_bytes_billed = (
        _byte_size(raw_cap) if raw_cap else DEFAULT_MAXIMUM_BYTES_BILLED
    )

    raw_credentials = _first(query, "credentials")
    return {
        "project": project,
        "database": database,
        "datasets": datasets,
        "schema": _first(query, "schema"),
        "location": _first(query, "location"),
        "credentials": (
            _decode_credentials(raw_credentials) if raw_credentials else None
        ),
        "credentials_file": _first(query, "credentials_file"),
        "maximum_bytes_billed": maximum_bytes_billed,
    }


def _query_parameter(value: Any) -> Any:
    """Bind one positional ``?`` parameter with an explicit BigQuery type.

    BigQuery rejects an untyped parameter, so the type is inferred here. ``bool``
    is tested before ``int`` because it is a subclass of it.
    """
    from google.cloud.bigquery import ScalarQueryParameter

    if value is None:
        return ScalarQueryParameter(None, "STRING", None)
    if isinstance(value, bool):
        return ScalarQueryParameter(None, "BOOL", value)
    if isinstance(value, int):
        return ScalarQueryParameter(None, "INT64", value)
    if isinstance(value, float):
        return ScalarQueryParameter(None, "FLOAT64", value)
    if isinstance(value, decimal.Decimal):
        return ScalarQueryParameter(None, "BIGNUMERIC", value)
    if isinstance(value, dt.datetime):
        kind = "TIMESTAMP" if value.tzinfo is not None else "DATETIME"
        return ScalarQueryParameter(None, kind, value)
    if isinstance(value, dt.date):
        return ScalarQueryParameter(None, "DATE", value)
    if isinstance(value, bytes):
        return ScalarQueryParameter(None, "BYTES", value)
    return ScalarQueryParameter(None, "STRING", str(value))


def _is_job_timeout(exc: BaseException) -> bool:
    """True when BigQuery stopped the job at its ``job_timeout_ms``."""
    errors = getattr(exc, "errors", None) or []
    if any(
        isinstance(error, dict) and error.get("reason") == "timeout" for error in errors
    ):
        return True
    return "timed out" in str(exc).lower()


def _description(option_value: Any) -> str | None:
    """Decode a ``TABLE_OPTIONS`` description, which arrives as a SQL literal."""
    if option_value is None or option_value is pd.NA:
        return None
    text = str(option_value)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = text.strip('"')
    return str(decoded) or None


class BigQueryDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by Google BigQuery.

    Parameters
    ----------
    connection_string:
        ``bigquery://project[/database][?dataset=project.dataset&schema=...]``
    schemas:
        Optional ingestion allowlist of dataset IDs. Empty or ``None`` means
        every dataset the URL names, or every dataset the billing project lists
        when it names none. A ``?schema=`` in the URL seeds the same allowlist.
    """

    supports_statement_timeout = True

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        self._settings = _parse_connection_string(connection_string)
        self._project = str(self._settings["project"])
        self._database = str(self._settings["database"])
        self._location: str | None = self._settings["location"]
        self._maximum_bytes_billed: int | None = self._settings["maximum_bytes_billed"]

        allowlist = [s.strip() for s in (schemas or []) if s and s.strip()]
        pinned = self._settings["schema"]
        if not allowlist and pinned:
            allowlist = [str(pinned)]
        wanted = {name.casefold() for name in allowlist}

        named: list[tuple[str, str]] = list(self._settings["datasets"])
        self._owners: dict[str, str] = {
            dataset.casefold(): owner for owner, dataset in named
        }
        if named:
            self._scope: list[tuple[str, str]] | None = [
                (owner, dataset)
                for owner, dataset in named
                if not wanted or dataset.casefold() in wanted
            ]
        elif allowlist:
            self._scope = [(self._project, name) for name in sorted(set(allowlist))]
        else:
            # Discovered from the billing project on first use.
            self._scope = None
        if allowlist:
            logger.info("BigQuery ingestion restricted to datasets: %s", allowlist)

        self._client: "Client | None" = None
        # Dataset (casefolded) -> wildcard name -> family, read on first use.
        self._families: dict[str, dict[str, ShardFamily]] | None = None

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _credentials(self) -> Any:
        """Build explicit credentials, or ``None`` to use Application Default.

        An explicit key that fails to load raises rather than falling back:
        quietly running as the pod's identity would read whatever *it* can see.
        """
        from google.cloud.bigquery import Client
        from google.oauth2 import service_account

        info = self._settings["credentials"]
        if info is not None:
            return service_account.Credentials.from_service_account_info(
                info, scopes=Client.SCOPE
            )
        path = self._settings["credentials_file"]
        if path:
            return service_account.Credentials.from_service_account_file(
                path, scopes=Client.SCOPE
            )
        return None

    @property
    def client(self) -> "Client":
        """The BigQuery client, created on first use.

        The library is imported lazily so importing this module stays cheap,
        and so constructing a connector never makes a network call.
        """
        if self._client is None:
            from google.cloud.bigquery import Client

            self._client = Client(
                project=self._project,
                credentials=self._credentials(),
                location=self._location,
            )
            logger.info("Opened BigQuery client for project %s", self._project)
        return self._client

    def _job_config(
        self,
        parameters: Optional[list],
        timeout_s: float | None,
    ) -> "QueryJobConfig":
        from google.cloud.bigquery import QueryJobConfig

        config = QueryJobConfig(
            maximum_bytes_billed=self._maximum_bytes_billed,
            labels=dict(_JOB_LABELS),
        )
        # An unqualified table name in generated SQL resolves the way the user
        # expects only when exactly one dataset is in scope.
        if self._scope is not None and len(self._scope) == 1:
            owner, dataset = self._scope[0]
            config.default_dataset = f"{owner}.{dataset}"
        if parameters:
            config.query_parameters = [_query_parameter(p) for p in parameters]
        if timeout_s is not None:
            config.job_timeout_ms = max(1, int(timeout_s * 1000))
        return config

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "bigquery"

    @property
    def database_name(self) -> str:
        return self._database

    def _owner(self, dataset: str) -> str:
        """The project holding *dataset*: the one the URL names, else billing."""
        return self._owners.get(dataset.casefold(), self._project)

    def qualify(self, schema: Optional[str], table: str) -> str:
        """Prepend the dataset's project: BigQuery names are ``project.dataset.table``.

        Raises:
            ValueError: *schema* is missing. A two-part name is read as
                ``dataset.table``, so ``project.table`` would put the project in
                the dataset position.
        """
        if not schema:
            raise ValueError(
                f"BigQuery requires a dataset to qualify {table!r} "
                f"in {self._database!r}"
            )
        family = self._family(schema, table)
        if family is not None:
            # Callers qualify a table to profile or probe it, and on a wildcard
            # that would scan every shard; the newest one stands for them all.
            table = family.latest
        return qualified_name(self._owner(schema), schema, table, dialect=self.dialect)

    def _qualify_datasets(self, sql: str) -> str:
        """Prefix ``dataset.table`` with the project of a dataset the URL names.

        SQL is regenerated only when a name actually changes, so a statement
        that already names its projects reaches BigQuery byte for byte. SQL
        sqlglot cannot parse is passed through for BigQuery to judge.
        """
        foreign = {
            name: owner
            for name, owner in self._owners.items()
            if owner != self._project
        }
        if not foreign:
            return sql
        try:
            statements = sqlglot.parse(sql, read="bigquery")
        except (sqlglot.errors.SqlglotError, ValueError):
            logger.debug("Could not parse SQL to qualify its datasets; sent as is")
            return sql

        changed = False
        for statement in statements:
            if statement is None:
                continue
            for table in statement.find_all(exp.Table):
                if table.catalog or not table.db:
                    continue
                owner = foreign.get(table.db.casefold())
                if owner is not None:
                    table.set("catalog", exp.to_identifier(owner))
                    changed = True
        if not changed:
            return sql
        return ";\n".join(
            s.sql(dialect="bigquery") for s in statements if s is not None
        )

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

        Parameters bind positionally to ``?`` placeholders. A statement that
        returns no rows (DDL, DML) yields an empty frame.

        With *timeout_s*, the job carries ``job_timeout_ms`` so BigQuery stops it
        server-side; the client waits a little longer before cancelling it
        itself, in case the server-side stop is late.

        A statement that only reads a table's first rows -- what profiling
        sends -- is read with ``tabledata.list`` instead, which bills nothing:
        as a query it would scan every row of the columns it names, ``LIMIT``
        or not, and a large table would hit the bytes-billed cap.
        """
        if not parameters:
            sample = self._table_sample(sql)
            if sample is not None:
                frame = self._read_rows(*sample, timeout_s=timeout_s)
                if frame is not None:
                    return frame
        job = self.client.query(
            self._qualify_datasets(sql),
            job_config=self._job_config(parameters, timeout_s),
        )
        started = time.monotonic()
        try:
            rows = job.result(
                timeout=None if timeout_s is None else timeout_s + _TIMEOUT_GRACE_S
            )
        except concurrent.futures.TimeoutError as exc:
            try:
                job.cancel()
            except Exception:
                logger.debug("Failed to cancel BigQuery job %s", job.job_id)
            raise StatementTimeout.since(timeout_s or 0.0, started) from exc
        except Exception as exc:
            if timeout_s is not None and _is_job_timeout(exc):
                raise StatementTimeout.since(timeout_s, started) from exc
            raise

        columns = [field.name for field in (rows.schema or [])]
        if not columns:
            return pd.DataFrame()
        return pd.DataFrame([list(row.values()) for row in rows], columns=columns)

    def _table_sample(self, sql: str) -> tuple[str, list[str] | None, int] | None:
        """``(table id, columns or None for all, limit)`` when *sql* reads first rows."""
        try:
            statement = sqlglot.parse_one(sql, read="bigquery")
        except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError):
            return None
        if not isinstance(statement, exp.Select):
            return None
        if any(v for k, v in statement.args.items() if k not in _SAMPLE_CLAUSES):
            return None
        source = statement.args.get("from_") or statement.args.get("from")
        limit = statement.args.get("limit")
        table = source.this if source is not None else None
        count = limit.expression if limit is not None else None
        if (
            not isinstance(table, exp.Table)
            or not table.db
            or "*" in table.name
            or not isinstance(count, exp.Literal)
            or not count.is_int
        ):
            return None
        columns: list[str] | None = []
        for item in statement.expressions:
            if isinstance(item, exp.Star):
                columns = None
            elif (
                isinstance(item, exp.Column) and not item.table and columns is not None
            ):
                columns.append(item.name)
            else:
                return None
        owner = table.catalog or self._owner(table.db)
        return f"{owner}.{table.db}.{table.name}", columns, int(count.this)

    def _read_rows(
        self,
        table_id: str,
        columns: list[str] | None,
        limit: int,
        *,
        timeout_s: float | None = None,
    ) -> pd.DataFrame | None:
        """The first *limit* rows of a stored table; ``None`` to run it as a query.

        Views and external tables have no stored rows to list, and anything the
        listing cannot answer falls back to the query it was asked as.
        """
        try:
            table = self.client.get_table(table_id)
            if table.table_type != "TABLE" or not table.schema:
                return None
            by_name = {field.name.casefold(): field for field in table.schema}
            if columns is None:
                fields = list(table.schema)
            elif all(name.casefold() in by_name for name in columns):
                fields = [by_name[name.casefold()] for name in columns]
            else:
                return None
            rows = self.client.list_rows(
                table, selected_fields=fields, max_results=limit, timeout=timeout_s
            )
            names = columns if columns is not None else [f.name for f in fields]
            return pd.DataFrame([list(row.values()) for row in rows], columns=names)
        except Exception:
            logger.debug("Listing %s failed; running it as a query", table_id)
            return None

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        """List the connection's datasets.

        Used by the connection UI to let the user pick which datasets to ingest,
        and by the connection test. Datasets the URL names are each fetched, so
        the test proves every one is readable; otherwise listing the billing
        project proves the credentials and the project both work.
        """
        named = self._settings["datasets"]
        if named:
            for owner, dataset in named:
                self.client.get_dataset(f"{owner}.{dataset}")
            return [dataset for _, dataset in named]
        return sorted(
            str(dataset.dataset_id)
            for dataset in self.client.list_datasets(project=self._project)
        )

    def _datasets(self) -> list[str]:
        """The datasets to introspect: the scope, or every listed dataset."""
        if self._scope is not None:
            return [dataset for _, dataset in self._scope]
        return self.get_schemas()

    def _information_schema(self, dataset: str, view: str) -> str:
        return (
            f"{qualified_name(self._owner(dataset), dataset, dialect=self.dialect)}"
            f".INFORMATION_SCHEMA.{view}"
        )

    def _shard_families(self) -> dict[str, dict[str, ShardFamily]]:
        """Every dataset's shard families, from metadata queries on first use."""
        if self._families is None:

            def sql(dataset: str) -> str:
                return f"""
                    SELECT table_schema, table_name
                    FROM {self._information_schema(dataset, "TABLES")}
                    WHERE table_type = 'BASE TABLE'
                """

            self._remember_families(
                self._per_dataset(sql, ["table_schema", "table_name"])
            )
        return self._families or {}

    def _remember_families(self, base_tables: pd.DataFrame) -> None:
        names: dict[str, list[str]] = {}
        for schema, table in base_tables[["table_schema", "table_name"]].itertuples(
            index=False, name=None
        ):
            names.setdefault(str(schema), []).append(str(table))
        self._families = {}
        for schema, tables in names.items():
            self._families[schema.casefold()] = {
                f.wildcard: f for f in shard_families(tables)
            }

    def _family(self, schema: str, table: str) -> ShardFamily | None:
        if "*" not in table:
            return None
        return self._shard_families().get(schema.casefold(), {}).get(table)

    def _shard_owners(self) -> dict[tuple[str, str], ShardFamily]:
        """``(dataset, shard)`` -> its family, casefolded dataset."""
        return {
            (schema, shard): family
            for schema, families in self._shard_families().items()
            for family in families.values()
            for shard in family.shards
        }

    def _collapse_shards(
        self, frame: pd.DataFrame, schema_col: str, table_col: str
    ) -> pd.DataFrame:
        """Rows of the newest shard renamed to its wildcard; other shards dropped."""
        owners = self._shard_owners()
        if frame.empty or not owners:
            return frame
        keep: list[bool] = []
        names: list[Any] = []
        for schema, table in frame[[schema_col, table_col]].itertuples(
            index=False, name=None
        ):
            family = owners.get((str(schema).casefold(), str(table)))
            keep.append(family is None or table == family.latest)
            names.append(table if family is None else family.wildcard)
        frame = frame.assign(**{table_col: names})
        return frame[keep].reset_index(drop=True)

    def _per_dataset(self, build_sql: Any, columns: list[str]) -> pd.DataFrame:
        """Run *build_sql(dataset)* on every dataset in scope and stack the results."""
        frames = [
            frame
            for frame in (self.execute(build_sql(ds)) for ds in self._datasets())
            if not frame.empty
        ]
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)

    def get_tables(self) -> pd.DataFrame:
        def sql(dataset: str) -> str:
            tables = self._information_schema(dataset, "TABLES")
            options = self._information_schema(dataset, "TABLE_OPTIONS")
            return f"""
                SELECT
                    t.table_schema,
                    t.table_name,
                    t.table_type,
                    o.option_value AS description
                FROM {tables} AS t
                LEFT JOIN {options} AS o
                    ON o.table_name = t.table_name
                   AND o.option_name = 'description'
                ORDER BY t.table_name
            """

        frame = self._per_dataset(sql, _TABLE_SCHEMA)
        self._remember_families(
            frame[frame["table_type"].astype(str).str.upper() == "BASE TABLE"]
        )
        if frame.empty:
            return frame
        frame["table_type"] = (
            frame["table_type"]
            .astype(str)
            .map(lambda kind: _TABLE_TYPES.get(kind.upper(), TableTypes.BASE_TABLE))
        )
        frame["description"] = frame["description"].map(_description)
        frame = self._collapse_shards(frame, "table_schema", "table_name")
        families = self._shard_families()
        frame["description"] = [
            f"{family.note()} {description or ''}".strip()
            if (family := families.get(str(schema).casefold(), {}).get(str(table)))
            else description
            for schema, table, description in frame[
                ["table_schema", "table_name", "description"]
            ].itertuples(index=False, name=None)
        ]
        return frame

    def get_columns(self) -> pd.DataFrame:
        """Return top-level columns.

        ``COLUMN_FIELD_PATHS`` also lists every nested ``STRUCT`` field; joining
        on ``field_path = column_name`` keeps only the top-level row, which is
        where a column's own description lives.
        """

        def sql(dataset: str) -> str:
            columns = self._information_schema(dataset, "COLUMNS")
            paths = self._information_schema(dataset, "COLUMN_FIELD_PATHS")
            return f"""
                SELECT
                    c.table_schema,
                    c.table_name,
                    c.column_name,
                    c.data_type,
                    c.is_nullable = 'YES' AS is_nullable,
                    c.ordinal_position,
                    p.description
                FROM {columns} AS c
                LEFT JOIN {paths} AS p
                    ON p.table_name = c.table_name
                   AND p.column_name = c.column_name
                   AND p.field_path = c.column_name
                ORDER BY c.table_name, c.ordinal_position
            """

        frame = self._collapse_shards(
            self._per_dataset(sql, _COLUMN_SCHEMA), "table_schema", "table_name"
        )
        suffixes = [
            {
                "table_schema": schema,
                "table_name": family.wildcard,
                "column_name": TABLE_SUFFIX_COLUMN,
                "data_type": "STRING",
                "is_nullable": False,
                "ordinal_position": int(
                    frame.loc[
                        (frame["table_schema"] == schema)
                        & (frame["table_name"] == family.wildcard),
                        "ordinal_position",
                    ].max()
                )
                + 1,
                "description": (
                    "Pseudo-column of the wildcard table: the part of each "
                    f"shard's name after '{family.prefix}'."
                ),
            }
            for schema in frame["table_schema"].drop_duplicates()
            for family in self._shard_families()
            .get(str(schema).casefold(), {})
            .values()
            if (
                (frame["table_schema"] == schema)
                & (frame["table_name"] == family.wildcard)
            ).any()
        ]
        if suffixes:
            frame = pd.concat([frame, pd.DataFrame(suffixes)], ignore_index=True)
        return frame

    def get_views(self) -> pd.DataFrame:
        def sql(dataset: str) -> str:
            return f"""
                SELECT table_schema, table_name, view_definition
                FROM {self._information_schema(dataset, "VIEWS")}
                ORDER BY table_name
            """

        return self._per_dataset(sql, _VIEW_SCHEMA)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Return recent successful ``SELECT`` jobs from ``JOBS_BY_PROJECT``.

        Job history is region-scoped, so it is read only when the connection
        names a location. It also needs ``bigquery.jobs.listAll``; without it the
        history is skipped rather than failing ingestion, as on Snowflake.
        """
        if not self._location:
            logger.info(
                "BigQuery connection %s names no location; skipping query history",
                self._project,
            )
            return pd.DataFrame(columns=_QUERY_SCHEMA)

        jobs = (
            f"{qualified_name(self._project, f'region-{self._location}', dialect=self.dialect)}"
            ".INFORMATION_SCHEMA.JOBS_BY_PROJECT"
        )
        window = f"TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {int(hours)} HOUR)"
        label_key, label_value = next(iter(_JOB_LABELS.items()))
        try:
            frame = self.execute(f"""
                SELECT end_time, query AS query_text
                FROM {jobs}
                WHERE creation_time >= {window}
                  AND end_time >= {window}
                  AND job_type = 'QUERY'
                  AND statement_type = 'SELECT'
                  AND state = 'DONE'
                  AND error_result IS NULL
                  AND NULLIF(TRIM(query), '') IS NOT NULL
                  AND NOT REGEXP_CONTAINS(LOWER(query), r'information_schema')
                  AND NOT EXISTS (
                      SELECT 1 FROM UNNEST(labels) AS label
                      WHERE label.key = '{label_key}' AND label.value = '{label_value}'
                  )
                ORDER BY end_time DESC
                LIMIT {_QUERY_HISTORY_LIMIT}
            """)
        except Exception:
            logger.exception("Failed to fetch BigQuery query history")
            return pd.DataFrame(columns=_QUERY_SCHEMA)
        if frame.empty:
            return pd.DataFrame(columns=_QUERY_SCHEMA)
        return frame[_QUERY_SCHEMA]

    def get_pks(self) -> pd.DataFrame:
        def sql(dataset: str) -> str:
            usage = self._information_schema(dataset, "KEY_COLUMN_USAGE")
            constraints = self._information_schema(dataset, "TABLE_CONSTRAINTS")
            return f"""
                SELECT
                    k.table_schema,
                    k.table_name,
                    k.column_name,
                    k.ordinal_position
                FROM {usage} AS k
                JOIN {constraints} AS c
                    ON c.constraint_name = k.constraint_name
                   AND c.table_name = k.table_name
                WHERE c.constraint_type = 'PRIMARY KEY'
                ORDER BY k.table_name, k.ordinal_position
            """

        return self._collapse_shards(
            self._per_dataset(sql, _PK_SCHEMA), "table_schema", "table_name"
        )

    def get_fks(self) -> pd.DataFrame:
        """Return foreign-key column pairs, read from each constrained table.

        ``TABLE_CONSTRAINTS`` finds the tables that declare a foreign key; the
        table resource then gives each key's referenced table and its exact
        referencing/referenced column pairs, including keys that point into
        another dataset.
        """

        def sql(dataset: str) -> str:
            constraints = self._information_schema(dataset, "TABLE_CONSTRAINTS")
            return f"""
                SELECT DISTINCT table_schema, table_name
                FROM {constraints}
                WHERE constraint_type = 'FOREIGN KEY'
            """

        constrained = self._per_dataset(sql, ["table_schema", "table_name"])
        rows: list[dict[str, str]] = []
        for schema, table in constrained.itertuples(index=False, name=None):
            resource = self.client.get_table(f"{self._owner(schema)}.{schema}.{table}")
            keys = getattr(resource.table_constraints, "foreign_keys", None) or []
            for key in keys:
                referenced = key.referenced_table
                for pair in key.column_references:
                    rows.append(
                        {
                            "table_schema": str(schema),
                            "table_name": str(table),
                            "column_name": pair.referencing_column,
                            "referenced_schema": referenced.dataset_id,
                            "referenced_table": referenced.table_id,
                            "referenced_column": pair.referenced_column,
                        }
                    )
        frame = self._collapse_shards(
            pd.DataFrame(rows, columns=_FK_SCHEMA), "table_schema", "table_name"
        )
        owners = self._shard_owners()
        frame["referenced_table"] = [
            family.wildcard
            if (family := owners.get((str(schema).casefold(), str(table))))
            else table
            for schema, table in frame[
                ["referenced_schema", "referenced_table"]
            ].itertuples(index=False, name=None)
        ]
        return frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify the credentials can run a job in the project."""
        self.execute("SELECT 1")

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Failed to close BigQuery client", exc_info=True)
            self._client = None
