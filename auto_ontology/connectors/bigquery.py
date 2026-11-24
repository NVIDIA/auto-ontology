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
  ``maximum_bytes_billed`` (40 GiB unless the URL overrides it); a query over
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
identity of the pod. A connection saved through the API always supplies a key
(see :mod:`auto_ontology.connectors.connection_string_factory`), so falling
back to the pod's identity is reserved for operator-written
``CONNECTION_STRINGS``.

Example
-------
::

    CONNECTION_STRINGS='bigquery://my-project?location=US&credentials=BASE64_KEY'
    CONNECTION_STRINGS='bigquery://my-project?schema=analytics&credentials_file=/secrets/sa.json'
    CONNECTION_STRINGS='bigquery://my-project/austin?dataset=bigquery-public-data.austin_311'
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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, TypeVar
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

_V = TypeVar("_V")

DEFAULT_MAXIMUM_BYTES_BILLED = 40 * 1024**3

# Attached to every job, so query history can skip the connector's own
# statements and billing exports can attribute them.
_JOB_LABELS = {"source": "auto-ontology"}

# Extra wall-clock allowed on top of a caller's statement timeout, so the
# server's ``job_timeout_ms`` fires first and the job is stopped rather than the
# client abandoning a job that keeps running (and billing).
_TIMEOUT_GRACE_S = 10.0
# How long a timed-out statement waits on its job's cancellation request.
_CANCEL_TIMEOUT_S = 5.0

_QUERY_HISTORY_LIMIT = 10_000

# How long a long-lived connector trusts its shard families before re-reading
# them, so a newly landed shard becomes the one a wildcard is qualified to.
_FAMILIES_TTL_S = 3600.0
_MAX_QUERY_JOB_HOURS = 6

_TABLE_TYPES = {
    "VIEW": TableTypes.VIEW,
    "MATERIALIZED VIEW": TableTypes.MATERIALIZED_VIEW,
}

# The clauses a first-rows read may have; anything else needs a real query.
_SAMPLE_CLAUSES = frozenset({"expressions", "from", "from_", "limit"})
# Time travel and TABLESAMPLE sit on the table, not the statement.
_SAMPLE_TABLE_PARTS = frozenset({"this", "db", "catalog", "alias"})

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
_SHARD_PREFIX = r"(?P<prefix>.*\D)"
_SHARD_NAME = re.compile(rf"^{_SHARD_PREFIX}(?P<suffix>\d{{4,8}})$")
# One shard, or a partial wildcard (``events_2021*``) over some of them.
_SHARD_REFERENCE = re.compile(rf"^{_SHARD_PREFIX}(?:\d{{4,8}}|\d{{1,7}}\*)$")


def wildcard_for(table: str) -> str | None:
    """The wildcard table (``events_*``) that *table* reads part of, if any.

    The catalog holds a date-sharded table only as its wildcard, so SQL naming
    a single shard or a partial wildcard is matched against that.
    """
    match = _SHARD_REFERENCE.match(table)
    return f"{match.group('prefix')}*" if match else None


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
    names: list[str],
    columns: dict[str, dict[str, str]] | None = None,
    views: Iterable[str] = (),
) -> list[ShardFamily]:
    """Group date-sharded table names into wildcard families of two shards or more.

    A wildcard query reads every shard with the newest one's schema: a column
    a shard lacks reads as NULL, but one only an older shard has cannot be
    selected, and a changed type fails the query. Given each table's
    *columns* (name to type), a shard with a column the newest does not have
    the same way stays a table of its own, listed in ``also_matches``.

    BigQuery rejects a wildcard that matches a view, even when a
    ``_TABLE_SUFFIX`` filter excludes it, so no family is formed over a prefix
    any of *views* (relations other than base tables) starts with.
    """
    unique = sorted(set(names))
    views = tuple(views)
    groups: dict[str, list[str]] = {}
    for name in unique:
        match = _SHARD_NAME.match(name)
        if match:
            groups.setdefault(match.group("prefix"), []).append(name)
    families = []
    for prefix, members in sorted(groups.items()):
        if any(view.startswith(prefix) for view in views):
            continue
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


def _by_dataset(items: dict[str, _V], dataset: str) -> _V | None:
    """The entry for *dataset*'s exact ID, else the only one differing in case.

    BigQuery dataset IDs are case-sensitive, so ``Sales`` and ``sales`` can
    both exist; a name in another case is tolerated only when unambiguous.
    """
    if dataset in items:
        return items[dataset]
    folded = [v for k, v in items.items() if k.casefold() == dataset.casefold()]
    return folded[0] if len(folded) == 1 else None


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
    return _service_account_key(text)


def _service_account_key(text: str) -> dict[str, Any]:
    """Parse a service-account JSON key, refusing any other kind of key."""
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
    size = int(float(match.group(1)) * _BYTE_UNITS[match.group(2)])
    if size < 1:
        raise ValueError(
            f"BigQuery maximum_bytes_billed must be at least one byte, got {text!r}"
        )
    return size


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
        previous = owners.setdefault(dataset, owner)
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


class BytesBilledLimitExceeded(RuntimeError):
    """BigQuery refused a query that would bill more than ``maximum_bytes_billed``.

    Nothing was billed: BigQuery estimates the scan before it runs the query.
    """

    def __init__(
        self, required: int | None, cap: int | None, errors: list | None = None
    ) -> None:
        self.required = required
        self.cap = cap
        self.errors = errors or []
        scan = (
            f"it would scan {_human_bytes(required)}"
            if required
            else "it scans too much"
        )
        limit = f", over the {_human_bytes(cap)} cap" if cap else ""
        super().__init__(
            f"Query exceeded limit for bytes billed: {scan}{limit}. Nothing was "
            "billed. Scan less: filter _TABLE_SUFFIX or the partition column, "
            "and select only the columns you need instead of *."
        )


def _human_bytes(size: int | None) -> str:
    value = float(size or 0)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


_BYTES_REQUIRED = re.compile(r"(\d+) or higher required", re.I)
_BYTES_LIMIT = re.compile(r"limit for bytes billed: (\d+)", re.I)


def _bytes_billed_limit(
    exc: BaseException, cap: int | None
) -> BytesBilledLimitExceeded | None:
    """*exc* restated with its sizes in GiB, when it is the bytes-billed refusal."""
    errors = getattr(exc, "errors", None) or []
    text = str(exc)
    if not (
        any(
            isinstance(error, dict)
            and error.get("reason") == "bytesBilledLimitExceeded"
            for error in errors
        )
        or "exceeded limit for bytes billed" in text.lower()
    ):
        return None
    required = _BYTES_REQUIRED.search(text)
    limit = _BYTES_LIMIT.search(text)
    return BytesBilledLimitExceeded(
        int(required.group(1)) if required else None,
        int(limit.group(1)) if limit else cap,
        list(errors),
    )


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
        named: list[tuple[str, str]] = list(self._settings["datasets"])
        self._owners: dict[str, str] = {dataset: owner for owner, dataset in named}
        if named:
            by_id = {dataset: (owner, dataset) for owner, dataset in named}
            wanted = {_by_dataset(by_id, name) for name in allowlist}
            self._scope: list[tuple[str, str]] | None = [
                pair for pair in named if not allowlist or pair in wanted
            ]
        elif allowlist:
            self._scope = [(self._project, name) for name in sorted(set(allowlist))]
        else:
            # Discovered from the billing project on first use.
            self._scope = None
        if allowlist:
            logger.info("BigQuery ingestion restricted to datasets: %s", allowlist)

        self._client: "Client | None" = None
        # Dataset -> wildcard name -> family, read on first use.
        self._families: dict[str, dict[str, ShardFamily]] | None = None
        self._families_read_at = 0.0

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
        path = self._settings["credentials_file"]
        if info is None and path:
            try:
                with open(path, encoding="utf-8") as key_file:
                    text = key_file.read()
            except (OSError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"BigQuery credentials file {path} cannot be read"
                ) from exc
            info = _service_account_key(text)
        if info is None:
            return None
        return service_account.Credentials.from_service_account_info(
            info, scopes=Client.SCOPE
        )

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

        config = QueryJobConfig(labels=dict(_JOB_LABELS))
        # The SDK sends None as the string "None", which BigQuery rejects.
        if self._maximum_bytes_billed is not None:
            config.maximum_bytes_billed = self._maximum_bytes_billed
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
        return _by_dataset(self._owners, dataset) or self._project

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
        if all(owner == self._project for owner in self._owners.values()):
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
                # Resolved against every named dataset, so an exact local
                # match is never traded for a case-twin in another project.
                owner = _by_dataset(self._owners, table.db)
                if owner is not None and owner != self._project:
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
        itself, in case the server-side stop is late. The listing below, the
        query that may follow it, and the wait for its result all share that
        one budget.

        A statement that only reads a table's first rows -- what profiling
        sends -- is read with ``tabledata.list`` instead, which bills nothing:
        as a query it would scan every row of the columns it names, ``LIMIT``
        or not, and a large table would hit the bytes-billed cap.
        """
        started = time.monotonic()

        def left() -> float | None:
            """The rest of the one budget every request of the statement shares."""
            if timeout_s is None:
                return None
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                raise StatementTimeout.since(timeout_s, started)
            return remaining

        if not parameters:
            sample = self._table_sample(sql)
            if sample is not None:
                frame = self._read_rows(*sample, timeout_s=left())
                if frame is not None:
                    return frame
        budget = left()
        job = self.client.query(
            self._qualify_datasets(sql),
            job_config=self._job_config(parameters, budget),
            timeout=budget,
        )
        try:
            budget = left()
            rows = job.result(
                timeout=None if budget is None else budget + _TIMEOUT_GRACE_S
            )
        except (concurrent.futures.TimeoutError, StatementTimeout) as exc:
            try:
                job.cancel(timeout=_CANCEL_TIMEOUT_S)
            except Exception:
                logger.debug("Failed to cancel BigQuery job %s", job.job_id)
            if isinstance(exc, StatementTimeout):
                raise
            raise StatementTimeout.since(timeout_s or 0.0, started) from exc
        except Exception as exc:
            if timeout_s is not None and _is_job_timeout(exc):
                raise StatementTimeout.since(timeout_s, started) from exc
            refused = _bytes_billed_limit(exc, self._maximum_bytes_billed)
            if refused is not None:
                raise refused from exc
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
            or any(v for k, v in table.args.items() if k not in _SAMPLE_TABLE_PARTS)
            or not isinstance(count, exp.Literal)
            or not count.is_int
        ):
            return None
        items = statement.expressions
        columns: list[str] | None = []
        if len(items) == 1 and isinstance(items[0], exp.Star):
            # EXCEPT and REPLACE live on the star; only a bare one reads as is.
            if any(items[0].args.values()):
                return None
            columns = None
        elif all(isinstance(i, exp.Column) and not i.table for i in items):
            columns = [i.name for i in items]
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
            table = self.client.get_table(table_id, timeout=timeout_s)
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
        """Every dataset's shard families, re-read once they are an hour old."""
        stale = time.monotonic() - self._families_read_at > _FAMILIES_TTL_S
        if self._families is None or stale:

            def sql(dataset: str) -> str:
                return f"""
                    SELECT table_schema, table_name, table_type
                    FROM {self._information_schema(dataset, "TABLES")}
                """

            self._remember_families(
                self._per_dataset(sql, ["table_schema", "table_name", "table_type"])
            )
        return self._families or {}

    def _remember_families(self, relations: pd.DataFrame) -> None:
        tables: dict[str, list[str]] = {}
        views: dict[str, list[str]] = {}
        for schema, name, kind in relations[
            ["table_schema", "table_name", "table_type"]
        ].itertuples(index=False, name=None):
            into = tables if str(kind).upper() == "BASE TABLE" else views
            into.setdefault(str(schema), []).append(str(name))
        # Assigned whole, so a concurrent reader never takes a half-built map
        # for the loaded cache.
        self._families_read_at = time.monotonic()
        self._families = {
            schema: {
                f.wildcard: f
                for f in shard_families(names, views=views.get(schema, ()))
            }
            for schema, names in tables.items()
        }

    def _family(self, schema: str, table: str) -> ShardFamily | None:
        if "*" not in table:
            return None
        return (_by_dataset(self._shard_families(), schema) or {}).get(table)

    def _shard_owners(self) -> dict[tuple[str, str], ShardFamily]:
        """``(dataset, shard)`` -> its family."""
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
            family = owners.get((str(schema), str(table)))
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
        self._remember_families(frame)
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
            if (family := families.get(str(schema), {}).get(str(table)))
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
            for family in self._shard_families().get(str(schema), {}).values()
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
        # creation_time is the partition key; a job that ended in the window can
        # have started up to the longest a query job may run before it.
        created = (
            f"TIMESTAMP_SUB(CURRENT_TIMESTAMP(), "
            f"INTERVAL {int(hours) + _MAX_QUERY_JOB_HOURS} HOUR)"
        )
        label_key, label_value = next(iter(_JOB_LABELS.items()))
        try:
            frame = self.execute(f"""
                SELECT end_time, query AS query_text
                FROM {jobs}
                WHERE creation_time >= {created}
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
                # The catalog names tables by dataset alone, so a key into
                # another project would land on this project's namesake.
                if referenced.project != self._owner(referenced.dataset_id):
                    continue
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
            if (family := owners.get((str(schema), str(table))))
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
