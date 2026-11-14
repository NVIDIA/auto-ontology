# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MongoDB and Amazon DocumentDB connector.

MongoDB has no SQL engine, so this connector brings one: every statement runs
in a private, in-process DuckDB database that is filled, for that statement
only, with the documents of the collections it reads. The connector therefore
reports the ``duckdb`` dialect, and everything downstream -- prompt rules,
sqlglot validation, profiling probes -- treats it like any other DuckDB
source. Amazon DocumentDB speaks the same wire protocol and is reached through
the same ``mongodb://`` URLs.

How documents become rows
-------------------------
* Each collection, and each view on servers that have them, is a table in
  schema ``main`` -- the schema DuckDB resolves bare and ``main.``-qualified
  names against, as for the DuckDB and SQLite connectors.
* Each top-level field is a column, typed from a ``$sample`` of
  ``sample_size`` documents. A field that holds one BSON type gets the matching
  DuckDB type; mixed scalars become ``VARCHAR``. A type held by fewer than one
  in a hundred of the sampled values is ignored, and the column keeps its type
  for as long as the connector runs.
* Embedded documents and arrays stay whole, as ``JSON`` columns read with
  ``->`` / ``->>``. Profiling already understands JSON columns and lists their
  keys the way it does for Postgres JSONB.
* ``_id`` is the primary key. MongoDB declares no foreign keys, so
  relationships come from semantic compilation, as they do for SQLite files
  without ``REFERENCES`` clauses.

How a statement runs
--------------------
1. sqlglot finds the collections the statement reads and the names it uses.
2. Only those fields are fetched. For a statement over a single collection,
   simple ``WHERE`` predicates and a bare ``LIMIT`` are also sent to the
   server. A pushed-down filter only ever *widens* what DuckDB filters again,
   so a predicate the connector cannot translate safely is left to DuckDB.
3. The documents are converted to the inferred column types and loaded as
   DuckDB tables. A value whose BSON type does not match its column reads as
   NULL rather than as a coerced value, which is what keeps step 2 safe.
4. The statement runs with DuckDB's external access disabled and its
   configuration locked, so generated SQL cannot read files, attach databases
   or install extensions.

Reading more than ``max_documents`` documents from one collection for one
statement raises :class:`MongoDBDocumentLimitError` instead of answering from
a truncated collection.

Connection strings
------------------
Any MongoDB URL, with the database to ingest in the path::

    mongodb://user:password@host:27017/shop?tls=true&authSource=admin
    mongodb+srv://user:password@cluster0.example.mongodb.net/shop
    mongodb://user:password@docdb.cluster-xyz.us-east-1.docdb.amazonaws.com:27017/shop
        ?tls=true&tlsCAFile=/certs/global-bundle.pem&replicaSet=rs0
        &readPreference=secondaryPreferred&retryWrites=false

Three options belong to the connector and are removed before the URL reaches
PyMongo: ``sample_size``, ``max_documents``, and ``tls_ca_data`` -- a
base64-encoded PEM bundle for deployments that cannot mount a ``tlsCAFile``.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

import duckdb
import pandas as pd
import pymongo
import sqlglot
from bson import DBRef, Decimal128, ObjectId, Timestamp
from bson.codec_options import DatetimeConversion
from bson.datetime_ms import DatetimeMS
from pymongo import MongoClient
from pymongo.errors import ExecutionTimeout, OperationFailure, PyMongoError
from sqlglot import exp

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout
from auto_ontology.utils.sql_identifiers import quoted_identifier

logger = logging.getLogger(__name__)

_DIALECT = "duckdb"

#: Schema every collection is reported under. DuckDB resolves a bare table
#: name against ``main``, which is what the duckdb prompt rules ask for.
SCHEMA = "main"

DEFAULT_SAMPLE_SIZE = 1_000
DEFAULT_MAX_DOCUMENTS = 100_000

# Inferred schemas are reused by ``execute`` for this long before the
# collection is sampled again, so fields added after the connector loaded show
# up without a restart. Ingestion always samples afresh.
_SCHEMA_TTL_SECONDS = 600.0
# Top-level fields kept per collection, most frequent first. A collection whose
# documents use their keys as data (one key per day, per user...) would
# otherwise turn every distinct key into a column.
_MAX_COLUMNS = 500
# A BSON type seen in less than this share of a field's sampled values does not
# change the column's type; such values read as NULL. Without it, one stray
# string in a thousand numbers decides the type by whether a random sample
# happened to catch it -- and the ingestion service and the server sample
# separately, so they would disagree about the column almost every other time.
_MINORITY_SHARE = 0.01
# Cap on one collection's ``$sample``. Sampling runs once per collection at
# ingestion, so it may take longer than a profiling query, but not forever.
_SAMPLE_TIME_LIMIT_MS = 60_000
# PyMongo waits 30s for a reachable server by default; a wrong host should
# fail a connection test faster than that. A URL that sets its own value wins.
_SERVER_SELECTION_TIMEOUT_MS = 10_000
# PyMongo waits for a socket read forever by default. Statements run under
# PyMongo's own per-statement timeout, which overrides this; it bounds the
# rest -- ingestion, and servers that refuse maxTimeMS -- when a connection
# stops answering. Longer than the sample's time limit, so a slow sample still
# ends with the server's error rather than a dropped socket.
_SOCKET_TIMEOUT_MS = 120_000
# Share of the statement timeout after which a timeout error counts as the
# statement's own; see _Deadline.nearly_over.
_DEADLINE_SHARE = 0.9
# Collections sampled at once when the whole database is introspected.
_SAMPLING_WORKERS = 8

_CONNECTOR_OPTIONS = frozenset({"sample_size", "max_documents", "tls_ca_data"})

# Applied when the in-process database is opened. External access is switched
# off right after, before anything else runs (DuckDB refuses to clear the temp
# directory once it is off): the generated SQL that reaches the database must
# not be able to touch the filesystem, the network or the extension directory.
# Without a temp directory, DuckDB would still let it read ``.tmp`` under the
# working directory, where other DuckDB connections spill.
_DUCKDB_CONFIG = {
    "temp_directory": "",
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
}

# Functions whose value differs between the connector evaluating a constant
# for pushdown and DuckDB evaluating it again.
_VOLATILE_FUNCTIONS = frozenset(
    {
        "currval",
        "gen_random_uuid",
        "nextval",
        "rand",
        "random",
        "setseed",
        "uuid",
        "uuidv4",
        "uuidv7",
    }
)
# Functions that read the clock. DuckDB evaluates them after the connector has
# fetched, so they are later there. A lower bound that only rises with time
# then only gets stricter, which is safe; anything else -- an upper bound, or a
# bound that can fall as time passes -- could drop documents DuckDB would keep.
# See _rises_with_the_clock.
_CLOCK_FUNCTIONS = frozenset(
    {
        "currentdate",
        "currentdatetime",
        "currenttime",
        "currenttimestamp",
        "get_current_time",
        "get_current_timestamp",
        "localtime",
        "localtimestamp",
        "now",
        "today",
        "transaction_timestamp",
    }
)

# Projection nodes that keep exactly one output row per document, so a bare
# ``LIMIT`` may be applied on the server.
_ROW_PRESERVING = (
    exp.Add,
    exp.Alias,
    exp.Boolean,
    exp.Cast,
    exp.Column,
    exp.DataType,
    exp.DataTypeParam,
    exp.Div,
    exp.Identifier,
    exp.JSONExtract,
    exp.JSONExtractScalar,
    exp.JSONPath,
    exp.JSONPathPart,
    exp.Literal,
    exp.Mul,
    exp.Neg,
    exp.Null,
    exp.Paren,
    exp.Star,
    exp.Sub,
    exp.TryCast,
)

# Comparisons the filter pushdown understands, as MongoDB operators. ``<>`` is
# absent on purpose: on an array field ``$ne`` excludes a document when *any*
# element is equal, and under a case-insensitive collection collation it
# excludes strings DuckDB would keep.
_COMPARISONS: dict[type, str] = {
    exp.EQ: "$eq",
    exp.GT: "$gt",
    exp.GTE: "$gte",
    exp.LT: "$lt",
    exp.LTE: "$lte",
}
# The same comparison with its operands swapped: ``5 < x`` is ``x > 5``.
_FLIPPED = {"$eq": "$eq", "$gt": "$lt", "$gte": "$lte", "$lt": "$gt", "$lte": "$gte"}
# Conversions that may be range-compared on the server. Strings may not: a
# collection's default collation can order them differently from DuckDB.
_ORDERED = frozenset({"int", "double", "date"})
# Conversions that never turn a non-null value into NULL or an array into a
# non-null cell, so ``IS NOT NULL`` can be sent as ``$ne: null``.
_STRICT = frozenset({"bool", "date", "double", "int", "objectid", "string"})

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_TIME_OF_DAY_FUNCTIONS = frozenset({"currenttime", "get_current_time", "localtime"})
# What DuckDB hands back for TIMESTAMP 'infinity' and '-infinity'.
_TIMESTAMP_SENTINELS = frozenset({datetime.max, datetime.min})
# Date and timestamp types: casting a rising clock to one keeps it rising. A
# time of day does not -- it wraps at midnight.
_INSTANT_TYPES = tuple(
    kind
    for kind in exp.DataType.TEMPORAL_TYPES
    if kind not in (exp.DataType.Type.TIME, exp.DataType.Type.TIMETZ)
)


class MongoDBDocumentLimitError(ValueError):
    """A statement would have loaded more documents than ``max_documents``.

    Raised rather than answering from the documents loaded so far: a count or
    a sum over a truncated collection is a wrong answer that looks right.
    """


# ---------------------------------------------------------------------------
# Connection strings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Settings:
    uri: str
    database: str
    uri_options: frozenset[str]
    sample_size: int
    max_documents: int
    tls_ca_pem: bytes | None


def _positive_int(name: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _decode_pem(raw: str) -> bytes:
    """Decode ``tls_ca_data``: base64 of a PEM bundle, standard or URL-safe.

    A ``+`` of standard base64 arrives as a space once a query string has been
    decoded, so spaces are put back before decoding.
    """
    text = raw.strip().replace(" ", "+").replace("-", "+").replace("_", "/")
    try:
        pem = base64.b64decode(text + "=" * (-len(text) % 4), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(
            "tls_ca_data must be base64 of a PEM certificate bundle"
        ) from None
    if b"-----BEGIN CERTIFICATE-----" not in pem:
        raise ValueError("tls_ca_data does not contain a PEM certificate")
    return pem


def _parse_connection_string(connection_string: str) -> _Settings:
    """Split a MongoDB URL into what PyMongo reads and what the connector reads."""
    parts = urlsplit(connection_string.strip())
    if parts.scheme.lower() not in ("mongodb", "mongodb+srv"):
        raise ValueError(
            "MongoDB connection strings start with mongodb:// or mongodb+srv://"
        )

    database = unquote(parts.path.lstrip("/"))
    if not database or "/" in database:
        raise ValueError(
            "The MongoDB connection string must name the database to ingest in its "
            "path, e.g. mongodb://host:27017/shop. If the user is defined in another "
            "database, add authSource=<that database>."
        )

    # The connector's options are cut out of the raw query rather than the
    # query being decoded and re-encoded, so PyMongo sees every other option
    # exactly as written.
    separator = ";" if ";" in parts.query and "&" not in parts.query else "&"
    kept: list[str] = []
    options: dict[str, str] = {}
    uri_options: set[str] = set()
    for item in re.split(r"[&;]", parts.query):
        if not item:
            continue
        key, _, value = item.partition("=")
        name = unquote(key).strip().lower()
        if name in _CONNECTOR_OPTIONS:
            options[name] = unquote(value)
        else:
            kept.append(item)
            uri_options.add(name)

    sample_size = _positive_int(
        "sample_size", options.get("sample_size", str(DEFAULT_SAMPLE_SIZE))
    )
    max_documents = _positive_int(
        "max_documents", options.get("max_documents", str(DEFAULT_MAX_DOCUMENTS))
    )
    tls_ca_pem = None
    if "tls_ca_data" in options:
        if "tlscafile" in uri_options:
            raise ValueError("Set either tlsCAFile or tls_ca_data, not both")
        tls_ca_pem = _decode_pem(options["tls_ca_data"])

    uri = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, separator.join(kept), parts.fragment)
    )
    return _Settings(
        uri=uri,
        database=database,
        uri_options=frozenset(uri_options),
        sample_size=sample_size,
        max_documents=max_documents,
        tls_ca_pem=tls_ca_pem,
    )


# ---------------------------------------------------------------------------
# Documents -> typed columns
# ---------------------------------------------------------------------------


def _kind(value: Any) -> str | None:
    """The BSON type family of a decoded value; ``None`` for null."""
    if value is None:
        return None
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, Decimal128):
        return "decimal"
    if isinstance(value, str):
        return "string"
    if isinstance(value, ObjectId):
        return "objectid"
    if isinstance(value, (datetime, DatetimeMS)):
        return "date"
    if isinstance(value, Timestamp):
        return "timestamp"
    if isinstance(value, uuid.UUID):
        return "uuid"
    if isinstance(value, (bytes, bytearray)):
        return "binary"
    if isinstance(value, (Mapping, DBRef)):
        return "document"
    if isinstance(value, (list, tuple)):
        return "array"
    return "other"


def _decimal_scale(value: Decimal128) -> int:
    exponent = value.to_decimal().as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


# Columns holding one kind: (DuckDB type, conversion). Decimal is built apart
# because its type carries the scale.
_SINGLE_KIND_TYPES = {
    "objectid": ("VARCHAR", "objectid"),
    "string": ("VARCHAR", "string"),
    "int": ("BIGINT", "int"),
    "double": ("DOUBLE", "double"),
    "bool": ("BOOLEAN", "bool"),
    "date": ("TIMESTAMP", "date"),
    "timestamp": ("TIMESTAMP", "timestamp"),
    "uuid": ("VARCHAR", "uuid"),
    "binary": ("VARCHAR", "binary"),
}
_NUMERIC_KINDS = frozenset({"int", "double", "decimal"})


def _column_type(kinds: set[str], decimal_scale: int) -> tuple[str, str]:
    """DuckDB type and conversion for a field seen holding *kinds*."""
    decimal_type = f"DECIMAL(38, {min(decimal_scale, 18)})"
    if not kinds:
        return "VARCHAR", "text"
    if kinds & {"document", "array"}:
        return "JSON", "json"
    if len(kinds) == 1:
        (kind,) = kinds
        if kind == "decimal":
            return decimal_type, "decimal"
        return _SINGLE_KIND_TYPES.get(kind, ("VARCHAR", "text"))
    if kinds <= _NUMERIC_KINDS:
        if "double" in kinds:
            return "DOUBLE", "double"
        return decimal_type, "decimal"  # integers and decimals
    if kinds <= {"date", "timestamp"}:
        return "TIMESTAMP", "datetime"
    return "VARCHAR", "text"


@dataclass(frozen=True)
class _Column:
    name: str
    sql_type: str
    conversion: str
    nullable: bool | None


#: The column of a collection or view that no sampled document described.
_PLACEHOLDER_ID = _Column("_id", "VARCHAR", "text", False)


@dataclass(frozen=True)
class _Collection:
    name: str
    is_view: bool
    columns: tuple[_Column, ...]
    inferred_at: float

    def column(self, name: str) -> _Column | None:
        """The column DuckDB would resolve *name* to: exact, else case-insensitive."""
        for column in self.columns:
            if column.name == name:
                return column
        folded = name.lower()
        for column in self.columns:
            if column.name.lower() == folded:
                return column
        return None


@dataclass
class _FieldStats:
    present: int = 0
    kinds: Counter[str] = field(default_factory=Counter)
    decimal_scale: int = 0

    def significant_kinds(self) -> set[str]:
        """The kinds held by at least :data:`_MINORITY_SHARE` of the values."""
        return {
            kind
            for kind, count in self.kinds.items()
            if count >= self.present * _MINORITY_SHARE
        }


def infer_columns(documents: Iterable[Mapping[str, Any]]) -> list[_Column]:
    """Type the top-level fields of *documents*.

    ``_id`` comes first, then fields in the order the documents first showed
    them. A field missing from, or null in, any sampled document is nullable;
    one present in all of them is reported as unknown rather than ``NOT
    NULL``, since a sample cannot prove that. A BSON type held by fewer than
    one in a hundred of a field's values does not decide its column type.

    DuckDB matches identifiers case-insensitively, quoted ones too, so of two
    fields that differ only in case the one present in more documents is kept.
    """
    stats: dict[str, _FieldStats] = {}
    total = 0
    for document in documents:
        total += 1
        for key, value in document.items():
            if not isinstance(key, str) or not key:
                continue
            entry = stats.get(key)
            if entry is None:
                entry = stats[key] = _FieldStats()
            kind = _kind(value)
            if kind is None:
                continue
            entry.present += 1
            entry.kinds[kind] += 1
            if kind == "decimal":
                entry.decimal_scale = max(entry.decimal_scale, _decimal_scale(value))

    first_seen = {name: index for index, name in enumerate(stats)}
    by_frequency = sorted(
        stats, key=lambda name: (name != "_id", -stats[name].present, first_seen[name])
    )
    kept: list[str] = []
    folded: set[str] = set()
    for name in by_frequency:
        if name.lower() in folded:
            logger.warning(
                "Skipping field %r: DuckDB cannot tell it apart from another field "
                "whose name differs only in case",
                name,
            )
            continue
        if len(kept) == _MAX_COLUMNS:
            logger.warning(
                "Keeping the %d most frequent of %d fields; the rest are not columns",
                _MAX_COLUMNS,
                len(stats),
            )
            break
        folded.add(name.lower())
        kept.append(name)

    columns = []
    for name in sorted(kept, key=lambda n: (n != "_id", first_seen[n])):
        entry = stats[name]
        sql_type, conversion = _column_type(
            entry.significant_kinds(), entry.decimal_scale
        )
        if name == "_id":
            nullable: bool | None = False
        elif entry.present < total:
            nullable = True
        else:
            nullable = None
        columns.append(_Column(name, sql_type, conversion, nullable))
    return columns


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _iso(value: datetime) -> str:
    return _naive_utc(value).isoformat() + "Z"


def _json_value(value: Any) -> Any:
    """A JSON-encodable rendering of a decoded BSON value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal128):
        number = value.to_decimal()
        return float(number) if number.is_finite() else None
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, DatetimeMS):  # outside the range datetime can hold
        return int(value)
    if isinstance(value, Timestamp):
        return _iso(value.as_datetime())
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, DBRef):
        reference = {"$ref": value.collection, "$id": _json_value(value.id)}
        if value.database:
            reference["$db"] = value.database
        return reference
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _text(value: Any) -> str | None:
    """A value of a mixed-type field, rendered as text, digit for digit."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Decimal128):
        return str(value.to_decimal())
    if isinstance(value, (Mapping, DBRef, list, tuple)):
        return json.dumps(_json_value(value), ensure_ascii=False)
    rendered = _json_value(value)
    return rendered if isinstance(rendered, str) else str(rendered)


def _to_int(value: Any) -> int | None:
    """A whole number that fits BIGINT; anything else is NULL."""
    number: int | None = None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = int(value)
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, Decimal128):
        decimal = value.to_decimal()
        if decimal.is_finite() and decimal == decimal.to_integral_value():
            number = int(decimal)
    if number is None or not _INT64_MIN <= number <= _INT64_MAX:
        return None
    return number


def _to_double(value: Any) -> float | None:
    """A number as DOUBLE. NaN reads as NULL, as it does in every other column."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal128):
        number = value.to_decimal()
        return float(number) if number.is_finite() else None
    return None


def _to_decimal(value: Any, scale: int) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, Decimal128):
        number = value.to_decimal()
    else:
        return None
    # copy_abs, not abs(): abs() rounds to the context's 28 digits, which would
    # push a 34-digit value that fits DECIMAL(38, 18) over the bound.
    if not number.is_finite() or number.copy_abs() >= Decimal(10) ** (38 - scale):
        return None
    return str(number)


def _scale_of(sql_type: str) -> int:
    match = re.search(r",\s*(\d+)\)", sql_type)
    return int(match.group(1)) if match else 0


def convert(column: _Column, value: Any) -> Any:
    """The cell *value* becomes in *column*.

    Strict for typed columns: a value of another BSON type reads as NULL,
    never as a coerced value. The filter pushdown relies on that -- MongoDB
    compares values of different BSON types as unequal, so the string ``"42"``
    in a ``BIGINT`` column must not read as ``42`` and then satisfy a
    predicate the server already applied.
    """
    if value is None:
        return None
    conversion = column.conversion
    if conversion == "json":
        return json.dumps(_json_value(value), ensure_ascii=False)
    if conversion == "string":
        return value if isinstance(value, str) else None
    if conversion == "objectid":
        if isinstance(value, ObjectId):
            return str(value)
        return value if isinstance(value, str) else None
    if conversion == "int":
        return _to_int(value)
    if conversion == "double":
        return _to_double(value)
    if conversion == "decimal":
        return _to_decimal(value, _scale_of(column.sql_type))
    if conversion == "bool":
        return value if isinstance(value, bool) else None
    if conversion in ("date", "timestamp", "datetime"):
        if conversion != "timestamp" and isinstance(value, datetime):
            return _naive_utc(value)
        if conversion != "date" and isinstance(value, Timestamp):
            return _naive_utc(value.as_datetime())
        return None
    if conversion == "uuid":
        return str(value) if isinstance(value, uuid.UUID) else None
    if conversion == "binary":
        if isinstance(value, (bytes, bytearray)):
            return base64.b64encode(bytes(value)).decode("ascii")
        return None
    return _text(value)


# Nullable extension dtypes: a plain float64 column would turn every missing
# value into NaN, which DuckDB keeps as NaN rather than NULL. Microsecond
# timestamps reach year 9999, which nanosecond ones do not.
_PANDAS_DTYPES = {
    "BIGINT": "Int64",
    "DOUBLE": "Float64",
    "BOOLEAN": "boolean",
    "TIMESTAMP": "datetime64[us]",
}


def _series(column: _Column, cells: list[Any]) -> Any:
    return pd.array(cells, dtype=_PANDAS_DTYPES.get(column.sql_type, object))


# ---------------------------------------------------------------------------
# SQL analysis
# ---------------------------------------------------------------------------


def _arg(node: exp.Expression, name: str) -> Any:
    """``node.args[name]``, under either spelling sqlglot has used for it."""
    value = node.args.get(f"{name}_")
    return value if value is not None else node.args.get(name)


def _parse_query(sql: str) -> exp.Query:
    try:
        statements = [s for s in sqlglot.parse(sql, read=_DIALECT) if s is not None]
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"Could not parse the SQL statement: {exc}") from exc
    if len(statements) != 1:
        raise ValueError("The MongoDB connector runs exactly one statement at a time")
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise ValueError(
            "The MongoDB connector is read-only: only SELECT statements can run"
        )
    return statement


def _referenced_names(
    statement: exp.Expression, relation_names: set[str]
) -> set[str] | None:
    """Lower-cased names the statement may read columns by; ``None`` for all.

    Deliberately a superset: every part of every column reference, whichever
    table it belongs to. Fetching a field nobody reads costs some transfer;
    missing one the statement reads breaks it. These need every column:
    ``SELECT *`` and ``t.*``; ``COLUMNS(...)``; ``#n`` positions;
    ``NATURAL JOIN``; ``PIVOT``, which groups by every column it is not told
    about; a column list on an alias, which renames by position; and a table
    used as a value (DuckDB reads ``SELECT t FROM t`` as the whole row).
    """
    for star in statement.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            return None
    if statement.find(exp.Columns, exp.PositionalColumn, exp.Pivot):
        return None
    for function in statement.find_all(exp.Anonymous):
        if str(function.name).lower() == "columns":
            return None
    for alias in statement.find_all(exp.TableAlias):
        if alias.columns:
            return None

    names: set[str] = set()
    for join in statement.find_all(exp.Join):
        if (join.text("method") or "").upper() == "NATURAL":
            return None
        for identifier in join.args.get("using") or []:
            names.add(identifier.name.lower())
    for column in statement.find_all(exp.Column):
        if not column.table and column.name.lower() in relation_names:
            return None
        names.update(part.name.lower() for part in column.parts)
    # sqlglot reads CURRENT_DATE and its kin as functions, but DuckDB binds a
    # column of that name first.
    for function in statement.find_all(exp.Func):
        if function.args.get("this") is None and not function.expressions:
            names.add(function.sql(dialect=_DIALECT).lower().rstrip("()"))
    return names


def _conjuncts(condition: exp.Expression) -> Iterable[exp.Expression]:
    while isinstance(condition, exp.Paren):
        condition = condition.this
    if isinstance(condition, exp.And):
        yield from _conjuncts(condition.this)
        yield from _conjuncts(condition.expression)
    else:
        yield condition


def _function_names(node: exp.Expression) -> set[str]:
    return {
        str(
            function.name if isinstance(function, exp.Anonymous) else function.key
        ).lower()
        for function in node.find_all(exp.Func)
    }


def _is_constant(node: exp.Expression, *, clock: bool) -> bool:
    """Whether *node* has one value the server may be given in its place.

    A ``COLLATE`` changes how the value compares, which a server-side filter
    cannot carry, so it disqualifies the node. *clock* allows functions that
    read the clock; see :data:`_CLOCK_FUNCTIONS` for when that is safe.
    """
    if node.find(
        exp.Column,
        exp.Star,
        exp.Placeholder,
        exp.Parameter,
        exp.Query,
        exp.AggFunc,
        exp.Window,
        exp.Collate,
    ):
        return False
    names = _function_names(node)
    if names & _VOLATILE_FUNCTIONS:
        return False
    return clock or not names & _CLOCK_FUNCTIONS


def _rises_with_the_clock(node: exp.Expression) -> bool:
    """Whether *node* can only stay level or go up as time passes.

    A lower bound read from the clock is safe only then: DuckDB evaluates it
    after the connector did, so it must come out the same or higher. Proven
    for the shapes "last N days" questions take -- the clock itself, shifted
    by a constant, truncated, or cast to a date or timestamp -- and refused
    for anything else, such as ``60 - date_part('second', now())``.
    """
    while isinstance(node, exp.Paren):
        node = node.this
    if not _function_names(node) & _CLOCK_FUNCTIONS:
        return True  # does not read the clock at all
    if isinstance(node, exp.Anonymous):  # now(), get_current_timestamp()...
        name = str(node.name).lower()
        return (
            name in _CLOCK_FUNCTIONS
            and name not in _TIME_OF_DAY_FUNCTIONS
            and not node.expressions
        )
    if isinstance(node, exp.Func) and node.key in _CLOCK_FUNCTIONS:
        # CURRENT_DATE, CURRENT_TIMESTAMP(3)... but not a time of day, which
        # wraps at midnight.
        return node.key not in _TIME_OF_DAY_FUNCTIONS
    if isinstance(node, exp.Add):
        return _rises_with_the_clock(node.this) and _rises_with_the_clock(
            node.expression
        )
    if isinstance(node, (exp.Sub, exp.DateAdd, exp.DateSub)):
        return _rises_with_the_clock(node.this) and not (
            _function_names(node.expression) & _CLOCK_FUNCTIONS
        )
    if isinstance(node, (exp.TimestampTrunc, exp.DateTrunc)):
        return _rises_with_the_clock(node.this)
    if isinstance(node, (exp.Cast, exp.TryCast)):
        return node.to.is_type(*_INSTANT_TYPES) and _rises_with_the_clock(node.this)
    return False


def _int_literal(node: exp.Expression | None) -> int | None:
    """A plain non-negative integer ``LIMIT``/``OFFSET``; ``None`` otherwise."""
    if node is None:
        return None
    # ``LIMIT 10 PERCENT`` and the like keep the number in ``expression`` and
    # the meaning elsewhere.
    if any(value for key, value in node.args.items() if key != "expression"):
        return None
    value = node.expression
    if isinstance(value, exp.Literal) and value.is_int:
        return int(value.this)
    return None


def _ceil_millisecond(value: datetime) -> datetime:
    """*value* rounded up to BSON's millisecond precision.

    PyMongo truncates a datetime to milliseconds, which moves an upper bound
    down past documents DuckDB would still find below it.
    """
    remainder = value.microsecond % 1000
    return value + timedelta(microseconds=1000 - remainder) if remainder else value


@dataclass(frozen=True)
class _Deadline:
    """When the statement running now has to stop; no timeout leaves it open."""

    started: float
    timeout_s: float | None

    def remaining_ms(self, cap: float | None = None) -> float | None:
        """Milliseconds left, at most *cap*; raises once none are left."""
        if self.timeout_s is None:
            return cap
        remaining = (self.started + self.timeout_s - time.monotonic()) * 1000
        if remaining <= 0:
            raise StatementTimeout(self.timeout_s)
        return remaining if cap is None else min(cap, remaining)

    def check(self) -> None:
        if self.timeout_s is not None:
            self.remaining_ms()

    def nearly_over(self) -> bool:
        """Whether a timeout raised now came from this deadline.

        PyMongo's own timeout stops an operation up to a round trip early. One
        raised well before the deadline -- server selection giving up after
        ``serverSelectionTimeoutMS`` -- is a connection failure, not this.
        """
        if self.timeout_s is None:
            return False
        elapsed = time.monotonic() - self.started
        return elapsed >= self.timeout_s * _DEADLINE_SHARE


#: Ingestion runs without a statement timeout.
_NO_DEADLINE = _Deadline(0.0, None)


@dataclass
class _Scan:
    """What to fetch from one collection for one statement."""

    collection: _Collection
    fields: list[str]
    filter: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None


class _Pushdown:
    """Translate the ``WHERE`` of a single-collection ``SELECT`` for the server.

    Each translated predicate must match every document whose row passes it
    in DuckDB; it may match more, because DuckDB applies the whole ``WHERE``
    again after loading. So only top-level ``AND`` conjuncts are considered,
    only columns whose conversion is strict take part, and constants are
    evaluated by DuckDB itself so the server compares against the very value
    DuckDB will.
    """

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        collection: _Collection,
        table: exp.Table,
    ) -> None:
        self._connection = connection
        self._collection = collection
        self._qualifiers = {table.name.lower(), table.alias_or_name.lower()}

    def filter(self, where: exp.Where | None) -> dict[str, Any]:
        if where is None:
            return {}
        clauses = [
            clause
            for condition in _conjuncts(where.this)
            if (clause := self._translate(condition))
        ]
        if not clauses:
            return {}
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _translate(self, condition: exp.Expression) -> dict[str, Any] | None:
        if isinstance(condition, exp.Not):
            inner = condition.this
            while isinstance(inner, exp.Paren):
                inner = inner.this
            # ``IS NOT NULL`` only. Never ``IS NULL``: a value of the wrong BSON
            # type reads as NULL too, and the server would not return it.
            if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
                column = self._column(inner.this)
                if column is not None and column.conversion in _STRICT:
                    return {column.name: {"$ne": None}}
            return None

        if isinstance(condition, exp.In):
            column = self._column(condition.this)
            # No DOUBLE: each value would have to become a range of its own.
            if (
                column is None
                or column.conversion == "double"
                or any(condition.args.get(k) for k in ("query", "unnest", "field"))
            ):
                return None
            values: list[Any] = []
            for item in condition.expressions:
                converted = self._values(column, item, lower=False)
                if converted is None:
                    return None
                values.extend(converted)
            return {column.name: {"$in": values}} if values else None

        if isinstance(condition, exp.Between):
            column = self._column(condition.this)
            # SYMMETRIC swaps the bounds when the first is the larger.
            if (
                column is None
                or column.conversion not in _ORDERED
                or condition.args.get("symmetric")
            ):
                return None
            low = self._bound(column, condition.args.get("low"), "$gte")
            high = self._bound(column, condition.args.get("high"), "$lte")
            if low is None or high is None:
                return None
            return {column.name: {**low, **high}}

        operator = _COMPARISONS.get(type(condition))
        if operator is None:
            return None
        column, operand = self._column(condition.this), condition.expression
        if column is None:
            column, operand = self._column(condition.expression), condition.this
            operator = _FLIPPED[operator]
        if column is None:
            return None
        if operator != "$eq":
            if column.conversion not in _ORDERED:
                return None
            bound = self._bound(column, operand, operator)
            return {column.name: bound} if bound else None
        if column.conversion == "double":
            low = self._bound(column, operand, "$gte", clock=False)
            high = self._bound(column, operand, "$lte")
            return {column.name: {**low, **high}} if low and high else None
        values = self._values(column, operand, lower=False)
        if not values:
            return None
        if len(values) > 1:  # an id that may be stored either way
            return {column.name: {"$in": values}}
        return {column.name: {"$eq": values[0]}}

    def _bound(
        self,
        column: _Column,
        node: exp.Expression | None,
        operator: str,
        *,
        clock: bool = True,
    ) -> dict[str, Any] | None:
        """One side of a range, widened so it can only let more documents in.

        DuckDB compares a DOUBLE column in DOUBLE, after both the constant and
        every stored value -- integers and decimals included -- have been
        rounded to the nearest double. The server compares the stored values
        exactly. A bound moved one step outwards still admits every value that
        rounds onto it. A time bound is rounded up to a whole millisecond for
        the same reason: PyMongo would otherwise truncate it.
        """
        lower = operator in ("$gt", "$gte")
        values = self._values(column, node, lower=lower and clock)
        if not values:
            return None
        value = values[0]
        if column.conversion == "double":
            if lower:
                return {"$gte": math.nextafter(value, -math.inf)}
            return {"$lte": math.nextafter(value, math.inf)}
        if column.conversion == "date" and not lower:
            try:
                value = _ceil_millisecond(value)
            except OverflowError:
                return None
        return {operator: value}

    def _column(self, node: exp.Expression | None) -> _Column | None:
        if not isinstance(node, exp.Column) or not isinstance(
            node.this, exp.Identifier
        ):
            return None
        if node.table and node.table.lower() not in self._qualifiers:
            return None
        column = self._collection.column(node.name)
        # A dotted or $-prefixed name is a path or an operator to the server,
        # not the top-level field it is here.
        if column is None or "." in column.name or column.name.startswith("$"):
            return None
        return column

    def _evaluate(self, sql: str) -> tuple[bool, Any]:
        try:
            row = self._connection.execute(f"SELECT {sql}").fetchone()
        except duckdb.Error:
            return False, None
        return True, row[0] if row else None

    def _values(
        self, column: _Column, node: exp.Expression | None, *, lower: bool
    ) -> list[Any] | None:
        """*node*'s value(s) as the server should compare them, or ``None``.

        *lower* says the value is a lower bound, the only place a value read
        from the clock is safe.
        """
        if node is None or not _is_constant(
            node, clock=lower and _rises_with_the_clock(node)
        ):
            return None
        rendered = node.sql(dialect=_DIALECT)
        conversion = column.conversion

        if conversion == "double":
            # DuckDB casts the constant to DOUBLE, whatever its own type.
            ok, value = self._evaluate(f"CAST(({rendered}) AS DOUBLE)")
            if ok and isinstance(value, float) and math.isfinite(value):
                return [value]
            return None

        ok, value = self._evaluate(rendered)
        if not ok or value is None:
            return None
        if conversion == "int":
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return [value] if _INT64_MIN <= value <= _INT64_MAX else None
            if isinstance(value, Decimal):
                # BIGINT against DECIMAL is exact in DuckDB, as on the server.
                try:
                    return [Decimal128(value)]
                except (ArithmeticError, TypeError, ValueError):
                    return None
            # Against a DOUBLE, DuckDB rounds the integers; the server does not.
            return None
        if conversion == "string":
            return [value] if isinstance(value, str) else None
        if conversion == "bool":
            return [value] if isinstance(value, bool) else None
        if conversion == "date":
            if isinstance(value, str):
                # DuckDB casts a string compared with a TIMESTAMP to TIMESTAMP.
                # Let it, so the server gets the instant DuckDB will use.
                ok, value = self._evaluate(f"CAST(({rendered}) AS TIMESTAMP)")
                if not ok:
                    return None
            if isinstance(value, datetime):
                value = _naive_utc(value)
            elif isinstance(value, date):
                value = datetime(value.year, value.month, value.day)
            else:
                return None
            # 'infinity' is not an instant the server could compare with.
            return None if value in _TIMESTAMP_SENTINELS else [value]
        if conversion == "objectid":
            if not isinstance(value, str):
                return None
            # The column reads ObjectIds and strings alike, so match both.
            return [ObjectId(value), value] if ObjectId.is_valid(value) else [value]
        return None


def _pushable_limit(select: exp.Select) -> int | None:
    """``LIMIT`` (plus ``OFFSET``) of a select that keeps one row per document."""
    for key, value in select.args.items():
        if value is None or value is False or value == []:
            continue
        if key not in ("expressions", "from", "from_", "limit", "offset"):
            return None
    limit = _int_literal(select.args.get("limit"))
    if limit is None or limit < 1:
        return None
    offset = select.args.get("offset")
    skip = _int_literal(offset) if offset is not None else 0
    if skip is None or skip < 0:
        return None
    for projection in select.expressions:
        for node in projection.walk():
            if not isinstance(node, _ROW_PRESERVING):
                return None
    return limit + skip


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class MongoDBDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` over one MongoDB or DocumentDB database.

    Parameters
    ----------
    connection_string:
        A ``mongodb://`` or ``mongodb+srv://`` URL whose path names the
        database. See the module docstring for the connector's own options.
    """

    supports_statement_timeout = True

    def __init__(self, connection_string: str) -> None:
        settings = _parse_connection_string(connection_string)
        self._database_name = settings.database
        self._sample_size = settings.sample_size
        self._max_documents = settings.max_documents
        self._ca_file: str | None = None

        options: dict[str, Any] = {
            "datetime_conversion": DatetimeConversion.DATETIME_AUTO,
            "tz_aware": False,
        }
        if "uuidrepresentation" not in settings.uri_options:
            options["uuidRepresentation"] = "standard"
        if "appname" not in settings.uri_options:
            options["appname"] = "auto-ontology"
        if "serverselectiontimeoutms" not in settings.uri_options:
            options["serverSelectionTimeoutMS"] = _SERVER_SELECTION_TIMEOUT_MS
        if not settings.uri_options & {"sockettimeoutms", "timeoutms"}:
            options["socketTimeoutMS"] = _SOCKET_TIMEOUT_MS
        if settings.tls_ca_pem is not None:
            self._ca_file = _write_ca_file(settings.tls_ca_pem)
            options["tlsCAFile"] = self._ca_file
            if not settings.uri_options & {"tls", "ssl"}:
                options["tls"] = True

        try:
            self._client: MongoClient = MongoClient(settings.uri, **options)
        except Exception:
            self._remove_ca_file()
            raise
        self._database = self._client.get_database(self._database_name)

        # Elastic DocumentDB clusters reject maxTimeMS. Learned from the first
        # rejection; the client-side deadline still applies without it.
        self._server_time_limits = True
        self._lock = threading.Lock()
        self._relations: dict[str, bool] = {}
        self._relations_at: float | None = None
        self._schemas: dict[str, _Collection] = {}
        logger.debug("MongoDB connector created (database=%r).", self._database_name)

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Statements run on DuckDB, so DuckDB is the dialect they are written in.
        Must be a member of ``sqlglot.dialects.DIALECTS`` -- callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return _DIALECT

    @property
    def database_name(self) -> str:
        return self._database_name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Check that the server is reachable and the database readable.

        Lists collections rather than sending ``ping``, which needs no
        credentials and would pass with a wrong password.
        """
        self._database.list_collection_names()

    def close(self) -> None:
        """Close the client and remove the temporary CA bundle, if any."""
        try:
            self._client.close()
        finally:
            self._remove_ca_file()

    def _remove_ca_file(self) -> None:
        if self._ca_file is None:
            return
        try:
            os.unlink(self._ca_file)
        except FileNotFoundError:
            pass
        self._ca_file = None

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_tables(self) -> pd.DataFrame:
        """Collections and views, sampled afresh."""
        snapshot = self._snapshot(refresh=True)
        rows = [
            {
                "table_schema": SCHEMA,
                "table_name": name,
                "table_type": (
                    TableTypes.VIEW if collection.is_view else TableTypes.BASE_TABLE
                ),
            }
            for name, collection in sorted(snapshot.items())
        ]
        return pd.DataFrame(rows, columns=["table_schema", "table_name", "table_type"])

    def get_columns(self) -> pd.DataFrame:
        snapshot = self._snapshot()
        rows = [
            {
                "table_schema": SCHEMA,
                "table_name": name,
                "column_name": column.name,
                "data_type": column.sql_type,
                "is_nullable": column.nullable,
                "ordinal_position": position,
            }
            for name, collection in sorted(snapshot.items())
            for position, column in enumerate(collection.columns, start=1)
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "table_schema",
                "table_name",
                "column_name",
                "data_type",
                "is_nullable",
                "ordinal_position",
            ],
        )

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """MongoDB keeps no SQL history."""
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        """Always empty: a MongoDB view is a pipeline, not SQL to parse.

        Views are still listed by :meth:`get_tables`, as ``view`` tables.
        """
        return pd.DataFrame(columns=["table_schema", "table_name", "view_definition"])

    def get_pks(self) -> pd.DataFrame:
        """``_id`` of every collection; views have no key of their own."""
        snapshot = self._snapshot()
        rows = [
            {
                "table_schema": SCHEMA,
                "table_name": name,
                "column_name": "_id",
                "ordinal_position": 1,
            }
            for name, collection in sorted(snapshot.items())
            if not collection.is_view and collection.column("_id") is not None
        ]
        return pd.DataFrame(
            rows,
            columns=["table_schema", "table_name", "column_name", "ordinal_position"],
        )

    def get_fks(self) -> pd.DataFrame:
        """Always empty: MongoDB has no declared references."""
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

    def _list_relations(
        self, *, refresh: bool = False, deadline: _Deadline | None = None
    ) -> dict[str, bool]:
        """Collection name -> whether it is a view."""
        with self._lock:
            fresh = (
                self._relations_at is not None
                and time.monotonic() - self._relations_at < _SCHEMA_TTL_SECONDS
            )
            if fresh and not refresh:
                return dict(self._relations)

        relations: dict[str, bool] = {}
        # Filtered here rather than with a ``type`` filter on the command:
        # DocumentDB does not support that key.
        infos = self._call(
            deadline or _NO_DEADLINE,
            lambda _limit: list(self._database.list_collections()),
        )
        for info in infos:
            name = info.get("name")
            if not isinstance(name, str) or not name or name.startswith("system."):
                continue
            kind = info.get("type", "collection")
            if kind not in ("collection", "view", "timeseries"):
                continue
            relations[name] = kind == "view"

        with self._lock:
            self._relations = relations
            self._relations_at = time.monotonic()
        return dict(relations)

    def _snapshot(self, *, refresh: bool = False) -> dict[str, _Collection]:
        """Every relation that could be typed, sampled in parallel.

        A relation that cannot be sampled is left out of every frame, per the
        :func:`auto_ontology.catalog.extract._validate_relation_column_coverage`
        contract. When none can be, the failure is not about one collection,
        so it is raised rather than reported as an empty database.
        """
        relations = self._list_relations(refresh=refresh)
        if not relations:
            logger.warning(
                "MongoDB database %r has no collections the connector can read",
                self._database_name,
            )
            return {}

        def load(item: tuple[str, bool]) -> tuple[str, _Collection | Exception]:
            name, is_view = item
            try:
                return name, self._collection(name, is_view, refresh=refresh)
            except Exception as exc:  # noqa: BLE001 -- reported below
                return name, exc

        workers = min(_SAMPLING_WORKERS, len(relations))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(load, sorted(relations.items())))

        snapshot: dict[str, _Collection] = {}
        failures: list[Exception] = []
        for name, result in results:
            if isinstance(result, Exception):
                logger.warning(
                    "Skipping MongoDB collection %r: sampling failed (%s)",
                    name,
                    result,
                )
                failures.append(result)
            else:
                snapshot[name] = result
        if failures and not snapshot:
            raise failures[0]
        return snapshot

    def _collection(
        self,
        name: str,
        is_view: bool,
        *,
        refresh: bool = False,
        deadline: _Deadline | None = None,
    ) -> _Collection:
        """The inferred schema of *name*, sampled when stale or on *refresh*.

        Outside *refresh*, a re-sample only adds columns: every column the
        previous sample had keeps its name and its type. The catalog, and the
        statements written against it, were built on those; a column that
        turned from BIGINT into VARCHAR between two statements would make
        ``SUM`` fail and ``>`` compare text.
        """
        with self._lock:
            cached = self._schemas.get(name)
        if (
            cached is not None
            and not refresh
            and time.monotonic() - cached.inferred_at < _SCHEMA_TTL_SECONDS
        ):
            return cached

        # An empty collection, or a view projecting everything away, still has
        # to be a table with a column.
        sampled = infer_columns(self._sample(name, deadline or _NO_DEADLINE))
        columns = sampled or [_PLACEHOLDER_ID]
        if cached is not None and not refresh:
            known = {column.name.lower() for column in cached.columns}
            columns = [*cached.columns]
            columns += [c for c in sampled if c.name.lower() not in known]

        collection = _Collection(name, is_view, tuple(columns), time.monotonic())
        with self._lock:
            self._schemas[name] = collection
        return collection

    def _scope(
        self, deadline: _Deadline, cap_ms: float | None = None
    ) -> tuple[AbstractContextManager[Any], int | None]:
        """The context one MongoDB operation runs in, and the maxTimeMS to send.

        Under a statement deadline, on a server that accepts maxTimeMS,
        PyMongo's own timeout bounds every wait in the operation -- server
        selection, each socket read, and the server's work, through a
        maxTimeMS it derives itself -- so none is sent explicitly. Otherwise
        only an explicit maxTimeMS of *cap_ms* bounds the server's work, and
        ``socketTimeoutMS`` bounds a connection that stops answering.
        """
        if deadline.timeout_s is not None and self._server_time_limits:
            return pymongo.timeout(deadline.remaining_ms(cap_ms) / 1000), None
        if cap_ms is None or not self._server_time_limits:
            return nullcontext(), None
        return nullcontext(), max(1, int(cap_ms))

    def _call(
        self,
        deadline: _Deadline,
        operation: Callable[[int | None], Any],
        cap_ms: float | None = None,
    ) -> Any:
        """Run *operation* in :meth:`_scope`, passing it the maxTimeMS to send.

        A server that refuses maxTimeMS -- an elastic DocumentDB cluster --
        refuses the one PyMongo derives as well, so the operation is retried
        once without either.
        """
        for attempt in range(2):
            scope, limit = self._scope(deadline, cap_ms)
            try:
                with scope:
                    return operation(limit)
            except ExecutionTimeout:
                raise
            except OperationFailure as exc:
                if attempt == 0 and self._time_limit_rejected(exc):
                    continue
                raise
        raise AssertionError("unreachable: the second attempt returns or raises")

    def _time_limit_rejected(self, exc: OperationFailure) -> bool:
        """Whether *exc* is the server refusing ``maxTimeMS`` itself.

        Not a statement that ran out of time: that error's text names
        ``MaxTimeMSExpired`` too, but means the limit worked.
        """
        if (
            not self._server_time_limits
            or isinstance(exc, ExecutionTimeout)
            or exc.code == 50
            or "maxtimems" not in str(exc).lower()
        ):
            return False
        logger.info(
            "MongoDB server for %r does not accept maxTimeMS; relying on the "
            "client-side deadline",
            self._database_name,
        )
        self._server_time_limits = False
        return True

    def _sample(self, name: str, deadline: _Deadline) -> list[dict[str, Any]]:
        collection = self._database.get_collection(name)
        pipeline = [{"$sample": {"size": self._sample_size}}]

        def sample(limit: int | None) -> list[dict[str, Any]]:
            options = {} if limit is None else {"maxTimeMS": limit}
            return list(collection.aggregate(pipeline, **options))

        def first(limit: int | None) -> list[dict[str, Any]]:
            options = {} if limit is None else {"max_time_ms": limit}
            return list(collection.find({}, limit=self._sample_size, **options))

        try:
            return self._call(deadline, sample, _SAMPLE_TIME_LIMIT_MS)
        except PyMongoError as exc:
            if getattr(exc, "timeout", False):
                deadline.check()  # the statement's time, not the sample's, is up
            elif not isinstance(exc, OperationFailure):
                raise
            logger.info(
                "$sample failed on %r (%s); typing the first documents instead",
                name,
                exc,
            )
        return self._call(deadline, first, _SAMPLE_TIME_LIMIT_MS)

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
        """Run one ``SELECT`` over the collections it names.

        Parameters
        ----------
        sql:
            A single DuckDB ``SELECT``.
        parameters:
            Optional positional parameters, bound by DuckDB.
        timeout_s:
            Stop after this many seconds and raise :class:`StatementTimeout`.
            Covers everything the statement does: listing and sampling
            collections, evaluating constants for pushdown, fetching the
            documents, and running the statement.
        """
        deadline = _Deadline(time.monotonic(), timeout_s)
        statement = _parse_query(sql)

        connection = self._open_duckdb()
        timer, fired = _arm_interrupt(connection, deadline)
        try:
            try:
                for scan in self._plan(statement, connection, deadline):
                    self._load(connection, scan, deadline)
                return _run(connection, sql, parameters, deadline)
            except PyMongoError as exc:
                if timeout_s is not None and exc.timeout and deadline.nearly_over():
                    raise StatementTimeout.since(timeout_s, deadline.started) from exc
                raise
            except duckdb.InterruptException as exc:
                if timeout_s is not None and fired.is_set():
                    raise StatementTimeout(timeout_s) from exc
                raise
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
            connection.close()

    def _open_duckdb(self) -> duckdb.DuckDBPyConnection:
        """A private, locked-down in-memory database for one statement.

        It is attached under the MongoDB database's name, so
        ``shop.main.orders`` and ``shop.orders`` resolve as well as
        ``main.orders`` and ``orders``.
        """
        connection = duckdb.connect(":memory:", config=dict(_DUCKDB_CONFIG))
        try:
            connection.execute("SET enable_external_access = false")
            name = quoted_identifier(self._database_name, _DIALECT)
            try:
                connection.execute(f"ATTACH ':memory:' AS {name}")
                connection.execute(f"USE {name}")
            except duckdb.Error:
                # A name DuckDB reserves (``memory``, ``temp``...): stay on the
                # default catalog, where bare and ``main.`` names still resolve.
                logger.debug("Cannot attach a DuckDB catalog named %s", name)
            connection.execute("SET TimeZone = 'UTC'")
            connection.execute("SET lock_configuration = true")
        except Exception:
            connection.close()
            raise
        return connection

    def _plan(
        self,
        statement: exp.Query,
        connection: duckdb.DuckDBPyConnection,
        deadline: _Deadline,
    ) -> list[_Scan]:
        tables = list(statement.find_all(exp.Table))

        relations = self._list_relations(deadline=deadline)
        deadline.check()
        refreshed = False
        references: list[tuple[exp.Table, str]] = []
        for table in tables:
            if not isinstance(table.this, exp.Identifier):
                continue  # a table function such as range(10)
            if _is_cte_reference(table):
                continue
            if not self._names_this_database(table):
                continue
            name = _match(table.name, relations)
            if name is None and not refreshed:
                # Possibly a collection created since the list was read.
                relations = self._list_relations(refresh=True, deadline=deadline)
                refreshed = True
                name = _match(table.name, relations)
            if name is not None:
                references.append((table, name))
        if not references:
            return []

        relation_names = {t.name.lower() for t in tables} | {
            t.alias_or_name.lower() for t in tables
        }
        names = _referenced_names(statement, relation_names)

        scans: dict[str, _Scan] = {}
        for _, name in references:
            if name in scans:
                continue
            collection = self._collection(
                name, relations.get(name, False), deadline=deadline
            )
            if names is None:
                fields = [c.name for c in collection.columns]
            else:
                fields = [
                    c.name
                    for c in collection.columns
                    if c.name == "_id" or c.name.lower() in names
                ] or [collection.columns[0].name]
            scans[name] = _Scan(collection, fields)

        single = _single_collection(statement, tables, references)
        if single is not None:
            table, name = single
            scan = scans[name]
            where = statement.args.get("where")
            scan.filter = _Pushdown(connection, scan.collection, table).filter(where)
            # A filter that may be a superset must not be paired with a
            # server-side limit: the server could return N documents DuckDB then
            # filters away while other matching ones exist.
            if where is None:
                scan.limit = _pushable_limit(statement)
        return list(scans.values())

    def _names_this_database(self, table: exp.Table) -> bool:
        """Whether a table reference can point into this database at all."""
        database = self._database_name.lower()
        catalog = table.catalog.lower()
        schema = table.db.lower()
        if catalog:
            return catalog == database and schema in ("", SCHEMA)
        return schema in ("", SCHEMA, database)

    def _too_many(self, collection: str) -> MongoDBDocumentLimitError:
        return MongoDBDocumentLimitError(
            f"Collection {collection!r} has more than {self._max_documents} "
            "documents for this statement, the most the MongoDB connector loads "
            "from one collection. Filter on a top-level field the server can "
            "apply, or raise max_documents in the connection string."
        )

    def _check_size(self, mongo: Any, scan: _Scan, deadline: _Deadline) -> None:
        """Fail before fetching what could only end in the document limit.

        Otherwise a statement over a large collection would stream
        ``max_documents`` documents just to raise. The metadata count is
        free; only a collection it puts over the limit is counted for real,
        on the server and stopping one past the limit, so nothing is
        transferred. Anything that goes wrong here leaves the limit to the
        fetch itself.
        """
        if scan.limit is not None and scan.limit <= self._max_documents:
            return
        if scan.collection.is_view:
            return

        def count(limit: int | None) -> int | None:
            if mongo.estimated_document_count() <= self._max_documents:
                return None
            options = {} if limit is None else {"maxTimeMS": limit}
            return mongo.count_documents(
                scan.filter, limit=self._max_documents + 1, **options
            )

        try:
            matched = self._call(deadline, count)
        except PyMongoError as exc:
            if exc.timeout:
                raise
            logger.debug("Counting %r before the fetch failed", mongo.name)
            return
        if matched is not None and matched > self._max_documents:
            raise self._too_many(scan.collection.name)

    def _load(
        self,
        connection: duckdb.DuckDBPyConnection,
        scan: _Scan,
        deadline: _Deadline,
    ) -> None:
        """Fetch *scan*'s documents and create its DuckDB table."""
        collection = scan.collection
        wanted = set(scan.fields)
        columns = [c for c in collection.columns if c.name in wanted]
        projection: dict[str, int] | None = {name: 1 for name in scan.fields}
        if any("." in name or name.startswith("$") for name in scan.fields):
            # The server reads such names as paths; fetch whole documents.
            projection = None
        elif "_id" not in wanted:
            projection["_id"] = 0

        mongo = self._database.get_collection(collection.name)
        self._check_size(mongo, scan, deadline)

        def fetch(limit: int | None) -> list[list[Any]]:
            cells: list[list[Any]] = [[] for _ in columns]
            options: dict[str, Any] = {}
            if scan.limit is not None:
                options["limit"] = scan.limit
            if limit is not None:
                options["max_time_ms"] = limit
            cursor = mongo.find(scan.filter, projection, **options)
            try:
                for count, document in enumerate(cursor, start=1):
                    if count > self._max_documents:
                        raise self._too_many(collection.name)
                    deadline.check()
                    for index, column in enumerate(columns):
                        cells[index].append(convert(column, document.get(column.name)))
            finally:
                cursor.close()
            return cells

        cells = self._call(deadline, fetch)
        frame = pd.DataFrame(
            {
                f"c{index}": _series(column, cells[index])
                for index, column in enumerate(columns)
            }
        )
        select_list = ", ".join(
            f"CAST(c{index} AS {column.sql_type}) AS "
            f"{quoted_identifier(column.name, _DIALECT)}"
            for index, column in enumerate(columns)
        )
        table = f"{SCHEMA}.{quoted_identifier(collection.name, _DIALECT)}"
        source = f"__ao_documents_{uuid.uuid4().hex}"
        connection.register(source, frame)
        try:
            connection.execute(
                f"CREATE TABLE {table} AS SELECT {select_list} FROM {source}"
            )
        finally:
            connection.unregister(source)


def _write_ca_file(pem: bytes) -> str:
    # mkstemp creates the file readable by this user only.
    handle, path = tempfile.mkstemp(prefix="auto-ontology-mongodb-", suffix=".pem")
    with os.fdopen(handle, "wb") as out:
        out.write(pem)
    return path


def _match(name: str, relations: Mapping[str, bool]) -> str | None:
    """The collection *name* resolves to: exact, else one case-insensitive match."""
    if name in relations:
        return name
    matches = [relation for relation in relations if relation.lower() == name.lower()]
    return matches[0] if len(matches) == 1 else None


def _single_collection(
    statement: exp.Query,
    tables: list[exp.Table],
    references: list[tuple[exp.Table, str]],
) -> tuple[exp.Table, str] | None:
    """The one collection a plain ``SELECT ... FROM c`` reads, if that is all."""
    if not isinstance(statement, exp.Select):
        return None
    if len(tables) != 1 or len(references) != 1:
        return None
    table, name = references[0]
    source = _arg(statement, "from")
    if source is None or source.this is not table:
        return None
    if table.args.get("sample") or _arg(statement, "with") is not None:
        return None
    # ``FROM c AS t(id, b, a)`` renames the fields by position, so the ``a``
    # in the WHERE is not the field the server knows as ``a``.
    alias = table.args.get("alias")
    if alias is not None and alias.columns:
        return None
    if any(
        statement.args.get(key) for key in ("joins", "laterals", "pivots", "sample")
    ):
        return None
    return table, name


def _is_cte_reference(table: exp.Table) -> bool:
    """Whether *table* names a CTE visible where it stands, not a collection.

    Resolved by scope, the way DuckDB does: a CTE is visible to the query that
    defines it and to the CTEs after it, and to its own body only under
    ``WITH RECURSIVE``. So in ``SELECT * FROM c WHERE EXISTS (WITH c AS (...)
    SELECT * FROM c)`` the outer ``c`` is still the collection, and in
    ``WITH orders AS (SELECT * FROM orders) ...`` the inner one is.
    """
    if table.db or table.catalog:
        return False
    name = table.name.lower()
    child: exp.Expression = table
    node = table.parent
    holder: exp.Expression | None = None
    while node is not None:
        if isinstance(node, exp.With):
            holder = child  # the CTE whose body holds the reference
        elif isinstance(node, exp.Query):
            with_ = _arg(node, "with")
            if with_ is not None:
                ctes = list(with_.expressions)
                names = [cte.alias_or_name.lower() for cte in ctes]
                if child is with_ and holder is not None:
                    position = next(i for i, cte in enumerate(ctes) if cte is holder)
                    if with_.args.get("recursive"):
                        position += 1
                    names = names[:position]
                if name in names:
                    return True
            holder = None
        child, node = node, node.parent
    return False


def _arm_interrupt(
    connection: duckdb.DuckDBPyConnection, deadline: _Deadline
) -> tuple[threading.Timer | None, threading.Event]:
    """Interrupt whatever DuckDB is running on *connection* at the deadline.

    Armed once for the whole statement, so constant evaluation for pushdown
    and loading the documents are covered as well as the statement itself.
    """
    fired = threading.Event()
    if deadline.timeout_s is None:
        return None, fired
    remaining = deadline.remaining_ms()

    def interrupt() -> None:
        fired.set()
        connection.interrupt()

    timer = threading.Timer(remaining / 1000, interrupt)
    timer.daemon = True
    timer.start()
    return timer, fired


def _run(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    parameters: Optional[list],
    deadline: _Deadline,
) -> pd.DataFrame:
    """Run *sql* on the loaded tables."""
    deadline.check()
    if parameters:
        connection.execute(sql, parameters)
    else:
        connection.execute(sql)
    types = [str(column[1]) for column in connection.description or []]
    frame = connection.df()

    # DuckDB returns JSON as text. Hand it on as Python values, the way
    # Postgres drivers return JSONB, which is what profiling expects.
    for index, sql_type in enumerate(types):
        if sql_type == "JSON":
            frame.isetitem(
                index,
                frame.iloc[:, index].map(
                    lambda value: json.loads(value) if isinstance(value, str) else value
                ),
            )
    return frame
