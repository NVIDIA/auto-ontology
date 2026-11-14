# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MongoDBDatabase against an in-process stand-in for the server.

The stand-in answers ``find`` with MongoDB's own matching rules for the
operators the connector sends -- type bracketing, and arrays matching when any
element does -- so the pushdown tests can compare every answer with the one
DuckDB gives when the server ignores the filter entirely. A pushed-down filter
is only correct if those two never differ.
"""

from __future__ import annotations

import base64
import copy
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterator

import duckdb
import pandas as pd
import pytest
import sqlglot
from bson import DBRef, Decimal128, ObjectId, Timestamp
from bson.datetime_ms import DatetimeMS
from pymongo.errors import (
    ExecutionTimeout,
    OperationFailure,
    ServerSelectionTimeoutError,
)
from pytest import MonkeyPatch

from auto_ontology.catalog.extract import create_dataframe
from auto_ontology.connectors import mongodb
from auto_ontology.connectors.base import StatementTimeout
from auto_ontology.connectors.mongodb import (
    MongoDBDatabase,
    MongoDBDocumentLimitError,
    convert,
    infer_columns,
)
from auto_ontology.connectors.registry import (
    CONNECTOR_REGISTRY,
    _redact,
    create_connector,
)

C1 = ObjectId("65a000000000000000000001")
C2 = ObjectId("65a000000000000000000002")
C3 = ObjectId("65a000000000000000000003")


def _documents() -> dict[str, list[dict[str, Any]]]:
    return {
        "customers": [
            {
                "_id": C1,
                "name": "Ada",
                "address": {"city": "Seoul", "zip": "04524"},
                "tags": ["vip"],
                "joined": datetime(2024, 1, 5),
                "score": 9.5,
                "active": True,
            },
            {
                "_id": C2,
                "name": "Grace",
                "address": {"city": "Busan"},
                "tags": [],
                "joined": datetime(2024, 2, 10),
                "score": 7,
                "active": False,
            },
            {
                "_id": C3,
                "name": "Linus",
                "joined": datetime(2024, 3, 1),
                "score": 8.25,
                "active": True,
            },
        ],
        "orders": [
            {
                "_id": ObjectId("65b000000000000000000001"),
                "customer_id": C1,
                "status": "paid",
                "qty": 2,
                "total": Decimal128("12.50"),
                "items": [{"sku": "a", "qty": 2}],
                "created_at": datetime(2024, 1, 10),
            },
            {
                "_id": ObjectId("65b000000000000000000002"),
                "customer_id": C1,
                "status": "shipped",
                "qty": 1,
                "total": Decimal128("5.25"),
                "items": [{"sku": "b", "qty": 1}],
                "created_at": datetime(2024, 2, 15),
            },
            {
                "_id": ObjectId("65b000000000000000000003"),
                "customer_id": C2,
                "status": "paid",
                "qty": 5,
                "total": Decimal128("40"),
                "items": [],
                "created_at": datetime(2024, 3, 20),
            },
            # Past the three-document sample: types the sample never saw.
            {
                "_id": ObjectId("65b000000000000000000004"),
                "customer_id": str(C3),
                "status": "cancelled",
                "qty": "3",
                "total": Decimal128("7.10"),
                "items": [],
                "created_at": datetime(2024, 3, 25),
            },
            {
                "_id": ObjectId("65b000000000000000000005"),
                "customer_id": C3,
                "status": ["paid"],
                "qty": 4.0,
                "total": 9,
                "items": None,
                "created_at": datetime(2024, 4, 1),
            },
        ],
        "paid_orders": [
            {"_id": ObjectId("65b000000000000000000001"), "status": "paid"},
            {"_id": ObjectId("65b000000000000000000003"), "status": "paid"},
        ],
        "status_counts": [{"status": "paid", "n": 2}],
        "events": [],
        "metrics": [
            {"_id": ObjectId("65c000000000000000000001"), "value": 1.5},
        ],
        "system.views": [{"_id": "shop.paid_orders"}],
    }


_TYPES = {
    "customers": "collection",
    "orders": "collection",
    "paid_orders": "view",
    "status_counts": "view",
    "events": "collection",
    "metrics": "timeseries",
    "system.views": "collection",
}


# ---------------------------------------------------------------------------
# Server stand-in
# ---------------------------------------------------------------------------


def _family(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float, Decimal128)):
        return "number"
    return type(value).__name__


def _comparable(value: Any) -> Any:
    if isinstance(value, Decimal128):
        return value.to_decimal()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Decimal(value)
    return value


def _compare(stored: Any, operator: str, operand: Any) -> bool:
    """MongoDB's rule: same BSON type family only, any element of an array."""
    candidates = stored if isinstance(stored, list) else [stored]
    for value in candidates:
        if value is None or _family(value) != _family(operand):
            continue
        left, right = _comparable(value), _comparable(operand)
        if {
            "$eq": left == right,
            "$gt": left > right,
            "$gte": left >= right,
            "$lt": left < right,
            "$lte": left <= right,
        }[operator]:
            return True
    return False


def _matches(document: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, condition in query.items():
        if key == "$and":
            if not all(_matches(document, clause) for clause in condition):
                return False
            continue
        value = document.get(key)
        if not isinstance(condition, dict):
            condition = {"$eq": condition}
        for operator, operand in condition.items():
            if operator == "$ne" and operand is None:
                # An array matches only when no element is null.
                if value is None or (isinstance(value, list) and None in value):
                    return False
            elif operator == "$in":
                if not any(_compare(value, "$eq", item) for item in operand):
                    return False
            elif operator in ("$eq", "$gt", "$gte", "$lt", "$lte"):
                if not _compare(value, operator, operand):
                    return False
            else:
                raise AssertionError(f"connector sent an unexpected {operator}")
    return True


def _project(document: dict[str, Any], projection: dict[str, int]) -> dict[str, Any]:
    keep = {name for name, flag in projection.items() if flag}
    if projection.get("_id", 1):
        keep.add("_id")
    return {name: value for name, value in document.items() if name in keep}


class FakeCursor:
    def __init__(self, documents: list[dict[str, Any]], delay: float) -> None:
        self._documents = documents
        self._delay = delay
        self.closed = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for document in self._documents:
            if self._delay:
                time.sleep(self._delay)
            yield document

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeServer:
    data: dict[str, list[dict[str, Any]]] = field(default_factory=_documents)
    types: dict[str, str] = field(default_factory=lambda: dict(_TYPES))
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    client_args: tuple[str, dict[str, Any]] | None = None
    ignore_filters: bool = False
    reject_time_limits: bool = False
    sample_error: Exception | None = None
    list_error: Exception | None = None
    sample_delay: float = 0.0
    delay: float = 0.0
    # Seconds of the pymongo.timeout() scope an operation runs in, if any.
    scope: float | None = None

    def finds(self, collection: str | None = None) -> list[tuple[Any, ...]]:
        return [
            call
            for call in self.calls
            if call[0] == "find" and collection in (None, call[1])
        ]

    def check_time_limit(self, value: Any) -> None:
        """Refuse maxTimeMS, explicit or the one pymongo.timeout() derives."""
        if (value is not None or self.scope is not None) and self.reject_time_limits:
            raise OperationFailure("Field 'maxTimeMS' is not supported", code=115)


class FakeCollection:
    def __init__(self, server: FakeServer, name: str) -> None:
        self._server = server
        self._name = name
        self.name = name

    def aggregate(self, pipeline: list[dict[str, Any]], **options: Any) -> Any:
        self._server.calls.append(
            ("aggregate", self._name, pipeline, options, self._server.scope)
        )
        self._server.check_time_limit(options.get("maxTimeMS"))
        time.sleep(self._server.sample_delay)
        if self._server.sample_error is not None:
            raise self._server.sample_error
        (stage,) = pipeline
        size = stage["$sample"]["size"]
        return iter(copy.deepcopy(self._server.data[self._name][:size]))

    def estimated_document_count(self) -> int:
        self._server.calls.append(
            ("estimated_document_count", self._name, self._server.scope)
        )
        self._server.check_time_limit(None)
        return len(self._server.data[self._name])

    def count_documents(
        self, query: dict[str, Any], limit: int = 0, **options: Any
    ) -> int:
        self._server.calls.append(("count_documents", self._name, query, limit))
        self._server.check_time_limit(options.get("maxTimeMS"))
        matched = sum(
            1
            for document in self._server.data[self._name]
            if self._server.ignore_filters or _matches(document, query)
        )
        return min(matched, limit) if limit else matched

    def find(
        self,
        query: dict[str, Any] | None = None,
        projection: dict[str, int] | None = None,
        limit: int = 0,
        max_time_ms: int | None = None,
    ) -> FakeCursor:
        self._server.calls.append(
            (
                "find",
                self._name,
                query,
                projection,
                limit,
                max_time_ms,
                self._server.scope,
            )
        )
        self._server.check_time_limit(max_time_ms)
        documents = [
            document
            for document in self._server.data[self._name]
            if self._server.ignore_filters or _matches(document, query or {})
        ]
        if limit:
            documents = documents[:limit]
        if projection is not None:
            documents = [_project(document, projection) for document in documents]
        return FakeCursor(copy.deepcopy(documents), self._server.delay)


class FakeDatabase:
    def __init__(self, server: FakeServer) -> None:
        self._server = server

    def list_collections(self) -> list[dict[str, Any]]:
        self._server.calls.append(("list_collections", self._server.scope))
        self._server.check_time_limit(None)
        if self._server.list_error is not None:
            raise self._server.list_error
        return [
            {"name": name, "type": kind} for name, kind in self._server.types.items()
        ]

    def list_collection_names(self) -> list[str]:
        self._server.calls.append(("list_collection_names",))
        return list(self._server.types)

    def get_collection(self, name: str) -> FakeCollection:
        return FakeCollection(self._server, name)


class FakeClient:
    def __init__(self, server: FakeServer, uri: str, **options: Any) -> None:
        server.client_args = (uri, options)
        self._server = server
        self.closed = False

    def get_database(self, name: str) -> FakeDatabase:
        assert name == "shop"
        return FakeDatabase(self._server)

    def close(self) -> None:
        self.closed = True


URL = "mongodb://reader:secret@localhost:27017/shop?authSource=admin&sample_size=3"


@pytest.fixture
def server(monkeypatch: MonkeyPatch) -> FakeServer:
    fake = FakeServer()
    monkeypatch.setattr(
        mongodb, "MongoClient", lambda uri, **options: FakeClient(fake, uri, **options)
    )

    @contextmanager
    def timeout(seconds: float) -> Iterator[None]:
        previous, fake.scope = fake.scope, seconds
        try:
            yield
        finally:
            fake.scope = previous

    monkeypatch.setattr(mongodb.pymongo, "timeout", timeout)
    return fake


@pytest.fixture
def connector(server: FakeServer) -> Iterator[MongoDBDatabase]:
    database = MongoDBDatabase(URL)
    yield database
    database.close()


def _same_answer(
    server: FakeServer, connector: MongoDBDatabase, sql: str
) -> pd.DataFrame:
    """Run *sql* with the pushdown, and again with the server ignoring filters."""
    pushed = connector.execute(sql)
    server.ignore_filters = True
    try:
        baseline = connector.execute(sql)
    finally:
        server.ignore_filters = False
    pd.testing.assert_frame_equal(pushed, baseline)
    return pushed


# ---------------------------------------------------------------------------
# Connection strings
# ---------------------------------------------------------------------------


def test_registry_maps_both_mongodb_schemes(server: FakeServer) -> None:
    assert CONNECTOR_REGISTRY["mongodb"] is MongoDBDatabase
    for url in (URL, "mongodb+srv://reader:secret@cluster0.example.net/shop"):
        connector = create_connector(url)
        try:
            assert isinstance(connector, MongoDBDatabase)
            assert connector.database_name == "shop"
            assert connector.dialect == "duckdb"
        finally:
            connector.close()


def test_connector_options_never_reach_pymongo(server: FakeServer) -> None:
    pem = b"-----BEGIN CERTIFICATE-----\nMIIBdummy\n-----END CERTIFICATE-----\n"
    data = base64.urlsafe_b64encode(pem).decode()
    connector = MongoDBDatabase(
        "mongodb://reader:secret@docdb.example.com:27017/shop?replicaSet=rs0"
        f"&max_documents=50&tls_ca_data={data}&readPreference=secondaryPreferred"
    )
    try:
        uri, options = server.client_args
        assert uri == (
            "mongodb://reader:secret@docdb.example.com:27017/shop"
            "?replicaSet=rs0&readPreference=secondaryPreferred"
        )
        assert options["tls"] is True
        assert options["appname"] == "auto-ontology"
        assert options["serverSelectionTimeoutMS"] == 10_000
        assert options["uuidRepresentation"] == "standard"
        ca_file = options["tlsCAFile"]
        with open(ca_file, "rb") as handle:
            assert handle.read() == pem
        assert os.stat(ca_file).st_mode & 0o077 == 0
    finally:
        connector.close()
    assert not os.path.exists(ca_file)


def test_settings_in_the_url_win_over_the_connector_defaults(
    server: FakeServer,
) -> None:
    connector = MongoDBDatabase(
        "mongodb://h/shop?appName=reports&serverSelectionTimeoutMS=500"
        "&uuidRepresentation=javaLegacy"
    )
    connector.close()
    _, options = server.client_args
    assert not {"appname", "serverSelectionTimeoutMS", "uuidRepresentation"} & set(
        options
    )


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("mongodb://localhost:27017/", "must name the database"),
        ("mongodb://localhost:27017", "must name the database"),
        ("postgresql://localhost/shop", "mongodb://"),
        ("mongodb://h/shop?sample_size=0", "sample_size"),
        ("mongodb://h/shop?max_documents=many", "max_documents"),
        ("mongodb://h/shop?tls_ca_data=bm90IGEgY2VydA", "PEM certificate"),
        (
            "mongodb://h/shop?tlsCAFile=/ca.pem&tls_ca_data=bm90IGEgY2VydA",
            "not both",
        ),
    ],
)
def test_invalid_connection_strings_are_rejected(
    server: FakeServer, url: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MongoDBDatabase(url)


def test_redact_hides_the_password_and_collapses_the_ca_bundle() -> None:
    redacted = _redact("mongodb://reader:pw123@h/shop?tls_ca_data=QUJDREVG&x=1")
    assert "pw123" not in redacted
    assert "QUJDREVG" not in redacted
    assert redacted.endswith("tls_ca_data=<certificates>&x=1")


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_tables_list_collections_views_and_time_series(
    connector: MongoDBDatabase,
) -> None:
    tables = connector.get_tables()
    assert tables.to_dict("records") == [
        {"table_schema": "main", "table_name": "customers", "table_type": "base table"},
        {"table_schema": "main", "table_name": "events", "table_type": "base table"},
        {"table_schema": "main", "table_name": "metrics", "table_type": "base table"},
        {"table_schema": "main", "table_name": "orders", "table_type": "base table"},
        {"table_schema": "main", "table_name": "paid_orders", "table_type": "view"},
        {"table_schema": "main", "table_name": "status_counts", "table_type": "view"},
    ]


def test_columns_are_typed_from_the_sample(connector: MongoDBDatabase) -> None:
    columns = connector.get_columns()
    customers = columns[columns["table_name"] == "customers"]
    assert customers[
        ["column_name", "data_type", "is_nullable", "ordinal_position"]
    ].to_dict("records") == [
        {
            "column_name": "_id",
            "data_type": "VARCHAR",
            "is_nullable": False,
            "ordinal_position": 1,
        },
        {
            "column_name": "name",
            "data_type": "VARCHAR",
            "is_nullable": None,
            "ordinal_position": 2,
        },
        {
            "column_name": "address",
            "data_type": "JSON",
            "is_nullable": True,
            "ordinal_position": 3,
        },
        {
            "column_name": "tags",
            "data_type": "JSON",
            "is_nullable": True,
            "ordinal_position": 4,
        },
        {
            "column_name": "joined",
            "data_type": "TIMESTAMP",
            "is_nullable": None,
            "ordinal_position": 5,
        },
        {
            "column_name": "score",
            "data_type": "DOUBLE",
            "is_nullable": None,
            "ordinal_position": 6,
        },
        {
            "column_name": "active",
            "data_type": "BOOLEAN",
            "is_nullable": None,
            "ordinal_position": 7,
        },
    ]
    types = dict(
        zip(
            columns.loc[columns["table_name"] == "orders", "column_name"],
            columns.loc[columns["table_name"] == "orders", "data_type"],
        )
    )
    assert types == {
        "_id": "VARCHAR",
        "customer_id": "VARCHAR",
        "status": "VARCHAR",
        "qty": "BIGINT",
        "total": "DECIMAL(38, 2)",
        "items": "JSON",
        "created_at": "TIMESTAMP",
    }
    # An empty collection still has its _id.
    events = columns[columns["table_name"] == "events"]
    assert events["column_name"].tolist() == ["_id"]


def test_keys_and_the_empty_frames(connector: MongoDBDatabase) -> None:
    assert sorted(connector.get_pks()["table_name"]) == [
        "customers",
        "events",
        "metrics",
        "orders",
    ]
    assert connector.get_fks().empty
    assert list(connector.get_fks().columns) == [
        "table_schema",
        "table_name",
        "column_name",
        "referenced_schema",
        "referenced_table",
        "referenced_column",
    ]
    assert connector.get_views().empty
    assert list(connector.get_queries().columns) == ["end_time", "query_text"]


def test_catalog_extraction_accepts_the_frames(connector: MongoDBDatabase) -> None:
    tables, columns, *_ = create_dataframe(connector)
    assert len(tables) == 6
    assert set(columns["table_name"]) == set(tables["table_name"])


def test_ingestion_samples_once_per_collection(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    create_dataframe(connector)
    sampled = [call[1] for call in server.calls if call[0] == "aggregate"]
    assert sorted(sampled) == [
        "customers",
        "events",
        "metrics",
        "orders",
        "paid_orders",
        "status_counts",
    ]


def test_a_collection_that_cannot_be_sampled_is_left_out(
    server: FakeServer, connector: MongoDBDatabase, monkeypatch: MonkeyPatch
) -> None:
    sample = MongoDBDatabase._sample

    def failing(self: MongoDBDatabase, name: str, *args: Any) -> list[dict[str, Any]]:
        if name == "orders":
            raise OperationFailure("not authorized on shop to execute command")
        return sample(self, name, *args)

    monkeypatch.setattr(MongoDBDatabase, "_sample", failing)
    tables, columns, *_ = create_dataframe(connector)
    assert "orders" not in set(tables["table_name"])
    assert "orders" not in set(columns["table_name"])


def test_a_database_that_cannot_be_sampled_at_all_raises(
    server: FakeServer, connector: MongoDBDatabase, monkeypatch: MonkeyPatch
) -> None:
    def failing(self: MongoDBDatabase, name: str, *args: Any) -> list[dict[str, Any]]:
        raise OperationFailure("not authorized on shop to execute command")

    monkeypatch.setattr(MongoDBDatabase, "_sample", failing)
    with pytest.raises(OperationFailure, match="not authorized"):
        connector.get_tables()


def test_sampling_falls_back_to_the_first_documents(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    server.sample_error = OperationFailure(
        "Unrecognized pipeline stage name: '$sample'"
    )
    columns = connector.get_columns()
    assert "customer_id" in set(columns["column_name"])
    assert ("find", "orders", {}, None, 3, 60_000, None) in server.calls


def test_ping_needs_a_readable_database(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    connector.ping()
    assert ("list_collection_names",) in server.calls


# ---------------------------------------------------------------------------
# Typing and conversion
# ---------------------------------------------------------------------------


def test_mixed_fields_get_a_type_that_holds_every_value() -> None:
    columns = infer_columns(
        [
            {"_id": 1, "a": 1, "b": 1, "c": 1, "d": datetime(2024, 1, 1), "e": [1]},
            {
                "_id": 2,
                "a": 2.5,
                "b": Decimal128("1.125"),
                "c": "x",
                "d": Timestamp(0, 1),
            },
            {"_id": 3, "e": "scalar", "": "skipped"},
        ]
    )
    assert [(c.name, c.sql_type, c.nullable) for c in columns] == [
        ("_id", "BIGINT", False),
        ("a", "DOUBLE", True),
        ("b", "DECIMAL(38, 3)", True),
        ("c", "VARCHAR", True),
        ("d", "TIMESTAMP", True),
        ("e", "JSON", True),
    ]


def test_names_differing_only_in_case_keep_the_more_frequent() -> None:
    columns = infer_columns(
        [{"_id": 1, "Name": "a"}, {"_id": 2, "name": "b"}, {"_id": 3, "name": "c"}]
    )
    assert [column.name for column in columns] == ["_id", "name"]


def test_values_of_another_type_read_as_null() -> None:
    (_, qty, when, flag, price) = infer_columns(
        [
            {
                "_id": 1,
                "qty": 1,
                "when": datetime(2024, 1, 1),
                "flag": True,
                "price": Decimal128("1.50"),
            }
        ]
    )
    assert convert(qty, "42") is None
    assert convert(qty, True) is None
    assert convert(qty, 3.0) == 3
    assert convert(qty, 3.5) is None
    assert convert(qty, Decimal128("2")) == 2
    assert convert(when, "2024-01-01") is None
    assert convert(when, DatetimeMS(2**62)) is None
    assert convert(
        when, datetime(2024, 1, 1, 9, tzinfo=timezone(timedelta(hours=9)))
    ) == datetime(2024, 1, 1)
    assert convert(flag, 1) is None
    assert convert(price, Decimal128("NaN")) is None
    assert convert(price, 7) == "7"


def test_nested_values_become_json() -> None:
    (_, nested) = infer_columns([{"_id": 1, "nested": {}}])
    rendered = convert(
        nested,
        {
            "id": ObjectId("65a000000000000000000001"),
            "at": datetime(2024, 1, 2, 3, 4, 5),
            "price": Decimal128("1.25"),
            "raw": b"\x00\x01",
            "ref": DBRef("customers", C1),
            "nan": float("nan"),
            "list": [1, "two", None],
        },
    )
    assert rendered == (
        '{"id": "65a000000000000000000001", "at": "2024-01-02T03:04:05Z", '
        '"price": 1.25, "raw": "AAE=", "ref": {"$ref": "customers", '
        '"$id": "65a000000000000000000001"}, "nan": null, "list": [1, "two", null]}'
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_profiling_sample_pushes_the_limit_and_reads_every_column(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    frame = connector.execute('SELECT * FROM "main"."orders" LIMIT 2')
    assert len(frame) == 2
    assert list(frame.columns) == [
        "_id",
        "customer_id",
        "status",
        "qty",
        "total",
        "items",
        "created_at",
    ]
    (find,) = server.finds("orders")
    assert find[2] == {}
    assert find[4] == 2
    assert frame.loc[0, "items"] == [{"sku": "a", "qty": 2}]


def test_distinct_probe_pushes_is_not_null(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    frame = _same_answer(
        server,
        connector,
        'SELECT DISTINCT "status" FROM "main"."orders" '
        'WHERE "status" IS NOT NULL ORDER BY "status" LIMIT 26',
    )
    assert frame["status"].tolist() == ["cancelled", "paid", "shipped"]
    query, projection, limit = server.finds("orders")[0][2:5]
    assert query == {"status": {"$ne": None}}
    assert projection == {"_id": 1, "status": 1}
    assert limit == 0


def test_filters_are_pushed_and_never_change_the_answer(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    frame = _same_answer(
        server,
        connector,
        "SELECT COUNT(*) AS n FROM orders o "
        "WHERE o.created_at >= DATE '2024-01-15' AND status IN ('paid', 'shipped') "
        "AND 1 < qty AND created_at BETWEEN '2024-01-01' AND '2024-12-31'",
    )
    assert frame["n"].tolist() == [1]
    assert server.finds("orders")[0][2] == {
        "$and": [
            {"created_at": {"$gte": datetime(2024, 1, 15)}},
            {"status": {"$in": ["paid", "shipped"]}},
            {"qty": {"$gt": 1}},
            {
                "created_at": {
                    "$gte": datetime(2024, 1, 1),
                    "$lte": datetime(2024, 12, 31),
                }
            },
        ]
    }


def test_values_the_sample_never_saw_do_not_leak_through_a_filter(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    frame = connector.execute("SELECT qty, status FROM orders ORDER BY created_at")
    qty, status = frame["qty"].tolist(), frame["status"].tolist()
    assert qty[:3] == [2, 1, 5]
    assert pd.isna(qty[3])  # "3": a string in a BIGINT column
    assert qty[4] == 4  # 4.0: a whole double
    assert pd.isna(status[4])  # ["paid"]: an array in a VARCHAR column
    # The server matches 4.0 but not "3" for qty > 1, exactly as DuckDB does.
    frame = _same_answer(
        server, connector, "SELECT COUNT(*) AS n FROM orders WHERE qty > 1"
    )
    assert frame["n"].tolist() == [3]


def test_relative_dates_are_evaluated_by_duckdb(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    connector.execute(
        "SELECT COUNT(*) FROM orders WHERE created_at >= current_date - INTERVAL 7 DAY"
    )
    query = server.finds("orders")[0][2]
    bound = query["created_at"]["$gte"]
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    assert bound.tzinfo is None
    assert timedelta(days=6) <= today - bound <= timedelta(days=8)


def test_an_id_matches_whichever_way_it_is_stored(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    frame = _same_answer(
        server,
        connector,
        f"SELECT COUNT(*) AS n FROM orders WHERE customer_id = '{C3}'",
    )
    assert frame["n"].tolist() == [2]
    assert server.finds("orders")[0][2] == {"customer_id": {"$in": [C3, str(C3)]}}


@pytest.mark.parametrize(
    "where",
    [
        "status <> 'paid'",
        "status IS NULL",
        "status > 'm'",
        "items IS NOT NULL",
        "status = 'paid' OR qty = 1",
        "status LIKE 'p%'",
        "qty > ?",
        "qty > random() * 10",
        "qty NOT IN (1, 2)",
        "CAST(created_at AS DATE) = DATE '2024-01-10'",
    ],
)
def test_predicates_that_cannot_be_sent_safely_stay_in_duckdb(
    server: FakeServer, connector: MongoDBDatabase, where: str
) -> None:
    sql = f"SELECT COUNT(*) AS n FROM orders WHERE {where}"
    parameters = [1] if "?" in where else None
    connector.execute(sql, parameters)
    assert server.finds("orders")[0][2] == {}


def test_a_join_reads_only_the_fields_it_names(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    frame = connector.execute(
        'SELECT 1 AS hit FROM "main"."orders" s JOIN "main"."customers" t '
        'ON s."customer_id" = t."_id" LIMIT 1'
    )
    assert frame["hit"].tolist() == [1]
    projections = {call[1]: call[3] for call in server.finds()}
    assert projections == {
        "orders": {"_id": 1, "customer_id": 1},
        "customers": {"_id": 1},
    }
    assert all(call[2] == {} and call[4] == 0 for call in server.finds())


def test_json_columns_come_back_as_python_values(
    connector: MongoDBDatabase,
) -> None:
    frame = connector.execute(
        "SELECT name, address->>'city' AS city, address FROM customers ORDER BY name"
    )
    assert frame["city"].tolist()[:2] == ["Seoul", "Busan"]
    assert frame["address"].tolist()[0] == {"city": "Seoul", "zip": "04524"}
    assert pd.isna(frame["city"].tolist()[2])


def test_numbers_and_timestamps_keep_their_types(connector: MongoDBDatabase) -> None:
    frame = connector.execute(
        "SELECT SUM(total) AS total, MAX(created_at) AS latest, "
        "SUM(qty) AS qty FROM orders"
    )
    assert frame["total"].tolist() == [pytest.approx(73.85)]
    assert frame["latest"].tolist() == [pd.Timestamp("2024-04-01")]
    assert frame["qty"].tolist() == [12]


@pytest.mark.parametrize(
    "table", ["orders", "main.orders", "shop.orders", "shop.main.orders", '"orders"']
)
def test_every_spelling_of_a_collection_resolves(
    connector: MongoDBDatabase, table: str
) -> None:
    assert connector.execute(f"SELECT COUNT(*) AS n FROM {table}")["n"].tolist() == [5]


def test_count_star_fetches_only_ids(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    connector.execute("SELECT COUNT(*) FROM orders")
    assert server.finds("orders")[0][3] == {"_id": 1}


def test_ctes_and_views_are_queried_in_duckdb(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    frame = connector.execute(
        "WITH paid AS (SELECT * FROM orders WHERE status = 'paid') "
        "SELECT (SELECT COUNT(*) FROM paid) AS paid, "
        "(SELECT COUNT(*) FROM paid_orders) AS view_rows, "
        "(SELECT COUNT(*) FROM status_counts) AS counts"
    )
    assert frame.to_dict("records") == [{"paid": 2, "view_rows": 2, "counts": 1}]
    assert all(call[2] == {} for call in server.finds())


def test_too_many_documents_raise_instead_of_truncating(
    server: FakeServer,
) -> None:
    connector = MongoDBDatabase(URL + "&max_documents=2")
    try:
        with pytest.raises(MongoDBDocumentLimitError, match="max_documents"):
            connector.execute("SELECT COUNT(*) FROM orders")
        assert len(connector.execute("SELECT * FROM orders LIMIT 2")) == 2
    finally:
        connector.close()


@pytest.mark.parametrize(
    "sql",
    ["DELETE FROM orders", "SELECT 1; SELECT 2", "CREATE TABLE x AS SELECT 1"],
)
def test_only_a_single_select_runs(connector: MongoDBDatabase, sql: str) -> None:
    with pytest.raises(ValueError):
        connector.execute(sql)


def test_generated_sql_cannot_reach_the_filesystem(
    connector: MongoDBDatabase,
) -> None:
    with pytest.raises(duckdb.Error):
        connector.execute("SELECT * FROM read_csv('/etc/hosts')")


def test_a_slow_fetch_times_out(server: FakeServer, connector: MongoDBDatabase) -> None:
    server.delay = 0.1
    with pytest.raises(StatementTimeout):
        connector.execute("SELECT COUNT(*) FROM orders", timeout_s=0.15)
    assert server.finds("orders")[0][6] is not None  # under pymongo.timeout()


def test_a_slow_statement_times_out(connector: MongoDBDatabase) -> None:
    started = time.monotonic()
    with pytest.raises(StatementTimeout):
        connector.execute(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
            "SELECT count(*) FROM n",
            timeout_s=0.3,
        )
    assert time.monotonic() - started < 5


def test_servers_without_max_time_ms_fall_back_to_the_client_deadline(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    server.reject_time_limits = True
    frame = connector.execute("SELECT COUNT(*) AS n FROM orders", timeout_s=30)
    assert frame["n"].tolist() == [5]
    # The first call ran under pymongo.timeout(), which sends maxTimeMS, and was
    # refused; everything after it runs without either.
    listed = [call for call in server.calls if call[0] == "list_collections"]
    assert listed[0][1] is not None
    assert listed[-1][1] is None
    sampled = [call for call in server.calls if call[0] == "aggregate"]
    assert all(call[3] == {} and call[4] is None for call in sampled)
    (fetch,) = server.finds("orders")
    assert fetch[5] is None and fetch[6] is None


def test_ingestion_sends_its_own_time_limit(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    connector.get_tables()
    sampled = [call for call in server.calls if call[0] == "aggregate"]
    assert all(call[3] == {"maxTimeMS": 60_000} and call[4] is None for call in sampled)


def test_every_wait_of_a_statement_is_bounded_by_its_timeout(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    connector.execute("SELECT COUNT(*) FROM orders WHERE qty > 0", timeout_s=30)
    scopes = [
        call[-1]
        for call in server.calls
        if call[0]
        in ("list_collections", "aggregate", "estimated_document_count", "find")
    ]
    assert scopes and all(scope is not None and scope <= 30 for scope in scopes)


def test_a_connection_failure_is_not_reported_as_a_timeout(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    server.list_error = ServerSelectionTimeoutError("No servers found yet")
    with pytest.raises(ServerSelectionTimeoutError):
        connector.execute("SELECT COUNT(*) FROM orders", timeout_s=30)


def test_a_resample_keeps_columns_the_catalog_already_has(
    server: FakeServer, connector: MongoDBDatabase, monkeypatch: MonkeyPatch
) -> None:
    connector.execute("SELECT COUNT(*) FROM customers")
    monkeypatch.setattr(mongodb, "_SCHEMA_TTL_SECONDS", 0.0)
    for document in server.data["customers"]:
        document.pop("score")
    frame = connector.execute("SELECT COUNT(score) AS n FROM customers")
    assert frame["n"].tolist() == [0]


# ---------------------------------------------------------------------------
# Comparisons the server makes differently from DuckDB
# ---------------------------------------------------------------------------


def _add(
    server: FakeServer,
    name: str,
    documents: list[dict[str, Any]],
    kind: str = "collection",
) -> None:
    server.data[name] = documents
    server.types[name] = kind


def test_double_bounds_admit_every_value_that_rounds_onto_them(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    # The sample (three documents) sees doubles only. The others hold a
    # decimal and a long whose doubles equal the constants, which DuckDB
    # compares as doubles and the server compares exactly.
    _add(
        server,
        "prices",
        [
            {"_id": 1, "price": 9.99, "w": 0.3},
            {"_id": 2, "price": 19.99, "w": 0.1},
            {"_id": 3, "price": 4.7, "w": 2.5},
            {"_id": 4, "price": Decimal128("9.99"), "w": Decimal128("0.1")},
            {"_id": 5, "price": 2**54 + 3, "w": 1},
        ],
    )
    for where, expected in [
        ("price = 9.99", [1, 4]),
        ("price IN (9.99, 19.99)", [1, 2, 4]),
        ("w >= 0.3", [1, 3, 5]),
        ("w <= 0.1", [2, 4]),
        ("price >= 18014398509481988e0", [5]),
        ("price BETWEEN 9.99 AND 19.99", [1, 2, 4]),
    ]:
        frame = _same_answer(
            server, connector, f"SELECT _id FROM prices WHERE {where} ORDER BY _id"
        )
        assert frame["_id"].tolist() == expected, where

    server.calls.clear()
    connector.execute("SELECT _id FROM prices WHERE price = 9.99")
    assert server.finds("prices")[0][2] == {
        "price": {
            "$gte": math.nextafter(9.99, -math.inf),
            "$lte": math.nextafter(9.99, math.inf),
        }
    }


@pytest.mark.parametrize(
    ("where", "pushed"),
    [
        ("qty >= 2e0", {}),
        ("qty > 2.5", {"qty": {"$gt": Decimal128("2.5")}}),
        ("qty < 0.12345678901234567890123456789012345", {}),
    ],
)
def test_integer_columns_take_only_exact_constants(
    server: FakeServer, connector: MongoDBDatabase, where: str, pushed: dict
) -> None:
    _same_answer(server, connector, f"SELECT COUNT(*) AS n FROM orders WHERE {where}")
    assert server.finds("orders")[0][2] == pushed


def test_a_collation_on_the_constant_stays_in_duckdb(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    server.data["orders"][1]["status"] = "PAID"
    frame = _same_answer(
        server,
        connector,
        "SELECT COUNT(*) AS n FROM orders WHERE status = 'paid' COLLATE NOCASE",
    )
    assert frame["n"].tolist() == [3]
    assert server.finds("orders")[0][2] == {}


def test_limit_percent_is_left_to_duckdb(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    frame = connector.execute("SELECT _id FROM orders LIMIT 40 PERCENT")
    assert len(frame) == 2
    assert server.finds("orders")[0][4] == 0


def test_time_bounds_survive_millisecond_rounding(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    frame = _same_answer(
        server,
        connector,
        "SELECT COUNT(*) AS n FROM orders "
        "WHERE created_at < TIMESTAMP '2024-01-10 00:00:00.0005'",
    )
    assert frame["n"].tolist() == [1]
    # PyMongo truncates to milliseconds, so the bound is rounded up first.
    assert server.finds("orders")[0][2] == {
        "created_at": {"$lt": datetime(2024, 1, 10, 0, 0, 0, 1000)}
    }


def test_the_clock_is_trusted_in_lower_bounds_only(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    connector.execute("SELECT COUNT(*) FROM orders WHERE created_at <= now()")
    assert server.finds("orders")[-1][2] == {}
    connector.execute(
        "SELECT COUNT(*) FROM orders WHERE created_at > now() - INTERVAL 10000 DAY"
    )
    assert set(server.finds("orders")[-1][2]["created_at"]) == {"$gt"}
    connector.execute("SELECT COUNT(*) FROM orders WHERE created_at < 'infinity'")
    assert server.finds("orders")[-1][2] == {}


# ---------------------------------------------------------------------------
# Typing that has to hold still
# ---------------------------------------------------------------------------


def test_one_stray_type_does_not_decide_the_column() -> None:
    documents = [{"_id": i, "qty": i} for i in range(200)] + [
        {"_id": 200, "qty": "n/a"}
    ]
    (_, qty) = infer_columns(documents)
    assert qty.sql_type == "BIGINT"
    (_, mixed) = infer_columns(
        [{"_id": 1, "v": 1}, {"_id": 2, "v": "x"}, {"_id": 3, "v": 2}]
    )
    assert mixed.sql_type == "VARCHAR"


def test_a_resample_does_not_change_a_known_column_type(
    server: FakeServer, connector: MongoDBDatabase, monkeypatch: MonkeyPatch
) -> None:
    connector.execute("SELECT COUNT(*) FROM orders")
    monkeypatch.setattr(mongodb, "_SCHEMA_TTL_SECONDS", 0.0)
    for document in server.data["orders"][:3]:
        document["qty"] = "n/a"
    frame = connector.execute("SELECT SUM(qty) AS q FROM orders")
    assert frame["q"].tolist() == [4]


def test_whole_numbers_beyond_bigint_read_as_null(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    (_, qty) = infer_columns([{"_id": 1, "qty": 1}])
    assert convert(qty, 2.0**63) is None
    assert convert(qty, Decimal128("1E+30")) is None
    assert convert(qty, -(2**63)) == -(2**63)
    server.data["orders"].append(
        {"_id": ObjectId(), "qty": 2.0**63, "created_at": datetime(2024, 5, 1)}
    )
    assert connector.execute("SELECT SUM(qty) AS q FROM orders")["q"].tolist() == [12]


def test_mixed_text_keeps_every_digit() -> None:
    (_, mixed) = infer_columns([{"_id": 1, "v": 1}, {"_id": 2, "v": "x"}])
    assert convert(mixed, float("nan")) == "nan"
    assert convert(mixed, 0.1) == "0.1"
    assert (
        convert(mixed, Decimal128("12345678901234567890.12"))
        == "12345678901234567890.12"
    )


# ---------------------------------------------------------------------------
# Statements that read columns without naming them
# ---------------------------------------------------------------------------


def test_positions_and_alias_column_lists_read_every_field(
    connector: MongoDBDatabase,
) -> None:
    frame = connector.execute(
        "SELECT #2 AS second FROM orders ORDER BY created_at LIMIT 1"
    )
    assert frame["second"].tolist() == [str(C1)]
    frame = connector.execute(
        "SELECT a FROM orders AS o(a, b) ORDER BY created_at LIMIT 1"
    )
    assert frame["a"].tolist() == ["65b000000000000000000001"]


def test_pivot_groups_by_fields_it_does_not_name(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    _add(
        server,
        "kpis",
        [
            {"region": "kr", "metric": "a", "v": 1},
            {"region": "us", "metric": "a", "v": 2},
        ],
        kind="view",
    )
    frame = connector.execute(
        "SELECT COUNT(*) AS n FROM (PIVOT kpis ON metric USING sum(v))"
    )
    assert frame["n"].tolist() == [2]


def test_a_field_named_like_a_keyword_is_fetched(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    _add(server, "snap", [{"_id": 1, "current_date": "2020-01-01"}])
    frame = connector.execute("SELECT current_date AS d FROM snap")
    assert frame["d"].tolist() == ["2020-01-01"]


# ---------------------------------------------------------------------------
# Time limits, sizes and the sandbox
# ---------------------------------------------------------------------------


def test_a_sample_that_runs_out_of_time_keeps_server_time_limits(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    server.sample_error = ExecutionTimeout(
        "operation exceeded time limit",
        code=50,
        details={"codeName": "MaxTimeMSExpired"},
    )
    assert "maxtimems" in str(server.sample_error).lower()
    connector.execute("SELECT COUNT(*) FROM orders", timeout_s=30)
    sample_fallback, fetch = server.finds("orders")
    assert sample_fallback[2:5] == ({}, None, 3)
    assert fetch[6] is not None  # still under pymongo.timeout()


def test_sampling_counts_against_the_statement_timeout(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    server.sample_delay = 0.3
    with pytest.raises(StatementTimeout):
        connector.execute("SELECT COUNT(*) FROM orders", timeout_s=0.2)
    (sampled,) = [call for call in server.calls if call[0] == "aggregate"]
    assert sampled[4] <= 0.2  # the pymongo.timeout() it ran under


def test_large_collections_fail_before_streaming(server: FakeServer) -> None:
    connector = MongoDBDatabase(URL + "&max_documents=2")
    try:
        with pytest.raises(MongoDBDocumentLimitError):
            connector.execute("SELECT COUNT(*) FROM orders WHERE qty > 0")
        assert not server.finds("orders")
        frame = connector.execute("SELECT COUNT(*) AS n FROM orders WHERE qty > 4")
        assert frame["n"].tolist() == [1]
    finally:
        connector.close()


def test_generated_sql_cannot_read_duckdb_temp_files(
    connector: MongoDBDatabase, tmp_path: Any, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tmp").mkdir()
    (tmp_path / ".tmp" / "spill.txt").write_text("secret")
    for sql in (
        "SELECT * FROM glob('.tmp/*')",
        "SELECT content FROM read_text('.tmp/spill.txt')",
    ):
        with pytest.raises(duckdb.Error):
            connector.execute(sql)


# ---------------------------------------------------------------------------
# Second review
# ---------------------------------------------------------------------------


def test_a_34_digit_decimal_that_fits_is_kept() -> None:
    (_, amount) = infer_columns(
        [
            {"_id": 1, "amount": Decimal128("1.000000000000000001")},
            {"_id": 2, "amount": Decimal128("5")},
        ]
    )
    assert amount.sql_type == "DECIMAL(38, 18)"
    value = "99999999999999999999.99999999999999"
    assert convert(amount, Decimal128(value)) == value


@pytest.mark.parametrize(
    ("where", "pushed"),
    [
        ("created_at >= date_trunc('month', current_date)", True),
        ("created_at >= CAST(now() AS DATE)", True),
        ("qty >= 60 - date_part('second', CURRENT_TIMESTAMP)", False),
        ("qty >= 0 - epoch(now())", False),
    ],
)
def test_only_lower_bounds_that_rise_with_the_clock_are_pushed(
    server: FakeServer, connector: MongoDBDatabase, where: str, pushed: bool
) -> None:
    connector.execute(f"SELECT COUNT(*) FROM orders WHERE {where}")
    assert bool(server.finds("orders")[0][2]) is pushed


@pytest.mark.parametrize(
    "expression", ["current_time", "localtime", "CAST(now() AS TIME)"]
)
def test_a_time_of_day_does_not_rise_with_the_clock(expression: str) -> None:
    node = sqlglot.parse_one(f"SELECT {expression}", read="duckdb").expressions[0]
    assert not mongodb._rises_with_the_clock(node)


def test_constant_evaluation_runs_under_the_statement_timeout(
    server: FakeServer, connector: MongoDBDatabase, monkeypatch: MonkeyPatch
) -> None:
    def endless(self: Any, sql: str) -> tuple[bool, Any]:
        try:
            self._connection.execute(
                "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
                "SELECT count(*) FROM n"
            ).fetchone()
        except duckdb.Error:
            return False, None
        return True, None

    monkeypatch.setattr(mongodb._Pushdown, "_evaluate", endless)
    started = time.monotonic()
    with pytest.raises(StatementTimeout):
        connector.execute("SELECT COUNT(*) FROM orders WHERE qty > 1", timeout_s=0.3)
    assert time.monotonic() - started < 5


def test_an_alias_column_list_keeps_the_filter_in_duckdb(
    server: FakeServer, connector: MongoDBDatabase
) -> None:
    # Renamed by position: this "status" is the customer_id field.
    frame = _same_answer(
        server,
        connector,
        f"SELECT COUNT(*) AS n FROM orders AS o(_id, status) WHERE status = '{C1}'",
    )
    assert frame["n"].tolist() == [2]
    assert server.finds("orders")[0][2] == {}


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        (
            "SELECT COUNT(*) AS n FROM orders "
            "WHERE EXISTS (WITH orders AS (SELECT 1 AS x) SELECT * FROM orders)",
            5,
        ),
        (
            "WITH orders AS (SELECT * FROM orders WHERE status = 'paid') "
            "SELECT COUNT(*) AS n FROM orders",
            2,
        ),
    ],
)
def test_a_cte_hides_a_collection_only_in_its_own_scope(
    connector: MongoDBDatabase, sql: str, expected: int
) -> None:
    assert connector.execute(sql)["n"].tolist() == [expected]


def test_redact_masks_mongodb_secrets_in_any_spelling() -> None:
    redacted = _redact(
        "mongodb://h/shop?TLSCertificateKeyFilePassword=key-pass"
        "&authMechanismProperties=AWS_SESSION_TOKEN:session-tok"
        "&tlsCertificateKeyFile%50assword=encoded-pass&tls=true"
    )
    for secret in ("key-pass", "session-tok", "encoded-pass"):
        assert secret not in redacted
    assert redacted.endswith("&tls=true")
