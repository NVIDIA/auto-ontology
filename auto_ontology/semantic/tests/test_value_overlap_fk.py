# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for semantic FK discovery from shared column values."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.connectors.sqlite import SQLiteDatabase
from auto_ontology.semantic import semantic_fk
from auto_ontology.semantic import value_overlap_fk as vo
from auto_ontology.semantic.models import FkHitSelection

_DEPOTS = [f"D{n // 10}-{100 + n}" for n in range(60)]


def _build(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE depots (depot_key TEXT, depot_use TEXT, area INTEGER)")
    # One depot is split between two zones, so its key repeats: 60 of 61 distinct.
    rows = [
        (key, "north" if n % 2 else "south", 10 + n) for n, key in enumerate(_DEPOTS)
    ]
    rows.append((_DEPOTS[0], "overflow", 5))
    conn.executemany("INSERT INTO depots VALUES (?, ?, ?)", rows)
    conn.execute(
        "CREATE TABLE shipments (shipment_id TEXT, drop_point TEXT, crates INTEGER, "
        "depot_ref TEXT, depot_title TEXT)"
    )
    shipments = []
    for n in range(40):
        # Half of the distinct drop points are known depots; the rest are not.
        place = _DEPOTS[n % 8] if n < 24 else f"CUSTOMER-{n % 8}"
        shipments.append((f"S{n:03d}", place, n % 12, _DEPOTS[n % 20], place))
    conn.executemany("INSERT INTO shipments VALUES (?, ?, ?, ?, ?)", shipments)
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path: Path) -> SQLiteDatabase:
    path = tmp_path / "logistics.sqlite"
    _build(path)
    return SQLiteDatabase(str(path))


def _col(
    name: str,
    data_type: str = "TEXT",
    *,
    attribute_id: str | None = None,
    is_unique: bool | None = False,
    has_semantic_fk: bool = False,
) -> dict[str, Any]:
    return {
        "id": name,
        "name": name.split(".")[1],
        "data_type": data_type,
        "description": "",
        "sample_values": [],
        "is_unique": is_unique,
        "attribute_id": attribute_id,
        "has_semantic_fk": has_semantic_fk,
    }


def _catalog(*, depot_ref_linked: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "id": "t-depots",
            "name": "depots",
            "schema_name": "main",
            "pk": [],
            "columns": [
                _col("depots.depot_key", attribute_id="attr-depot-key"),
                _col("depots.depot_use", attribute_id="attr-depot-use"),
                _col("depots.area", "INTEGER", attribute_id="attr-area"),
            ],
        },
        {
            "id": "t-shipments",
            "name": "shipments",
            "schema_name": "main",
            "pk": ["shipment_id"],
            "columns": [
                _col(
                    "shipments.shipment_id",
                    attribute_id="attr-shipment",
                    is_unique=True,
                ),
                _col("shipments.drop_point", attribute_id="attr-drop-point"),
                _col("shipments.crates", "INTEGER", attribute_id="attr-crates"),
                _col("shipments.depot_ref", has_semantic_fk=depot_ref_linked),
                # Holds depot keys too, but a label column is never a reference.
                _col("shipments.depot_title", attribute_id="attr-depot-title"),
            ],
        },
    ]


def _run(db: SQLiteDatabase, selected_id: str | None, **catalog: Any):
    prompts: list[str] = []
    writes: list[tuple[str, str]] = []

    def _invoke(_llm, messages, _schema):
        prompts.append(messages[1].content)
        return FkHitSelection(selected_id=selected_id)

    with (
        patch.object(
            vo, "fetch_columns_for_value_overlap", return_value=_catalog(**catalog)
        ),
        patch.object(vo, "get_llm_client"),
        patch.object(vo, "invoke_with_structured_output", side_effect=_invoke),
        patch.object(
            vo,
            "merge_semantic_fk",
            side_effect=lambda s, a: writes.append((s, a)) or True,
        ),
    ):
        written = vo.discover_value_overlap_fks("logistics", db)
    return written, prompts, writes


def test_links_a_column_whose_values_are_a_nearly_unique_key(
    db: SQLiteDatabase,
) -> None:
    written, prompts, writes = _run(db, "depots.depot_key")

    assert written == 1
    assert writes == [("shipments.drop_point", "attr-depot-key")]
    assert len(prompts) == 1
    assert prompts[0].startswith("Column: shipments.drop_point")
    assert "column_id=depots.depot_key" in prompts[0]
    assert "holds 8 of the column's 16 sampled values (50%)" in prompts[0]


def test_writes_nothing_when_the_model_declines(db: SQLiteDatabase) -> None:
    written, prompts, writes = _run(db, None)

    assert written == 0
    assert writes == []
    assert len(prompts) == 1


def test_ignores_an_unknown_id_from_the_model(db: SQLiteDatabase) -> None:
    written, _prompts, writes = _run(db, "depots.depot_use")

    assert written == 0
    assert writes == []


def test_a_column_that_is_not_linked_yet_is_considered(db: SQLiteDatabase) -> None:
    _written, prompts, _writes = _run(db, None, depot_ref_linked=False)

    assert sorted(p.splitlines()[0].split(" ")[1] for p in prompts) == [
        "shipments.depot_ref",
        "shipments.drop_point",
    ]


def test_disabled_discovery_reads_nothing(db: SQLiteDatabase) -> None:
    with (
        patch.object(vo, "_ENABLED", False),
        patch.object(vo, "fetch_columns_for_value_overlap") as fetch,
    ):
        assert vo.discover_value_overlap_fks("logistics", db) == 0
    fetch.assert_not_called()


def test_no_connector_writes_nothing() -> None:
    with patch.object(vo, "fetch_columns_for_value_overlap") as fetch:
        assert vo.discover_value_overlap_fks("logistics", None) == 0
    fetch.assert_not_called()


def test_short_values_never_make_a_candidate() -> None:
    target = vo._Column("t.code", "code", "TEXT", "", (), "attr", "t", "t", None)
    source = vo._Column("s.flag", "flag", "TEXT", "", (), None, "s", "s", None)
    index = {value: [target] for value in ("Y", "N", "12", "7")}

    assert vo.candidates_for(source, {"Y", "N", "12", "7"}, index) == []


def test_targets_on_the_same_table_are_skipped() -> None:
    target = vo._Column("t.key", "key", "TEXT", "", (), "attr", "t", "t", None)
    source = vo._Column("t.ref", "ref", "TEXT", "", (), None, "t", "t", None)
    values = {f"K-{n}" for n in range(10)}

    assert vo.candidates_for(source, values, {v: [target] for v in values}) == []


@pytest.mark.parametrize(
    ("non_null", "distinct", "is_unique", "expected"),
    [
        (100, 98, None, True),
        (100, 97, None, False),
        (10, 10, None, False),
        (3, 3, True, True),
        (0, 0, True, False),
    ],
)
def test_is_key_like(
    non_null: int, distinct: int, is_unique: bool | None, expected: bool
) -> None:
    assert vo.is_key_like(non_null, distinct, is_unique) is expected


@pytest.mark.parametrize(
    ("source_values", "shared", "expected"),
    [
        (553, 108, True),
        (553, 40, False),
        (3, 3, True),
        (40, 4, False),
        (0, 0, False),
    ],
)
def test_overlap_accepted(source_values: int, shared: int, expected: bool) -> None:
    assert vo.overlap_accepted(source_values, shared) is expected


@pytest.mark.parametrize(
    ("name", "samples", "expected"),
    [
        ("REGION_NAME", ("North",), True),
        ("productTitle", ("Lamp",), True),
        ("STATUS_DESC", ("OPEN",), True),
        ("store_label", (), True),
        ("location", ("Main Street", "Harbor Road", "D1-101"), True),
        ("drop_point", ("D1-101", "D2-205", "Main Street"), False),
        ("customer_code", ("C-1", "C-2"), False),
        ("named_key", ("K1",), False),
    ],
)
def test_is_label_column(name: str, samples: tuple[str, ...], expected: bool) -> None:
    assert vo.is_label_column(name, samples) is expected


@pytest.mark.parametrize(
    ("data_type", "expected"),
    [
        ("varchar(255)", True),
        ("TEXT", True),
        ("bigint", True),
        ("int(11)", True),
        ("uuid", True),
        ("interval", False),
        ("decimal(10,2)", False),
        ("date", False),
    ],
)
def test_is_joinable_type(data_type: str, expected: bool) -> None:
    assert vo.is_joinable_type(data_type) is expected


def test_near_unique_check_on_a_live_column(db: SQLiteDatabase) -> None:
    assert vo.is_near_unique_column(db, "main", "depots", "depot_key") is True
    assert vo.is_near_unique_column(db, "main", "depots", "depot_use") is False
    assert vo.shared_value_counts(
        db, ("main", "shipments", "depot_ref"), ("main", "depots", "depot_key")
    ) == (20, 20)


def _vdb_hit(attr_id: str, table: str, column: str) -> dict[str, Any]:
    return {
        "metadata": {
            "id": attr_id,
            "table_id": f"t-{table}",
            "table_name": table,
            "schema_name": "main",
            "source_column": column,
            "is_unique": False,
        },
        "_distance": 0.4,
        "text": f"ColumnAttribute {attr_id}",
    }


def test_vdb_path_keeps_a_nearly_unique_key_holding_the_fk_values(
    db: SQLiteDatabase,
) -> None:
    retriever = MagicMock()
    retriever.query.return_value = [
        _vdb_hit("attr-depot-key", "depots", "depot_key"),
        _vdb_hit("attr-depot-use", "depots", "depot_use"),
    ]
    column = {
        "id": "shipments.depot_ref",
        "name": "depot_ref",
        "table_id": "t-shipments",
        "table_name": "shipments",
        "schema_name": "main",
    }

    kept = semantic_fk._near_unique_hits(column, retriever, "logistics", db)

    assert [hit["metadata"]["id"] for hit in kept] == ["attr-depot-key"]
    assert retriever.query.call_args.kwargs["vdb_kwargs"]["where"]["is_unique"] is False


def test_vdb_path_without_a_connector_adds_nothing() -> None:
    retriever = MagicMock()

    assert (
        semantic_fk._near_unique_hits(
            {"name": "depot_ref", "table_name": "shipments"},
            retriever,
            "logistics",
            None,
        )
        == []
    )
    retriever.query.assert_not_called()
