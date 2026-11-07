# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.text_to_sql import table_twins
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_tables_for_prompt,
)

_SHARED = ["DEPOT_KEY", "DEPOT_NAME", "REGION", "CAPACITY", "OPENED_ON"]


def _table(table_id, name, columns, schema="ops"):
    return {
        "id": table_id,
        "name": name,
        "schema_name": schema,
        "database_name": "logistics",
        "columns": [{"name": c, "data_type": "text"} for c in columns],
    }


def _catalog():
    return [
        _table("1", "DEPOTS", _SHARED),
        _table("2", "DEPOTS_HIST", _SHARED + ["FISCAL_PERIOD"]),
        _table("3", "SHIPMENTS", ["SHIPMENT_ID", "DEPOT_KEY", "WEIGHT", "SHIPPED_ON"]),
        _table("4", "DEPOTS_OLD", _SHARED),
        _table("5", "DEPOTS_COPY", _SHARED, schema="archive"),
    ]


def _patch(monkeypatch, legacy=frozenset({"4"})):
    catalog = _catalog()
    monkeypatch.setattr(
        table_twins, "fetch_tables_for_legacy_detection", lambda db: catalog
    )
    monkeypatch.setattr(table_twins, "fetch_legacy_table_ids", lambda db: set(legacy))
    monkeypatch.setattr(
        table_twins,
        "fetch_tables_by_ids",
        lambda ids: [dict(t) for t in catalog if t["id"] in ids],
    )


def test_a_missing_twin_is_added_and_both_sides_name_their_own_columns(
    monkeypatch,
) -> None:
    _patch(monkeypatch)
    in_scope = [_table("1", "DEPOTS", _SHARED), _table("3", "SHIPMENTS", [])]

    tables, added = table_twins.attach_twins(in_scope, "logistics")

    assert added == ["DEPOTS_HIST"]
    assert tables[0]["twins"] == [
        {"name": "DEPOTS_HIST", "only_here": [], "only_there": ["FISCAL_PERIOD"]}
    ]
    assert tables[-1]["twin_of"] == "DEPOTS"
    assert tables[-1]["twins"][0]["only_here"] == ["FISCAL_PERIOD"]
    assert "twins" not in tables[1]


def test_legacy_tables_and_other_schemas_are_never_twins(monkeypatch) -> None:
    _patch(monkeypatch)

    tables, _ = table_twins.attach_twins([_table("1", "DEPOTS", _SHARED)], "logistics")

    assert [t["name"] for t in tables[0]["twins"]] == ["DEPOTS_HIST"]


def test_twins_already_in_scope_are_annotated_but_not_added_again(monkeypatch) -> None:
    _patch(monkeypatch)
    in_scope = [
        _table("1", "DEPOTS", _SHARED),
        _table("2", "DEPOTS_HIST", _SHARED + ["FISCAL_PERIOD"]),
    ]

    tables, added = table_twins.attach_twins(in_scope, "logistics")

    assert added == []
    assert len(tables) == 2
    assert tables[1]["twins"][0]["only_here"] == ["FISCAL_PERIOD"]


def test_an_added_twin_renders_only_its_own_columns(monkeypatch) -> None:
    _patch(monkeypatch)
    tables, _ = table_twins.attach_twins([_table("1", "DEPOTS", _SHARED)], "logistics")

    text = format_tables_for_prompt(tables, dialect="mysql")
    base, twin = text.split("\n\n")

    assert "twins: DEPOTS_HIST (only here: none; only there: FISCAL_PERIOD)" in base
    assert "Shares every other column with DEPOTS" in twin
    assert "FISCAL_PERIOD" in twin
    assert "DEPOT_NAME" not in twin
