# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for linking key columns that hold the same identifier."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from auto_ontology.connectors.sqlite import SQLiteDatabase
from auto_ontology.semantic import equivalent_keys as ek
from auto_ontology.semantic.models import KeyEquivalenceJudgement
from auto_ontology.semantic.value_overlap_fk import ProfiledColumn

_ALL = [f"D-{100 + n}" for n in range(50)]
_ACTIVE = _ALL[:30]


def _build(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE depots_all (depot_code TEXT, opened TEXT)")
    conn.executemany(
        "INSERT INTO depots_all VALUES (?, ?)",
        [(code, f"2020-01-{n + 1:02d}") for n, code in enumerate(_ALL)],
    )
    conn.execute(
        "CREATE TABLE depots_active (depot_code TEXT, opened TEXT, bays INTEGER)"
    )
    conn.executemany(
        "INSERT INTO depots_active VALUES (?, ?, ?)",
        [(code, f"2020-01-{n + 1:02d}", n) for n, code in enumerate(_ACTIVE)],
    )
    conn.execute("CREATE TABLE shipments (shipment_id TEXT, depot_code TEXT)")
    conn.executemany(
        "INSERT INTO shipments VALUES (?, ?)",
        [(f"S{n:03d}", _ACTIVE[n % 20]) for n in range(40)],
    )
    conn.execute("CREATE TABLE closures (closure_id TEXT, depot_code TEXT)")
    conn.executemany(
        "INSERT INTO closures VALUES (?, ?)",
        [(f"C{n:03d}", _ALL[30 + n % 20]) for n in range(40)],
    )
    conn.execute("CREATE TABLE depot_pairs (from_depot TEXT, to_depot TEXT)")
    conn.executemany(
        "INSERT INTO depot_pairs VALUES (?, ?)",
        [(_ACTIVE[n], _ACTIVE[n + 1]) for n in range(20)],
    )
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
) -> dict[str, Any]:
    return {
        "id": name,
        "name": name.split(".")[1],
        "data_type": data_type,
        "description": "",
        "sample_values": [],
        "is_unique": is_unique,
        "attribute_id": attribute_id,
        "has_semantic_fk": attribute_id is None,
    }


def _table(name: str, pk: list[str], *columns: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"t-{name}",
        "name": name,
        "schema_name": "main",
        "pk": pk,
        "columns": list(columns),
    }


def _catalog() -> list[dict[str, Any]]:
    return [
        _table(
            "depots_all",
            ["depot_code"],
            _col("depots_all.depot_code", attribute_id="attr-all-code", is_unique=True),
            _col("depots_all.opened", attribute_id="attr-all-opened", is_unique=True),
        ),
        _table(
            "depots_active",
            ["depot_code"],
            _col(
                "depots_active.depot_code",
                attribute_id="attr-active-code",
                is_unique=True,
            ),
            _col(
                "depots_active.opened",
                attribute_id="attr-active-opened",
                is_unique=True,
            ),
            _col("depots_active.bays", "INTEGER", attribute_id="attr-bays"),
        ),
        _table(
            "shipments",
            ["shipment_id"],
            _col("shipments.shipment_id", attribute_id="attr-shipment", is_unique=True),
            _col("shipments.depot_code"),
        ),
        _table(
            "closures",
            ["closure_id"],
            _col("closures.closure_id", attribute_id="attr-closure", is_unique=True),
            _col("closures.depot_code"),
        ),
        # Every column a reference, none an attribute owner.
        _table(
            "depot_pairs",
            [],
            _col("depot_pairs.from_depot"),
            _col("depot_pairs.to_depot"),
        ),
    ]


def _fk(source: str, target: str) -> dict[str, Any]:
    source_table, source_column = source.split(".")
    target_table, target_column = target.split(".")
    return {
        "source_table_id": f"t-{source_table}",
        "source_table_name": source_table,
        "source_schema_name": "main",
        "source_column_id": source,
        "source_column_name": source_column,
        "target_attribute_id": f"attr-{target_table.split('_')[1]}-code",
        "target_table_id": f"t-{target_table}",
        "target_table_name": target_table,
        "target_schema_name": "main",
        "target_column_id": target,
        "target_column_name": target_column,
    }


def _existing() -> list[dict[str, Any]]:
    return [
        _fk("shipments.depot_code", "depots_all.depot_code"),
        _fk("closures.depot_code", "depots_all.depot_code"),
        _fk("depot_pairs.from_depot", "depots_all.depot_code"),
    ]


def _run(
    db: SQLiteDatabase | None,
    *,
    judge=lambda human: "opened" not in human,
    existing: list[dict[str, Any]] | None = None,
):
    prompts: list[str] = []
    writes: list[tuple[str, str]] = []

    def _invoke(_llm, messages, _schema):
        human = messages[1].content
        prompts.append(human)
        return KeyEquivalenceJudgement(equivalent=judge(human))

    with (
        patch.object(ek, "fetch_columns_for_value_overlap", return_value=_catalog()),
        patch.object(
            ek,
            "fetch_semantic_fk_join_groups",
            return_value=_existing() if existing is None else existing,
        ),
        patch.object(ek, "get_llm_client"),
        patch.object(ek, "invoke_with_structured_output", side_effect=_invoke),
        patch.object(
            ek,
            "merge_semantic_fk",
            side_effect=lambda src, attr: writes.append((src, attr)) or True,
        ),
    ):
        written = ek.link_equivalent_keys("logistics", db)
    return written, writes, prompts


def test_links_the_keys_and_the_columns_referencing_them(db) -> None:
    written, writes, prompts = _run(db)

    assert sorted(writes) == [
        ("depots_active.depot_code", "attr-all-code"),
        ("shipments.depot_code", "attr-active-code"),
    ]
    assert written == 2
    # The date columns share values too; the LLM sees them and rejects them.
    assert any("opened" in prompt for prompt in prompts)
    assert any("depot_code" in prompt for prompt in prompts)


def test_a_column_whose_values_miss_the_other_key_is_not_extended(db) -> None:
    _, writes, _ = _run(db)

    assert all(src != "closures.depot_code" for src, _ in writes)


@pytest.mark.parametrize(
    ("source_values", "shared", "expected"),
    [(40, 40, True), (40, 20, True), (40, 19, False), (3, 3, True), (40, 0, False)],
)
def test_a_reference_needs_most_of_its_values_in_the_other_key(
    source_values: int, shared: int, expected: bool
) -> None:
    assert ek._references(source_values, shared) is expected


def test_a_table_of_pure_references_keeps_one_target(db) -> None:
    _, writes, _ = _run(db)

    assert all(not src.startswith("depot_pairs.") for src, _ in writes)


def test_a_column_already_joining_the_other_table_is_left_alone(db) -> None:
    existing = _existing() + [
        {
            **_fk("shipments.depot_code", "depots_active.depot_code"),
            "target_attribute_id": "attr-active-code",
        }
    ]
    _, writes, _ = _run(db, existing=existing)

    assert ("shipments.depot_code", "attr-active-code") not in writes


def test_nothing_is_written_when_the_llm_rejects_every_pair(db) -> None:
    written, writes, prompts = _run(db, judge=lambda _human: False)

    assert written == 0
    assert writes == []
    assert prompts


def test_an_llm_failure_counts_as_a_rejection(db) -> None:
    def _boom(_human):
        raise RuntimeError("down")

    written, writes, _ = _run(db, judge=_boom)

    assert written == 0
    assert writes == []


def test_no_connector_is_a_no_op() -> None:
    written, writes, prompts = _run(None)

    assert (written, writes, prompts) == (0, [], [])


def test_disabled_is_a_no_op(db, monkeypatch) -> None:
    monkeypatch.setattr(ek, "_ENABLED", False)
    written, writes, prompts = _run(db)

    assert (written, writes, prompts) == (0, [], [])


def _key(table: str, name: str, data_type: str = "TEXT") -> ProfiledColumn:
    return ProfiledColumn(
        id=f"{table}.{name}",
        name=name,
        data_type=data_type,
        description="",
        sample_values=(),
        attribute_id=f"attr-{table}-{name}",
        table_id=f"t-{table}",
        table_name=table,
        schema_name="main",
    )


def test_candidate_pairs_put_the_smaller_key_inside() -> None:
    big, small = _key("big", "code"), _key("small", "code")
    values = {
        big.id: {f"v{n}" for n in range(20)},
        small.id: {f"v{n}" for n in range(8)},
    }

    [pair] = ek.candidate_pairs([big, small], values)

    assert (pair.inner, pair.outer) == (small, big)
    assert (pair.shared, pair.inner_values, pair.outer_values) == (8, 8, 20)


@pytest.mark.parametrize(
    ("other", "other_values"),
    [
        (_key("small", "code", "INTEGER"), {f"v{n}" for n in range(8)}),
        (_key("big", "alt"), {f"v{n}" for n in range(8)}),
        (_key("small", "code"), {f"v{n}" for n in range(4)}),
        (
            _key("small", "code"),
            {f"v{n}" for n in range(5)} | {f"w{n}" for n in range(10)},
        ),
    ],
    ids=["type mismatch", "same table", "too few shared", "too little contained"],
)
def test_candidate_pairs_skip(other: ProfiledColumn, other_values: set[str]) -> None:
    big = _key("big", "code")
    values = {big.id: {f"v{n}" for n in range(20)}, other.id: other_values}

    assert ek.candidate_pairs([big, other], values) == []


def test_canonical_pairs_prefer_matching_names_on_a_tie() -> None:
    outer = _key("all", "code")
    alias = ek.KeyPair(_key("active", "code_key"), outer, 10, 10, 20)
    named = ek.KeyPair(_key("active", "code"), outer, 10, 10, 20)

    [chosen] = ek.canonical_pairs([alias, named], set()).values()

    assert chosen is named


def test_canonical_pairs_prefer_the_fuller_containment() -> None:
    outer = _key("all", "code")
    partial = ek.KeyPair(_key("active", "code"), outer, 6, 10, 20)
    full = ek.KeyPair(_key("active", "ref"), outer, 10, 10, 20)

    [chosen] = ek.canonical_pairs([partial, full], set()).values()

    assert chosen is full
