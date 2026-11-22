# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for linking unmatched FK columns directly to same-name FK columns."""

from __future__ import annotations

from typing import Any

import pytest

from auto_ontology.semantic import direct_fks


def _column(table: str, name: str = "course_id", **extra: Any) -> dict[str, Any]:
    return {
        "id": f"{table}.{name}",
        "name": name,
        "table_id": table,
        "table_name": table,
        "schema_name": "school",
        "distinct_values": "[1, 2, 3, 4, 5]",
        **extra,
    }


class _Warehouse:
    """Records join probes and answers them from a set of joining table pairs."""

    def __init__(self, joining: set[frozenset[str]]) -> None:
        self.joining = joining
        self.probes: list[tuple[str, str]] = []

    def probe(
        self,
        connector: Any,
        col: dict[str, Any],
        target: dict[str, Any],
        *,
        stage: str,
    ) -> bool:
        assert stage == "link_direct_fks"
        self.probes.append((col["table_name"], target["table_name"]))
        return frozenset({col["table_name"], target["table_name"]}) in self.joining


@pytest.fixture
def wired(monkeypatch):
    state: dict[str, Any] = {
        "candidates": [],
        "peers": [],
        "families": [],
        "groups": [],
        "warehouse": _Warehouse(set()),
        "probed_values": {},
        "stored_values": [],
    }

    def link(column_ids: list[str]) -> str:
        state["groups"].append(sorted(column_ids))
        return f"group-{len(state['groups'])}"

    monkeypatch.setattr(
        direct_fks, "find_unlinked_fk_columns", lambda _db: state["candidates"]
    )
    monkeypatch.setattr(direct_fks, "fetch_direct_fk_peers", lambda _db: state["peers"])
    monkeypatch.setattr(
        direct_fks, "fetch_table_families", lambda _db: state["families"]
    )
    monkeypatch.setattr(direct_fks, "link_direct_fk_columns", link)
    monkeypatch.setattr(direct_fks, "is_semantic_fk_db_query_prevented", lambda: False)
    monkeypatch.setattr(direct_fks, "_resolve_connector", lambda _db: object())
    monkeypatch.setattr(
        direct_fks,
        "join_probe_returns_rows",
        lambda *args, **kwargs: state["warehouse"].probe(*args, **kwargs),
    )
    monkeypatch.setattr(
        direct_fks,
        "probe_ascending_distinct_values",
        lambda _connector, _schema, table, _column: state["probed_values"].get(table),
    )
    monkeypatch.setattr(
        direct_fks,
        "store_column_distinct_values",
        lambda table_id, values: state["stored_values"].append((table_id, values)),
    )
    return state


def test_a_column_with_few_distinct_values_is_not_grouped(wired) -> None:
    wired["candidates"] = [
        _column("lessons", distinct_values='["A", "B", "C"]'),
        _column("grades"),
        _column("rooms"),
    ]
    wired["warehouse"] = _Warehouse(
        {frozenset({"lessons", "grades"}), frozenset({"grades", "rooms"})}
    )

    assert direct_fks.link_direct_fks("db") == 2
    assert wired["groups"] == [["grades.course_id", "rooms.course_id"]]
    assert all("lessons" not in probe for probe in wired["warehouse"].probes)


def test_a_linked_peer_with_few_values_still_joins(wired) -> None:
    wired["candidates"] = [_column("rooms")]
    wired["peers"] = [_column("lessons", group_id=None, distinct_values='["A"]')]
    wired["warehouse"] = _Warehouse({frozenset({"rooms", "lessons"})})

    assert direct_fks.link_direct_fks("db") == 1


def test_a_column_with_no_stored_values_is_probed_and_stored(wired) -> None:
    wired["candidates"] = [
        _column("lessons", distinct_values=None),
        _column("grades", distinct_values=None),
    ]
    wired["probed_values"] = {"lessons": [1, 2, 3, 4, 5, 6], "grades": [1, 2]}
    wired["warehouse"] = _Warehouse({frozenset({"lessons", "grades"})})

    assert direct_fks.link_direct_fks("db") == 0
    assert wired["stored_values"] == [
        ("lessons", {"course_id": [1, 2, 3, 4, 5, 6]}),
        ("grades", {"course_id": [1, 2]}),
    ]


def test_unlinked_columns_that_join_form_one_group(wired) -> None:
    wired["candidates"] = [_column("lessons"), _column("grades"), _column("rooms")]
    wired["warehouse"] = _Warehouse(
        {frozenset({"lessons", "grades"}), frozenset({"grades", "rooms"})}
    )

    assert direct_fks.link_direct_fks("db") == 3
    assert wired["groups"] == [
        ["grades.course_id", "lessons.course_id", "rooms.course_id"]
    ]


def test_a_column_that_joins_nothing_is_left_alone(wired) -> None:
    wired["candidates"] = [_column("lessons"), _column("grades"), _column("rooms")]
    wired["warehouse"] = _Warehouse({frozenset({"lessons", "grades"})})

    assert direct_fks.link_direct_fks("db") == 2
    assert wired["groups"] == [["grades.course_id", "lessons.course_id"]]


def test_names_match_after_normalization(wired) -> None:
    wired["candidates"] = [
        _column("lessons", "Course_ID"),
        _column("grades", "courseid"),
    ]
    wired["warehouse"] = _Warehouse({frozenset({"lessons", "grades"})})

    assert direct_fks.link_direct_fks("db") == 2


def test_different_names_are_never_probed(wired) -> None:
    wired["candidates"] = [
        _column("lessons", "course_id"),
        _column("grades", "room_id"),
    ]

    assert direct_fks.link_direct_fks("db") == 0
    assert wired["warehouse"].probes == []


def test_generic_names_are_skipped(wired) -> None:
    wired["candidates"] = [_column("lessons", "id"), _column("grades", "id")]
    wired["warehouse"] = _Warehouse({frozenset({"lessons", "grades"})})

    assert direct_fks.link_direct_fks("db") == 0
    assert wired["warehouse"].probes == []


def test_an_unlinked_column_joins_an_existing_group(wired) -> None:
    wired["candidates"] = [_column("rooms")]
    wired["peers"] = [
        _column("lessons", group_id="g1"),
        _column("grades", group_id="g1"),
    ]
    wired["warehouse"] = _Warehouse({frozenset({"rooms", "grades"})})

    assert direct_fks.link_direct_fks("db") == 1
    assert wired["groups"] == [
        ["grades.course_id", "lessons.course_id", "rooms.course_id"]
    ]
    # Joining one member reached the whole group, so the other is not probed.
    assert wired["warehouse"].probes == [("rooms", "grades")]


def test_a_linked_peer_is_probed_before_an_unlinked_one(wired) -> None:
    wired["candidates"] = [_column("lessons"), _column("grades")]
    wired["peers"] = [_column("rooms", group_id=None)]
    wired["warehouse"] = _Warehouse(set())

    direct_fks.link_direct_fks("db")

    assert wired["warehouse"].probes[:2] == [
        ("lessons", "rooms"),
        ("lessons", "grades"),
    ]


def test_linked_peers_alone_are_not_regrouped(wired) -> None:
    wired["candidates"] = [_column("rooms", "room_id")]
    wired["peers"] = [
        _column("lessons", group_id=None),
        _column("grades", group_id=None),
    ]
    wired["warehouse"] = _Warehouse({frozenset({"lessons", "grades"})})

    assert direct_fks.link_direct_fks("db") == 0
    assert wired["groups"] == []
    assert wired["warehouse"].probes == []


def test_another_familys_copy_is_never_joined(wired) -> None:
    wired["candidates"] = [_column("old_lessons"), _column("new_grades")]
    wired["families"] = [
        {"table_id": "old_lessons", "family_id": "old", "base_name": "LESSONS"},
        {"table_id": "old_grades", "family_id": "old", "base_name": "GRADES"},
        {"table_id": "new_grades", "family_id": "new", "base_name": "GRADES"},
    ]
    wired["warehouse"] = _Warehouse({frozenset({"old_lessons", "new_grades"})})

    assert direct_fks.link_direct_fks("db") == 0
    assert wired["warehouse"].probes == []


def test_skipped_when_warehouse_queries_are_prevented(wired, monkeypatch) -> None:
    wired["candidates"] = [_column("lessons"), _column("grades")]
    monkeypatch.setattr(direct_fks, "is_semantic_fk_db_query_prevented", lambda: True)

    assert direct_fks.link_direct_fks("db") == 0
    assert wired["warehouse"].probes == []


def test_skipped_without_a_connector(wired, monkeypatch) -> None:
    wired["candidates"] = [_column("lessons"), _column("grades")]
    monkeypatch.setattr(direct_fks, "_resolve_connector", lambda _db: None)

    assert direct_fks.link_direct_fks("db") == 0
    assert wired["warehouse"].probes == []
