# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Term family detection, on hand-built catalog rows and data edges."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from auto_ontology.semantic.term_families import (
    detect_term_families,
    resolve_term_families,
)

_COLUMNS = {
    "ROOMS": ["ROOM_ID", "BUILDING_ID", "FLOOR", "CAPACITY", "NAME"],
    "BUILDING": ["BUILDING_ID", "ADDRESS", "CITY", "FLOORS", "NAME"],
    "STAFF": ["STAFF_ID", "ROOM_ID", "FIRST_NAME", "LAST_NAME", "ROLE"],
    "VISITS": ["VISIT_ID", "ROOM_ID", "STAFF_ID", "VISITED_AT", "REASON"],
}


def _table(
    name: str, columns: list[str] | None = None, schema: str = "dw"
) -> dict[str, Any]:
    base = name.split("_", 1)[-1]
    return {
        "table_id": f"{schema}.{name}",
        "schema_id": schema,
        "table_name": name,
        "term_id": f"term:{schema}.{name}",
        "columns": columns if columns is not None else _COLUMNS.get(base, ["ID"]),
    }


def _edge(
    source: str, source_column: str, target: str, target_column: str, schema="dw"
) -> dict[str, Any]:
    return {
        "source_table_id": f"{schema}.{source}",
        "source_column": source_column,
        "target_table_id": f"{schema}.{target}",
        "target_column": target_column,
        "origin": "fk",
    }


def _copies(prefix: str, bases: list[str], schema: str = "dw") -> list[dict]:
    return [_table(f"{prefix}_{base}", schema=schema) for base in bases]


def _members(families, prefix: str) -> set[str]:
    family = next(f for f in families if f.prefix == prefix)
    return set(family.members)


def test_three_twins_make_a_family_pair() -> None:
    bases = ["ROOMS", "BUILDING", "STAFF"]
    families = detect_term_families(_copies("OLD", bases) + _copies("NEW", bases), [])

    assert [f.prefix for f in families] == ["NEW", "OLD"]
    assert _members(families, "OLD") == {
        "term:dw.OLD_ROOMS",
        "term:dw.OLD_BUILDING",
        "term:dw.OLD_STAFF",
    }
    old = next(f for f in families if f.prefix == "OLD")
    assert old.members["term:dw.OLD_ROOMS"] == "ROOMS"


def test_two_twins_are_not_enough() -> None:
    bases = ["ROOMS", "BUILDING"]
    assert detect_term_families(_copies("OLD", bases) + _copies("NEW", bases), []) == []


def test_columns_below_the_bar_are_not_twins() -> None:
    bases = ["ROOMS", "BUILDING", "STAFF"]
    rows = _copies("OLD", bases) + _copies("NEW", ["ROOMS", "BUILDING"])
    rows.append(_table("NEW_STAFF", ["STAFF_ID", "EMAIL", "PHONE", "TEAM", "LEVEL"]))

    assert detect_term_families(rows, []) == []


def test_prefixed_column_names_match_after_stripping() -> None:
    rows = []
    for prefix in ("OLD", "NEW"):
        for base in ("ROOMS", "BUILDING", "STAFF"):
            columns = [f"{prefix}_{c}" for c in _COLUMNS[base]]
            rows.append(_table(f"{prefix}_{base}", columns))

    assert len(detect_term_families(rows, [])) == 2


def test_history_tables_are_never_twins() -> None:
    bases = ["ROOMS", "BUILDING", "ROOMS_HIST"]
    rows = _copies("OLD", bases) + _copies("NEW", bases)

    assert detect_term_families(rows, []) == []


def test_three_versions_share_one_family_per_prefix() -> None:
    bases = ["ROOMS", "BUILDING", "STAFF"]
    rows = _copies("V1", bases) + _copies("V2", bases) + _copies("V3", bases)

    families = detect_term_families(rows, [])

    assert [f.prefix for f in families] == ["V1", "V2", "V3"]
    assert all(len(f.members) == 3 for f in families)


def test_twins_must_share_a_schema() -> None:
    bases = ["ROOMS", "BUILDING", "STAFF"]
    rows = _copies("OLD", bases, schema="a") + _copies("NEW", bases, schema="b")

    assert detect_term_families(rows, []) == []


def test_a_word_prefix_without_twins_is_no_family() -> None:
    rows = [
        _table("ORDER_ITEMS"),
        _table("ORDER_HEADERS"),
        _table("ORDER_LINES"),
        _table("CUSTOMER_ITEMS", ["X"]),
    ]

    assert detect_term_families(rows, []) == []


def test_a_mirrored_edge_counts_and_relaxes_the_twin_bar() -> None:
    """ROOMS is a twin only through the mirrored edge, and that edge is the third
    piece of evidence: two twins plus one mirrored edge."""
    rows = [
        _table("OLD_ROOMS", ["ROOM_ID", "BUILDING_ID", "FLOOR", "CAPACITY"]),
        _table("NEW_ROOMS", ["ROOM_ID", "BUILDING_ID", "FLOOR", "SEATS"]),
        *_copies("OLD", ["BUILDING"]),
        *_copies("NEW", ["BUILDING"]),
    ]
    edges = [
        _edge("OLD_ROOMS", "BUILDING_ID", "OLD_BUILDING", "BUILDING_ID"),
        _edge("NEW_ROOMS", "BUILDING_ID", "NEW_BUILDING", "BUILDING_ID"),
    ]

    assert detect_term_families(rows, []) == []
    families = detect_term_families(rows, edges)
    assert _members(families, "NEW") == {"term:dw.NEW_ROOMS", "term:dw.NEW_BUILDING"}
    assert all(f.mirrored_edges == 1 for f in families)


def test_an_edge_on_different_columns_is_not_mirrored() -> None:
    rows = _copies("OLD", ["ROOMS", "BUILDING"]) + _copies("NEW", ["ROOMS", "BUILDING"])
    edges = [
        _edge("OLD_ROOMS", "BUILDING_ID", "OLD_BUILDING", "BUILDING_ID"),
        _edge("NEW_ROOMS", "ROOM_ID", "NEW_BUILDING", "BUILDING_ID"),
    ]

    assert detect_term_families(rows, edges) == []


def test_mirrored_edges_ignore_direction_and_prefixed_columns() -> None:
    rows = _copies("OLD", ["ROOMS", "BUILDING"]) + _copies("NEW", ["ROOMS", "BUILDING"])
    edges = [
        _edge("OLD_ROOMS", "OLD_BUILDING_ID", "OLD_BUILDING", "BUILDING_ID"),
        _edge("NEW_BUILDING", "BUILDING_ID", "NEW_ROOMS", "NEW_BUILDING_ID"),
    ]

    assert len(detect_term_families(rows, edges)) == 2


def test_expansion_follows_edges_within_the_prefix_only() -> None:
    bases = ["ROOMS", "BUILDING", "STAFF"]
    rows = _copies("OLD", bases) + _copies("NEW", bases)
    rows += [
        _table("OLD_AUDIT"),
        _table("OLD_AUDIT_LOG"),
        _table("OLD_ROOMS_HIST"),
        _table("OLD_ISOLATED"),
        _table("DIM_DATE"),
        _table("DIM_CALENDAR"),
    ]
    edges = [
        _edge("OLD_AUDIT", "ROOM_ID", "OLD_ROOMS", "ROOM_ID"),
        _edge("OLD_AUDIT_LOG", "AUDIT_ID", "OLD_AUDIT", "AUDIT_ID"),
        _edge("OLD_ROOMS_HIST", "ROOM_ID", "OLD_ROOMS", "ROOM_ID"),
        _edge("OLD_STAFF", "DATE_ID", "DIM_DATE", "DATE_ID"),
        _edge("DIM_CALENDAR", "DATE_ID", "DIM_DATE", "DATE_ID"),
    ]

    families = detect_term_families(rows, edges)

    assert _members(families, "OLD") == {
        "term:dw.OLD_ROOMS",
        "term:dw.OLD_BUILDING",
        "term:dw.OLD_STAFF",
        "term:dw.OLD_AUDIT",
        "term:dw.OLD_AUDIT_LOG",
        "term:dw.OLD_ROOMS_HIST",
    }
    old = next(f for f in families if f.prefix == "OLD")
    assert (old.twins, old.expanded) == (3, 3)
    assert [f.prefix for f in families] == ["NEW", "OLD"]


def test_an_unconfirmed_prefix_never_expands() -> None:
    rows = _copies("OLD", ["ROOMS", "BUILDING"]) + _copies("NEW", ["ROOMS"])
    rows.append(_table("OLD_AUDIT"))
    edges = [
        _edge("OLD_AUDIT", "ROOM_ID", "OLD_ROOMS", "ROOM_ID"),
        _edge("OLD_ROOMS", "BUILDING_ID", "OLD_BUILDING", "BUILDING_ID"),
    ]

    assert detect_term_families(rows, edges) == []


def test_a_term_shared_by_two_families_goes_to_the_first_only() -> None:
    bases = ["ROOMS", "BUILDING", "STAFF"]
    rows = _copies("OLD", bases) + _copies("NEW", bases)
    rows.append({**_table("NEW_ROOMS"), "term_id": "term:dw.OLD_ROOMS"})

    families = detect_term_families(rows, [])

    owners = [f.prefix for f in families if "term:dw.OLD_ROOMS" in f.members]
    assert owners == ["NEW"]


def test_resolve_writes_the_detected_families() -> None:
    bases = ["ROOMS", "BUILDING", "STAFF"]
    rows = _copies("OLD", bases) + _copies("NEW", bases)
    module = "auto_ontology.semantic.term_families"
    with (
        patch(f"{module}.fetch_family_candidates", return_value=rows),
        patch(f"{module}.fetch_family_edges", return_value=[]),
        patch(f"{module}.replace_term_families", return_value=6) as replace,
    ):
        assert resolve_term_families("warehouse") == 2

    database, families = replace.call_args.args
    assert database == "warehouse"
    assert [(schema, prefix, len(m)) for schema, prefix, m in families] == [
        ("dw", "NEW", 3),
        ("dw", "OLD", 3),
    ]


def test_resolve_re_embeds_only_tables_whose_family_changed() -> None:
    module = "auto_ontology.semantic.term_families"
    before = [
        {"table_id": "kept", "prefix": "NEW"},
        {"table_id": "moved", "prefix": "OLD"},
        {"table_id": "dropped", "prefix": "OLD"},
    ]
    after = [
        {"table_id": "kept", "prefix": "NEW"},
        {"table_id": "moved", "prefix": "NEW"},
        {"table_id": "joined", "prefix": "NEW"},
    ]
    embedder = MagicMock()
    with (
        patch(f"{module}.fetch_family_candidates", return_value=[]),
        patch(f"{module}.fetch_family_edges", return_value=[]),
        patch(f"{module}.replace_term_families", return_value=0),
        patch(f"{module}.fetch_table_families", side_effect=[before, after]),
        patch(
            f"{module}.fetch_terms_and_attributes_for_table",
            side_effect=lambda table_id: ([], [{"id": f"attr:{table_id}"}]),
        ) as fetch,
    ):
        resolve_term_families("warehouse", embedder)

    assert [c.args[0] for c in fetch.call_args_list] == ["dropped", "joined", "moved"]
    assert [c.args[0] for c in embedder.vdb.delete_by_id.call_args_list] == [
        "attr:dropped",
        "attr:joined",
        "attr:moved",
    ]
    assert embedder.embed_column_attributes.call_count == 3
