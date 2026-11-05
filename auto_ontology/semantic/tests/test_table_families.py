# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for table-family detection and FK realignment."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from auto_ontology.semantic import table_families as tf


def _table(name: str, columns: list[str], table_id: str | None = None) -> dict:
    return {
        "id": table_id or name,
        "name": name,
        "schema_name": "warehouse",
        "columns": columns,
    }


def _catalog() -> list[dict]:
    """Two families, OLD_ and NEW_, plus a history table and an unrelated one."""
    return [
        _table("OLD_BUILDING", ["OLD_BUILDING_KEY", "BUILDING_NAME", "SITE"]),
        _table(
            "NEW_BUILDING",
            ["NEW_BUILDING_KEY", "BUILDING_NAME", "SITE", "BUILDING_NUMBER"],
        ),
        _table("OLD_ROOMS", ["ROOM", "OLD_BUILDING_KEY", "AREA"]),
        _table("NEW_ROOMS", ["ROOM", "NEW_BUILDING_KEY", "AREA", "ZONE_KEY"]),
        _table("OLD_ORGANIZATION", ["ORGANIZATION_KEY", "NAME", "REGION_CODE"]),
        _table("NEW_ORGANIZATION", ["NEW_ORGANIZATION_KEY", "NAME"]),
        _table("NEW_BUILDING_HIST", ["NEW_BUILDING_KEY", "BUILDING_NAME"]),
        _table("PAY_DEPARTMENT", ["DEPARTMENT_CODE", "NAME"]),
    ]


def test_detects_the_pair_and_leaves_history_tables_out() -> None:
    [pair] = tf.detect_family_pairs(_catalog())

    assert pair.prefixes == ("NEW_", "OLD_")
    assert sorted(pair.twins) == ["BUILDING", "ORGANIZATION", "ROOMS"]


def test_too_few_twins_is_not_a_family() -> None:
    tables = [t for t in _catalog() if "ORGANIZATION" not in t["name"]]

    assert tf.detect_family_pairs(tables) == []


def test_tables_without_a_prefix_or_overlap_are_ignored() -> None:
    tables = _catalog() + [
        _table("HR_BUILDING", ["EMPLOYEE", "BADGE"]),
        _table("HR_ROOMS", ["EMPLOYEE", "DESK"]),
        _table("HR_ORGANIZATION", ["UNIT"]),
        _table("PLAIN", ["A"]),
    ]

    assert [p.prefixes for p in tf.detect_family_pairs(tables)] == [("NEW_", "OLD_")]


def _fk_row(source: str, column: str, target: str, target_column: str) -> dict:
    return {
        "source_table_id": source,
        "source_table_name": source,
        "source_schema_name": "warehouse",
        "source_column_id": f"{source}.{column}",
        "source_column_name": column,
        "target_attribute_id": f"attr:{target}.{target_column}",
        "target_table_id": target,
        "target_table_name": target,
        "target_schema_name": "warehouse",
        "target_column_name": target_column,
    }


def _connector(coverage: list[int]) -> MagicMock:
    connector = MagicMock()
    connector.dialect = "mysql"
    connector.qualify.side_effect = lambda schema, table: f"{schema}.{table}"
    connector.execute.side_effect = [pd.DataFrame({"n": [n]}) for n in coverage]
    return connector


def test_crossing_fk_moves_to_the_twin_in_its_own_family() -> None:
    [pair] = tf.detect_family_pairs(_catalog())
    row = _fk_row("NEW_ROOMS", "NEW_BUILDING_KEY", "OLD_BUILDING", "OLD_BUILDING_KEY")
    with (
        patch.object(tf, "fetch_semantic_fk_join_groups", return_value=[row]),
        patch.object(tf, "find_column_id_by_table_and_name", return_value="col:twin"),
        patch.object(
            tf, "find_column_attribute_by_column_id", return_value="attr:twin"
        ),
        patch.object(tf, "merge_semantic_fk", return_value=True) as merge,
        patch.object(tf, "set_semantic_fk_join_default") as default,
        patch.object(tf, "delete_semantic_fk") as delete,
    ):
        assert (
            tf.keep_semantic_fks_in_family("warehouse", [pair], _connector([8, 9])) == 1
        )

    merge.assert_called_once_with("NEW_ROOMS.NEW_BUILDING_KEY", "attr:twin")
    default.assert_called_once_with(
        "NEW_ROOMS", "attr:twin", "NEW_ROOMS.NEW_BUILDING_KEY"
    )
    delete.assert_called_once_with(
        "NEW_ROOMS.NEW_BUILDING_KEY", "attr:OLD_BUILDING.OLD_BUILDING_KEY"
    )


def test_crossing_fk_is_dropped_when_the_twin_covers_less() -> None:
    [pair] = tf.detect_family_pairs(_catalog())
    row = _fk_row("NEW_ROOMS", "NEW_BUILDING_KEY", "OLD_BUILDING", "OLD_BUILDING_KEY")
    with (
        patch.object(tf, "fetch_semantic_fk_join_groups", return_value=[row]),
        patch.object(tf, "find_column_id_by_table_and_name", return_value="col:twin"),
        patch.object(
            tf, "find_column_attribute_by_column_id", return_value="attr:twin"
        ),
        patch.object(tf, "merge_semantic_fk") as merge,
        patch.object(tf, "delete_semantic_fk") as delete,
    ):
        tf.keep_semantic_fks_in_family("warehouse", [pair], _connector([9, 3]))

    merge.assert_not_called()
    delete.assert_called_once()


def test_crossing_fk_whose_twin_is_its_own_table_is_dropped() -> None:
    [pair] = tf.detect_family_pairs(_catalog())
    row = _fk_row(
        "OLD_BUILDING", "PARENT_BUILDING_NUMBER", "NEW_BUILDING", "NEW_BUILDING_KEY"
    )
    with (
        patch.object(tf, "fetch_semantic_fk_join_groups", return_value=[row]),
        patch.object(tf, "merge_semantic_fk") as merge,
        patch.object(tf, "delete_semantic_fk") as delete,
    ):
        assert tf.keep_semantic_fks_in_family("warehouse", [pair], _connector([])) == 1

    merge.assert_not_called()
    delete.assert_called_once()


def test_fks_inside_a_family_or_outside_any_are_left_alone() -> None:
    [pair] = tf.detect_family_pairs(_catalog())
    rows = [
        _fk_row("NEW_ROOMS", "NEW_BUILDING_KEY", "NEW_BUILDING", "NEW_BUILDING_KEY"),
        _fk_row("PAY_DEPARTMENT", "BUILDING", "OLD_BUILDING", "OLD_BUILDING_KEY"),
    ]
    with (
        patch.object(tf, "fetch_semantic_fk_join_groups", return_value=rows),
        patch.object(tf, "delete_semantic_fk") as delete,
    ):
        assert tf.keep_semantic_fks_in_family("warehouse", [pair], _connector([])) == 0

    delete.assert_not_called()
