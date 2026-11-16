# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for table-family detection and FK realignment."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from auto_ontology.semantic import table_families as tf


def _table(
    name: str,
    columns: list[str],
    table_id: str | None = None,
    schema: str = "warehouse",
) -> dict:
    return {
        "id": table_id or name,
        "name": name,
        "schema_name": schema,
        "columns": columns,
    }


def _catalog(schema: str = "warehouse", ids: bool = False) -> list[dict]:
    """Two families, OLD_ and NEW_, plus a history table and an unrelated one.

    With *ids*, table ids carry the schema, so two calls do not collide.
    """
    tables = [
        ("OLD_BUILDING", ["OLD_BUILDING_KEY", "BUILDING_NAME", "SITE"]),
        (
            "NEW_BUILDING",
            ["NEW_BUILDING_KEY", "BUILDING_NAME", "SITE", "BUILDING_NUMBER"],
        ),
        ("OLD_ROOMS", ["ROOM", "OLD_BUILDING_KEY", "AREA"]),
        ("NEW_ROOMS", ["ROOM", "NEW_BUILDING_KEY", "AREA", "ZONE_KEY"]),
        ("OLD_ORGANIZATION", ["ORGANIZATION_KEY", "NAME", "REGION_CODE"]),
        ("NEW_ORGANIZATION", ["NEW_ORGANIZATION_KEY", "NAME"]),
        ("NEW_BUILDING_HIST", ["NEW_BUILDING_KEY", "BUILDING_NAME"]),
        ("PAY_DEPARTMENT", ["DEPARTMENT_CODE", "NAME"]),
    ]
    return [
        _table(name, columns, f"{schema}.{name}" if ids else None, schema)
        for name, columns in tables
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


def test_families_do_not_span_schemas() -> None:
    """``live.NEW_*`` beside ``archive.OLD_*`` are independent schemas."""
    tables = [
        t for t in _catalog("live", ids=True) if not t["name"].startswith("OLD_")
    ] + [t for t in _catalog("archive", ids=True) if not t["name"].startswith("NEW_")]

    assert tf.detect_family_pairs(tables) == []


def test_same_named_tables_in_two_schemas_do_not_overwrite_each_other() -> None:
    tables = _catalog("live", ids=True) + _catalog("archive", ids=True)

    pairs = tf.detect_family_pairs(tables)

    assert [(p.schema, p.prefixes) for p in pairs] == [
        ("archive", ("NEW_", "OLD_")),
        ("live", ("NEW_", "OLD_")),
    ]
    for pair in pairs:
        ids = {t["id"] for twin in pair.twins.values() for t in twin.values()}
        assert all(i.startswith(f"{pair.schema}.") for i in ids)


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
        patch.object(tf, "find_column_id_in_table", return_value="col:twin"),
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


def test_twin_column_is_looked_up_in_the_twin_table_not_by_name() -> None:
    """A same-named table in another schema must not answer for the twin."""
    tables = _catalog()
    for table in tables:
        table["id"] = f"warehouse.{table['name']}"
    [pair] = tf.detect_family_pairs(tables)
    row = _fk_row("NEW_ROOMS", "NEW_BUILDING_KEY", "OLD_BUILDING", "OLD_BUILDING_KEY")
    row["source_table_id"] = "warehouse.NEW_ROOMS"
    row["target_table_id"] = "warehouse.OLD_BUILDING"
    with (
        patch.object(tf, "fetch_semantic_fk_join_groups", return_value=[row]),
        patch.object(tf, "find_column_id_in_table", return_value="col:twin") as find,
        patch.object(
            tf, "find_column_attribute_by_column_id", return_value="attr:twin"
        ),
        patch.object(tf, "merge_semantic_fk", return_value=True),
        patch.object(tf, "set_semantic_fk_join_default"),
        patch.object(tf, "delete_semantic_fk"),
    ):
        tf.keep_semantic_fks_in_family("warehouse", [pair], _connector([8, 9]))

    find.assert_called_once_with("warehouse.NEW_BUILDING", "NEW_BUILDING_KEY")


def test_crossing_fk_is_dropped_when_the_twin_covers_less() -> None:
    [pair] = tf.detect_family_pairs(_catalog())
    row = _fk_row("NEW_ROOMS", "NEW_BUILDING_KEY", "OLD_BUILDING", "OLD_BUILDING_KEY")
    with (
        patch.object(tf, "fetch_semantic_fk_join_groups", return_value=[row]),
        patch.object(tf, "find_column_id_in_table", return_value="col:twin"),
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


def _keep_case(
    connector: MagicMock | None, attr: str | None = "attr:twin", merged: bool = True
) -> tuple[int, MagicMock]:
    [pair] = tf.detect_family_pairs(_catalog())
    row = _fk_row("NEW_ROOMS", "NEW_BUILDING_KEY", "OLD_BUILDING", "OLD_BUILDING_KEY")
    with (
        patch.object(tf, "fetch_semantic_fk_join_groups", return_value=[row]),
        patch.object(tf, "find_column_id_in_table", return_value="col:twin"),
        patch.object(tf, "find_column_attribute_by_column_id", return_value=attr),
        patch.object(tf, "merge_semantic_fk", return_value=merged),
        patch.object(tf, "set_semantic_fk_join_default"),
        patch.object(tf, "delete_semantic_fk") as delete,
    ):
        handled = tf.keep_semantic_fks_in_family("warehouse", [pair], connector)
    return handled, delete


def test_crossing_fk_is_kept_when_coverage_cannot_be_checked() -> None:
    failing = _connector([])
    failing.execute.side_effect = TimeoutError("statement timeout")

    handled, delete = _keep_case(failing)

    assert handled == 0
    delete.assert_not_called()


def test_crossing_fk_is_kept_without_a_connector() -> None:
    handled, delete = _keep_case(None)

    assert handled == 0
    delete.assert_not_called()


def test_crossing_fk_is_kept_when_the_twin_column_has_no_attribute() -> None:
    handled, delete = _keep_case(_connector([8, 9]), attr=None)

    assert handled == 0
    delete.assert_not_called()


def test_crossing_fk_is_kept_when_the_moved_edge_cannot_be_written() -> None:
    handled, delete = _keep_case(_connector([8, 9]), merged=False)

    assert handled == 0
    delete.assert_not_called()


def _schema_fk_row(source_schema: str, target_schema: str) -> dict:
    row = _fk_row("NEW_ROOMS", "NEW_BUILDING_KEY", "OLD_BUILDING", "OLD_BUILDING_KEY")
    row.update(
        source_table_id=f"{source_schema}.NEW_ROOMS",
        source_schema_name=source_schema,
        target_table_id=f"{target_schema}.OLD_BUILDING",
        target_schema_name=target_schema,
    )
    return row


def test_each_schemas_family_handles_its_own_crossing_fks() -> None:
    """Pairs with the same prefixes in two schemas must not shadow each other."""
    pairs = tf.detect_family_pairs(
        _catalog("live", ids=True) + _catalog("archive", ids=True)
    )
    rows = [_schema_fk_row("live", "live"), _schema_fk_row("archive", "archive")]
    with (
        patch.object(tf, "fetch_semantic_fk_join_groups", return_value=rows),
        patch.object(tf, "find_column_id_in_table", return_value="col:twin") as find,
        patch.object(
            tf, "find_column_attribute_by_column_id", return_value="attr:twin"
        ),
        patch.object(tf, "merge_semantic_fk", return_value=True),
        patch.object(tf, "set_semantic_fk_join_default"),
        patch.object(tf, "delete_semantic_fk"),
    ):
        handled = tf.keep_semantic_fks_in_family(
            "warehouse", pairs, _connector([8, 9, 8, 9])
        )

    assert handled == 2
    assert [c.args[0] for c in find.call_args_list] == [
        "live.NEW_BUILDING",
        "archive.NEW_BUILDING",
    ]


def test_an_fk_into_another_schemas_family_is_left_alone() -> None:
    pairs = tf.detect_family_pairs(
        _catalog("live", ids=True) + _catalog("archive", ids=True)
    )
    with (
        patch.object(
            tf,
            "fetch_semantic_fk_join_groups",
            return_value=[_schema_fk_row("live", "archive")],
        ),
        patch.object(tf, "delete_semantic_fk") as delete,
    ):
        assert tf.keep_semantic_fks_in_family("warehouse", pairs, _connector([])) == 0

    delete.assert_not_called()


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
