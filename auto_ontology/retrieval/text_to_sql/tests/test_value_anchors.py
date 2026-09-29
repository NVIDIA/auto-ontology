# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.text_to_sql.value_anchors import (
    split_schema_named_anchors,
)

_TABLES = [
    {
        "name": "enrollment",
        "columns": [
            {"name": "student_id"},
            {"name": "Headcount (Full-Time)"},
            {"name": "Headcount (Grade 12)"},
        ],
    },
    {
        "name": "students",
        "columns": [{"name": "Region"}, {"name": "Attendance Type"}, {"name": "Grade"}],
    },
]


def _anchor(phrase: str, tbl: str, col: str, stored_value: str) -> dict:
    return {
        "phrase": phrase,
        "kind": "value",
        "tbl": tbl,
        "col": col,
        "stored_value": stored_value,
    }


def test_phrase_inside_a_column_name_is_not_a_value() -> None:
    # A question asking for the highest full-time headcount is naming
    # enrollment."Headcount (Full-Time)", not filtering on the attendance
    # type that happens to store the same words.
    anchor = _anchor("full-time", "students", "Attendance Type", "Full-Time")

    kept, naming_schema = split_schema_named_anchors([anchor], _TABLES)

    assert kept == []
    assert naming_schema == [anchor]


def test_phrase_matching_a_whole_table_name_is_not_a_value() -> None:
    anchor = _anchor("students", "students", "Region", "Students Landing")

    kept, naming_schema = split_schema_named_anchors([anchor], _TABLES)

    assert kept == []
    assert naming_schema == [anchor]


def test_value_absent_from_every_schema_name_is_kept() -> None:
    anchor = _anchor("northside", "students", "Region", "Northside")

    kept, naming_schema = split_schema_named_anchors([anchor], _TABLES)

    assert kept == [anchor]
    assert naming_schema == []


def test_digits_alone_are_kept_despite_matching_a_column_name() -> None:
    # A question filtering on grade 12 means the value, which the "12" of
    # "Headcount (Grade 12)" must not take away.
    anchor = _anchor("12", "students", "Grade", "12")

    kept, naming_schema = split_schema_named_anchors([anchor], _TABLES)

    assert kept == [anchor]
    assert naming_schema == []


def test_phrase_matches_whole_tokens_only() -> None:
    whole = _anchor("region", "students", "Region", "Region")
    partial = _anchor("head", "students", "Region", "Headland")

    kept, naming_schema = split_schema_named_anchors([whole, partial], _TABLES)

    assert naming_schema == [whole]
    assert kept == [partial]


def test_no_tables_in_scope_leaves_every_anchor_alone() -> None:
    anchors = [_anchor("full-time", "students", "Attendance Type", "Full-Time")]

    assert split_schema_named_anchors(anchors, []) == (anchors, [])
    assert split_schema_named_anchors(anchors, None) == (anchors, [])
    assert split_schema_named_anchors(None, _TABLES) == ([], [])
