# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the join_columns Neo4j-property (de)serialization helpers."""

from __future__ import annotations

from gsf.utils.join_columns import dump_join_columns, parse_join_columns


def test_dump_join_columns_produces_a_json_string() -> None:
    """Neo4j properties can't be a list of maps, so this must be a string."""
    dumped = dump_join_columns([{"source": "customer_id", "target": "id"}])

    assert isinstance(dumped, str)
    assert parse_join_columns(dumped) == [{"source": "customer_id", "target": "id"}]


def test_dump_join_columns_round_trips_empty_list() -> None:
    assert parse_join_columns(dump_join_columns([])) == []


def test_parse_join_columns_accepts_a_raw_list() -> None:
    """Backward-compat: property values written before this fix are plain lists."""
    raw = [{"source": "a", "target": "b"}]

    assert parse_join_columns(raw) is raw


def test_parse_join_columns_handles_missing_or_invalid_values() -> None:
    assert parse_join_columns(None) == []
    assert parse_join_columns("") == []
    assert parse_join_columns("not json") == []
    assert parse_join_columns('"a string, not a list"') == []
