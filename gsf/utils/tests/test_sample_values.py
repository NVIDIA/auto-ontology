# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Column.sample_values normalization."""

from __future__ import annotations

from gsf.utils.sample_values import (
    dump_sample_values,
    parse_sample_values,
    stringify_sample_values,
)


def test_parse_sample_values_drops_none() -> None:
    """JSON null / Python None must not reach consumers as a value."""
    assert parse_sample_values(["a", None, "b"]) == ["a", "b"]
    assert parse_sample_values('["a", null, "b"]') == ["a", "b"]
    assert parse_sample_values([None, None]) == []


def test_parse_sample_values_handles_legacy_json_and_native_list() -> None:
    assert parse_sample_values(["1", "2"]) == ["1", "2"]
    assert parse_sample_values('["1", "2"]') == ["1", "2"]
    assert parse_sample_values(None) is None
    assert parse_sample_values([]) == []
    assert parse_sample_values("not json") is None
    assert parse_sample_values(42) is None


def test_parse_sample_values_preserves_scalar_types() -> None:
    """A numeric column must stay distinguishable from a text one."""
    assert parse_sample_values([0, 1.5, True, "1"]) == [0, 1.5, True, "1"]
    assert [type(v) for v in parse_sample_values([0, 1.5, True, "1"]) or []] == [
        int,
        float,
        bool,
        str,
    ]


def test_parse_sample_values_decodes_legacy_json_scalar_types() -> None:
    assert parse_sample_values('[0, 1.5, true, "1"]') == [0, 1.5, True, "1"]


def test_stringify_sample_values_renders_scalars() -> None:
    assert stringify_sample_values([0, 1.5, True, "a"]) == ["0", "1.5", "True", "a"]
    assert stringify_sample_values(None) is None
    assert stringify_sample_values([]) == []


def test_stringify_sample_values_renders_containers_as_json() -> None:
    """Python repr would leak single quotes into prompts and API responses."""
    assert stringify_sample_values([{"a": 1}, ["x", "y"]]) == ['{"a": 1}', '["x", "y"]']


def test_stringify_sample_values_applies_max_len_to_rendered_form() -> None:
    """The cap measures display text, so a non-string is judged once rendered."""
    assert stringify_sample_values(["short", "x" * 31], max_len=30) == ["short"]
    assert stringify_sample_values([12345], max_len=30) == ["12345"]
    assert stringify_sample_values([10**40], max_len=30) == []


def test_dump_sample_values_round_trips_scalar_types() -> None:
    """What profiling saw is what a later read gets back."""
    values = [10, 1.5, True, "text"]
    assert parse_sample_values(dump_sample_values(values)) == values
    assert [type(v) for v in parse_sample_values(dump_sample_values(values)) or []] == [
        int,
        float,
        bool,
        str,
    ]


def test_dump_sample_values_round_trips_mixed_types_and_containers() -> None:
    """A single column can hold either: JSONB and VARIANT take any JSON value,
    and a SQLite column declared without an affinity keeps whatever was
    inserted."""
    values = ["open", 1, True, {"k": 1}, ["a", "b"]]
    assert parse_sample_values(dump_sample_values(values)) == values


def test_dump_sample_values_drops_none() -> None:
    assert parse_sample_values(dump_sample_values(["a", None, "b"])) == ["a", "b"]


def test_dump_sample_values_yields_none_when_nothing_is_storable() -> None:
    """The caller skips the write, leaving what a previous run established."""
    assert dump_sample_values([]) is None
    assert dump_sample_values([None, None]) is None


def test_dump_sample_values_stores_what_a_prompt_would_render() -> None:
    """Stored and rendered forms have to agree, containers included."""
    values = [{"k": 1}, ["a", "b"], 10]
    assert stringify_sample_values(
        dump_sample_values(values)
    ) == stringify_sample_values(values)
