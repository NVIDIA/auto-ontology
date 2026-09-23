# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for which columns admit a hand-edited sample value."""

from __future__ import annotations

import pytest

from auto_ontology.utils.column_types import (
    sample_values_edit_error,
    sample_values_editable,
)


@pytest.mark.parametrize(
    "data_type",
    [
        "text",
        "TEXT",
        "  varchar  ",
        "varchar(255)",
        "character varying",
        "character varying(50)",
        "nvarchar2(20)",
        "char(1)",
        "longtext",
        "STRING",
        "clob",
        "citext",
        "enum('a','b')",
    ],
)
def test_text_columns_are_editable(data_type: str) -> None:
    """Any text a user types is a value the column could already hold."""
    assert sample_values_editable(data_type) is True


@pytest.mark.parametrize("data_type", ["json", "jsonb", "VARIANT", "xml"])
def test_semi_structured_columns_are_editable(data_type: str) -> None:
    """A JSONB or VARIANT value may be a bare string, so text fits them too."""
    assert sample_values_editable(data_type) is True


@pytest.mark.parametrize(
    "data_type",
    [
        "integer",
        "int",
        "bigint",
        "numeric",
        "numeric(10,2)",
        "double precision",
        "money",
        "boolean",
        "date",
        "timestamp without time zone",
        "uuid",
        "bytea",
        "object",
        "array",
    ],
)
def test_non_text_columns_are_not_editable(data_type: str) -> None:
    """These hold no arbitrary string, so profiled samples stay as they are."""
    assert sample_values_editable(data_type) is False


@pytest.mark.parametrize("data_type", ["interval", "point", "inet", "integer"])
def test_type_names_are_matched_whole(data_type: str) -> None:
    """A substring search would read `interval` and `point` as containing `int`.

    Their being rejected is the point: the guard must not accept a type merely
    because a text type's name occurs inside it, nor the reverse.
    """
    assert sample_values_editable(data_type) is False


@pytest.mark.parametrize("data_type", ["text[]", "varchar[]", "_text"])
def test_array_columns_are_not_editable(data_type: str) -> None:
    """An array holds no single string a text box could stand for."""
    assert sample_values_editable(data_type) is False


@pytest.mark.parametrize("data_type", [None, "", "   ", "some_unknown_type"])
def test_unknown_types_are_not_editable(data_type: str | None) -> None:
    """An unvouched type costs a column its edit affordance, not its samples."""
    assert sample_values_editable(data_type) is False


def test_edit_error_is_silent_for_an_editable_column() -> None:
    assert sample_values_edit_error("varchar(10)") is None
    assert sample_values_edit_error("jsonb") is None


def test_edit_error_names_the_offending_type() -> None:
    """The message reaches the user as the detail of a 422, so it must say why."""
    assert sample_values_edit_error("timestamp") == (
        "Sample values of a column typed timestamp cannot be edited; only "
        "text-typed columns (text, varchar, json, …) accept them."
    )


@pytest.mark.parametrize("data_type", ["integer", "int", "interval"])
def test_edit_error_reads_as_english_for_a_vowel_initial_type(data_type: str) -> None:
    """An indefinite article in front of the type name would read "a integer"."""
    error = sample_values_edit_error(data_type)
    assert error is not None
    assert f"typed {data_type}" in error


def test_edit_error_covers_a_column_with_no_declared_type() -> None:
    error = sample_values_edit_error(None)
    assert error is not None
    assert "no declared type" in error
