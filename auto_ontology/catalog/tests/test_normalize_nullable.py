# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``is_nullable`` coercion at the ingest boundary.

``catalog_column.is_nullable`` is a real boolean. It used to hold
``information_schema``'s ``'YES'``/``'NO'`` strings, and since both are truthy,
``bool(stored)`` reported every column in the catalog as nullable — 112 of 218
wrong on the fixture, with nothing failing to show it. These tests pin the
conversion so it cannot silently regress to a string.
"""

from __future__ import annotations

import pandas as pd
import pytest

from auto_ontology.catalog.normalize import coerce_nullable, normalize_columns


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        ("YES", True),
        ("NO", False),
        ("yes", True),
        ("no", False),
        (" NO ", False),
        ("TRUE", True),
        ("FALSE", False),
        ("t", True),
        ("f", False),
        ("1", True),
        ("0", False),
        (None, None),
        ("", None),
    ],
)
def test_coercion_covers_what_connectors_actually_return(raw, expected) -> None:
    assert coerce_nullable(raw) is expected


def test_the_string_no_is_false_not_true() -> None:
    """The whole bug in one assertion.

    ``bool('NO')`` is ``True``. Anything that reads the stored value without
    converting gets this backwards for every NOT NULL column in the catalog.
    """
    assert bool("NO") is True
    assert coerce_nullable("NO") is False


def test_an_unrecognised_value_raises_rather_than_guessing() -> None:
    """Defaulting is what made the original bug invisible."""
    with pytest.raises(ValueError, match="must be a boolean"):
        coerce_nullable("maybe")


def test_numpy_bools_from_a_dataframe_round_trip() -> None:
    """Connectors hand back DataFrame cells, not Python scalars."""
    series = pd.DataFrame({"x": [True, False]})["x"]
    assert coerce_nullable(series.iloc[0]) is True
    assert coerce_nullable(series.iloc[1]) is False


def test_missing_values_stay_none() -> None:
    for missing in (None, pd.NA, float("nan")):
        assert coerce_nullable(missing) is None


def test_normalize_columns_produces_a_nullable_boolean_dtype() -> None:
    """ "boolean", not "bool": a connector may not be able to determine it."""
    df = normalize_columns(
        pd.DataFrame(
            {
                "table_schema": ["s", "s", "s"],
                "table_name": ["t", "t", "t"],
                "column_name": ["a", "b", "c"],
                "ordinal_position": [1, 2, 3],
                "data_type": ["int", "int", "int"],
                "is_nullable": [True, False, None],
                "description": [None, None, None],
            }
        )
    )
    assert df["is_nullable"].dtype == "boolean"
    assert df["is_nullable"].tolist()[:2] == [True, False]
    assert pd.isna(df["is_nullable"].iloc[2])


def test_normalize_columns_still_accepts_the_sql_standard_spelling() -> None:
    """A third-party connector returning 'YES'/'NO' must not poison the store.

    The in-repo connectors all convert at their own edge; this is the safety
    net for one that does not.
    """
    df = normalize_columns(
        pd.DataFrame(
            {
                "table_schema": ["s", "s"],
                "table_name": ["t", "t"],
                "column_name": ["a", "b"],
                "ordinal_position": [1, 2],
                "data_type": ["int", "int"],
                "is_nullable": ["NO", "YES"],
                "description": [None, None],
            }
        )
    )
    assert df["is_nullable"].tolist() == [False, True]


def test_a_missing_column_normalizes_to_null_not_false() -> None:
    """Absent means undetermined; callers read that as nullable."""
    df = normalize_columns(
        pd.DataFrame(
            {
                "table_schema": ["s"],
                "table_name": ["t"],
                "column_name": ["a"],
                "ordinal_position": [1],
                "data_type": ["int"],
                "description": [None],
            }
        )
    )
    assert pd.isna(df["is_nullable"].iloc[0])
