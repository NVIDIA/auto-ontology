# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for comparing FK and key columns by their ascending distinct values."""

from __future__ import annotations

import pytest

from auto_ontology.semantic.fk_values import (
    containment,
    contains,
    is_value_search_ambiguous,
)

_LIMIT = 5


def test_complete_key_list_must_hold_every_fk_value() -> None:
    key = [10, 11, 12]

    assert contains([10, 12], key, limit=_LIMIT)
    assert containment([10, 12, 50], key, limit=_LIMIT) == pytest.approx(2 / 3)
    assert not contains([10, 12, 50], key, limit=_LIMIT)


def test_cut_key_list_ignores_fk_values_beyond_its_range() -> None:
    key = [10, 11, 12, 13, 14]

    assert containment([10, 14, 900, 901], key, limit=_LIMIT) == 1.0
    assert containment([900, 901], key, limit=_LIMIT) is None
    assert not contains([900, 901], key, limit=_LIMIT)


def test_a_few_orphans_still_count_as_contained() -> None:
    key = list(range(100, 120))
    fk = list(range(100, 119)) + [500]

    assert contains(fk, key, limit=50)


def test_numeric_key_matches_numbers_stored_as_text() -> None:
    assert contains(["100", "101"], [100, 101, 102], limit=_LIMIT)
    assert contains([100.0], [100, 101], limit=_LIMIT)


def test_text_key_compares_rendered_values() -> None:
    assert contains([100, 101], ["100", "101", "102"], limit=_LIMIT)
    assert not contains(["A01", "B07"], ["A01", "A02"], limit=_LIMIT)


def test_text_range_uses_text_order() -> None:
    key = ["A01", "A02", "A03", "A04", "A05"]

    assert containment(["A01", "Z99"], key, limit=_LIMIT) == 1.0


@pytest.mark.parametrize("fk, key", [(None, [1]), ([], [1]), ([1], None), ([1], [])])
def test_nothing_to_compare(fk: list | None, key: list | None) -> None:
    assert containment(fk, key, limit=_LIMIT) is None
    assert not contains(fk, key, limit=_LIMIT)


@pytest.mark.parametrize(
    ("name", "values", "ambiguous"),
    [
        ("customer_ref", [1, 2, 3], True),
        ("flag", ["Y", "N"], True),
        ("customer_ref", [1, 2, 1000], False),
        ("customer_name", ["Alice Smith"], True),
        ("ProductTitle", ["Widget Pro"], True),
        ("store_code", ["ST-001", "ST-002"], False),
        ("customer_ref", None, True),
    ],
)
def test_value_search_ambiguity(
    name: str, values: list | None, ambiguous: bool
) -> None:
    assert is_value_search_ambiguous(name, values) is ambiguous
