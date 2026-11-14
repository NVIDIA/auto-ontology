# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the ascending distinct values of a FK column and a candidate key.

Both sides store only their lowest distinct values (``catalog_column.
distinct_values``), so neither list is the whole column. Comparison is
therefore restricted to the range the key's list covers: a FK value above the
key's highest stored value may well be in the key, only beyond the cut. When
the key's list is shorter than the cap it *is* the whole column, and every FK
value has to be found in it.
"""

from __future__ import annotations

import re
from typing import Any

from auto_ontology.utils.sample_values import render_sample_value

# Share of the FK's comparable values the key must contain. Below 1.0 so a
# handful of orphaned rows, routine in loaded data, does not hide the match.
MIN_CONTAINMENT = 0.9
# Values this short (``1``, ``2``, ``Y``) occur in every small key, so a
# column of nothing else proves nothing and is kept out of the value-only
# search.
_MIN_VALUE_LENGTH = 3
_LABEL_TOKENS = frozenset(
    {"name", "title", "label", "description", "desc", "comment", "note", "notes"}
)
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _comparable(
    fk_values: list[Any], key_values: list[Any]
) -> tuple[list[Any], list[Any]]:
    """Put both lists in the order the key's list was cut in.

    Numeric when the key column is numeric and every FK value reads as a
    number, so ``"42"`` stored in a text FK still finds ``42``; otherwise the
    rendered text, which is also the order a text key was probed in.
    """
    if key_values and all(_is_number(v) for v in key_values):
        fk_numbers = [_as_number(v) for v in fk_values]
        if all(n is not None for n in fk_numbers):
            return fk_numbers, [float(v) for v in key_values]
    return (
        [render_sample_value(v) for v in fk_values],
        [render_sample_value(v) for v in key_values],
    )


def containment(
    fk_values: list[Any] | None,
    key_values: list[Any] | None,
    *,
    limit: int,
) -> float | None:
    """Share of the FK's comparable values found in the key, or ``None``.

    ``None`` when there is nothing to compare: either list is empty, or every
    FK value lies beyond the key's stored range.
    """
    if not fk_values or not key_values:
        return None
    fk, key = _comparable(fk_values, key_values)
    if len(key_values) >= limit:
        ceiling = max(key)
        fk = [v for v in fk if v <= ceiling]
    if not fk:
        return None
    present = set(key)
    return sum(1 for v in fk if v in present) / len(fk)


def contains(
    fk_values: list[Any] | None,
    key_values: list[Any] | None,
    *,
    limit: int,
) -> bool:
    """Whether the key's values hold the FK's, within the key's stored range."""
    share = containment(fk_values, key_values, limit=limit)
    return share is not None and share >= MIN_CONTAINMENT


def is_value_search_ambiguous(column_name: str, fk_values: list[Any] | None) -> bool:
    """Whether a value-only match for this column would be a coincidence.

    True for columns whose values are all very short — small integers and
    flags coincide with every surrogate key — and for label columns, whose
    text matches other label columns rather than the key they describe.
    """
    if not fk_values:
        return True
    words = _CAMEL_BOUNDARY.sub("_", column_name).lower()
    if set(_TOKEN_SPLIT.split(words)) & _LABEL_TOKENS:
        return True
    return all(len(render_sample_value(v)) < _MIN_VALUE_LENGTH for v in fk_values)
