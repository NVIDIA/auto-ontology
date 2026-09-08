# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dialect-aware SQL identifier quoting."""

from __future__ import annotations

import pytest

from gsf.utils.sql_identifiers import qualified_name, quoted_identifier


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("sqlite", '"Sales Orders"'),
        ("postgres", '"Sales Orders"'),
        ("mysql", "`Sales Orders`"),
        ("spark", "`Sales Orders`"),
        (None, '"Sales Orders"'),
        ("not-a-dialect", '"Sales Orders"'),
    ],
)
def test_quoted_identifier_per_dialect(dialect: str | None, expected: str) -> None:
    # Backticks on MySQL/Spark, double quotes elsewhere; an unknown dialect
    # still quotes, since a bare identifier is what loses the table.
    assert quoted_identifier("Sales Orders", dialect) == expected


def test_qualified_name_joins_parts() -> None:
    assert qualified_name("nvapp", "raw", "t", dialect="spark") == "`nvapp`.`raw`.`t`"


def test_qualified_name_skips_empty_parts() -> None:
    # A two-level engine passes no catalog; an unqualified table passes no
    # schema. Neither may leave a stray leading dot.
    assert qualified_name(None, "raw", "t", dialect="spark") == "`raw`.`t`"
    assert qualified_name(None, None, "t", dialect="spark") == "`t`"
    assert qualified_name("", "raw", "t", dialect="spark") == "`raw`.`t`"
