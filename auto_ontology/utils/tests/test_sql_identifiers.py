# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dialect-aware SQL identifier quoting."""

from __future__ import annotations

import pytest

from auto_ontology.utils.sql_identifiers import (
    qualified_name,
    quoted_identifier,
    quoted_if_needed,
)


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


@pytest.mark.parametrize(
    ("name", "dialect", "expected"),
    [
        ("orders", "postgres", "orders"),
        ("sales-data", "postgres", '"sales-data"'),
        (
            "01a1d278-8f97-7755-8907-45833c5c34a5",
            "postgres",
            '"01a1d278-8f97-7755-8907-45833c5c34a5"',
        ),
        ("kubeType", "postgres", '"kubeType"'),
        ("select", "postgres", '"select"'),
        ("as", "postgres", '"as"'),
        ("my col", "postgres", '"my col"'),
        ("ORDERS", "snowflake", "ORDERS"),
        ("orders", "snowflake", '"orders"'),
        ("sales-data", "spark", "`sales-data`"),
        ("orders", "spark", "orders"),
        ("Orders", "mysql", "Orders"),
        ("orders", None, "orders"),
        ("sales-data", "no-such-dialect", '"sales-data"'),
    ],
)
def test_quoted_if_needed(name: str, dialect: str | None, expected: str) -> None:
    assert quoted_if_needed(name, dialect) == expected
