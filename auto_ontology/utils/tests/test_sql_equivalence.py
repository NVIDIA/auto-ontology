# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which SQL expressions count as the same expression."""

from __future__ import annotations

import pytest

from auto_ontology.utils.sql_equivalence import sql_equivalence_key


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("SELECT  sum(amount)\n FROM p", "select SUM(amount) from p"),
        ("SELECT A.B.X AS M FROM A.B", "select a.b.x as m from a.b"),
        ("SELECT x FROM t WHERE s = 'x", "select   x from t where S = 'X"),
    ],
    ids=["formatting-and-keywords", "unquoted-identifiers", "unparseable-fallback"],
)
def test_equivalent(left: str, right: str) -> None:
    assert sql_equivalence_key(left) == sql_equivalence_key(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("SELECT x FROM t WHERE s = 'Open'", "SELECT x FROM t WHERE s = 'OPEN'"),
        ("SELECT x FROM t WHERE s = 'A  B'", "SELECT x FROM t WHERE s = 'A B'"),
        ('SELECT x AS "Margin" FROM t', 'SELECT x AS "margin" FROM t'),
    ],
    ids=["literal-case", "literal-whitespace", "quoted-identifier"],
)
def test_distinct(left: str, right: str) -> None:
    assert sql_equivalence_key(left) != sql_equivalence_key(right)
