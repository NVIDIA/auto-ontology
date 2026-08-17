# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The embedding columns must not escape into API responses.

``embedding`` is 2048 floats -- about 8KB per row. It is storage for vector
search, not a property of the entity, and several DAL reads mean "give me every
property of this node" and are serialised straight into an HTTP response.

Adding the column made three such reads start returning it. Two were caught by
the golden oracle because they happened to be recorded; the other three were
found only by grepping for ``select(table)``. This test covers the ones the
oracle does not, so the next embeddable table cannot quietly reintroduce it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gsf.dal import schema as s
from gsf.vdb.entity_store import LABEL_TABLES

_DAL = Path(__file__).resolve().parents[1]

#: `select(<table>)` with no column list -- the form that takes every column.
_BARE_SELECT = re.compile(
    r"select\(\s*(?:s\.)?("
    + "|".join(t.name for t in LABEL_TABLES.values())
    + r"|table)\s*\)"
)


def test_embedded_tables_declare_their_internals() -> None:
    """Every embeddable table carries exactly the three internal columns."""
    for label, table in LABEL_TABLES.items():
        missing = s.INTERNAL_COLUMNS - set(table.c.keys())
        assert not missing, f"{label} ({table.name}) is missing {sorted(missing)}"


def test_public_columns_excludes_every_internal_column() -> None:
    for label, table in LABEL_TABLES.items():
        names = {c.name for c in s.public_columns(table)}
        leaked = names & s.INTERNAL_COLUMNS
        assert not leaked, f"{label}: public_columns leaks {sorted(leaked)}"
        # And it must not have thrown the baby out: id always survives.
        assert "id" in names, f"{label}: public_columns dropped id"


def test_no_bare_select_of_an_embeddable_table() -> None:
    """A bare ``select(table)`` on an embeddable table returns the vector.

    Use ``select(*s.public_columns(table))``. This is a source scan because the
    failure is silent -- the response is merely 8KB larger per row, which no
    assertion elsewhere notices.
    """
    offenders: list[str] = []
    for path in _DAL.rglob("*.py"):
        if "tests" in path.parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if _BARE_SELECT.search(line):
                offenders.append(f"{path.relative_to(_DAL)}:{number}: {line.strip()}")

    assert not offenders, (
        "bare select() of an embeddable table returns the 2048-float embedding "
        "column; use select(*s.public_columns(table)):\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("label", sorted(LABEL_TABLES))
def test_internal_columns_are_nullable(label: str) -> None:
    """A row exists before it is embedded, so these must be optional.

    If any were NOT NULL, inserting a catalog row would require an embedding it
    cannot have yet, and ingestion would fail before the embed step ran.
    """
    table = LABEL_TABLES[label]
    for name in sorted(s.INTERNAL_COLUMNS):
        assert table.c[name].nullable, f"{label}.{name} must be nullable"
