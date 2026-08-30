# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The history table is declared twice, and the two have to agree.

``history.py`` creates it at service startup (``CREATE TABLE IF NOT EXISTS``)
so a deployment gets it without waiting on a migration; ``gsf.dal.schema``
declares it so Alembic's autogenerate doesn't read it as drift and propose
``drop_table`` for it — which is what ``include_object`` in ``alembic/env.py``
warns any stray ``public`` table invites.

Two declarations of one table is a duplication that can drift silently: the
column list disagreeing costs nothing at startup (the CREATE is skipped once
the table exists) and shows up later as a write failing against a column that
isn't there. So they are compared here rather than trusted.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("sqlalchemy")

from gsf.dal import schema as s  # noqa: E402
from gsf.ingestion_service import history  # noqa: E402


def test_status_values_match_the_schemas_check_constraint() -> None:
    """`schema.py` restates these rather than importing them from here."""
    assert (s.RUN_SUCCEEDED, s.RUN_FAILED) == (
        history.RUN_SUCCEEDED,
        history.RUN_FAILED,
    )


def test_the_startup_create_matches_the_declared_table() -> None:
    """Same columns, same order, same nullability."""
    columns = re.findall(
        r"^\s+(\w+)\s+(BIGSERIAL|TIMESTAMPTZ|TEXT)(.*)$",
        history._CREATE_TABLE_SQL,
        re.MULTILINE,
    )
    assert [name for name, _, _ in columns] == [
        column.name for column in s.semantic_compilation_history.columns
    ]

    not_null = {name for name, _, rest in columns if "NOT NULL" in rest.upper()}
    assert not_null == {
        column.name
        for column in s.semantic_compilation_history.columns
        if not column.nullable and not column.primary_key
    }


def test_the_startup_create_names_the_same_status_values() -> None:
    """The CHECK is spelled out in SQL on one side and a constraint on the other."""
    for value in (history.RUN_SUCCEEDED, history.RUN_FAILED):
        assert f"'{value}'" in history._CREATE_TABLE_SQL
    assert history._VALID_OUTCOMES == {s.RUN_SUCCEEDED, s.RUN_FAILED}
