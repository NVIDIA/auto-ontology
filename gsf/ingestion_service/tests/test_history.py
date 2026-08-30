# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic compilation history.

The table is declared in :mod:`gsf.dal.schema` and created by a migration, like
every other GSF table. It used to create itself at service startup instead,
which is worth remembering only because this module must not go back to it: a
`public` table the schema does not declare reads as drift to Alembic, and
autogenerate proposes ``drop_table`` for it (see ``include_object`` in
``alembic/env.py``). ``test_the_module_does_not_create_its_own_table`` is that
rule, written down.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import CheckConstraint  # noqa: E402

from gsf.dal import schema as s  # noqa: E402
from gsf.ingestion_service import history  # noqa: E402


def test_the_module_does_not_create_its_own_table() -> None:
    """Creating it here would put it back outside the schema's source of truth."""
    assert "CREATE TABLE" not in inspect.getsource(history).upper()


def test_status_values_come_from_the_check_constraint() -> None:
    """One definition, imported — not two that agree today.

    A write of a value the constraint doesn't permit fails at runtime, so these
    being the same object is what keeps the writer honest.
    """
    assert history.RUN_SUCCEEDED is s.RUN_SUCCEEDED
    assert history.RUN_FAILED is s.RUN_FAILED
    assert history._VALID_OUTCOMES == {s.RUN_SUCCEEDED, s.RUN_FAILED}


def test_the_check_constraint_permits_exactly_those_values() -> None:
    """The constraint is written as SQL text, so it can drift from the values."""
    # Selected by type, not by name: the schema's naming convention rewrites
    # "status_value" into "ck_semantic_compilation_history_status_value".
    constraint = next(
        c
        for c in s.semantic_compilation_history.constraints
        if isinstance(c, CheckConstraint)
    )
    condition = str(constraint.sqltext)
    for value in (s.RUN_SUCCEEDED, s.RUN_FAILED):
        assert f"'{value}'" in condition
    # NULL is the third permitted state — an unfinished pass — and dropping it
    # from the constraint would make `record_run_start` fail on insert.
    assert "IS NULL" in condition.upper()
