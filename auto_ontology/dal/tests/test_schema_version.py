# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The startup schema gate, and the three states it has to tell apart.

The distinction that matters is between *unmigrated* and *unreachable*. Both
yield "no revision", and conflating them tells an operator whose database is
down to run a migration against it — advice that cannot work. So the message is
asserted here, not just the pass/fail.

No database needed: the states are constructed directly. The live paths are
covered by ``test_reports_the_real_revision``, which skips without a store.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("sqlalchemy")

from auto_ontology.dal import schema_version as sv  # noqa: E402


def test_matching_revisions_are_current() -> None:
    state = sv.SchemaState(expected="abc123", applied="abc123")
    assert state.current


def test_a_behind_database_is_not_current_and_says_both_revisions() -> None:
    state = sv.SchemaState(expected="new", applied="old")
    assert not state.current
    assert "old" in state.detail and "new" in state.detail
    assert "alembic upgrade head" in state.detail


def test_an_unmigrated_database_is_told_to_migrate() -> None:
    state = sv.SchemaState(expected="abc123", applied=None)
    assert not state.current
    assert "no alembic_version table" in state.detail
    assert "alembic upgrade head" in state.detail


def test_an_unreachable_database_is_not_told_to_migrate() -> None:
    """The distinction this module exists to preserve.

    Telling someone to migrate a database nothing can connect to sends them
    down a path that cannot work, and hides the actual fault.
    """
    state = sv.SchemaState(
        expected="abc123", applied=None, unreachable="connection refused"
    )
    assert not state.current
    assert "could not read the applied revision" in state.detail
    assert "connection refused" in state.detail
    assert "alembic upgrade head" not in state.detail


def test_unknown_expected_revision_is_not_treated_as_fine() -> None:
    """A packaging mistake that hides the migrations must not read as healthy."""
    state = sv.SchemaState(expected=None, applied="abc123")
    assert not state.current
    assert "cannot determine the expected revision" in state.detail


def test_require_current_schema_raises_with_an_actionable_message() -> None:
    state = sv.SchemaState(expected="abc123", applied=None)
    with pytest.raises(sv.SchemaOutOfDateError) as excinfo:
        sv._raise_for(state)
    message = str(excinfo.value)
    assert "Refusing to start" in message
    assert "alembic upgrade head" in message


def test_missing_table_is_distinguished_from_other_failures() -> None:
    class UndefinedTable(Exception):
        pass

    assert sv._is_missing_table(UndefinedTable("relation does not exist"))
    assert not sv._is_missing_table(OSError("connection refused"))


def test_missing_table_is_found_through_a_wrapped_exception() -> None:
    """SQLAlchemy wraps the driver error, so the cause chain has to be walked."""

    class UndefinedTable(Exception):
        pass

    wrapper = RuntimeError("(psycopg.errors.UndefinedTable) ...")
    wrapper.__cause__ = UndefinedTable("relation does not exist")
    assert sv._is_missing_table(wrapper)


def test_reports_the_real_revision(monkeypatch) -> None:
    """Against a live migrated store, expected and applied agree."""
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    sv.head_revision.cache_clear()
    state = sv.schema_state()
    if state.unreachable is not None:
        pytest.skip(f"store unreachable: {state.unreachable}")
    assert state.expected, "the build should always know its own head revision"
    assert state.current, state.detail
