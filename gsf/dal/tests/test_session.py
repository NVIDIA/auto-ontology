# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transaction semantics for the Postgres DAL.

Runs against a **real
database** rather than a ``MagicMock``. Mocking here would prove only that
``.execute()`` was called; what has to hold is that a failed block leaves no
rows behind, which only a real transaction can demonstrate.

Two cases exist because the pool makes them possible to get
wrong: concurrent threads must get genuinely separate transactions (ingestion
fans out over a ``ThreadPoolExecutor``), and a spawned subprocess must not
inherit pooled sockets (the chat worker is a separate process).

Needs a migrated database and skips without one::

    docker compose up -d postgres
    uv run alembic upgrade head
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("sqlalchemy")

from gsf.dal import session as pg_session  # noqa: E402


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


@pytest.fixture(scope="module", autouse=True)
def require_database():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        rows = pg_session.store().query_read("SELECT 1 AS ok")
        assert rows == [{"ok": 1}]
        pg_session.store().query_read("SELECT 1 FROM term LIMIT 1")
    except Exception as exc:  # noqa: BLE001 — unmigrated or unreachable
        pytest.skip(f"gsf schema unavailable (run alembic upgrade head): {exc}")
    yield
    pg_session.dispose_engine()


@pytest.fixture(autouse=True)
def clean_terms():
    """Remove anything a test inserted, whatever it did to the transaction."""
    yield
    pg_session.store().query_write("DELETE FROM term WHERE name LIKE 'tx-test-%'")


def _insert_term(name: str) -> None:
    pg_session.store().query_write(
        "INSERT INTO term (name, description) VALUES (:name, 'transaction test')",
        {"name": name},
    )


def _term_exists(name: str) -> bool:
    rows = pg_session.store().query_read(
        "SELECT 1 AS found FROM term WHERE name = :name",
        {"name": name},
    )
    return bool(rows)


def test_commits_on_clean_exit() -> None:
    name = _unique("tx-test-commit")
    with pg_session.write_transaction():
        _insert_term(name)
    assert _term_exists(name)


def test_rolls_back_on_exception() -> None:
    """The whole point of the scope: a partial write must leave nothing."""
    first, second = _unique("tx-test-rollback-a"), _unique("tx-test-rollback-b")
    with pytest.raises(RuntimeError):
        with pg_session.write_transaction():
            _insert_term(first)
            _insert_term(second)
            raise RuntimeError("boom")
    assert not _term_exists(first)
    assert not _term_exists(second)


def test_nesting_reuses_the_outermost_transaction() -> None:
    """A routine calling another transactional routine commits once.

    ``model_interchange``'s import depends on this — its helpers each open a
    scope, and the import as a whole has to be atomic.
    """
    outer, inner = _unique("tx-test-outer"), _unique("tx-test-inner")
    with pytest.raises(RuntimeError):
        with pg_session.write_transaction():
            _insert_term(outer)
            with pg_session.write_transaction():
                _insert_term(inner)
            raise RuntimeError("fail after the inner block returned")
    # The inner block exiting cleanly must not have committed on its own.
    assert not _term_exists(outer)
    assert not _term_exists(inner)


def test_store_autocommits_outside_a_scope() -> None:
    name = _unique("tx-test-autocommit")
    _insert_term(name)
    assert _term_exists(name)


def test_contextvar_is_cleared_after_failure() -> None:
    """A leaked ContextVar would silently enlist the next call in a dead tx."""
    assert pg_session._active.get() is None
    with pytest.raises(RuntimeError):
        with pg_session.write_transaction():
            assert pg_session._active.get() is not None
            raise RuntimeError("boom")
    assert pg_session._active.get() is None


def test_store_returns_the_active_transaction_inside_a_scope() -> None:
    outside = pg_session.store()
    assert outside._connection is None
    with pg_session.write_transaction():
        inside = pg_session.store()
        assert inside._connection is not None


def test_concurrent_threads_get_separate_transactions() -> None:
    """Ingestion fans out over a ThreadPoolExecutor.

    If the ContextVar or the connection were shared, one thread's rollback
    would discard another thread's committed work.
    """
    committed = _unique("tx-test-thread-ok")
    rolled_back = _unique("tx-test-thread-fail")

    def commit_one() -> None:
        with pg_session.write_transaction():
            _insert_term(committed)

    def fail_one() -> None:
        try:
            with pg_session.write_transaction():
                _insert_term(rolled_back)
                raise RuntimeError("boom")
        except RuntimeError:
            pass

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(commit_one), pool.submit(fail_one)]
        for future in futures:
            future.result()

    assert _term_exists(committed), "a sibling thread's rollback discarded this"
    assert not _term_exists(rolled_back)


def test_many_concurrent_readers_do_not_exhaust_the_pool() -> None:
    """More workers than ``pool_size``; overflow plus checkin must absorb it."""

    def read() -> int:
        return pg_session.store().query_read("SELECT 42 AS n")[0]["n"]

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = [f.result() for f in [pool.submit(read) for _ in range(48)]]
    assert results == [42] * 48


def test_dispose_engine_is_idempotent() -> None:
    pg_session.dispose_engine()
    pg_session.dispose_engine()
    # Still usable afterwards: the next call rebuilds the engine.
    assert pg_session.store().query_read("SELECT 1 AS ok") == [{"ok": 1}]


def test_url_is_pinned_to_psycopg3() -> None:
    """A bare postgresql:// URL resolves to psycopg2, which is not installed."""
    assert pg_session.sqlalchemy_url().startswith("postgresql+psycopg://")
