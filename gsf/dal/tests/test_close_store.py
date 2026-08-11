# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`gsf.dal.close_store`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from gsf.catalog.store.neo4j import connection

from gsf.dal import close_store, neo4j_tx


@pytest.fixture(autouse=True)
def _restore_globals():
    """Leave both module globals as they were found."""
    shared, driver = connection._conn, neo4j_tx._driver
    connection._conn = None
    neo4j_tx._driver = None
    yield
    connection._conn, neo4j_tx._driver = shared, driver


def test_closes_both_connections() -> None:
    """The shared auto-commit connection *and* the transaction driver."""
    shared, driver = MagicMock(), MagicMock()
    connection._conn, neo4j_tx._driver = shared, driver

    close_store()

    shared.close.assert_called_once_with()
    driver.close.assert_called_once_with()
    assert connection._conn is None
    assert neo4j_tx._driver is None


def test_is_idempotent() -> None:
    """Calling it twice must not raise — lifespan teardown can run on a
    process that never opened a connection."""
    close_store()
    close_store()


def test_closes_driver_even_if_shared_connection_raises() -> None:
    """A failing close must not strand the other connection.

    Shutdown is best-effort: the process is going away either way, and leaking
    the driver because the singleton misbehaved is strictly worse than logging.
    """
    shared, driver = MagicMock(), MagicMock()
    shared.close.side_effect = RuntimeError("bolt already gone")
    connection._conn, neo4j_tx._driver = shared, driver

    close_store()

    driver.close.assert_called_once_with()
    assert connection._conn is None
    assert neo4j_tx._driver is None


def test_only_closes_what_is_open() -> None:
    """A driver opened without the shared singleton still gets closed."""
    driver = MagicMock()
    neo4j_tx._driver = driver

    close_store()

    driver.close.assert_called_once_with()
