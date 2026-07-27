# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the opt-in Neo4j write-transaction scope."""

from __future__ import annotations

from typing import Self
from unittest.mock import MagicMock, patch

import pytest

from gsf.dal import neo4j_tx


class _FakeSession:
    def __init__(self, tx: MagicMock) -> None:
        self._tx = tx

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def begin_transaction(self) -> MagicMock:
        return self._tx


def _driver_returning(tx: MagicMock) -> MagicMock:
    driver = MagicMock()
    driver.session.return_value = _FakeSession(tx)
    return driver


def test_commits_and_routes_queries_through_the_transaction() -> None:
    tx = MagicMock()
    tx.run.return_value = [{"id": "live-1"}]

    with (
        patch.object(neo4j_tx, "_get_driver", return_value=_driver_returning(tx)),
        neo4j_tx.write_transaction(),
    ):
        assert neo4j_tx.graph().query_write("CREATE (n)") == [{"id": "live-1"}]

    tx.run.assert_called_once_with("CREATE (n)", {})
    tx.commit.assert_called_once()
    tx.rollback.assert_not_called()


def test_rolls_back_when_the_block_raises() -> None:
    tx = MagicMock()

    with (
        patch.object(neo4j_tx, "_get_driver", return_value=_driver_returning(tx)),
        pytest.raises(RuntimeError, match="boom"),
        neo4j_tx.write_transaction(),
    ):
        neo4j_tx.graph().query_write("CREATE (n)")
        raise RuntimeError("boom")

    tx.rollback.assert_called_once()
    tx.commit.assert_not_called()


def test_nesting_reuses_the_outer_transaction() -> None:
    tx = MagicMock()

    with (
        patch.object(neo4j_tx, "_get_driver", return_value=_driver_returning(tx)),
        neo4j_tx.write_transaction(),
    ):
        outer = neo4j_tx.graph()
        with neo4j_tx.write_transaction():
            assert neo4j_tx.graph() is outer

    tx.commit.assert_called_once()


def test_falls_back_to_the_shared_connection_outside_a_transaction() -> None:
    shared = MagicMock()

    with patch.object(neo4j_tx, "get_neo4j_conn", return_value=shared):
        assert neo4j_tx.graph() is shared


def test_scope_is_cleared_after_a_failed_transaction() -> None:
    tx = MagicMock()
    shared = MagicMock()

    with (
        patch.object(neo4j_tx, "_get_driver", return_value=_driver_returning(tx)),
        pytest.raises(RuntimeError),
        neo4j_tx.write_transaction(),
    ):
        raise RuntimeError("boom")

    with patch.object(neo4j_tx, "get_neo4j_conn", return_value=shared):
        assert neo4j_tx.graph() is shared
