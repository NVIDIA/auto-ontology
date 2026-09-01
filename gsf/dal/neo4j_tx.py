# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in single-transaction scope for multi-statement Neo4j writes.

:func:`get_neo4j_conn` runs every statement in its own auto-commit
transaction, so a routine that issues many writes leaves a partial graph
behind when one of them fails. Call sites that must be all-or-nothing route
their queries through :func:`graph` and wrap the work in
:func:`write_transaction`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from neo4j import GraphDatabase, Transaction

_DATABASE = "neo4j"
_active_tx: ContextVar[TransactionConn | None] = ContextVar("_active_tx", default=None)
_driver = None


class TransactionConn:
    """``Neo4jConnection``-shaped adapter bound to one explicit transaction."""

    def __init__(self, tx: Transaction) -> None:
        self._tx = tx

    def query(self, query: str, parameters: dict[str, Any] | None = None, **_: Any):
        return [dict(record) for record in self._tx.run(query, parameters or {})]

    def query_write(self, query: str, parameters: dict[str, Any] | None = None):
        return self.query(query, parameters)

    def query_read(self, query: str, parameters: dict[str, Any] | None = None):
        return self.query(query, parameters)


def graph():
    """Return the active transaction, or the shared auto-commit connection."""
    active = _active_tx.get()
    return active if active is not None else get_neo4j_conn()


def _get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
            notifications_min_severity="OFF",
        )
    return _driver


@contextmanager
def write_transaction() -> Iterator[None]:
    """Run every :func:`graph` query in the block in one transaction.

    Commits on clean exit and rolls back on any exception. Nesting reuses the
    outermost transaction so the whole block stays a single unit of work.
    """
    if _active_tx.get() is not None:
        yield
        return

    with _get_driver().session(database=_DATABASE) as session:
        tx = session.begin_transaction()
        token = _active_tx.set(TransactionConn(tx))
        try:
            yield
        except BaseException:
            tx.rollback()
            raise
        else:
            tx.commit()
        finally:
            _active_tx.reset(token)
