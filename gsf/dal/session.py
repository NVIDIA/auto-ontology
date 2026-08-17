# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection and transaction scope for the Postgres DAL.

Two entry points, used by every read and write in the DAL:

* :func:`store` returns the active transaction if there is one, otherwise a
  handle that runs each statement in its own autocommitted transaction — the
  same split the DAL relies on for single-statement writes.
* :func:`write_transaction` makes every statement in the block one unit of
  work. Nesting reuses the outermost scope, so a routine that calls another
  transactional routine still commits once.

**Pooling.** The DAL is called from three places with different concurrency
shapes: FastAPI request handlers, ingestion ``ThreadPoolExecutor`` threads
(``write_to_graph`` fans out), and a *spawned* chat-worker subprocess
(:mod:`gsf.server.chat.worker`). SQLAlchemy's ``QueuePool`` is thread-safe, and
the ``ContextVar`` holding the active transaction is per-thread and per-task, so
two threads inside ``write_transaction`` get genuinely separate transactions
rather than trampling one connection.

The subprocess is the case worth naming: a pool cannot be inherited across a
fork, and file descriptors carried into a child produce corruption that only
shows under load. :func:`dispose_engine` exists for that reason and is called on
process boundaries; ``spawn`` (which is what the worker uses) starts a fresh
interpreter and so builds its own engine anyway.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import Engine

from gsf.infra.postgres import get_postgres_connection_string

logger = logging.getLogger(__name__)

#: Postgres schema owning every GSF catalog and semantic table.
#:
#: GSF is the application, so it holds the default schema. The other two owners
#: are named explicitly and stay out of it: Prisma has ``frontend``
#: (``schemas = ["frontend"]`` in ``frontend/prisma/schema.prisma``) and
#: langchain_postgres has ``vdb`` (``gsf.vdb.VDB_SCHEMA``). Nothing else writes
#: here, which is what lets Alembic treat "in this schema but not in
#: ``METADATA``" as drift.
#:
#: Still set on the MetaData rather than via ``search_path``: search_path is
#: per-session and would be a silent hazard on a pooled connection, and every
#: statement being schema-qualified is what makes the ownership legible.
SCHEMA = "public"

_engine: Engine | None = None
_active: ContextVar[StoreConn | None] = ContextVar("_active_pg_conn", default=None)


def sqlalchemy_url() -> str:
    """The shared Postgres URL, pinned to the psycopg 3 driver.

    ``get_postgres_connection_string()`` returns a bare ``postgresql://`` URL,
    which SQLAlchemy resolves to **psycopg2** — a package this repo does not
    install. The rewrite mirrors ``gsf/vdb/postgres.py``'s ``_to_async_url``,
    which does the same thing for asyncpg.
    """
    url = get_postgres_connection_string()
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


def get_engine() -> Engine:
    """The process-wide engine, built on first use."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            sqlalchemy_url(),
            pool_size=int(os.environ.get("GSF_PG_POOL_SIZE", "5")),
            max_overflow=int(os.environ.get("GSF_PG_MAX_OVERFLOW", "5")),
            # Recycle below any middlebox idle timeout, and check liveness
            # before handing a connection out. Without pre_ping, a connection
            # killed server-side surfaces as a failed query on an unrelated
            # request rather than as a reconnect.
            pool_recycle=1800,
            pool_pre_ping=True,
            # A TCP connect must not hang forever. Every other Postgres consumer
            # in the repo passes connect_timeout=3; without it here, a
            # blackholed database makes the health endpoint -- which is the
            # *liveness* probe as well as readiness -- hang until the kubelet
            # kills a backend that is otherwise fine.
            # `options` pins the search_path for the life of every connection in
            # the pool, set at connect time rather than by a statement someone
            # has to remember to issue. The MetaData deliberately does not name
            # the schema (see `gsf/dal/schema.py`), so this is what decides where
            # unqualified SQL resolves -- and pinning it here means it cannot
            # drift per session, which was the original objection to relying on
            # search_path at all.
            connect_args={
                "connect_timeout": 3,
                "options": f"-csearch_path={SCHEMA}",
            },
            future=True,
        )
    return _engine


def dispose_engine() -> None:
    """Close every pooled connection. Idempotent.

    Called from ``gsf.dal.close_store()`` on shutdown, and before handing a
    process to a child, since pooled sockets must not be shared across a fork.
    """
    global _engine
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            logger.warning("dispose_engine: engine failed to dispose", exc_info=True)
        finally:
            _engine = None


def _rows(result: Any) -> list[dict[str, Any]]:
    """Normalise a result to ``list[dict]``."""
    if result.returns_rows:
        return [dict(row) for row in result.mappings()]
    return []


class StoreConn:
    """A thin handle over a SQLAlchemy connection.

    The ``query_read`` / ``query_write`` split is naming only: it documents
    intent at the call site, and Postgres needs no routing hint.
    """

    def __init__(self, connection: Connection | None = None) -> None:
        self._connection = connection

    def query(
        self, statement: Any, parameters: dict[str, Any] | None = None, **_: Any
    ) -> list[dict[str, Any]]:
        compiled = text(statement) if isinstance(statement, str) else statement
        if self._connection is not None:
            return _rows(self._connection.execute(compiled, parameters or {}))
        # No open transaction: one autocommitted statement, connection returned
        # to the pool immediately. This is why a multi-statement write that
        # must be all-or-nothing has to go through write_transaction().
        with (
            get_engine()
            .connect()
            .execution_options(isolation_level="AUTOCOMMIT") as connection
        ):
            return _rows(connection.execute(compiled, parameters or {}))

    def query_read(
        self, statement: Any, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.query(statement, parameters)

    def query_write(
        self, statement: Any, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.query(statement, parameters)


def store() -> StoreConn:
    """Return the active transaction, or an autocommitting handle."""
    active = _active.get()
    return active if active is not None else StoreConn()


@contextmanager
def write_transaction() -> Iterator[None]:
    """Run every :func:`store` query in the block in one transaction.

    Commits on clean exit, rolls back on any exception. Nesting reuses the
    outermost transaction so the whole block stays a single unit of work —
    ``model_interchange``'s import depends on this.
    """
    if _active.get() is not None:
        yield
        return

    with get_engine().connect() as connection:
        transaction = connection.begin()
        token = _active.set(StoreConn(connection))
        try:
            yield
        except BaseException:
            transaction.rollback()
            raise
        else:
            transaction.commit()
        finally:
            _active.reset(token)
