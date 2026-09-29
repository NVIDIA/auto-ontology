# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`auto_ontology.dal.close_store`.

Small, and worth keeping: it is the shutdown hook the app's lifespan calls, and
the failure it guards against is one nothing else would notice — a process that
exits holding pooled sockets open.

It closes one thing, so the suite is correspondingly small.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from auto_ontology.dal import close_store
from auto_ontology.dal import session as pg_session


def test_disposes_the_engine(monkeypatch) -> None:
    engine = MagicMock()
    monkeypatch.setattr(pg_session, "_engine", engine)

    close_store()

    engine.dispose.assert_called_once_with()
    assert pg_session._engine is None


def test_is_idempotent(monkeypatch) -> None:
    """Lifespan teardown can run on a process that never opened a connection."""
    monkeypatch.setattr(pg_session, "_engine", None)
    close_store()
    close_store()
