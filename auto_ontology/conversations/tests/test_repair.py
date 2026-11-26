# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repairing a stored graph narrows itself where a mistake would stick."""

from __future__ import annotations

import types
from typing import Any

import pytest
import sqlalchemy as sa

from auto_ontology.conversations.semantic import repair


def test_deleting_an_edge_matches_its_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """A semantic edge with the same endpoints and role is a different row."""
    edge = sa.Table(
        "conversation_edge",
        sa.MetaData(),
        sa.Column("source_id", sa.Text, primary_key=True),
        sa.Column("target_id", sa.Text, primary_key=True),
        sa.Column("type", sa.Text, primary_key=True),
        sa.Column("role", sa.Text, primary_key=True),
    )
    statements: list[Any] = []

    class Store:
        def query_write(self, statement: Any) -> list[Any]:
            statements.append(statement)
            return [("supersedes",)]

    sys_modules = __import__("sys").modules
    schema = types.ModuleType("auto_ontology.dal.schema")
    schema.conversation_edge = edge  # type: ignore[attr-defined]
    session = types.ModuleType("auto_ontology.dal.session")
    session.store = Store  # type: ignore[attr-defined]
    dal = types.ModuleType("auto_ontology.dal")
    dal.schema = schema  # type: ignore[attr-defined]
    dal.session = session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys_modules, "auto_ontology.dal", dal)
    monkeypatch.setitem(sys_modules, "auto_ontology.dal.schema", schema)
    monkeypatch.setitem(sys_modules, "auto_ontology.dal.session", session)

    draft = object()
    row = {
        "source_id": "a",
        "target_id": "b",
        "type": "knowledge",
        "role": "supersedes",
    }
    assert repair._delete({id(draft): row}, [draft]) == 1

    [statement] = statements
    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "conversation_edge.type = 'knowledge'" in sql
