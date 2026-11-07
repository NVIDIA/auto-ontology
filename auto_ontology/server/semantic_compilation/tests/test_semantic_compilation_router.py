# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic compilation reset route forwards optional database_name scope."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.server.semantic_compilation import router as sc_router


def test_reset_without_database_name_resets_every_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []
    monkeypatch.setattr(sc_router, "trigger_semantic_reset", seen.append)

    result = asyncio.run(sc_router.reset())

    assert result == {"status": "accepted"}
    assert seen == [None]


def test_reset_forwards_database_name_to_ingestion_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []
    monkeypatch.setattr(sc_router, "trigger_semantic_reset", seen.append)

    result = asyncio.run(sc_router.reset(database_name="pagila"))

    assert result == {"status": "accepted"}
    assert seen == ["pagila"]


def test_reset_route_accepts_database_name_query_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []
    monkeypatch.setattr(sc_router, "trigger_semantic_reset", seen.append)
    app = FastAPI()
    app.include_router(sc_router.router, prefix="/api")
    client = TestClient(app)

    unscoped = client.post("/api/semantic-compilation/reset")
    scoped = client.post(
        "/api/semantic-compilation/reset", params={"database_name": "pagila"}
    )

    assert unscoped.status_code == 202
    assert scoped.status_code == 202
    assert seen == [None, "pagila"]
