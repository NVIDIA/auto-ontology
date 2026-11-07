# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``POST /semantic-compilation/reset`` honours ``database_name``.

The DAL, the ingestion service and the proxy all scope a semantic reset to one
database when asked. This route is the only way through the public API, so if
it drops the parameter a request to reset one database silently wipes every
database's semantic layer and custom analyses.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.server.semantic_compilation import router as module


@pytest.fixture
def resets(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    calls: list[str | None] = []
    monkeypatch.setattr(
        module,
        "trigger_semantic_reset",
        lambda database_name=None: calls.append(database_name),
    )
    return calls


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(module.router, prefix="/api")
    return TestClient(app)


def test_scoped_reset_forwards_the_database(
    client: TestClient, resets: list[str | None]
) -> None:
    resp = client.post(
        "/api/semantic-compilation/reset", params={"database_name": "sales"}
    )
    assert resp.status_code == 202
    assert resp.json() == {"status": "accepted"}
    assert resets == ["sales"]


def test_unscoped_reset_still_resets_everything(
    client: TestClient, resets: list[str | None]
) -> None:
    resp = client.post("/api/semantic-compilation/reset")
    assert resp.status_code == 202
    assert resets == [None]
