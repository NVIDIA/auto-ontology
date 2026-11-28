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

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.connectors import registry
from auto_ontology.dal import connections
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


def test_reset_proxy_runs_outside_the_event_loop(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_calls: list[str | None] = []

    def blocking_reset(database_name: str | None = None) -> None:
        # Synchronous HTTP must run in FastAPI's worker, not its event loop.
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        worker_calls.append(database_name)

    monkeypatch.setattr(module, "trigger_semantic_reset", blocking_reset)
    resp = client.post(
        "/api/semantic-compilation/reset", params={"database_name": "sales"}
    )
    assert resp.status_code == 202
    assert resp.json() == {"status": "accepted"}
    assert worker_calls == ["sales"]


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool | str | None]:
    monkeypatch.setattr(module, "is_semantic_compilation_running", lambda: False)
    monkeypatch.setattr(
        module, "get_last_successful_run", lambda: "2026-11-01T12:00:00Z"
    )
    monkeypatch.setattr(module, "get_last_failure_if_most_recent", lambda: None)
    return {
        "running": False,
        "last_success_at": "2026-11-01T12:00:00Z",
        "last_failure_at": None,
    }


def test_unscoped_status_keeps_the_deployment_response(
    client: TestClient,
    scheduler: dict[str, bool | str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "semantic_layer_calculated", lambda: True)
    response = client.get("/api/semantic-compilation/status")
    assert response.status_code == 200
    assert response.json() == {"calculated": True, **scheduler}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sales", (True, True, True, True)),
        ("stock", (True, True, False, False)),
        ("archived", (True, False, True, False)),
        ("Sales", (False, False, False, False)),
    ],
)
def test_scoped_status_uses_only_the_requested_database(
    client: TestClient,
    scheduler: dict[str, bool | str | None],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    expected: tuple[bool, bool, bool, bool],
) -> None:
    monkeypatch.setattr(
        module,
        "database_catalog_status",
        lambda name: {
            "sales": (True, True),
            "stock": (True, False),
            "archived": (True, True),
        }.get(name, (False, False)),
    )
    monkeypatch.setattr(connections, "list_connections", lambda: [])
    monkeypatch.setenv(
        "CONNECTION_STRINGS", "postgresql://host/sales,postgresql://host/stock"
    )
    monkeypatch.setattr(
        module,
        "shared_vdb",
        lambda _: SimpleNamespace(
            has_embeddings_for_database=lambda name: name == "sales"
        ),
    )
    response = client.get(
        "/api/semantic-compilation/status", params={"database_name": name}
    )
    ingested, connected, compiled, searchable = expected
    assert response.status_code == 200
    assert response.json() == {
        "calculated": compiled,
        **scheduler,
        "database": {
            "name": name,
            "ingested": ingested,
            "connected": connected,
            "compiled": compiled,
            "searchable": searchable,
        },
    }


@pytest.mark.parametrize("cache_loaded", [False, True])
def test_scoped_status_reports_configured_sources_without_connecting(
    client: TestClient,
    scheduler: dict[str, bool | str | None],
    monkeypatch: pytest.MonkeyPatch,
    cache_loaded: bool,
) -> None:
    cached = [SimpleNamespace(database_name="previous")] if cache_loaded else None
    monkeypatch.setattr(registry, "_connectors", cached)
    monkeypatch.setattr(connections, "list_connections", lambda: [])
    monkeypatch.setenv(
        "CONNECTION_STRINGS",
        "postgresql://unreachable/analytics,snowflake://user:pass@account"
        "?warehouse=wh&database=Sales%20Data",
    )
    monkeypatch.setattr(module, "database_catalog_status", lambda _: (True, True))
    monkeypatch.setattr(
        module,
        "shared_vdb",
        lambda _: SimpleNamespace(has_embeddings_for_database=lambda _: True),
    )
    attempted: list[str] = []

    def create_connector(connection_string: str, **kwargs: object) -> None:
        attempted.append(connection_string)
        raise ConnectionError("source unavailable")

    monkeypatch.setattr(registry, "create_connector", create_connector)
    for name, configured in [
        ("analytics", True),
        ("Sales Data", True),
        ("previous", False),
    ]:
        response = client.get(
            "/api/semantic-compilation/status", params={"database_name": name}
        )
        assert response.status_code == 200
        assert response.json() == {
            "calculated": True,
            **scheduler,
            "database": {
                "name": name,
                "ingested": True,
                "connected": configured,
                "compiled": True,
                "searchable": True,
            },
        }
    assert attempted == []
    assert registry._connectors is cached
