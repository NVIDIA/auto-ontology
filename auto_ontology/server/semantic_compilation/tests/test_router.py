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
    monkeypatch.setattr(
        registry,
        "_connectors",
        [SimpleNamespace(database_name=name) for name in ("sales", "stock")],
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


def test_scoped_status_survives_an_unreachable_source_on_a_cold_cache(
    client: TestClient,
    scheduler: dict[str, bool | str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "_connectors", None)
    monkeypatch.setattr(connections, "list_connections", lambda: [])
    monkeypatch.setenv(
        "CONNECTION_STRINGS", "sqlite:unreachable,sqlite:healthy,sqlite:unvisited"
    )
    monkeypatch.setattr(module, "database_catalog_status", lambda _: (True, True))
    monkeypatch.setattr(
        module,
        "shared_vdb",
        lambda _: SimpleNamespace(has_embeddings_for_database=lambda _: True),
    )
    attempted: list[str] = []
    closed: list[str] = []

    class HealthProbe:
        def __init__(self, name: str) -> None:
            self.database_name = name

        def __enter__(self) -> HealthProbe:
            return self

        def __exit__(self, *exc: object) -> None:
            closed.append(self.database_name)

    def create_connector(connection_string: str, **kwargs: object) -> HealthProbe:
        name = connection_string.removeprefix("sqlite:")
        attempted.append(name)
        if name == "unreachable":
            raise ConnectionError("source unavailable")
        return HealthProbe(name)

    monkeypatch.setattr(registry, "create_connector", create_connector)
    response = client.get(
        "/api/semantic-compilation/status", params={"database_name": "healthy"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "calculated": True,
        **scheduler,
        "database": {
            "name": "healthy",
            "ingested": True,
            "connected": True,
            "compiled": True,
            "searchable": True,
        },
    }
    assert attempted == ["unreachable", "healthy"]
    assert closed == ["healthy"]
    assert registry._connectors is None

    attempted.clear()
    closed.clear()
    response = client.get(
        "/api/semantic-compilation/status", params={"database_name": "unreachable"}
    )
    assert response.status_code == 200
    assert response.json()["database"]["connected"] is False
    assert attempted == ["unreachable", "healthy", "unvisited"]
    assert closed == ["healthy", "unvisited"]
