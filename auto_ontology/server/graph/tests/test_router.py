# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Status codes of the AGE graph routes, with the graph reads stubbed."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.graph import age
from auto_ontology.server.graph.router import router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(age.ENABLED_ENV, "true")


def test_status_reports_disabled_without_touching_the_database(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(age.ENABLED_ENV, raising=False)
    with patch("auto_ontology.graph.queries.graph_status") as status:
        response = client.get("/api/graph/status")
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    status.assert_not_called()


def test_status_when_enabled(client: TestClient, enabled: None) -> None:
    payload = {
        "graph": "g",
        "built": True,
        "synced_at": "t",
        "vertices": {"Term": 2},
        "edges": {},
    }
    with patch("auto_ontology.graph.queries.graph_status", return_value=payload):
        body = client.get("/api/graph/status").json()
    assert body["enabled"] is True
    assert body["vertices"] == {"Term": 2}


def test_neighborhood_is_503_when_disabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(age.ENABLED_ENV, raising=False)
    assert client.get("/api/graph/terms/t1/neighborhood").status_code == 503


@pytest.mark.parametrize(
    ("raised", "code"),
    [(age.GraphNotBuilt("no graph"), 409), (age.AgeUnavailable("no age"), 503)],
)
def test_neighborhood_errors(
    client: TestClient, enabled: None, raised: Exception, code: int
) -> None:
    with patch("auto_ontology.graph.queries.term_neighborhood", side_effect=raised):
        assert client.get("/api/graph/terms/t1/neighborhood").status_code == code


def test_neighborhood_unknown_term_is_404(client: TestClient, enabled: None) -> None:
    with patch("auto_ontology.graph.queries.term_neighborhood", return_value=None):
        assert client.get("/api/graph/terms/nope/neighborhood").status_code == 404


def test_neighborhood_validates_depth(client: TestClient, enabled: None) -> None:
    assert client.get("/api/graph/terms/t1/neighborhood?depth=4").status_code == 422
    assert client.get("/api/graph/terms/t1/neighborhood?max_nodes=0").status_code == 422


def test_neighborhood_passes_arguments(client: TestClient, enabled: None) -> None:
    result = {
        "term_id": "t1",
        "depth": 2,
        "truncated": False,
        "nodes": [{"id": "t1", "label": "Term", "properties": {"name": "Order"}}],
        "edges": [],
    }
    with patch(
        "auto_ontology.graph.queries.term_neighborhood", return_value=result
    ) as walk:
        body = client.get(
            "/api/graph/terms/t1/neighborhood?depth=2&max_nodes=50"
        ).json()
    walk.assert_called_once_with("t1", depth=2, max_nodes=50)
    assert body["nodes"][0]["label"] == "Term"
