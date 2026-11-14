# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reset route must forward ``database_name`` to the ingestion proxy.

Every layer below the API server scopes semantic reset by an optional
``database_name`` — the DAL (``delete_semantic_layer``), the ingestion service
(``POST /semantic/reset``), and the backend proxy
(``trigger_semantic_reset``) all accept it. The route handler is the last hop,
and a handler that declares no parameter makes FastAPI silently *drop* the
query string: ``POST /api/semantic-compilation/reset?database_name=mydb``
would answer 202 and wipe every database's semantic layer instead of one.

These tests pin the forwarding at the HTTP boundary, with the proxy patched
out, so they need neither a database nor a running ingestion service.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.server.semantic_compilation.router import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_reset_forwards_database_name_to_the_proxy() -> None:
    client = _client()
    with patch(
        "auto_ontology.server.semantic_compilation.router.trigger_semantic_reset"
    ) as mock_reset:
        response = client.post("/api/semantic-compilation/reset?database_name=sales")

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    mock_reset.assert_called_once_with("sales")


def test_reset_without_database_name_resets_every_database() -> None:
    client = _client()
    with patch(
        "auto_ontology.server.semantic_compilation.router.trigger_semantic_reset"
    ) as mock_reset:
        response = client.post("/api/semantic-compilation/reset")

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    mock_reset.assert_called_once_with(None)
