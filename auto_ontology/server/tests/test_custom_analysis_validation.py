# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CustomAnalysis SQL validation responses."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from auto_ontology.server.custom_analyses import service
from auto_ontology.server.datasources.router import router
from auto_ontology.server.sql_utils import SqlParseError


def test_sql_parse_error_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_sql(_sql: str) -> dict:
        raise SqlParseError("bad SQL")

    monkeypatch.setattr(service, "validate_custom_analysis_sql", reject_sql)
    app = FastAPI()
    app.include_router(router, prefix="/api")

    with TestClient(app) as client:
        response = client.post(
            "/api/custom-analyses/validate",
            json={"sql": "SELECT missing FROM orders"},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "bad SQL"}
