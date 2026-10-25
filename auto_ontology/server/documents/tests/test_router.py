# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The documents routes, against a fake engine and DAL.

What is under test is the route's own work: which uploads it refuses before
touching the disk, that an accepted one is answered before it is ingested,
what an edit may change, and which stored files a delete may remove.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.server.documents import router as documents

NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


class FakeEngine:
    @contextmanager
    def connect(self) -> Iterator[object]:
        yield object()

    @contextmanager
    def begin(self) -> Iterator[object]:
        yield object()


def _row(document_id: str = "doc-1", **overrides: Any) -> dict[str, Any]:
    row = {
        "id": document_id,
        "name": "report.docx",
        "description": None,
        "file_type": "docx",
        "status": "ready",
        "error": None,
        "ingested_at": NOW,
        "path": "/elsewhere/report.docx",
    }
    row.update(overrides)
    return row


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setattr(documents, "get_engine", lambda: FakeEngine())
    monkeypatch.setattr(documents.config, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(documents.ingest, "missing_dependency", lambda _path: None)
    app = FastAPI()
    app.include_router(documents.router, prefix="/api")
    with TestClient(app) as test_client:
        yield test_client


def test_list_passes_the_search_and_reports_the_total(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def list_documents(_conn: object, **kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        return [_row(summary="internal")]

    monkeypatch.setattr(documents.dal, "list_documents", list_documents)
    monkeypatch.setattr(documents.dal, "count_documents", lambda _c, **_k: 7)

    response = client.get("/api/documents", params={"query": "rep", "limit": 20})

    assert response.status_code == 200
    body = response.json()
    assert seen == {"search": "rep", "newest_first": True, "skip": 0, "limit": 20}
    assert body["total"] == 7 and body["count"] == 1
    assert "path" not in body["data"][0] and "summary" not in body["data"][0]


def test_an_unsupported_extension_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/documents", files={"file": ("notes.exe", b"x", "application/x")}
    )
    assert response.status_code == 400
    assert ".exe" in response.json()["detail"]


def test_a_missing_extractor_is_a_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(documents.ingest, "missing_dependency", lambda _p: "docling")
    response = client.post(
        "/api/documents", files={"file": ("report.docx", b"x", "application/x")}
    )
    assert response.status_code == 503
    assert "docling" in response.json()["detail"]


def test_an_oversized_file_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(documents, "MAX_UPLOAD_BYTES", 4)
    response = client.post(
        "/api/documents", files={"file": ("notes.txt", b"12345", "text/plain")}
    )
    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_an_upload_is_stored_answered_and_then_ingested(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inserted: dict[str, Any] = {}
    ingested: list[tuple[str, str]] = []

    def insert_document(_conn: object, **kwargs: Any) -> str:
        inserted.update(kwargs)
        return "doc-9"

    monkeypatch.setattr(documents.dal, "insert_document", insert_document)
    monkeypatch.setattr(
        documents.dal, "unique_document_name", lambda _c, name: f"{name} (renamed)"
    )
    monkeypatch.setattr(
        documents.dal,
        "get_document",
        lambda _c, _id: _row("doc-9", status="processing", path=inserted["path"]),
    )
    monkeypatch.setattr(
        documents.ingest,
        "run_document_ingest",
        lambda document_id, path: ingested.append((document_id, path)),
    )

    response = client.post(
        "/api/documents", files={"file": ("../notes.txt", b"hello", "text/plain")}
    )

    assert response.status_code == 202
    assert response.json()["data"]["status"] == "processing"
    stored = Path(inserted["path"])
    assert stored.name == "notes.txt"
    assert stored.parent.parent == tmp_path.resolve()
    assert stored.read_bytes() == b"hello"
    assert inserted["name"] == "notes.txt (renamed)"
    assert inserted["status"] == "processing"
    assert ingested == [("doc-9", str(stored))]


def test_an_edit_trims_and_clears(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def update_document(_c: object, _id: str, values: dict[str, Any]) -> dict:
        seen.update(values)
        return _row(name=values["name"])

    monkeypatch.setattr(documents.dal, "update_document", update_document)

    response = client.patch(
        "/api/documents/doc-1", json={"name": "  Q3 report ", "description": "  "}
    )

    assert response.status_code == 200
    assert seen == {"name": "Q3 report", "description": None}
    assert "path" not in response.json()["data"]


def test_a_blank_name_is_refused(client: TestClient) -> None:
    response = client.patch("/api/documents/doc-1", json={"name": "   "})
    assert response.status_code == 400


def test_an_edit_of_a_missing_document_is_a_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(documents.dal, "update_document", lambda *_a: None)
    response = client.patch("/api/documents/gone", json={"description": "x"})
    assert response.status_code == 404


def test_delete_removes_an_uploaded_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stored = tmp_path.resolve() / "abc" / "notes.txt"
    stored.parent.mkdir()
    stored.write_text("hello")
    deleted: list[str] = []
    monkeypatch.setattr(
        documents.dal, "get_document", lambda _c, _id: _row(path=str(stored))
    )
    monkeypatch.setattr(
        documents.dal,
        "delete_document",
        lambda _c, document_id: deleted.append(document_id),
    )

    response = client.delete("/api/documents/doc-1")

    assert response.status_code == 200
    assert deleted == ["doc-1"]
    assert not stored.parent.exists()


def test_delete_leaves_a_file_ingested_from_elsewhere(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "corpus" / "report.docx"
    outside.parent.mkdir(exist_ok=True)
    outside.write_text("keep me")
    monkeypatch.setattr(
        documents.dal, "get_document", lambda _c, _id: _row(path=str(outside))
    )
    monkeypatch.setattr(documents.dal, "delete_document", lambda _c, _id: None)

    assert client.delete("/api/documents/doc-1").status_code == 200
    assert outside.exists()


def test_delete_of_a_missing_document_is_a_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(documents.dal, "get_document", lambda _c, _id: None)
    assert client.delete("/api/documents/gone").status_code == 404
