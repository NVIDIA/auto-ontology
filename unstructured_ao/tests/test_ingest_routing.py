# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

import unstructured_ao.config as config
import unstructured_ao.ingest as ingest
from unstructured_ao.ingest import _extractor_for_file
from unstructured_ao.text_extract import extract_chunks


@pytest.mark.parametrize(
    ("path", "extractor"),
    [
        ("document.pdf", "hosted"),
        ("document.docx", "docling"),
        ("slides.pptx", "docling"),
        ("page.html", "docling"),
        ("notes.txt", "text"),
    ],
)
def test_auto_routes_by_extension(
    monkeypatch: pytest.MonkeyPatch, path: str, extractor: str
) -> None:
    monkeypatch.setattr(config, "EXTRACTOR", "auto")
    assert _extractor_for_file(path) == extractor


def test_hosted_rejects_non_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EXTRACTOR", "hosted")
    with pytest.raises(ValueError, match="does not support"):
        _extractor_for_file("document.docx")


def test_unknown_extractor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EXTRACTOR", "unknown")
    with pytest.raises(ValueError, match="Unknown EXTRACTOR"):
        _extractor_for_file("document.pdf")


def test_plain_text_extraction(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("  first line\nsecond line  ", encoding="utf-8")

    assert extract_chunks(str(path)) == [
        {
            "text": "first line\nsecond line",
            "source_id": str(path),
            "element_type": "text",
            "page_number": None,
        }
    ]


def test_missing_dependency_names_the_extractor_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    monkeypatch.setattr(config, "EXTRACTOR", "auto")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert ingest.missing_dependency("report.docx") == "docling"
    assert ingest.missing_dependency("notes.txt") is None


class _FakeConnection:
    @contextmanager
    def begin(self) -> Iterator[object]:
        yield self

    def commit(self) -> None:
        pass


class _FakeEngine:
    def __init__(self) -> None:
        self.connection = _FakeConnection()

    @contextmanager
    def connect(self) -> Iterator[_FakeConnection]:
        yield self.connection


def _record_statuses(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    import auto_ontology.dal.session as session

    statuses: list[tuple] = []
    monkeypatch.setattr(session, "get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(
        ingest,
        "get_document",
        lambda _c, document_id: {"id": document_id, "status": "processing"},
    )
    monkeypatch.setattr(ingest, "_acquire_document_lock", lambda *_a: True)
    monkeypatch.setattr(ingest, "_release_document_lock", lambda *_a: None)
    monkeypatch.setattr(
        ingest,
        "set_document_status",
        lambda _c, document_id, status, error=None: statuses.append(
            (document_id, status, error)
        ),
    )
    return statuses


def test_background_ingest_marks_the_document_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = _record_statuses(monkeypatch)
    monkeypatch.setattr(config, "EXTRACT_ON_UPLOAD", False)
    stored: list[int] = []
    monkeypatch.setattr(ingest, "_extract_chunks_from_file", lambda _p: [{"text": "a"}])
    monkeypatch.setattr(
        ingest, "_store_chunks", lambda _c, _id, _n, chunks: stored.append(len(chunks))
    )

    ingest.run_document_ingest("doc-1", "notes.txt")

    assert stored == [1]
    assert statuses == [("doc-1", "ready", None)]


def _ingest_one_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EXTRACT_ON_UPLOAD", True)
    monkeypatch.setattr(ingest, "_extract_chunks_from_file", lambda _p: [{"text": "a"}])
    monkeypatch.setattr(ingest, "_store_chunks", lambda *_a: None)


def test_background_ingest_goes_on_to_extract_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = _record_statuses(monkeypatch)
    _ingest_one_chunk(monkeypatch)
    extracted: list[str] = []

    def extract(document_id: str, conn: object) -> None:
        extracted.append(document_id)
        ingest.set_document_status(conn, document_id, "ready")

    monkeypatch.setattr(ingest, "extract_document_semantics", extract)

    ingest.run_document_ingest("doc-1", "notes.txt")

    assert extracted == ["doc-1"]
    assert statuses == [("doc-1", "extracting", None), ("doc-1", "ready", None)]


def test_recovery_resumes_an_extracting_document_without_reingesting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = _record_statuses(monkeypatch)
    monkeypatch.setattr(
        ingest,
        "get_document",
        lambda _c, document_id: {"id": document_id, "status": "extracting"},
    )
    monkeypatch.setattr(
        ingest,
        "ingest_document",
        lambda *_a, **_k: pytest.fail("an extracting document was re-ingested"),
    )

    def extract(document_id: str, conn: object) -> None:
        ingest.set_document_status(conn, document_id, "ready")

    monkeypatch.setattr(ingest, "extract_document_semantics", extract)

    ingest.run_document_ingest("doc-1", "notes.txt")

    assert statuses == [("doc-1", "ready", None)]


def test_startup_recovery_resumes_every_pending_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auto_ontology.dal.session as session

    monkeypatch.setattr(session, "get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(
        ingest,
        "list_pending_documents",
        lambda _c: [
            {"id": "doc-1", "path": "/uploads/one.txt", "status": "processing"},
            {"id": "doc-2", "path": "/uploads/two.pdf", "status": "extracting"},
        ],
    )
    resumed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ingest,
        "run_document_ingest",
        lambda document_id, path: resumed.append((document_id, path)),
    )

    ingest.recover_pending_document_ingests()

    assert resumed == [
        ("doc-1", "/uploads/one.txt"),
        ("doc-2", "/uploads/two.pdf"),
    ]


def test_startup_recovery_stops_between_documents_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import auto_ontology.dal.session as session

    monkeypatch.setattr(session, "get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(
        ingest,
        "list_pending_documents",
        lambda _c: [
            {"id": "doc-1", "path": "/uploads/one.txt", "status": "processing"},
            {"id": "doc-2", "path": "/uploads/two.pdf", "status": "processing"},
        ],
    )
    stop = threading.Event()
    resumed: list[str] = []

    def resume(document_id: str, _path: str) -> None:
        resumed.append(document_id)
        stop.set()

    monkeypatch.setattr(ingest, "run_document_ingest", resume)

    ingest.recover_pending_document_ingests(stop)

    assert resumed == ["doc-1"]


def test_a_failed_semantic_pass_marks_the_document_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = _record_statuses(monkeypatch)
    _ingest_one_chunk(monkeypatch)

    def extract(_document_id: str, _conn: object) -> None:
        raise RuntimeError("Term extraction failed")

    monkeypatch.setattr(ingest, "extract_document_semantics", extract)

    ingest.run_document_ingest("doc-1", "notes.txt")

    assert statuses[0] == ("doc-1", "extracting", None)
    _id, status, error = statuses[1]
    assert status == "failed"
    assert error.startswith("Term and attribute extraction failed")


def test_background_ingest_refuses_a_document_over_the_word_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = _record_statuses(monkeypatch)
    monkeypatch.setattr(config, "MAX_DOCUMENT_WORDS", 3)
    monkeypatch.setattr(
        ingest, "_extract_chunks_from_file", lambda _p: [{"text": "one two"}] * 2
    )
    monkeypatch.setattr(ingest, "_store_chunks", lambda *_a: pytest.fail("stored"))

    ingest.run_document_ingest("doc-1", "notes.txt")

    ((document_id, status, error),) = statuses
    assert (document_id, status) == ("doc-1", "failed")
    assert "4 words" in error


@pytest.mark.parametrize(
    ("name", "taken", "expected"),
    [
        ("report.pdf", set(), "report.pdf"),
        ("report.pdf", {"report.pdf"}, "report (1).pdf"),
        ("Report.PDF", {"report.pdf", "report (1).pdf"}, "Report (2).PDF"),
        ("report.pdf", {"report.pdf", "report (2).pdf"}, "report (1).pdf"),
        ("archive.tar.gz", {"archive.tar.gz"}, "archive.tar (1).gz"),
        ("README", {"readme"}, "README (1)"),
        (".env", {".env"}, ".env (1)"),
    ],
)
def test_next_free_name(name: str, taken: set[str], expected: str) -> None:
    from unstructured_ao.dal import next_free_name

    assert next_free_name(name, taken) == expected


def test_background_ingest_records_an_empty_extraction_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = _record_statuses(monkeypatch)
    monkeypatch.setattr(ingest, "_extract_chunks_from_file", lambda _p: [])

    ingest.run_document_ingest("doc-1", "notes.txt")

    assert len(statuses) == 1
    document_id, status, error = statuses[0]
    assert (document_id, status) == ("doc-1", "failed")
    assert "No content" in error
