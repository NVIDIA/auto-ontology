# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for unstructured documents."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel

from auto_ontology.dal.session import get_engine
from auto_ontology.server.models import Document
from auto_ontology.server.pagination import LIMIT_QUERY, SKIP_QUERY
from auto_ontology.server.responses import (
    DocumentPageResponse,
    DocumentResponse,
    IdResponse,
)
from unstructured_ao import config
from unstructured_ao import dal
from unstructured_ao import ingest

router = APIRouter()

#: Largest upload accepted. Mirrored by the upload dialog, which shows it; this
#: is the half a caller bypassing the UI still meets.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

_COPY_BLOCK_BYTES = 1024 * 1024


class DocumentUpdate(BaseModel):
    """Fields left out are left as they are; an empty description clears it."""

    name: str | None = None
    description: str | None = None


def _public(row: dict) -> dict:
    """*row* cut to the ``Document`` fields. Response models allow extra keys,
    so anything else — the server-side ``path`` above all — would go out."""
    return {k: v for k, v in row.items() if k in Document.model_fields}


def _not_found(document_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Document {document_id!r} not found")


@router.get("/documents", response_model=DocumentPageResponse)
def list_documents(
    query: str | None = Query(
        default=None,
        description="Case-insensitive substring filter on name and description.",
    ),
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """Return documents, newest first.

    ``status`` tells a caller which ones are still being ingested, so the
    settings page knows to read the list again until none are.
    """
    with get_engine().connect() as conn:
        rows = dal.list_documents(
            conn, search=query, newest_first=True, skip=skip, limit=limit
        )
        total = dal.count_documents(conn, search=query)
    return {"data": [_public(r) for r in rows], "count": len(rows), "total": total}


@router.post("/documents", status_code=202, response_model=DocumentResponse)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    """Store an uploaded file and ingest it in the background.

    Answers 202 with the row as ``processing`` as soon as the file is stored:
    extraction and embedding take seconds to minutes, longer than a request
    should stay open. The row turns ``ready`` or ``failed`` when they finish;
    a document whose extracted text runs past the word limit is ``failed``,
    since its words are only known once it has been extracted.

    A name another document already holds gets the lowest free ``" (n)"``
    suffix before its extension, so two uploads of ``report.pdf`` list as
    ``report.pdf`` and ``report (1).pdf``. One file per request: the settings
    page sends a batch as a request each, so one bad file fails alone.

    400 for an extension the configured extractor does not read, 413 for a
    file over the size limit, and 503 when the extractor this file needs is not
    installed -- refused here rather than accepted and failed in the
    background.
    """
    name = Path(file.filename or "").name
    extension = Path(name).suffix.lower()
    supported = ingest.supported_extensions()
    if not name or extension not in supported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type {extension or '(none)'!r}; expected one "
                f"of {', '.join(sorted(supported))}"
            ),
        )
    missing = ingest.missing_dependency(name)
    if missing is not None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ingesting {extension} files needs the {missing!r} package; "
                "install the 'unstructured' extra"
            ),
        )

    too_large = HTTPException(
        status_code=413,
        detail=f"{name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit",
    )
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise too_large

    target = Path(config.UPLOAD_DIR).expanduser().resolve() / str(uuid.uuid4()) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    # Copied a block at a time, so a file near the limit is never held whole in
    # memory, and counted as it goes, since ``size`` is the client's claim.
    written = 0
    with target.open("wb") as out:
        while block := file.file.read(_COPY_BLOCK_BYTES):
            written += len(block)
            if written > MAX_UPLOAD_BYTES:
                break
            out.write(block)
    if written > MAX_UPLOAD_BYTES:
        shutil.rmtree(target.parent, ignore_errors=True)
        raise too_large

    try:
        with get_engine().begin() as conn:
            document_id = dal.insert_document(
                conn,
                name=dal.unique_document_name(conn, name),
                path=str(target),
                file_type=extension.lstrip("."),
                status="processing",
            )
            row = dal.get_document(conn, document_id)
    except Exception:
        shutil.rmtree(target.parent, ignore_errors=True)
        raise

    background_tasks.add_task(ingest.run_document_ingest, document_id, str(target))
    return {"data": _public(row)}


@router.post(
    "/documents/{document_id}/retry",
    status_code=202,
    response_model=DocumentResponse,
)
def retry_document(document_id: str, background_tasks: BackgroundTasks) -> dict:
    """Retry a failed document from its stored file.

    A failure during semantic extraction resumes from the committed chunks.
    Earlier failures restart extraction and embedding from the original upload.
    Only failed rows are accepted, preventing a retry from racing a live ingest
    or needlessly rebuilding a ready document.
    """
    with get_engine().begin() as conn:
        row = dal.get_document(conn, document_id)
        if row is None:
            raise _not_found(document_id)
        if row["status"] != "failed":
            raise HTTPException(
                status_code=409,
                detail=f"Document {document_id!r} is not failed and cannot be retried",
            )

        path = Path(row["path"]).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(
                status_code=409,
                detail="The stored document file is unavailable; upload it again.",
            )
        missing = ingest.missing_dependency(path.name)
        if missing is not None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Ingesting {path.suffix.lower()} files needs the {missing!r} "
                    "package; install the 'unstructured' extra"
                ),
            )

        status = (
            "extracting" if dal.has_document_chunks(conn, document_id) else "processing"
        )
        dal.set_document_status(conn, document_id, status)
        row = dal.get_document(conn, document_id)

    background_tasks.add_task(ingest.run_document_ingest, document_id, str(path))
    return {"data": _public(row)}


@router.patch("/documents/{document_id}", response_model=DocumentResponse)
def update_document(document_id: str, body: DocumentUpdate) -> dict:
    """Rename a document or change its description.

    400 for a blank name: the name is what the list shows, so a row with none
    could not be told apart from the next.
    """
    values = body.model_dump(exclude_unset=True)
    if "name" in values:
        name = (values["name"] or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Document name is required")
        values["name"] = name
    if "description" in values:
        values["description"] = (values["description"] or "").strip() or None

    with get_engine().begin() as conn:
        row = dal.update_document(conn, document_id, values)
    if row is None:
        raise _not_found(document_id)
    return {"data": _public(row)}


@router.delete("/documents/{document_id}", response_model=IdResponse)
def delete_document(document_id: str) -> dict:
    """Delete a document with its chunks, terms, attributes and vectors.

    The stored file goes too when it is one this API uploaded; a document
    ingested from a directory on the server leaves that directory alone.
    """
    with get_engine().begin() as conn:
        row = dal.get_document(conn, document_id)
        if row is None:
            raise _not_found(document_id)
        dal.delete_document(conn, document_id)

    upload_root = Path(config.UPLOAD_DIR).expanduser().resolve()
    stored = Path(row["path"]).parent
    if stored.parent == upload_root:
        shutil.rmtree(stored, ignore_errors=True)
    return {"data": {"id": document_id}}
