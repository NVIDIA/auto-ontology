# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAL for the unstructured document / chunk / term / attribute tables.

Uses SQLAlchemy Core (no ORM), matching auto-ontology's DAL conventions.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Connection, and_, case, func, literal, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from auto_ontology.dal.schema import (
    unstructured_attribute,
    unstructured_attribute__chunk,
    unstructured_attribute_family,
    unstructured_attribute_family__attribute,
    unstructured_chunk,
    unstructured_document,
    unstructured_term,
)
from unstructured_ao import config


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def insert_document(
    conn: Connection,
    *,
    name: str,
    path: str,
    file_type: str,
    summary: str | None = None,
    publication_date: str | None = None,
    description: str | None = None,
    status: str = "ready",
) -> str:
    row = conn.execute(
        unstructured_document.insert().returning(unstructured_document.c.id),
        {
            "name": name,
            "path": path,
            "file_type": file_type,
            "summary": summary,
            "publication_date": publication_date,
            "description": description,
            "status": status,
        },
    ).scalar_one()
    return row


def get_document_by_path(conn: Connection, path: str) -> dict[str, Any] | None:
    row = (
        conn.execute(
            select(unstructured_document).where(unstructured_document.c.path == path)
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


#: What the settings page shows of a document; ``path`` is a server path and
#: stays internal.
_DOCUMENT_LIST_COLUMNS = (
    unstructured_document.c.id,
    unstructured_document.c.name,
    unstructured_document.c.description,
    unstructured_document.c.file_type,
    unstructured_document.c.status,
    unstructured_document.c.error,
    unstructured_document.c.ingested_at,
)


def _document_filter(search: str | None, without_term: bool) -> Any:
    conditions = []
    if search:
        pattern = f"%{search}%"
        conditions.append(
            or_(
                unstructured_document.c.name.ilike(pattern),
                unstructured_document.c.description.ilike(pattern),
            )
        )
    if without_term:
        conditions.append(
            ~select(unstructured_term.c.id)
            .where(unstructured_term.c.document_id == unstructured_document.c.id)
            .exists()
        )
    return and_(literal(True), *conditions)


def list_documents(
    conn: Connection,
    *,
    search: str | None = None,
    without_term: bool = False,
    newest_first: bool = False,
    skip: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Ingested documents, oldest first unless *newest_first*.

    *search* matches name and description, case-insensitively.  With
    *without_term*, only those the semantic stage has not processed — which is
    what makes a second ``run_semantic`` run resume rather than redo.  *limit*
    ``None`` returns every match.  Rows include the server ``path``; callers
    that answer clients must drop it.
    """
    ingested_at = unstructured_document.c.ingested_at
    rows = (
        conn.execute(
            select(unstructured_document)
            .where(_document_filter(search, without_term))
            .order_by(
                ingested_at.desc() if newest_first else ingested_at,
                unstructured_document.c.id,
            )
            .offset(skip)
            .limit(limit)
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def count_documents(
    conn: Connection, *, search: str | None = None, without_term: bool = False
) -> int:
    return conn.execute(
        select(func.count())
        .select_from(unstructured_document)
        .where(_document_filter(search, without_term))
    ).scalar_one()


def _split_name(name: str) -> tuple[str, str]:
    """``("report", ".pdf")`` for ``report.pdf``; a dotfile has no extension."""
    stem, dot, extension = name.rpartition(".")
    if not dot or not stem:
        return name, ""
    return stem, f".{extension}"


def next_free_name(name: str, taken: set[str]) -> str:
    """*name*, or ``"<stem> (n)<suffix>"`` with the lowest *n* not in *taken*.

    *taken* holds lower-cased names, so ``Report.docx`` beside ``report.docx``
    still reads as a second copy.
    """
    if name.lower() not in taken:
        return name
    stem, suffix = _split_name(name)
    n = 1
    while f"{stem} ({n}){suffix}".lower() in taken:
        n += 1
    return f"{stem} ({n}){suffix}"


def unique_document_name(conn: Connection, name: str) -> str:
    """*name* made unique among stored documents — see :func:`next_free_name`.

    The transaction-scoped lock keeps choosing a suffix and inserting the row
    one atomic operation for callers that insert before committing.  One lock
    covers every spelling because ``report.pdf`` and an explicit
    ``report (1).pdf`` upload can otherwise choose the same candidate.
    """
    conn.execute(
        text(
            "SELECT pg_advisory_xact_lock("
            "hashtext('auto_ontology.unstructured_document.name'))"
        )
    )
    stem, suffix = _split_name(name)
    escaped = stem.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    taken = {
        row.lower()
        for row in conn.execute(
            select(unstructured_document.c.name).where(
                or_(
                    func.lower(unstructured_document.c.name) == name.lower(),
                    unstructured_document.c.name.ilike(
                        f"{escaped} (%){suffix}", escape="\\"
                    ),
                )
            )
        ).scalars()
    }
    return next_free_name(name, taken)


def list_pending_documents(conn: Connection) -> list[dict[str, Any]]:
    """Documents whose background pipeline did not reach a terminal state."""
    rows = (
        conn.execute(
            select(
                unstructured_document.c.id,
                unstructured_document.c.path,
                unstructured_document.c.status,
            ).where(unstructured_document.c.status.in_(("processing", "extracting")))
        )
        .mappings()
        .all()
    )
    return [dict(row) for row in rows]


def get_document(conn: Connection, document_id: str) -> dict[str, Any] | None:
    row = (
        conn.execute(
            select(*_DOCUMENT_LIST_COLUMNS, unstructured_document.c.path).where(
                unstructured_document.c.id == document_id
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


def update_document(
    conn: Connection, document_id: str, values: dict[str, Any]
) -> dict[str, Any] | None:
    """Set the given ``name``/``description`` fields; ``None`` when not found."""
    allowed = {k: v for k, v in values.items() if k in ("name", "description")}
    if allowed:
        conn.execute(
            unstructured_document.update()
            .where(unstructured_document.c.id == document_id)
            .values(**allowed)
        )
    return get_document(conn, document_id)


def set_document_status(
    conn: Connection, document_id: str, status: str, *, error: str | None = None
) -> None:
    conn.execute(
        unstructured_document.update()
        .where(unstructured_document.c.id == document_id)
        .values(status=status, error=error)
    )


def delete_document(conn: Connection, document_id: str) -> None:
    """Delete the document, its downstream rows, and its vectors.

    Membership cascades with the attributes, but a family spans documents, so
    only the families this document touched that are left with fewer than two
    members are deleted.  Larger survivors keep their labels until
    ``relate.build`` regenerates them.
    """
    from unstructured_ao.vectorstore import delete_document_vectors

    member = unstructured_attribute_family__attribute
    touched = select(member.c.family_id).where(
        member.c.attribute_id.in_(
            select(unstructured_attribute.c.id)
            .join(
                unstructured_term,
                unstructured_term.c.id == unstructured_attribute.c.term_id,
            )
            .where(unstructured_term.c.document_id == document_id)
        )
    )
    family_ids = list(conn.execute(touched.distinct()).scalars())

    delete_document_vectors(conn, document_id)
    conn.execute(
        unstructured_document.delete().where(unstructured_document.c.id == document_id)
    )

    if family_ids:
        survivors = (
            select(func.count())
            .where(member.c.family_id == unstructured_attribute_family.c.id)
            .scalar_subquery()
        )
        conn.execute(
            unstructured_attribute_family.delete().where(
                unstructured_attribute_family.c.id.in_(family_ids),
                survivors < 2,
            )
        )


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


def insert_chunk(
    conn: Connection,
    *,
    document_id: str,
    content: str,
    content_type: str,
    page_number: int | None,
    chunk_index: int,
    source_metadata: dict[str, Any] | None = None,
    chunk_id: str | None = None,
) -> str:
    """Insert a chunk and return its id.

    Pass *chunk_id* explicitly when the id must match a vector-store row already
    inserted with the same id (e.g. after upserting into langchain_pg_embedding).
    """
    row = conn.execute(
        unstructured_chunk.insert().returning(unstructured_chunk.c.id),
        {
            "id": chunk_id or str(uuid.uuid4()),
            "document_id": document_id,
            "content": content,
            "content_type": content_type,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "source_metadata": source_metadata,
        },
    ).scalar_one()
    return row


def get_document_chunks(conn: Connection, document_id: str) -> list[dict[str, Any]]:
    rows = (
        conn.execute(
            select(unstructured_chunk)
            .where(unstructured_chunk.c.document_id == document_id)
            .order_by(unstructured_chunk.c.chunk_index)
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


def insert_term(
    conn: Connection,
    *,
    document_id: str,
    name: str,
    description: str | None = None,
) -> str:
    row = conn.execute(
        unstructured_term.insert().returning(unstructured_term.c.id),
        {"document_id": document_id, "name": name, "description": description},
    ).scalar_one()
    return row


def _vector_table_exists(conn: Connection) -> bool:
    """Whether the attribute vector table has been created yet.

    It is created lazily by the first indexing write, so a document extracted
    before then has a term to delete but no vectors.
    """
    return bool(
        conn.execute(
            text("SELECT to_regclass(:qualified)"),
            {"qualified": f"{config.ATTRIBUTE_SCHEMA}.{config.ATTRIBUTE_TABLE}"},
        ).scalar()
    )


def delete_term(conn: Connection, term_id: str) -> None:
    """Delete a term, cascading to its attributes and their chunk links.

    How the semantic stage re-runs: a document has at most one term, so a
    second extraction pass has to clear the first one before inserting.

    The attributes' embeddings go too.  The vector table has no foreign key to
    cascade through, so without this a re-extraction would leave rows pointing
    at attribute ids that no longer exist — harmless to results, which join
    back to ``unstructured_attribute``, but they would occupy hits and make a
    ``k`` of 10 return fewer than 10 live attributes.  Issued on *conn* so it
    commits or rolls back with the delete it belongs to.
    """
    if _vector_table_exists(conn):
        conn.execute(
            text(
                f"DELETE FROM {config.ATTRIBUTE_SCHEMA}.{config.ATTRIBUTE_TABLE} "  # noqa: S608 — identifiers are module constants
                f"WHERE term_id = :term_id"
            ),
            {"term_id": term_id},
        )
    conn.execute(unstructured_term.delete().where(unstructured_term.c.id == term_id))


def get_document_term(conn: Connection, document_id: str) -> dict[str, Any] | None:
    row = (
        conn.execute(
            select(unstructured_term).where(
                unstructured_term.c.document_id == document_id
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def insert_attribute(
    conn: Connection,
    *,
    term_id: str,
    name: str,
    description: str | None,
    attribute_type: str,
) -> str:
    if attribute_type not in ("sql", "ref"):
        raise ValueError(
            f"attribute_type must be 'sql' or 'ref', got {attribute_type!r}"
        )
    row = conn.execute(
        unstructured_attribute.insert().returning(unstructured_attribute.c.id),
        {
            "term_id": term_id,
            "name": name,
            "description": description,
            "attribute_type": attribute_type,
        },
    ).scalar_one()
    return row


def get_term_attributes(conn: Connection, term_id: str) -> list[dict[str, Any]]:
    rows = (
        conn.execute(
            select(unstructured_attribute).where(
                unstructured_attribute.c.term_id == term_id
            )
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def get_document_attributes(conn: Connection, document_id: str) -> list[dict[str, Any]]:
    """All attributes for the term of *document_id*."""
    term = get_document_term(conn, document_id)
    if not term:
        return []
    return get_term_attributes(conn, term["id"])


# ---------------------------------------------------------------------------
# Attribute → chunk links
# ---------------------------------------------------------------------------


def insert_attribute_chunk_link(
    conn: Connection,
    *,
    attribute_id: str,
    chunk_id: str,
    score: float | None = None,
    source: str = "vector",
) -> None:
    """Link *attribute_id* to *chunk_id*, upserting on the composite key.

    Extraction and vector search both reach the same chunk routinely, and a
    plain insert would abort the whole document's transaction on the second
    one.  When they collide the llm link wins — the model demonstrably read
    that chunk — while the vector distance is kept for scoring.
    """
    if source not in ("llm", "vector"):
        raise ValueError(f"source must be 'llm' or 'vector', got {source!r}")

    stmt = pg_insert(unstructured_attribute__chunk).values(
        attribute_id=attribute_id,
        chunk_id=chunk_id,
        score=str(score) if score is not None else None,
        source=source,
    )
    conn.execute(
        stmt.on_conflict_do_update(
            index_elements=["attribute_id", "chunk_id"],
            set_={
                "score": func.coalesce(
                    stmt.excluded.score, unstructured_attribute__chunk.c.score
                ),
                "source": case(
                    (
                        or_(
                            stmt.excluded.source == "llm",
                            unstructured_attribute__chunk.c.source == "llm",
                        ),
                        literal("llm"),
                    ),
                    else_=stmt.excluded.source,
                ),
            },
        )
    )
