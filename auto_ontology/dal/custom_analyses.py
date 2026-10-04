# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``CustomAnalysis`` reads and writes.

A CustomAnalysis is a user-authored, named SQL statement. The zone rule here is
the same all-or-nothing one SqlAttribute uses, and for a sharper reason: the
analysis's **SQL text is returned to the caller**, so an analysis touching one
out-of-zone table would leak that table's name and columns even if no rows ever
came back.

Orchestration stays in ``auto_ontology/server/custom_analyses/service.py``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, literal, select

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.dal.sql_fragments import column_description_expr, name_contains
from auto_ontology.dal.users import resolve_accessible_catalog_ids
from auto_ontology.server.sql_utils import SqlParseError

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)


class CustomAnalysisNameConflict(Exception):
    """Raised when a write would collide with another CustomAnalysis name."""


class CustomAnalysisSqlConflict(Exception):
    """Raised when the submitted SQL is already linked to a different analysis."""


class CustomAnalysisSqlError(SqlParseError):
    """Raised when the SQL can't be parsed against the current catalog."""


# ---------------------------------------------------------------------------
# Shared traversal
# ---------------------------------------------------------------------------


def _sql_text():
    """The analysis's SQL, from its lowest-id statement.

    An analysis carries one statement in practice, but the link table permits
    more; taking the lowest id keeps one row per analysis and keeps repeated
    reads agreeing with each other.
    """
    return (
        select(s.sql_query.c.sql_full_query)
        .select_from(
            s.custom_analysis__sql.join(
                s.sql_query, s.sql_query.c.id == s.custom_analysis__sql.c.sql_query_id
            )
        )
        .where(s.custom_analysis__sql.c.analysis_id == s.custom_analysis.c.id)
        .order_by(s.sql_query.c.id)
        .limit(1)
        .correlate(s.custom_analysis)
        .scalar_subquery()
    )


def _has_sql():
    """An analysis with no statement is invisible to every read here.

    The SQL's ``HAS_SQL`` match was not optional except in
    ``fetch_custom_analyses_with_sql``, which used ``OPTIONAL MATCH`` — that
    difference is real and preserved.
    """
    return (
        select(literal(1))
        .where(s.custom_analysis__sql.c.analysis_id == s.custom_analysis.c.id)
        .correlate(s.custom_analysis)
        .exists()
    )


def _out_of_zone(table_ids: list[str]):
    """True when the analysis's SQL touches a table outside *table_ids*."""
    return (
        select(literal(1))
        .select_from(
            s.custom_analysis__sql.join(
                s.sql_query__table,
                s.sql_query__table.c.sql_query_id
                == s.custom_analysis__sql.c.sql_query_id,
            )
        )
        .where(
            s.custom_analysis__sql.c.analysis_id == s.custom_analysis.c.id,
            s.sql_query__table.c.table_id.notin_(table_ids),
        )
        .correlate(s.custom_analysis)
        .exists()
    )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _matching(zone_ids: list[str] | None, search: str | None) -> list:
    """The WHERE every read of the visible analysis set applies.

    One function so :func:`list_custom_analyses` and
    :func:`count_custom_analyses` cannot come to disagree about what matches —
    a page and a total taken from different filters would leave the list asking
    for rows that are not there.
    """
    conditions = [_has_sql()]
    if zone_ids is not None:
        resolved = resolve_accessible_catalog_ids(zone_ids)
        conditions.append(~_out_of_zone(list(resolved["table_ids"])))
    if search and search.strip():
        conditions.append(name_contains(s.custom_analysis.c.name, search))
    return conditions


def list_custom_analyses(
    zone_ids: list[str] | None = None,
    *,
    search: str | None = None,
    skip: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Analyses with their SQL text, paged.

    *zone_ids* is an authorization boundary, not a relevance filter: an analysis
    touching even one out-of-zone table is excluded outright, because returning
    its SQL would disclose the names and columns of tables the caller may not
    see. ``None`` returns everything, for admin callers.

    *search*, when given, keeps the analyses whose name contains it,
    case-insensitively.

    Ordered by name case-insensitively then by id — the id breaks ties between
    same-named analyses, without which a page boundary could repeat one and
    skip another. *skip* and *limit* select a window of that order, and *limit*
    omitted returns every matching analysis. Pair with
    :func:`count_custom_analyses` for the total.
    """
    statement = (
        select(
            s.custom_analysis.c.id,
            s.custom_analysis.c.name,
            s.custom_analysis.c.description,
            _sql_text().label("sql"),
        )
        .where(*_matching(zone_ids, search))
        .order_by(func.lower(s.custom_analysis.c.name), s.custom_analysis.c.id)
        .offset(skip or None)
    )
    if limit is not None:
        statement = statement.limit(limit)

    return [dict(r) for r in store().query_read(statement)]


def count_custom_analyses(
    zone_ids: list[str] | None = None,
    *,
    search: str | None = None,
) -> int:
    """The unpaged size of :func:`list_custom_analyses`, from the same filter."""
    rows = store().query_read(
        select(func.count(s.custom_analysis.c.id).label("total")).where(
            *_matching(zone_ids, search)
        )
    )
    return int(rows[0]["total"]) if rows else 0


def find_analysis_by_name(
    name: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    """Another analysis already using *name*, or ``None``."""
    statement = select(s.custom_analysis.c.id, s.custom_analysis.c.name).where(
        s.custom_analysis.c.name == name
    )
    if exclude_id is not None:
        statement = statement.where(s.custom_analysis.c.id != exclude_id)
    rows = store().query_read(statement.order_by(s.custom_analysis.c.id).limit(1))
    return {"id": rows[0]["id"], "name": rows[0]["name"]} if rows else None


def find_analysis_by_sql(
    sql: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    """Another analysis already linked to this exact SQL, or ``None``.

    Exact text, not normalised — unlike ``find_attr_by_expression``, which
    collapses whitespace and case. The difference is inherited: this compared
    ``sql_full_query`` directly, and loosening it here would start rejecting
    saves that succeed today.
    """
    statement = (
        select(s.custom_analysis.c.id, s.custom_analysis.c.name)
        .select_from(
            s.custom_analysis.join(
                s.custom_analysis__sql,
                s.custom_analysis__sql.c.analysis_id == s.custom_analysis.c.id,
            ).join(
                s.sql_query, s.sql_query.c.id == s.custom_analysis__sql.c.sql_query_id
            )
        )
        .where(s.sql_query.c.sql_full_query == sql)
    )
    if exclude_id is not None:
        statement = statement.where(s.custom_analysis.c.id != exclude_id)
    rows = store().query_read(statement.order_by(s.custom_analysis.c.id).limit(1))
    return {"id": rows[0]["id"], "name": rows[0]["name"]} if rows else None


def get_custom_analysis_by_id(analysis_id: str) -> str | None:
    """The analysis id if it exists, else ``None`` — an existence check."""
    rows = store().query_read(
        select(s.custom_analysis.c.id)
        .where(s.custom_analysis.c.id == analysis_id)
        .limit(1)
    )
    return rows[0]["id"] if rows else None


def fetch_custom_analyses() -> list[dict[str, str]]:
    """Analyses as domain rules: ``{name, description}``, SQL folded into the text.

    The ``description`` here is a *rendered blob*, not the column — description
    and SQL joined by a newline — because the caller feeds it to a prompt as
    one piece of guidance. An analysis with neither is dropped, since a rule
    with an empty body is noise in a prompt.

    Returns ``[]`` on failure: missing domain rules degrade a prompt, a raised
    exception loses the request.
    """
    try:
        rows = store().query_read(
            select(
                s.custom_analysis.c.name,
                s.custom_analysis.c.description,
                _sql_text().label("sql_code"),
            )
            .where(_has_sql())
            .order_by(s.custom_analysis.c.name)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch custom analyses: %s", exc)
        return []

    rules: list[dict[str, str]] = []
    for row in rows:
        if not row["name"]:
            continue
        parts = []
        if row["description"]:
            parts.append(row["description"])
        if row["sql_code"]:
            parts.append(f"SQL: {row['sql_code']}")
        if parts:
            rules.append({"name": row["name"], "description": "\n".join(parts)})
    logger.info("Fetched %d custom analyses as domain rules", len(rules))
    return rules


def fetch_custom_analyses_with_sql(analysis_ids: list[str]) -> list[dict[str, str]]:
    """``{id, name, description, sql}`` per analysis, values stripped.

    Unlike the reads above, an analysis with **no** statement still comes back
    with an empty ``sql`` — the SQL used ``OPTIONAL MATCH`` here alone.
    Preserved: the caller asked for specific ids and a silently missing entry is
    harder to notice than an empty one.
    """
    if not analysis_ids:
        return []
    try:
        rows = store().query_read(
            select(
                s.custom_analysis.c.id,
                s.custom_analysis.c.name,
                s.custom_analysis.c.description,
                _sql_text().label("sql_text"),
            ).where(s.custom_analysis.c.id.in_(list(analysis_ids)))
        )
    except Exception:
        logger.warning("fetch_custom_analyses_with_sql: query failed", exc_info=True)
        return []

    return [
        {
            "id": row["id"] or "",
            "name": (row["name"] or "").strip(),
            "description": (row["description"] or "").strip(),
            "sql": (row["sql_text"] or "").strip(),
        }
        for row in rows
    ]


def fetch_tables_from_custom_analyses(analysis_ids: list[str]) -> list[dict[str, Any]]:
    """The tables these analyses' SQL references, with column summaries."""
    if not analysis_ids:
        return []
    try:
        rows = store().query_read(
            select(
                s.catalog_table.c.id,
                s.catalog_table.c.name,
                s.catalog_table.c.description,
                s.catalog_table.c.pk,
                s.catalog_database.c.name.label("database_name"),
                s.catalog_schema.c.name.label("schema_name"),
                s.catalog_column.c.name.label("column_name"),
                s.catalog_column.c.data_type,
                s.catalog_column.c.ordinal_position,
                column_description_expr().label("column_description"),
            )
            .select_from(
                s.custom_analysis__sql.join(
                    s.sql_query__table,
                    s.sql_query__table.c.sql_query_id
                    == s.custom_analysis__sql.c.sql_query_id,
                )
                .join(
                    s.catalog_table,
                    s.catalog_table.c.id == s.sql_query__table.c.table_id,
                )
                .join(
                    s.catalog_schema,
                    s.catalog_schema.c.id == s.catalog_table.c.schema_id,
                )
                .join(
                    s.catalog_database,
                    s.catalog_database.c.id == s.catalog_schema.c.database_id,
                )
                .join(
                    s.catalog_column,
                    s.catalog_column.c.table_id == s.catalog_table.c.id,
                )
            )
            .where(s.custom_analysis__sql.c.analysis_id.in_(list(analysis_ids)))
            .distinct()
            .order_by(s.catalog_table.c.id, s.catalog_column.c.ordinal_position)
        )
    except Exception:
        logger.warning("fetch_tables_from_custom_analyses: query failed", exc_info=True)
        return []

    tables: dict[str, dict[str, Any]] = {}
    for row in rows:
        table = tables.setdefault(
            row["id"],
            {
                "id": row["id"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "database_name": row["database_name"] or "",
                "schema_name": row["schema_name"] or "",
                "label": "Table",
                # The prediction graph keys its entities on this: a table
                # that arrives without a pk reaches KumoRFM with no identity,
                # which costs it every edge and makes it unusable in
                # `FOR EACH`. It has to survive every path to relevant_tables.
                "pk": row.get("pk") or [],
                "columns": [],
            },
        )
        if row["column_name"]:
            table["columns"].append(
                {
                    "name": row["column_name"],
                    "data_type": row["data_type"],
                    "description": row["column_description"],
                }
            )
    return list(tables.values())


def custom_analysis_exists(database_name: str | None = None) -> bool:
    """Is there at least one custom analysis, optionally in *database_name*?

    Scoped by *database_name* to analyses whose statement references a table in
    that database; any analysis at all satisfies the unscoped check.

    A ``LIMIT 1`` existence probe, so retrieval can skip the embedding search
    outright when nothing could match. **Fails open** — a failed probe returns
    ``True`` rather than silently suppressing a search that might have hit.
    """
    statement = select(literal(1)).select_from(s.custom_analysis).limit(1)
    if database_name:
        statement = statement.where(
            select(literal(1))
            .select_from(
                s.custom_analysis__sql.join(
                    s.sql_query__table,
                    s.sql_query__table.c.sql_query_id
                    == s.custom_analysis__sql.c.sql_query_id,
                )
                .join(
                    s.catalog_table,
                    s.catalog_table.c.id == s.sql_query__table.c.table_id,
                )
                .join(
                    s.catalog_schema,
                    s.catalog_schema.c.id == s.catalog_table.c.schema_id,
                )
                .join(
                    s.catalog_database,
                    s.catalog_database.c.id == s.catalog_schema.c.database_id,
                )
            )
            .where(
                s.custom_analysis__sql.c.analysis_id == s.custom_analysis.c.id,
                s.catalog_database.c.name == database_name,
            )
            .correlate(s.custom_analysis)
            .exists()
        )
    try:
        return bool(store().query_read(statement))
    except Exception:
        logger.warning("custom_analysis_exists: query failed", exc_info=True)
        return True


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def detach_existing_sql_edges(analysis_id: str) -> None:
    """Unlink every statement from the analysis, leaving the statements alone."""
    store().query_write(
        delete(s.custom_analysis__sql).where(
            s.custom_analysis__sql.c.analysis_id == analysis_id
        )
    )


def delete_custom_analysis_node(analysis_id: str) -> None:
    """Delete the analysis **and the statements it owns**.

    The SQL's ``DETACH DELETE ca, sql`` deleted both, and that is preserved
    rather than left to the cascade — which would only remove the link row. An
    analysis's statement is not shared: it is parsed from text the user typed
    into this analysis, so leaving it behind accumulates unreachable rows that
    still show up in query-history reads.
    """
    owned = select(s.custom_analysis__sql.c.sql_query_id).where(
        s.custom_analysis__sql.c.analysis_id == analysis_id
    )

    # One unit: deleting the analysis first drops the link rows by cascade, so
    # a failure before the second delete loses the only route back to those
    # statements. They then survive as exactly the unreachable rows this
    # function exists to prevent.
    with write_transaction():
        sql_ids = [r["sql_query_id"] for r in store().query_read(owned)]
        store().query_write(
            delete(s.custom_analysis).where(s.custom_analysis.c.id == analysis_id)
        )
        if sql_ids:
            store().query_write(
                delete(s.sql_query).where(s.sql_query.c.id.in_(sql_ids))
            )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _custom_analysis_docs(analysis_id: str | None) -> list[dict[str, Any]]:
    """Embedding-ready docs.

    The text format is what gets embedded, so a changed separator silently
    invalidates every stored vector for these analyses and nothing downstream
    fails — retrieval just quietly degrades. Reproduced literally, including a
    blank description being omitted rather than rendered as an empty clause.
    """
    statement = (
        select(
            s.custom_analysis.c.id,
            s.custom_analysis.c.name,
            s.custom_analysis.c.description,
            _sql_text().label("sql_text"),
        )
        .where(_has_sql())
        .order_by(s.custom_analysis.c.id)
    )
    if analysis_id is not None:
        statement = statement.where(s.custom_analysis.c.id == analysis_id)

    docs: list[dict[str, Any]] = []
    for row in store().query_read(statement):
        description = row["description"]
        text = f"custom_analysis: {row['name']}"
        if description is not None and str(description).strip():
            text += f", description: {description}"
        if row["sql_text"] is not None:
            text += f", sql: {row['sql_text']}"
        docs.append(
            {
                "text": text,
                "name": row["name"],
                "label": "CustomAnalysis",
                "id": row["id"],
            }
        )
    return docs


def embed_custom_analyses(
    embed_params: "EmbedParams",
    vdb: "VDB",
    analysis_id: str | None = None,
    database_name: str | None = None,
) -> None:
    """Embed CustomAnalysis docs and append them to *vdb*."""
    import pandas as pd
    from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
    from nemo_retriever.operators.vdb import IngestVdbOperator

    from auto_ontology.utils.embedding import _embed_with_retry

    docs = _custom_analysis_docs(analysis_id)
    if not docs:
        logger.info(
            "No CustomAnalysis rows found for analysis_id=%r; skipping VDB upsert.",
            analysis_id,
        )
        return

    rows = []
    for item in docs:
        node_id = item.get("id")
        # A stored row key, not a reference to the store -- see the note in
        # pg/pql_analyses.py.
        path = (
            f"auto_ontology:{node_id}"
            if node_id is not None
            else "auto_ontology:unknown"
        )
        tabular_fields = {
            "id": node_id,
            "label": item.get("label", ""),
            "name": item.get("name", ""),
            "source_path": path,
            "database_name": database_name,
        }
        rows.append(
            {
                "text": (item.get("text") or "").strip(),
                "_embed_modality": "text",
                "path": path,
                "page_number": -1,
                "metadata": {
                    **tabular_fields,
                    "content_metadata": dict(tabular_fields),
                },
            }
        )
    df = pd.DataFrame(rows)

    before = time.time()
    # Through the same retry the catalog embeds use. The library catches per
    # frame, so one refused request zeroes every row it was given -- and this
    # frame is usually a handful of rows, so a single transient 5xx empties it
    # and aborts the whole ingest at whichever database it landed on.
    embedded = _embed_with_retry(
        lambda part: embed_text_main_text_embed(
            part,
            model_name=embed_params.model_name,
            embed_invoke_url=embed_params.embed_invoke_url,
            api_key=embed_params.api_key,
            embed_modality=embed_params.embed_modality,
        ),
        df,
        label=f"CustomAnalysis {database_name or '<all>'}",
    )

    with_embeddings = [
        row
        for row in (embedded.to_dict(orient="records") if embedded is not None else [])
        if (row.get("metadata") or {}).get("embedding")
    ]
    if not with_embeddings:
        raise RuntimeError(
            f"Embedding step produced 0/{len(df)} CustomAnalysis rows "
            f"with embeddings; check upstream embed errors (often a transient "
            f"{embed_params.embed_invoke_url} 5xx)."
        )

    IngestVdbOperator(vdb=vdb)(with_embeddings)
    logger.info(
        "Embedded and appended %d/%d CustomAnalysis row(s) via %s in %.2fs.",
        len(with_embeddings),
        len(embedded),
        type(vdb).__name__,
        time.time() - before,
    )


def fetch_database_name_for_analysis(analysis_id: str) -> str | None:
    """Return the database whose tables an analysis's SQL references.

    Retrieval filters semantic hits on ``database_name``
    (:func:`auto_ontology.retrieval.data_access.semantic_search.search_semantic_index`),
    so an analysis embedded without one is invisible to every scoped search.

    The relational path mirrors the edges this replaced -- analysis -> its SQL
    -> the tables that SQL reads -> schema -> database.

    ``DISTINCT ... LIMIT 1`` is deliberate, and lossy: an analysis whose SQL
    joins across two databases has two answers and this returns one of them.
    That matches the behaviour being ported rather than quietly improving on
    it; the column it feeds holds a single value.
    """
    rows = store().query_read(
        select(s.catalog_database.c.name)
        .select_from(
            s.custom_analysis__sql.join(
                s.sql_query__table,
                s.sql_query__table.c.sql_query_id
                == s.custom_analysis__sql.c.sql_query_id,
            )
            .join(
                s.catalog_table,
                s.catalog_table.c.id == s.sql_query__table.c.table_id,
            )
            .join(
                s.catalog_schema,
                s.catalog_schema.c.id == s.catalog_table.c.schema_id,
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.custom_analysis__sql.c.analysis_id == analysis_id)
        .distinct()
        # Stable across calls: without it two databases would alternate, and the
        # embedding's scope would depend on plan order.
        .order_by(s.catalog_database.c.name)
        .limit(1)
    )
    return str(rows[0]["name"]) if rows else None
