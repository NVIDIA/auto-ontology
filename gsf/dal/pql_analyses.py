# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``PqlAnalysis`` reads and writes — the predictive twin of CustomAnalysis.

A PqlAnalysis is a verified natural-language question paired with a KumoRFM PQL
query. Unlike a CustomAnalysis it does not parse into a statement/table
subgraph — PQL resolves its tables against the prediction graph at predict time
— so the PQL text is a plain column with no link table behind it.

These are kept out of SQL text-to-SQL retrieval entirely; they are embedded and
retrieved only as few-shot examples for PQL generation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from gsf.dal import schema as s
from gsf.dal.session import store
from gsf.semantic.constants import LABEL_PQL_ANALYSIS

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

_LABEL = LABEL_PQL_ANALYSIS


class PqlAnalysisNameConflict(Exception):
    """Raised when a write would collide with another PqlAnalysis name."""


class PqlAnalysisPqlConflict(Exception):
    """Raised when the submitted PQL is already linked to a different analysis."""


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_pql_analyses() -> list[dict[str, Any]]:
    """Every PqlAnalysis as ``{id, name, description, pql}``, by name."""
    return [
        dict(r)
        for r in store().query_read(
            select(
                s.pql_analysis.c.id,
                s.pql_analysis.c.name,
                s.pql_analysis.c.description,
                s.pql_analysis.c.pql,
            ).order_by(s.pql_analysis.c.name)
        )
    ]


def _find_conflict(column, value: str, exclude_id: str | None):
    """Another analysis already using *value* in *column*, or ``None``.

    *exclude_id* is the analysis being edited — without it, saving one unchanged
    reports a conflict with itself.
    """
    statement = select(s.pql_analysis.c.id, s.pql_analysis.c.name).where(
        column == value
    )
    if exclude_id is not None:
        statement = statement.where(s.pql_analysis.c.id != exclude_id)
    rows = store().query_read(statement.order_by(s.pql_analysis.c.id).limit(1))
    return {"id": rows[0]["id"], "name": rows[0]["name"]} if rows else None


def find_pql_analysis_by_name(
    name: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    return _find_conflict(s.pql_analysis.c.name, name, exclude_id)


def find_pql_analysis_by_pql(
    pql: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    return _find_conflict(s.pql_analysis.c.pql, pql, exclude_id)


def get_pql_analysis_by_id(analysis_id: str) -> str | None:
    """The analysis id if it exists, else ``None`` — an existence check."""
    rows = store().query_read(
        select(s.pql_analysis.c.id).where(s.pql_analysis.c.id == analysis_id).limit(1)
    )
    return rows[0]["id"] if rows else None


def fetch_pql_analyses_by_ids(analysis_ids: list[str]) -> dict[str, dict[str, str]]:
    """``{id: {id, name, description, pql}}``.

    Values are stripped, and a missing one becomes ``""`` — these go straight
    into a prompt, where ``None`` would render as the word "None".

    Returns ``{}`` on failure rather than raising: this decorates retrieval
    results, and losing the examples beats losing the answer.
    """
    if not analysis_ids:
        return {}
    try:
        rows = store().query_read(
            select(
                s.pql_analysis.c.id,
                s.pql_analysis.c.name,
                s.pql_analysis.c.description,
                s.pql_analysis.c.pql,
            ).where(s.pql_analysis.c.id.in_(list(analysis_ids)))
        )
    except Exception:
        logger.warning("fetch_pql_analyses_by_ids: query failed", exc_info=True)
        return {}

    return {
        row["id"]: {
            "id": row["id"],
            "name": (row["name"] or "").strip(),
            "description": (row["description"] or "").strip(),
            "pql": (row["pql"] or "").strip(),
        }
        for row in rows
        if row["id"]
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def upsert_pql_analysis_node(
    analysis_id: str,
    name: str,
    description: str,
    pql: str,
) -> None:
    """Create or overwrite an analysis by id.

    The caller supplies the id — this is an upsert *by id*, not a merge by name,
    so the service layer can decide identity before writing. Every field is
    assigned rather than coalesced: it is a PUT, and the conflict checks above
    have already run.
    """
    statement = insert(s.pql_analysis).values(
        id=analysis_id, name=name, description=description, pql=pql
    )
    store().query_write(
        statement.on_conflict_do_update(
            index_elements=[s.pql_analysis.c.id],
            set_={
                "name": statement.excluded.name,
                "description": statement.excluded.description,
                "pql": statement.excluded.pql,
            },
        )
    )


def delete_pql_analysis_node(analysis_id: str) -> None:
    """Delete the analysis. Nothing hangs off it, so nothing cascades."""
    store().query_write(
        delete(s.pql_analysis).where(s.pql_analysis.c.id == analysis_id)
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _pql_analysis_docs(analysis_id: str | None) -> list[dict[str, Any]]:
    """Embedding-ready docs — **the question text only**.

    The PQL body is deliberately not embedded. Retrieval here is
    question-to-question similarity, so the PQL is payload, fetched by id once a
    match is found. Embedding it would push the vector toward query syntax and
    away from the question a user actually asks.
    """
    statement = select(
        s.pql_analysis.c.id, s.pql_analysis.c.name, s.pql_analysis.c.description
    )
    if analysis_id is not None:
        statement = statement.where(s.pql_analysis.c.id == analysis_id)

    docs: list[dict[str, Any]] = []
    for row in store().query_read(statement.order_by(s.pql_analysis.c.id)):
        description = row["description"]
        text = row["name"]
        if description is not None and str(description).strip():
            text += f": {description}"
        docs.append({"text": text, "name": row["name"], "id": row["id"]})
    return docs


def embed_pql_analyses(
    embed_params: "EmbedParams",
    vdb: "VDB",
    analysis_id: str | None = None,
    database_name: str | None = None,
) -> None:
    """Embed PqlAnalysis docs and append them to *vdb*."""
    import pandas as pd
    from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
    from nemo_retriever.operators.vdb import IngestVdbOperator

    docs = _pql_analysis_docs(analysis_id)
    if not docs:
        logger.info(
            "No PqlAnalysis rows found for analysis_id=%r; skipping VDB upsert.",
            analysis_id,
        )
        return

    rows = []
    for item in docs:
        node_id = item.get("id")
        # An opaque provenance key. Nothing matches on it -- deletes go
        # through `metadata["id"]` -- but changing it leaves rows already
        # written carrying the old value.
        path = f"gsf:{node_id}" if node_id is not None else "gsf:unknown"
        tabular_fields = {
            "id": node_id,
            "label": _LABEL,
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
    embedded = embed_text_main_text_embed(
        df,
        model_name=embed_params.model_name,
        embed_invoke_url=embed_params.embed_invoke_url,
        api_key=embed_params.api_key,
        embed_modality=embed_params.embed_modality,
    )

    with_embeddings = [
        row
        for row in embedded.to_dict(orient="records")
        if (row.get("metadata") or {}).get("embedding")
    ]
    if not with_embeddings:
        raise RuntimeError(
            f"Embedding step produced 0/{len(embedded)} PqlAnalysis rows with "
            f"embeddings; check upstream embed errors (often a transient "
            f"{embed_params.embed_invoke_url} 5xx)."
        )

    IngestVdbOperator(vdb=vdb)(with_embeddings)
    logger.info(
        "Embedded and appended %d/%d PqlAnalysis row(s) via %s in %.2fs.",
        len(with_embeddings),
        len(embedded),
        type(vdb).__name__,
        time.time() - before,
    )
