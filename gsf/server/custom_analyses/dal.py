# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Access Layer — Neo4j queries for ``CustomAnalysis`` nodes.

Writes mirror the inner loop of
``nemo_retriever.tabular_data.dev_tools.enrich_graph.add_custom_analyses``
(parse SQL → set ``Sql`` ``match_props`` to ``sql_full_query`` → append
``HAS_SQL`` edge → ``add_query``) so the resulting ``Sql`` node carries
the full property set (``nodes_count``, ``join_count``, ``union_count``,
``count_<month>_<year>``, ...) and the ``Sql -[:SQL]-> Table/Column``
edges expected by the rest of the stack. ``add_custom_analyses`` itself
isn't reusable from the API path — it's a JSON-driven, batch-oriented
helper that also embeds into LanceDB — so we reproduce its inner loop
here and add the API-only concerns on top: strict natural-key
uniqueness on ``id``, ``name`` and ``sql`` (any collision is rejected
upfront so the graph stays consistent and one analysis can never be
silently merged into another by ``POST``/``PUT``).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import (
    Edges,
    Labels,
    Props,
)
from nemo_retriever.tabular_data.ingestion.services.queries import parse_query_single
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from gsf.retrieval.data_access.graph_schemas import (
    get_all_schemas_ids,
    get_schemas_by_ids,
)
from nemo_retriever.operators.vdb import IngestVdbOperator
from nemo_retriever.models.inference.runtime import embed_text_main_text_embed

from gsf.connectors import get_connectors

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)


class CustomAnalysisNameConflict(Exception):
    """Raised when a write would collide with another CustomAnalysis name.

    ``name`` is treated as a natural key alongside ``id`` and ``sql``;
    two ``CustomAnalysis`` nodes sharing the same ``name`` would break
    list ordering and any future lookup-by-name, so both ``POST`` (new
    create with an already-used name) and ``PUT`` (rename into a slot
    held by a different analysis) fail with this error instead of
    silently merging records.
    """


class CustomAnalysisSqlConflict(Exception):
    """Raised when the submitted SQL is already linked to a different analysis.

    SQL text is treated as a natural key alongside ``id`` and ``name``:
    each ``Sql`` node (matched by ``sql_full_query``) is allowed exactly
    one inbound ``HAS_SQL`` edge from a ``CustomAnalysis``. Without this
    guard two analyses could fan-out from the same ``Sql`` node — they
    would share the ``Sql -[:SQL]-> Table/Column`` graph and retrieval
    would no longer be able to attribute results to a single analysis,
    and editing one analysis' SQL would silently move the other one too.
    """


class CustomAnalysisSqlError(Exception):
    """Raised when the SQL can't be parsed against the current catalog.

    Covers two failure modes from ``parse_query_single``:

    * the parser itself raises (syntax error, unsupported dialect
      construct, sqlglot blow-up) — mirrored from
      ``SQLValidationAgent._sql_parse_validation`` in NeMo-Retriever,
      which wraps the same call in ``try/except`` and returns
      ``{"error": str(error)}``;
    * the parser returns ``None`` because the SQL doesn't resolve to
      any table the graph already knows about (typos, missing
      ingestion, ...).

    Either way we refuse the write: without a valid AST and at least
    one resolved table the ``Sql -[:SQL]-> Table/Column`` edges that
    downstream retrieval relies on can't be produced.
    """


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def list_custom_analyses() -> list[dict[str, Any]]:
    """Return every ``CustomAnalysis`` that has a linked ``Sql`` node.

    Each row contains ``id``, ``name``, ``description`` and ``sql`` — the
    ``sql_full_query`` of the related :class:`Sql` node reached via
    ``CustomAnalysis -[:HAS_SQL]-> Sql``.

    The ``HAS_SQL`` join is mandatory: a ``CustomAnalysis`` without an
    attached ``Sql`` node is unusable (retrieval can't surface it, the UI
    can't render it) and would leak ``sql: null`` to the API. Our own
    write path (:func:`create_custom_analysis` / :func:`update_custom_analysis`)
    only commits a record after the SQL parses, so dangling records can
    only come from external writers — NeMo-Retriever's ``enrich_graph``,
    which only warns on parse ``None``, or direct Cypher. Hiding those
    here keeps the API contract simple (``sql`` is always a string) at
    the cost of needing direct DB access to inspect or clean them up.

    NeMo-Retriever exposes ``fetch_custom_analyses`` for a similar read,
    but it (a) drops the ``id`` we need to address rows from the UI and
    (b) folds ``description`` and ``sql`` into a single string aimed at
    the LLM prompt. We keep this dedicated read instead of forcing
    those choices on every consumer.
    """
    neo4j_conn = get_neo4j_conn()

    rows = neo4j_conn.query_read(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS})-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        WITH ca, sql
        ORDER BY ca.name

        RETURN collect({{
            id: ca.id,
            name: ca.name,
            description: ca.description,
            sql: sql.sql_full_query
        }}) AS analyses
        """,
    )

    return rows[0]["analyses"]


# ---------------------------------------------------------------------------
# Write helpers (private)
# ---------------------------------------------------------------------------


def _get_dialects() -> list[str]:
    """Return SQL dialects from active connectors (NeMo multi-connector order)."""
    connectors = get_connectors()
    dialects = [c.dialect for c in connectors if getattr(c, "dialect", None)]
    if not dialects:
        return ["generic", "ansi", "postgres"]
    return dialects


def _get_schemas() -> dict:
    """Return the full catalog snapshot for ``parse_query_single``."""
    schemas_ids = get_all_schemas_ids()
    return get_schemas_by_ids(schemas_ids)


def _validate_sql(sql: str, dialects: list[str], schemas: dict) -> Any:
    """Validate ``sql`` against *schemas*, returning a query object.

    Pure validation step: no graph writes happen here. Callers MUST run
    this before any mutating call (``_detach_existing_sql_edges``,
    ``_persist_analysis_with_sql``) so a parse failure can't leave the
    graph in a half-updated state — e.g. ``update_custom_analysis`` used
    to detach the old ``HAS_SQL`` edge first, then parse the new SQL, so
    a 422 would orphan the ``CustomAnalysis`` from any ``Sql`` node.

    ``parse_query_single`` can fail in two ways:

    * raise (sqlglot syntax error, unsupported dialect construct, ...) —
      mirrored from ``SQLValidationAgent._sql_parse_validation`` in
      NeMo-Retriever, which wraps the same call in ``try/except`` and
      returns ``{"error": str(error)}``;
    * return ``None`` when the SQL parses but doesn't resolve to any
      table the graph already knows about (typos, missing ingestion,
      ...).

    ``enrich_graph`` lets the raise propagate and only warns on
    ``None``; the API path can't do either — a 500 leaks the parser's
    internals to the UI, and a silently dropped write would leave the
    UI thinking the analysis was saved. Both cases are converted to
    :class:`CustomAnalysisSqlError` so the caller gets a 422 with a
    message it can render.

    Note: this leaves a gap NeMo-Retriever is expected to close
    upstream — sqlglot silently parses garbage like ``"fghcghv"`` as
    a bare ``Column`` expression, so ``parse_query_single`` returns
    ``None`` and the 422 reads "doesn't reference any table" rather
    than "not valid SQL". Once the parser surfaces that distinction
    we can map it to a clearer message here without changing the API.
    """
    try:
        query_obj = parse_query_single(sql=sql, dialects=dialects, schemas=schemas)
    except Exception as exc:
        # `parse_query_single` -> sqlglot can raise a variety of
        # exception types for syntax / dialect issues; NeMo-Retriever's
        # own validation agent uses the same broad `except Exception`.
        raise CustomAnalysisSqlError(
            f"SQL parse error: {exc}",
        ) from exc

    if query_obj is None:
        raise CustomAnalysisSqlError(
            "SQL doesn't reference any table known to the catalog; "
            "ingest the schema first or check the query",
        )

    return query_obj


def _find_analysis_by_name(
    name: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    """Return ``{id, name}`` of an analysis already using ``name``, or None.

    ``exclude_id`` skips one specific analysis from the check — used by
    :func:`update_custom_analysis` so re-saving the same record with an
    unchanged name doesn't collide with itself; :func:`create_custom_analysis`
    passes ``None`` since a new record can't legitimately collide with
    itself. ``LIMIT 1`` because we only need to surface one conflicting
    analysis to the UI.

    Lives here rather than as part of NeMo-Retriever because that
    library only exposes by-id lookups (``get_item_by_id``); there is
    no helper for "by-name, excluding one id".
    """
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (other:{Labels.CUSTOM_ANALYSIS} {{name: $name}})
        WHERE $exclude_id IS NULL OR other.id <> $exclude_id
        RETURN other.id AS id, other.name AS name
        LIMIT 1
        """,
        {"name": name, "exclude_id": exclude_id},
    )
    if not rows:
        return None
    return {"id": rows[0]["id"], "name": rows[0]["name"]}


def _find_analysis_by_sql(
    sql: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    """Return ``{id, name}`` of an analysis already linked to ``sql``, or None.

    ``Sql`` nodes are matched by ``sql_full_query`` (same key
    :func:`_persist_analysis_with_sql` writes), so a non-empty result
    means the same SQL text is already attached to another
    ``CustomAnalysis``. ``exclude_id`` skips one specific analysis from
    the check — used so an update keeping the same SQL on the same
    record doesn't collide with itself. ``LIMIT 1`` because we only
    need to surface one conflicting analysis to the UI.
    """
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (other:{Labels.CUSTOM_ANALYSIS})
              -[:{Edges.HAS_SQL}]->(:{Labels.SQL} {{sql_full_query: $sql}})
        WHERE $exclude_id IS NULL OR other.id <> $exclude_id
        RETURN other.id AS id, other.name AS name
        LIMIT 1
        """,
        {"sql": sql, "exclude_id": exclude_id},
    )
    if not rows:
        return None
    return {"id": rows[0]["id"], "name": rows[0]["name"]}


def _detach_existing_sql_edges(analysis_id: str) -> None:
    """Drop every ``HAS_SQL`` edge leaving the given CustomAnalysis.

    The pipeline's ``add_query`` uses ``apoc.merge.relationship.eager`` with
    no identity props for ``HAS_SQL``, so without this cleanup re-pointing
    an analysis at a different ``Sql`` would *add* a second edge instead of
    replacing the old one — leaving the previous ``Sql`` dangling under the
    same analysis. NeMo-Retriever has no public helper for edge deletion,
    hence the local Cypher.
    """
    get_neo4j_conn().query_write(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: $analysis_id}})
              -[r:{Edges.HAS_SQL}]->(:{Labels.SQL})
        DELETE r
        """,
        {"analysis_id": analysis_id},
    )


def _persist_analysis_with_sql(
    analysis_node: Neo4jNode,
    sql: str,
    query_obj: Any,
) -> dict[str, Any]:
    """Link a pre-parsed ``query_obj`` to ``analysis_node`` via HAS_SQL.

    Mirrors the inner loop of
    :func:`nemo_retriever.tabular_data.dev_tools.enrich_graph.add_custom_analyses`
    — that helper is the canonical NeMo-Retriever pattern for "ingest a
    CustomAnalysis with its SQL" (it produces the right ``Sql`` node
    properties: ``nodes_count``, ``join_count``, ``union_count``, the
    per-month ``count_*`` keys, and the full
    ``Sql -[:SQL]-> Table/Column`` fan-out used by retrieval). We can't
    call ``add_custom_analyses`` directly because it reads JSON from disk,
    embeds into LanceDB, and doesn't return per-entry results, so we
    reproduce its match_props → append HAS_SQL edge → ``add_query``
    sequence verbatim.

    ``query_obj`` must already be the result of :func:`_validate_sql`
    — that split lets callers validate the SQL before doing any graph
    writes (so a parse failure on update doesn't strand the analysis
    without a ``HAS_SQL`` edge).

    The ``Sql`` node is matched by ``sql_full_query`` (same as
    ``add_custom_analyses``) so two analyses pointing at the same text
    reuse the same node instead of creating a duplicate.
    """
    query_obj.sql_node.match_props = {"sql_full_query": sql}

    edge_props = {Props.ANALYSIS_ID: analysis_node.get_id()}
    query_obj.edges.append((analysis_node, query_obj.sql_node, edge_props))

    add_query(query_obj.get_edges())

    props = analysis_node.get_properties()
    return {
        "id": analysis_node.get_id(),
        "name": props["name"],
        "description": props["description"],
        "sql": sql,
    }


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------


def create_custom_analysis(
    name: str,
    description: str,
    sql: str,
) -> dict[str, Any]:
    """Create a fresh ``CustomAnalysis`` linked to its ``Sql`` node.

    Strict insert: a duplicate ``name`` or ``sql`` is rejected upfront
    so the graph never ends up with two analyses sharing a natural key.
    Updating an existing analysis is the job of
    :func:`update_custom_analysis` (addressed by ``id``).

    Raises :class:`CustomAnalysisNameConflict` when ``name`` is already
    used by any ``CustomAnalysis``.

    Raises :class:`CustomAnalysisSqlConflict` when ``sql`` is already
    attached to any ``CustomAnalysis`` (see the class docstring for the
    rationale).

    Raises :class:`CustomAnalysisSqlError` when the SQL can't be resolved
    against the current catalog (see :class:`CustomAnalysisSqlError`).

    Returns ``{id, name, description, sql}`` — the same shape used by
    :func:`list_custom_analyses`.
    """
    name_conflict = _find_analysis_by_name(name, exclude_id=None)
    if name_conflict is not None:
        raise CustomAnalysisNameConflict(
            f"another CustomAnalysis already uses name {name!r} "
            f"(id={name_conflict['id']!r})",
        )

    sql_conflict = _find_analysis_by_sql(sql, exclude_id=None)
    if sql_conflict is not None:
        raise CustomAnalysisSqlConflict(
            f"this SQL is already used by CustomAnalysis {sql_conflict['name']!r} "
            f"(id={sql_conflict['id']!r})",
        )

    query_obj = _validate_sql(sql, _get_dialects(), _get_schemas())
    analysis_node = Neo4jNode(
        name=name,
        label=Labels.CUSTOM_ANALYSIS,
        props={"name": name, "description": description},
        match_props={"name": name},
    )

    row = _persist_analysis_with_sql(analysis_node, sql, query_obj)

    from gsf.utils import get_embed_params
    from gsf.vdb import get_semantic_vdb

    vdb = get_semantic_vdb()
    _embed_custom_analyses(
        embed_params=get_embed_params(),
        vdb=vdb,
        analysis_id=row["id"],
    )

    return row


def update_custom_analysis(
    analysis_id: str,
    name: str,
    description: str,
    sql: str,
) -> dict[str, Any] | None:
    """Replace name/description/sql of an existing ``CustomAnalysis`` by id.

    Returns the updated row (same shape as :func:`list_custom_analyses`),
    or ``None`` when no ``CustomAnalysis`` with ``analysis_id`` exists.

    Raises :class:`CustomAnalysisNameConflict` when ``name`` is already
    used by a *different* ``CustomAnalysis`` — renaming into that slot
    would silently merge two analyses on the next create-by-name and is
    rejected.

    Raises :class:`CustomAnalysisSqlConflict` when ``sql`` is already
    linked to a *different* ``CustomAnalysis`` — SQL text is unique per
    analysis. Re-saving the same SQL on the analysis being updated is
    explicitly allowed (the owning record is excluded from the check).

    Raises :class:`CustomAnalysisSqlError` when the SQL can't be resolved
    against the current catalog.
    """
    # NeMo-Retriever exposes `get_item_by_id` for this kind of probe, but
    # in the installed version it calls `Neo4jConnection.query_read_only`,
    # which doesn't exist (the real method is `query_read`), so the helper
    # raises `AttributeError` on every call. Until that's fixed upstream
    # we run the same one-row existence check directly via `query_read`.
    existing = get_neo4j_conn().query_read(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: $analysis_id}})
        RETURN ca.id AS id
        LIMIT 1
        """,
        {"analysis_id": analysis_id},
    )
    if not existing:
        return None

    name_conflict = _find_analysis_by_name(name, exclude_id=analysis_id)
    if name_conflict is not None:
        raise CustomAnalysisNameConflict(
            f"another CustomAnalysis already uses name {name!r} "
            f"(id={name_conflict['id']!r})",
        )

    sql_conflict = _find_analysis_by_sql(sql, exclude_id=analysis_id)
    if sql_conflict is not None:
        raise CustomAnalysisSqlConflict(
            f"this SQL is already used by CustomAnalysis {sql_conflict['name']!r} "
            f"(id={sql_conflict['id']!r})",
        )

    # Validate the new SQL BEFORE touching the graph: a parse failure
    # here used to land after `_detach_existing_sql_edges`, orphaning
    # the analysis from any `Sql` node and leaving the read endpoint
    # to return `sql: null` for an otherwise valid-looking record.
    query_obj = _validate_sql(sql, _get_dialects(), _get_schemas())

    _detach_existing_sql_edges(analysis_id)

    analysis_node = Neo4jNode(
        name=name,
        label=Labels.CUSTOM_ANALYSIS,
        props={"name": name, "description": description},
        match_props={"id": analysis_id},
        existing_id=analysis_id,
        override_existing_props={"name": name, "description": description},
    )

    row = _persist_analysis_with_sql(analysis_node, sql, query_obj)

    # `IngestVdbOperator` appends, so re-embedding without first dropping
    # the stale row would leave two VDB entries for this analysis_id and
    # double-weight it at retrieval time.
    from gsf.utils import get_embed_params
    from gsf.vdb import get_semantic_vdb

    vdb = get_semantic_vdb()
    vdb.delete_by_id(analysis_id)
    _embed_custom_analyses(
        embed_params=get_embed_params(),
        vdb=vdb,
        analysis_id=analysis_id,
    )

    return row


def delete_custom_analysis(analysis_id: str) -> dict[str, str] | None:
    """Remove a CustomAnalysis, its Sql node, and its VDB embedding.

    Returns ``{"id": analysis_id}`` on success, or ``None`` when no
    ``CustomAnalysis`` with ``analysis_id`` exists (caller maps to 404).

    Cypher uses ``DETACH DELETE ca, sql`` so both nodes and every edge
    they participate in vanish in one statement — including the
    ``Sql -[:SQL]-> Table/Column`` edges that retrieval walks.
    ``Table`` / ``Column`` nodes themselves are kept (they belong to the
    schema, not the analysis). Sharing of an ``Sql`` node across
    analyses is already prevented at write time by
    :class:`CustomAnalysisSqlConflict` (see
    :func:`_find_analysis_by_sql`), so this never strands another
    analysis.

    Graph delete happens before the VDB delete so a Neo4j failure
    leaves both stores pointing at the same (still-present) record;
    if Neo4j succeeds and the VDB delete throws, the orphan VDB row
    will be cleaned up on the next ingest of *database_name* with ``reset=True``
    (see :func:`gsf.vdb.get_semantic_vdb`).
    """
    existing = get_neo4j_conn().query_read(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: $analysis_id}})
        RETURN ca.id AS id
        LIMIT 1
        """,
        {"analysis_id": analysis_id},
    )
    if not existing:
        return None

    get_neo4j_conn().query_write(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: $analysis_id}})
              -[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        DETACH DELETE ca, sql
        """,
        {"analysis_id": analysis_id},
    )

    from gsf.vdb import get_semantic_vdb

    get_semantic_vdb().delete_by_id(analysis_id)

    return {"id": analysis_id}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed_custom_analyses(
    embed_params: "EmbedParams",
    vdb: "VDB",
    analysis_id: str | None = None,
    database_name: str | None = None,
) -> None:
    """Fetch ``CustomAnalysis`` docs from Neo4j, embed them, and append to *vdb*.

    Shapes every matched ``CustomAnalysis`` into the same 5-column
    DataFrame the main pipeline produces, then uses the same embedder
    (:func:`nemo_retriever.text_embed.runtime.embed_text_main_text_embed`) and
    writes the embedded rows through *vdb* in append mode, so existing
    ``Table``/``Column`` rows are preserved.

    ``CustomAnalysis`` is treated as a single global pool — the graph
    isn't sliced per database here because nothing downstream reads
    the row back by database name: ``PostgresVDB.delete_by_database``
    explicitly excludes the ``CustomAnalysis`` label (see
    ``gsf/vdb/postgres.py``), and retrieval matches on label + vector
    similarity, not on database.

    When *analysis_id* is given, the Cypher match is narrowed to that
    one ``CustomAnalysis`` — used by the create/update paths to embed
    only the row just written instead of re-embedding every analysis.
    ``None`` (the default) embeds every ``CustomAnalysis`` linked to
    an ``Sql`` node, used for full ingests. Because writes are
    append-only (see operator note above), passing *analysis_id* for
    an analysis already present in the VDB would duplicate its row;
    callers updating an existing analysis must delete the stale VDB
    row first.
    """
    import pandas as pd

    query = f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS})-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        WHERE $analysis_id IS NULL OR ca.id = $analysis_id
        WITH DISTINCT ca, sql,
             CASE
                 WHEN ca.description IS NOT NULL AND trim(toString(ca.description)) <> ''
                 THEN ca.description
                 ELSE ''
             END AS desc,
             CASE
                 WHEN sql.sql_full_query IS NOT NULL
                 THEN ', sql: ' + sql.sql_full_query
                 ELSE ''
             END AS sql_text
        RETURN collect({{
            text: 'custom_analysis: ' + ca.name +
                  CASE WHEN desc <> '' THEN ', description: ' + desc ELSE '' END +
                  sql_text,
            name: ca.name,
            label: labels(ca)[0],
            id: ca.id
        }}) AS docs
    """
    result = get_neo4j_conn().query_read(
        query,
        parameters={"analysis_id": analysis_id},
    )
    docs = result[0].get("docs") if result else None
    if not docs:
        logger.info(
            "No CustomAnalysis rows found for analysis_id=%r; skipping VDB upsert.",
            analysis_id,
        )
        return

    rows = []
    for item in docs:
        node_id = item.get("id")
        path = f"neo4j:{node_id}" if node_id is not None else "neo4j:unknown"
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
    embedded = embed_text_main_text_embed(
        df,
        model_name=embed_params.model_name,
        embed_invoke_url=embed_params.embed_invoke_url,
        api_key=embed_params.api_key,
        embed_modality=embed_params.embed_modality,
    )

    # nemo_retriever's embed runtime swallows HTTP failures (e.g. 502 Bad
    # Gateway from integrate.api.nvidia.com) and returns rows without an
    # `embedding` column. Detect that here so we don't silently report success
    # while PostgresVDB.write_to_index skips every row.
    with_embeddings = [
        row
        for row in embedded.to_dict(orient="records")
        if (row.get("metadata") or {}).get("embedding")
    ]
    if not with_embeddings:
        raise RuntimeError(
            f"Embedding step produced 0/{len(embedded)} CustomAnalysis rows "
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
