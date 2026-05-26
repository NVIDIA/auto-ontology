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

from typing import Any

from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import (
    Edges,
    Labels,
    Props,
)
from nemo_retriever.tabular_data.ingestion.services.queries import parse_query_single
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.tabular_data.retrieval.data_access.graph_schemas import (
    get_all_schemas_ids,
    get_schemas_by_ids,
)

from gsf.server.chat.helpers import get_connector


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


def _get_dialect_and_schemas() -> tuple[str, dict]:
    """Resolve the ``(dialect, schemas)`` pair needed by ``parse_query_single``.

    ``dialect`` is read from the active connector (currently always the
    one configured by ``CONNECTION_STRINGS``); ``schemas`` is the catalog
    snapshot the SQL parser uses to resolve table/column references.
    """
    connector = get_connector()
    if connector is None:
        raise CustomAnalysisSqlError(
            "no source-DB connector configured; set CONNECTION_STRINGS",
        )
    schemas_ids = get_all_schemas_ids()
    schemas = get_schemas_by_ids(schemas_ids)
    return connector.dialect, schemas


def _parse_sql_or_raise(sql: str) -> Any:
    """Validate ``sql`` against the current catalog, returning a query object.

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
    dialect, schemas = _get_dialect_and_schemas()

    try:
        query_obj = parse_query_single(sql=sql, dialect=dialect, schemas=schemas)
    except Exception as exc:
        # `parse_query_single` -> sqlglot can raise a variety of
        # exception types for syntax / dialect issues; NeMo-Retriever's
        # own validation agent uses the same broad `except Exception`.
        raise CustomAnalysisSqlError(
            f"SQL parse error (dialect={dialect!r}): {exc}",
        ) from exc

    if query_obj is None:
        raise CustomAnalysisSqlError(
            "SQL doesn't reference any table known to the catalog "
            f"(dialect={dialect!r}); ingest the schema first or check the query",
        )

    return query_obj


def _find_analysis_with_name(
    name: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    """Return ``{id, name}`` of an analysis already using ``name``, or None.

    ``exclude_id`` skips one specific analysis from the check — used by
    :func:`update_custom_analysis` so re-saving the same record with an
    unchanged name doesn't collide with itself; :func:`create_custom_analysis`
    passes ``None`` since a new record can't legitimately collide with
    itself. ``LIMIT 1`` because we only need to surface one conflicting
    owner to the UI.

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


def _find_analysis_owning_sql(
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
    need to surface one conflicting owner to the UI.
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

    ``query_obj`` must already be the result of :func:`_parse_sql_or_raise`
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
    name_owner = _find_analysis_with_name(name, exclude_id=None)
    if name_owner is not None:
        raise CustomAnalysisNameConflict(
            f"another CustomAnalysis already uses name {name!r} "
            f"(id={name_owner['id']!r})",
        )

    sql_owner = _find_analysis_owning_sql(sql, exclude_id=None)
    if sql_owner is not None:
        raise CustomAnalysisSqlConflict(
            f"this SQL is already used by CustomAnalysis {sql_owner['name']!r} "
            f"(id={sql_owner['id']!r})",
        )

    query_obj = _parse_sql_or_raise(sql)

    analysis_node = Neo4jNode(
        name=name,
        label=Labels.CUSTOM_ANALYSIS,
        props={"name": name, "description": description},
        match_props={"name": name},
    )

    row = _persist_analysis_with_sql(analysis_node, sql, query_obj)

    from dev_tools.evaluation.enrich_graph import _embed_custom_analyses

    from gsf.ingestion_service.ingest import EMBED_PARAMS
    from gsf.vdb import get_vdb

    connector = get_connector()

    _embed_custom_analyses(
        database_name=connector.database_name,
        embed_params=EMBED_PARAMS,
        vdb=get_vdb(),
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

    name_owner = _find_analysis_with_name(name, exclude_id=analysis_id)
    if name_owner is not None:
        raise CustomAnalysisNameConflict(
            f"another CustomAnalysis already uses name {name!r} "
            f"(id={name_owner['id']!r})",
        )

    sql_owner = _find_analysis_owning_sql(sql, exclude_id=analysis_id)
    if sql_owner is not None:
        raise CustomAnalysisSqlConflict(
            f"this SQL is already used by CustomAnalysis {sql_owner['name']!r} "
            f"(id={sql_owner['id']!r})",
        )

    # Validate the new SQL BEFORE touching the graph: a parse failure
    # here used to land after `_detach_existing_sql_edges`, orphaning
    # the analysis from any `Sql` node and leaving the read endpoint
    # to return `sql: null` for an otherwise valid-looking record.
    query_obj = _parse_sql_or_raise(sql)

    _detach_existing_sql_edges(analysis_id)

    analysis_node = Neo4jNode(
        name=name,
        label=Labels.CUSTOM_ANALYSIS,
        props={"name": name, "description": description},
        match_props={"id": analysis_id},
        existing_id=analysis_id,
        override_existing_props={"name": name, "description": description},
    )

    return _persist_analysis_with_sql(analysis_node, sql, query_obj)
