# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j read/write for Term nodes.

ColumnAttribute and SemanticFK operations live in gsf/dal/attributes.py.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.dal.attributes import fetch_column_attribute_columns_map
from gsf.dal.users import resolve_accessible_catalog_ids, resolve_table_filter
from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_REPRESENTS,
    REL_SEMANTIC_FK,
    SEMANTIC_SOURCE,
)
from gsf.server.zones.constants import (
    LABEL_ZONE_DISABLED,
    REL_ZONE_OF,
    ZONE_LABEL_PATTERN,
)
from gsf.utils.sample_values import parse_sample_values

logger = logging.getLogger(__name__)

# Shared RETURN projection for ColumnAttribute rows — keep fetch_column_attributes
# and fetch_column_attributes_by_term_id in sync when adding/removing fields.
# Both queries pair this with an `OPTIONAL MATCH (col:Column)-[:REL_HAS_ATTRIBUTE]->(attr)`
# clause so `col.sample_values` (profiled at ingestion time) is available.
_COLUMN_ATTRIBUTE_FIELDS = """attr.id            AS id,
               attr.name          AS name,
               attr.description   AS description,
               attr.term_name     AS term_name,
               attr.source_column AS source_column,
               attr.datatype      AS datatype,
               attr.table_id      AS table_id,
               col.sample_values  AS sample_values,
               coalesce(attr.certified, false) AS certified"""


def _with_parsed_sample_values(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize ``sample_values`` on each row via ``parse_sample_values``."""
    for row in rows:
        row["sample_values"] = parse_sample_values(row.get("sample_values"))
    return rows


# All-or-nothing table visibility for a Term: a Term represented by any table
# outside the accessible set is hidden entirely. Expects a `$table_ids` param
# and binds `term`. See fetch_all_terms for the rationale.
_TERM_TABLE_SCOPE_CONDITION = (
    f"NOT EXISTS {{"
    f" (other:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)"
    f" WHERE NOT other.id IN $table_ids"
    f" }}"
)

# Maps the `flags` list built by _certification_flags_clause onto the
# three-state status the frontend renders.
_CERTIFICATION_CASE = """CASE
                   WHEN size([f IN flags WHERE f]) = size(flags) THEN 'certified'
                   WHEN size([f IN flags WHERE f]) = 0           THEN 'pending'
                   ELSE 'partial'
               END"""


def _certification_flags_clause(*, zone_scoped: bool) -> str:
    """Cypher fragment building a Term's aggregate certification ``flags`` list.

    This is the single definition of a Term's aggregate certification —
    ``fetch_all_terms`` (list cards), ``get_full_term_by_id`` (detail page) and
    ``get_term_certification`` (certification writes) all project it through
    :data:`_CERTIFICATION_CASE`, so those three surfaces can never disagree.

    The flags are the Term's own name/description booleans plus one boolean per
    column and sql attribute. Because the Term flags are always present the
    list is never empty, and a Term with no attributes is decided by its own
    two flags alone.

    When *zone_scoped*, each attribute comprehension applies the same
    visibility rule as the endpoint that lists those attributes — plain
    ``table_id`` membership for ColumnAttribute (see
    ``fetch_column_attribute_counts``) and all-or-nothing over the tables a
    SqlAttribute's SQL touches (see ``_sql_attr_zone_filter``) — so the
    aggregate never reflects an attribute the caller isn't allowed to see.
    Expects `term` in scope and a ``$table_ids`` param.
    """
    if zone_scoped:
        column_where = "WHERE ca.table_id IN $table_ids "
        sql_where = (
            f"WHERE NOT EXISTS {{"
            f" (sa)-[:{Edges.HAS_SQL}]->(:{Labels.SQL})"
            f"-[:{Edges.SQL}]->(tbl:{Labels.TABLE})"
            f" WHERE NOT tbl.id IN $table_ids"
            f" }} "
        )
    else:
        column_where = ""
        sql_where = ""
    return f"""[coalesce(term.name_certified, false),
              coalesce(term.description_certified, false)]
             + [(term)<-[:{REL_PROPERTY_OF}]-(ca:{LABEL_COLUMN_ATTRIBUTE})
                {column_where}| coalesce(ca.certified, false)]
             + [(term)<-[:{REL_PROPERTY_OF}]-(sa:{LABEL_SQL_ATTRIBUTE})
                {sql_where}| coalesce(sa.certified, false)] AS flags"""


def get_term_certification(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> str | None:
    """Return one Term's aggregate certification status, or None when missing.

    Certification writes call this to hand the caller a freshly recomputed
    aggregate, so the Terms list card can be updated without the frontend
    duplicating the rollup rule (or refetching the whole list).
    """
    if zone_ids is None:
        term_filter = ""
        params: dict[str, Any] = {"term_id": term_id}
    else:
        term_filter = f"WHERE {_TERM_TABLE_SCOPE_CONDITION}"
        params = {
            "term_id": term_id,
            "table_ids": list(resolve_accessible_catalog_ids(zone_ids)["table_ids"]),
        }

    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        {term_filter}
        WITH term, {_certification_flags_clause(zone_scoped=zone_ids is not None)}
        RETURN {_CERTIFICATION_CASE} AS certification
        LIMIT 1
        """,
        params,
    )
    return rows[0]["certification"] if rows else None


def semantic_layer_calculated() -> bool:
    """True if at least one semantic Term exists in the graph."""
    rows = get_neo4j_conn().query_read(
        f"MATCH (term:{LABEL_TERM} {{source: $source}}) RETURN term.id AS id LIMIT 1",
        {"source": SEMANTIC_SOURCE},
    )
    return bool(rows)


def table_has_term(table_id: str) -> bool:
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        RETURN term.name AS name LIMIT 1
        """,
        {"table_id": table_id, "source": SEMANTIC_SOURCE},
    )
    return bool(rows)


def get_term_for_table(table_id: str) -> str | None:
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        RETURN term.name AS name LIMIT 1
        """,
        {"table_id": table_id, "source": SEMANTIC_SOURCE},
    )
    return rows[0]["name"] if rows else None


def get_term_id_for_table(table_id: str) -> str | None:
    """Return the Term id linked via REPRESENTS, or None."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        RETURN term.id AS id LIMIT 1
        """,
        {"table_id": table_id, "source": SEMANTIC_SOURCE},
    )
    return rows[0]["id"] if rows else None


def get_term_record_for_table(table_id: str) -> dict[str, str] | None:
    """Return ``{id, name, description}`` for the Term that REPRESENTS *table_id*.

    Returns ``None`` when the table has no REPRESENTS Term.
    """
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        RETURN term.id AS id,
               term.name AS name,
               coalesce(term.description, '') AS description
        LIMIT 1
        """,
        {"table_id": table_id, "source": SEMANTIC_SOURCE},
    )
    return dict(rows[0]) if rows else None


def get_slim_term_by_id(term_id: str) -> dict[str, str] | None:
    """Return ``{id, name}`` of a Term, or None."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{LABEL_TERM} {{id: $term_id}})
        RETURN t.id AS id, t.name AS name LIMIT 1
        """,
        {"term_id": term_id},
    )
    return rows[0] if rows else None


def update_term(
    term_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    name_certified: bool | None = None,
    description_certified: bool | None = None,
) -> dict[str, Any] | None:
    """Update a Term and return old/new values, or None when missing."""
    rows = get_neo4j_conn().query_write(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        WITH term, term.name AS old_name
        SET term.name = coalesce($name, term.name),
            term.description = coalesce($description, term.description),
            term.name_certified = coalesce($name_certified, term.name_certified),
            term.description_certified =
                coalesce($description_certified, term.description_certified)
        WITH term, old_name
        OPTIONAL MATCH (attr:{LABEL_COLUMN_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->(term)
        SET attr.term_name = term.name
        RETURN term.id AS id,
               old_name,
               term.name AS name,
               term.description AS description,
               coalesce(term.name_certified, false)        AS name_certified,
               coalesce(term.description_certified, false) AS description_certified
        """,
        {
            "term_id": term_id,
            "name": name,
            "description": description,
            "name_certified": name_certified,
            "description_certified": description_certified,
        },
    )
    return dict(rows[0]) if rows else None


def merge_term(
    name: str,
    description: str,
    table_id: str,
    synonyms: list[str] | None = None,
) -> str | None:
    """Merge the Term node and return its persistent ``id`` (UUID)."""
    rows = get_neo4j_conn().query_write(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})
        MERGE (term:{LABEL_TERM} {{name: $name, source: $source}})
        ON CREATE SET term.id = randomUUID()
        SET term.description = $description,
            term.synonyms = $synonyms
        MERGE (t)-[:{REL_REPRESENTS}]->(term)
        RETURN term.id AS id
        """,
        {
            "table_id": table_id,
            "name": name,
            "description": description,
            "synonyms": synonyms or [],
            "source": SEMANTIC_SOURCE,
        },
    )
    return rows[0]["id"] if rows else None


def fetch_term_synonyms(attr_ids: list[str]) -> dict[str, list[str]]:
    """Fetch synonyms for Terms connected to the given ColumnAttribute IDs.

    Returns a mapping of term_name -> list[synonym].
    """
    if not attr_ids:
        return {}
    query = """
    UNWIND $attr_ids AS attr_id
    MATCH (attr:ColumnAttribute {id: attr_id})-[:PROPERTY_OF]->(term:Term)
    WHERE term.synonyms IS NOT NULL AND size(term.synonyms) > 0
    RETURN DISTINCT term.name AS term_name, term.synonyms AS synonyms
    """
    try:
        rows = get_neo4j_conn().query_read(query, {"attr_ids": attr_ids})
    except Exception:
        logger.warning("fetch_term_synonyms: Neo4j query failed", exc_info=True)
        return {}
    result: dict[str, list[str]] = {}
    for row in rows:
        name = row.get("term_name")
        syns = row.get("synonyms") or []
        if name and syns:
            result[name] = [s for s in syns if s]
    return result


def fetch_all_terms(
    zone_ids: list[str] | None = None,
    search: str | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Scan semantic Term nodes in Neo4j.

    *zone_ids* is a hard authorization boundary (the zones the requesting
    user has been granted access to), not a relevance filter.  A Term can be
    represented by more than one Table (see ``merge_term``), so when
    *zone_ids* is supplied a term is only returned when **every** table that
    represents it is reachable through those zones — a term that also
    represents an out-of-zone table is excluded entirely, mirroring the
    all-or-nothing rule used for CustomAnalysis
    (see ``gsf.dal.custom_analyses.list_custom_analyses``).  Pass ``None``
    (or omit) to return all terms (admin / internal callers).

    *search* is a case-insensitive substring filter applied to the term's
    name. Pass ``None`` (or omit, or an empty string) to skip filtering.

    Each term row additionally carries a ``zones`` list, resolved via
    ``fetch_term_zones_map`` — the same attribute → column → table → zone
    path used by ``get_full_term_by_id`` — so callers (the ``/terms`` list
    and the Exploration graph) render Zone chips from one response instead
    of a second per-page request.

    Each row also carries the aggregate ``certification`` status (see
    ``_certification_flags_clause``), zone-scoped to the same boundary, so
    the Terms list renders a per-card badge without fetching attributes.

    Pass a pre-resolved *data_ids_by_zone* (see
    ``resolve_accessible_catalog_ids``) when the caller already resolved
    *zone_ids* for this request — e.g.
    ``gsf.dal.exploration.fetch_semantic_exploration_graph`` — to skip the
    repeat Neo4j round trip this function would otherwise make on its own.
    """
    conn = get_neo4j_conn()
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)

    term_params: dict[str, Any] = {"source": SEMANTIC_SOURCE}
    term_conditions: list[str] = []

    if data_ids_by_zone is not None:
        term_params["table_ids"] = list(data_ids_by_zone["table_ids"])
        term_conditions.append(_TERM_TABLE_SCOPE_CONDITION)

    if search:
        term_params["search"] = search.strip().lower()
        term_conditions.append("toLower(term.name) CONTAINS $search")

    term_filter = f"WHERE {' AND '.join(term_conditions)}" if term_conditions else ""

    terms = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        {term_filter}
        OPTIONAL MATCH (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
        WITH term, collect(DISTINCT sch.name) AS schemas
        WITH term, schemas,
             {_certification_flags_clause(zone_scoped=data_ids_by_zone is not None)}
        RETURN term.name AS name, term.description AS description,
               term.synonyms AS synonyms, term.id AS id,
               schemas AS schema_names,
               coalesce(term.name_certified, false)        AS name_certified,
               coalesce(term.description_certified, false) AS description_certified,
               {_CERTIFICATION_CASE} AS certification
        """,
        term_params,
    )
    zones_by_term = fetch_term_zones_map(zone_ids)
    return [{**dict(row), "zones": zones_by_term.get(row["id"], [])} for row in terms]


def fetch_all_terms_and_attributes(
    zone_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan every semantic Term and ColumnAttribute node in Neo4j.

    Used only by ``embed_all_semantic_nodes`` for a one-shot bulk re-embed of
    the whole graph: unlike ``fetch_all_terms``, this also runs a second,
    global Neo4j scan for every ColumnAttribute (needed to rebuild each
    Term's embedding text). Callers that only need Term rows — the
    ``/terms`` list, the Exploration semantic graph — should call
    ``fetch_all_terms`` instead and skip that second query entirely. Pass
    ``None`` (or omit) to scan the whole graph (admin / internal callers).
    """
    conn = get_neo4j_conn()
    terms = fetch_all_terms(zone_ids=zone_ids)

    attr_filter, attr_params = resolve_table_filter(
        zone_ids,
        "t.id",
        extra_params={"source": SEMANTIC_SOURCE},
    )
    attrs = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        {attr_filter}
        OPTIONAL MATCH (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
        RETURN attr.name AS name,
               attr.description AS description,
               attr.term_name AS term_name,
               attr.source_column AS source_column,
               col.name AS column_name,
               col.sample_values AS sample_values,
               attr.id AS id,
               sch.name AS schema_name
        """,
        attr_params,
    )
    return terms, attrs


def get_full_term_by_id(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Return a single Term node by its id with table count and zones, or None.

    *zone_ids* is a hard authorization boundary, not a relevance filter. A
    Term can be represented by more than one Table (see ``merge_term``), so
    when *zone_ids* is supplied the term is only returned when **every**
    table that represents it is reachable through those zones — matching the
    all-or-nothing rule used by ``fetch_all_terms`` for the
    ``/terms`` list, so a viewer can't bypass list-level zone scoping by
    requesting a term directly by id.  Returns ``None`` (→ 404) when the
    check fails.  Pass ``None`` to skip the check (admin / internal callers).

    ``zones`` is resolved via **both** attribute paths a term can carry
    content through: ColumnAttribute → Column → Table, and SqlAttribute →
    Sql → Table (a term's SqlAttribute can reference tables no
    ColumnAttribute touches, e.g. a custom cross-table formula) — a term
    participates in a zone when either path reaches a table belonging to
    it. When *zone_ids* is supplied, the returned ``zones`` are additionally
    restricted to that set, so a viewer never sees zone names/colors they
    don't have access to — disabled zones are excluded outright in that
    case, so a viewer never sees a disabled zone chip even if its id ended
    up in *zone_ids*. Pass ``None`` to skip the check and include disabled
    zones (with ``enabled: False``) so admins can see and manage them.
    """
    conn = get_neo4j_conn()
    if zone_ids is None:
        term_filter = ""
        term_params: dict[str, Any] = {"term_id": term_id}
    else:
        table_ids = list(resolve_accessible_catalog_ids(zone_ids)["table_ids"])
        term_filter = f"WHERE {_TERM_TABLE_SCOPE_CONDITION}"
        term_params = {"term_id": term_id, "table_ids": table_ids}

    rows = conn.query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        {term_filter}
        OPTIONAL MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)
        WITH term, collect(DISTINCT CASE WHEN t IS NULL THEN NULL ELSE {{
                 id: t.id,
                 name: t.name,
                 schema_id: sch.id,
                 db_id: db.id
             }} END) AS raw_tables
        WITH term, [tbl IN raw_tables WHERE tbl IS NOT NULL] AS tables
        WITH term, tables,
             {_certification_flags_clause(zone_scoped=zone_ids is not None)}
        RETURN term.name AS name, term.description AS description,
               term.synonyms AS synonyms, term.id AS id,
               size(tables) AS table_count,
               tables AS tables,
               coalesce(term.name_certified, false)        AS name_certified,
               coalesce(term.description_certified, false) AS description_certified,
               {_CERTIFICATION_CASE} AS certification
        LIMIT 1
        """,
        term_params,
    )
    if not rows:
        return None
    result = dict(rows[0])

    zone_filter = (
        ""
        if zone_ids is None
        else f"WHERE z.id IN $zone_ids AND NOT z:{LABEL_ZONE_DISABLED}"
    )
    zone_params: dict[str, Any] = {"term_id": term_id}
    if zone_ids is not None:
        zone_params["zone_ids"] = zone_ids

    # Each branch's zone MATCH is chained through `item` (found by walking
    # CONTAINS backwards from `t`, 0..2 hops) rather than matched independently
    # and filtered via WHERE afterwards — see fetch_table_zones_map in
    # gsf/dal/exploration.py for why the disconnected-pattern version forces a
    # cartesian product between every zone/item pair and every candidate `t`.
    zone_rows = conn.query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MATCH (term)<-[:{REL_PROPERTY_OF}]-(:{LABEL_COLUMN_ATTRIBUTE})
              <-[:{REL_HAS_ATTRIBUTE}]-(:{Labels.COLUMN})
              <-[:{Edges.CONTAINS}]-(t:{Labels.TABLE})
        MATCH (item)-[:{Edges.CONTAINS}*0..2]->(t)
        MATCH (z:{ZONE_LABEL_PATTERN})-[:{REL_ZONE_OF}]->(item)
        {zone_filter}
        RETURN DISTINCT z.id    AS id,
                        z.name  AS name,
                        z.color AS color,
                        NOT z:{LABEL_ZONE_DISABLED} AS enabled

        UNION

        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MATCH (term)<-[:{REL_PROPERTY_OF}]-(:{LABEL_SQL_ATTRIBUTE})
              -[:{Edges.HAS_SQL}]->(:{Labels.SQL})
              -[:{Edges.SQL}]->(t:{Labels.TABLE})
        MATCH (item)-[:{Edges.CONTAINS}*0..2]->(t)
        MATCH (z:{ZONE_LABEL_PATTERN})-[:{REL_ZONE_OF}]->(item)
        {zone_filter}
        RETURN DISTINCT z.id    AS id,
                        z.name  AS name,
                        z.color AS color,
                        NOT z:{LABEL_ZONE_DISABLED} AS enabled
        """,
        zone_params,
    )
    result["zones"] = sorted((dict(r) for r in zone_rows), key=lambda z: z["name"])
    return result


def fetch_term_zones_map(
    zone_ids: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{term_id: [zone, ...]}`` for every Term reachable via *zone_ids*.

    Mirrors the per-term zone resolution in ``get_full_term_by_id`` — the
    union of the ColumnAttribute (attribute → column → table → zone) and
    SqlAttribute (attribute → sql → table → zone) paths — but scans every
    Term in one query, so the Exploration graph and the Terms list can
    render Zone chips without a per-node request. When *zone_ids* is
    supplied, the returned zone names/colors are restricted to that set and
    disabled zones are excluded outright — a viewer never sees a disabled
    zone chip, even if its id ended up in *zone_ids*. Pass ``None`` to
    return zones for every term, disabled included, so admins can see and
    manage them (admin / internal callers).
    """
    zone_filter = (
        ""
        if zone_ids is None
        else f"WHERE z.id IN $zone_ids AND NOT z:{LABEL_ZONE_DISABLED}"
    )
    params: dict[str, Any] = {}
    if zone_ids is not None:
        params["zone_ids"] = zone_ids

    # See the comment in get_full_term_by_id: chaining every MATCH through
    # `item` (rather than matching `t` and zone/item independently and
    # filtering via WHERE) avoids a cartesian product — this scans every
    # Term in the graph, so it matters even more here than there.
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (term:{LABEL_TERM})<-[:{REL_PROPERTY_OF}]-(:{LABEL_COLUMN_ATTRIBUTE})
              <-[:{REL_HAS_ATTRIBUTE}]-(:{Labels.COLUMN})
              <-[:{Edges.CONTAINS}]-(t:{Labels.TABLE})
        MATCH (item)-[:{Edges.CONTAINS}*0..2]->(t)
        MATCH (z:{ZONE_LABEL_PATTERN})-[:{REL_ZONE_OF}]->(item)
        {zone_filter}
        RETURN DISTINCT term.id AS term_id,
                        z.id     AS id,
                        z.name   AS name,
                        z.color  AS color,
                        NOT z:{LABEL_ZONE_DISABLED} AS enabled

        UNION

        MATCH (term:{LABEL_TERM})<-[:{REL_PROPERTY_OF}]-(:{LABEL_SQL_ATTRIBUTE})
              -[:{Edges.HAS_SQL}]->(:{Labels.SQL})
              -[:{Edges.SQL}]->(t:{Labels.TABLE})
        MATCH (item)-[:{Edges.CONTAINS}*0..2]->(t)
        MATCH (z:{ZONE_LABEL_PATTERN})-[:{REL_ZONE_OF}]->(item)
        {zone_filter}
        RETURN DISTINCT term.id AS term_id,
                        z.id     AS id,
                        z.name   AS name,
                        z.color  AS color,
                        NOT z:{LABEL_ZONE_DISABLED} AS enabled
        """,
        params,
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(row["term_id"], []).append(
            {
                "id": row["id"],
                "name": row["name"],
                "color": row["color"],
                "enabled": row["enabled"],
            }
        )
    for term_id, zones in result.items():
        result[term_id] = sorted(zones, key=lambda z: z["name"])
    return result


def fetch_table_schema_map(database_name: str) -> dict[str, str]:
    """Return ``{table_name_lower: schema_name}`` for every table in *database_name*.

    Used by the SqlAttribute suggester to qualify bare table names in
    generated SELECT statements with their canonical schema prefix.
    """
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (db:{Labels.DB} {{name: $db_name}})-[:{Edges.CONTAINS}]->
              (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
        RETURN t.name AS table_name, sch.name AS schema_name
        """,
        {"db_name": database_name},
    )
    return {
        row["table_name"].lower(): row["schema_name"]
        for row in rows
        if row.get("table_name") and row.get("schema_name")
    }


def fetch_terms_with_sqls() -> list[dict[str, Any]]:
    """Return every semantic Term with ingestion SQL from its connected tables.

    Each row contains:
      term_id, term_name, term_description,
      sqls — list of {sql_text, props} where *props* holds all Sql node
              properties (including count_monthly_YYYY_MM counters).

    Ingestion-created Sql nodes point directly to their referenced tables.
    Sql nodes created for SqlAttributes and CustomAnalyses additionally have
    an incoming HAS_SQL relationship from their owner; those are excluded to
    prevent generated semantic SQL from feeding subsequent suggestions.

    Only terms that have at least one associated ingestion query are returned.
    """
    return get_neo4j_conn().query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{source: $source}})
        MATCH (t:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)
        MATCH (sql:{Labels.SQL})-[:{Edges.SQL}]->(t)
        WHERE NOT EXISTS {{ (sql)<-[:{Edges.HAS_SQL}]-() }}
        WITH term,
             collect({{sql_text: sql.sql_full_query,
                       sql_id:   sql.id,
                       props:    properties(sql)}}) AS sqls
        RETURN term.id          AS term_id,
               term.name        AS term_name,
               term.description AS term_description,
               sqls
        """,
        {"source": SEMANTIC_SOURCE},
    )


def fetch_terms_and_attributes_for_table(
    table_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (terms, attrs) written for a single table — used for inline embedding."""
    conn = get_neo4j_conn()
    params = {"table_id": table_id, "source": SEMANTIC_SOURCE}
    terms = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        OPTIONAL MATCH (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
        RETURN term.name AS name, term.description AS description,
               term.synonyms AS synonyms, term.id AS id,
               collect(DISTINCT sch.name) AS schema_names
        """,
        params,
    )
    attrs = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        OPTIONAL MATCH (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
        RETURN attr.name AS name,
               attr.description AS description,
               attr.term_name AS term_name,
               attr.source_column AS source_column,
               col.sample_values AS sample_values,
               attr.id AS id,
               sch.name AS schema_name
        """,
        params,
    )
    return terms, attrs


def fetch_term_and_column_attributes_for_embedding(
    term_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return one Term and its ColumnAttributes for semantic VDB re-embedding."""
    conn = get_neo4j_conn()
    params = {"term_id": term_id}
    terms = conn.query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        OPTIONAL MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)
        RETURN term.name AS name,
               term.description AS description,
               term.synonyms AS synonyms,
               term.id AS id,
               head(collect(DISTINCT db.name)) AS database_name
        LIMIT 1
        """,
        params,
    )
    if not terms:
        return None, []

    attrs = conn.query_read(
        f"""
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->
              (term:{LABEL_TERM} {{id: $term_id}})
        // ColumnAttribute embeddings include sample values from the owning Column.
        OPTIONAL MATCH (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->(attr)
        RETURN attr.name AS name,
               attr.description AS description,
               attr.term_name AS term_name,
               attr.source_column AS source_column,
               col.sample_values AS sample_values,
               attr.id AS id
        """,
        params,
    )
    return dict(terms[0]), [dict(attr) for attr in attrs]


def fetch_column_attribute_embedding_contexts_by_column_id(
    column_id: str,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Return Term + ColumnAttribute contexts affected by a Column sample update."""
    conn = get_neo4j_conn()
    rows = conn.query_read(
        f"""
        MATCH (col:{Labels.COLUMN} {{id: $column_id}})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        OPTIONAL MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col)
        RETURN term.name AS term_name,
               term.description AS term_description,
               term.synonyms AS term_synonyms,
               term.id AS term_id,
               head(collect(DISTINCT db.name)) AS database_name,
               collect({{
                   name: attr.name,
                   description: attr.description,
                   term_name: attr.term_name,
                   source_column: attr.source_column,
                   sample_values: col.sample_values,
                   id: attr.id
               }}) AS attrs
        """,
        {"column_id": column_id},
    )
    contexts: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for row in rows:
        term_id = row.get("term_id")
        if not term_id:
            continue
        contexts.append(
            (
                {
                    "name": row.get("term_name") or "",
                    "description": row.get("term_description") or "",
                    "synonyms": row.get("term_synonyms") or [],
                    "id": term_id,
                    "database_name": row.get("database_name") or "",
                },
                [dict(attr) for attr in row.get("attrs") or []],
            )
        )
    return contexts


def fetch_column_attribute_counts(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return per-term ColumnAttribute counts, zone-scoped when zone_ids are provided.

    A ColumnAttribute is owned by exactly one table (via its ``table_id``
    property), so a plain membership filter is sufficient here — no
    all-or-nothing check is needed, matching ``fetch_column_attributes``.

    Pass a pre-resolved *data_ids_by_zone* (see
    ``resolve_accessible_catalog_ids``) when the caller already resolved
    *zone_ids* for this request, to skip a repeat Neo4j round trip.

    Each entry is ``{term_id: str, count: int}``. Terms with zero
    ColumnAttributes are omitted.
    """
    table_filter, params = resolve_table_filter(
        zone_ids,
        "attr.table_id",
        extra_params={"source": SEMANTIC_SOURCE},
        data_ids_by_zone=data_ids_by_zone,
    )
    return get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        {table_filter}
        RETURN term.id AS term_id, count(DISTINCT attr) AS count
        """,
        params,
    )


def _fetch_attr_zones_by_table(
    rows: list[dict[str, Any]],
    zone_ids: list[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{table_id: [zone, ...]}`` for the owning tables of *rows*.

    A ColumnAttribute is owned by exactly one table (``attr.table_id``), so its
    zones are its table's zones. This reuses ``fetch_table_zones_map`` rather
    than duplicating the table→zone resolution. Imported locally because
    ``gsf.dal.exploration`` imports from this module (avoids a circular import).
    """
    from gsf.dal.exploration import fetch_table_zones_map

    table_ids = list({row["table_id"] for row in rows if row.get("table_id")})
    if not table_ids:
        return {}
    return fetch_table_zones_map(zone_ids=zone_ids, table_ids=table_ids)


def fetch_column_attributes_by_term_id(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return ColumnAttribute nodes for a single Term.

    *zone_ids* is a hard authorization boundary; when supplied, only
    attributes owned by tables reachable through those zones are returned —
    a ColumnAttribute is owned by exactly one table, so a plain membership
    filter is sufficient here (no all-or-nothing check needed).

    Each attribute includes ``primary_column`` (HAS_ATTRIBUTE owner) and
    ``referenced_columns`` (SEMANTIC_FK sources) with catalog path ids for
    navigation.
    """
    table_filter, params = resolve_table_filter(
        zone_ids,
        "attr.table_id",
        extra_params={"term_id": term_id, "source": SEMANTIC_SOURCE},
    )
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{term_name: term.name, source: $source}})
        {table_filter}
        OPTIONAL MATCH (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->(attr)
        RETURN {_COLUMN_ATTRIBUTE_FIELDS}
        ORDER BY attr.name
        """,
        params,
    )
    attr_ids = [row["id"] for row in rows]
    zones_by_table = _fetch_attr_zones_by_table(rows, zone_ids)
    columns_by_attr = fetch_column_attribute_columns_map(attr_ids)
    empty_columns = {"primary_column": None, "referenced_columns": []}
    return [
        {
            **row,
            "zones": zones_by_table.get(row["table_id"], []),
            **columns_by_attr.get(row["id"], empty_columns),
        }
        for row in _with_parsed_sample_values(rows)
    ]


def find_column_attribute_by_column_id(column_id: str) -> str | None:
    """Return the id of the ColumnAttribute connected to a given Column, or None."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (col:{Labels.COLUMN} {{id: $col_id}})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        RETURN attr.id AS id LIMIT 1
        """,
        {"col_id": column_id, "source": SEMANTIC_SOURCE},
    )
    return rows[0]["id"] if rows else None


def fetch_related_terms(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return Term nodes related to *term_id* by co-location in the same table.

    Two terms are considered related when they are both connected to the same
    Table node.  A term is connected to a table through any of three paths:

    * REPRESENTS — the table directly represents the term;
    * PROPERTY_OF — the table owns a column whose ColumnAttribute is
      PROPERTY_OF the term (``Table → Column → ColumnAttribute → Term``);
    * SEMANTIC_FK — the table owns a foreign-key column that points, via a
      SEMANTIC_FK edge, to a ColumnAttribute of the term
      (``Table → Column → [SEMANTIC_FK] → ColumnAttribute → Term``).  This is
      what links, e.g. ``Request`` (represented by ``requests``) to ``User``
      when ``requests.creator_id`` references ``users.id``.

    Step 1 — collect every table connected to *term_id* (all three paths).
    Step 2 — collect every other term connected to those same tables (all three).

    *zone_ids* is a hard authorization boundary, not a relevance filter.
    When supplied: Step 1 only considers tables reachable through those
    zones, so sharing a table the caller can't see never surfaces a related
    term.  Step 2 additionally requires that a candidate related term's own
    REPRESENTS-tables are *all* within *zone_ids* — matching the
    all-or-nothing rule used by ``fetch_all_terms`` for the
    ``/terms`` list — so a related term the caller couldn't otherwise open
    (it would 404 via ``get_full_term_by_id``) is never shown as a chip.
    Pass ``None`` to skip zone scoping (admin / internal callers).
    """
    conn = get_neo4j_conn()
    params: dict[str, Any] = {"term_id": term_id}

    if zone_ids is None:
        step1_filter = ""
    else:
        params["zone_table_ids"] = list(
            resolve_accessible_catalog_ids(zone_ids)["table_ids"]
        )
        step1_filter = "AND ta.id IN $zone_table_ids"

    # Step 1: tables for this term
    table_rows = conn.query_read(
        f"""
        MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{id: $term_id}})
        WHERE true {step1_filter}
        RETURN ta.id AS table_id
        UNION
        MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->(:{LABEL_COLUMN_ATTRIBUTE})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM} {{id: $term_id}})
        WHERE true {step1_filter}
        RETURN ta.id AS table_id
        """,
        params,
    )
    if not table_rows:
        return []

    shared_table_ids = [r["table_id"] for r in table_rows if r.get("table_id")]
    if not shared_table_ids:
        return []

    term_b_filter = (
        ""
        if zone_ids is None
        else (
            f"AND NOT EXISTS {{"
            f" (other:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term_b)"
            f" WHERE NOT other.id IN $zone_table_ids"
            f" }}"
        )
    )
    step2_params: dict[str, Any] = {
        "shared_table_ids": shared_table_ids,
        "term_id": term_id,
    }
    if zone_ids is not None:
        step2_params["zone_table_ids"] = params["zone_table_ids"]

    # Step 2: other terms in those tables
    term_rows = conn.query_read(
        f"""
        MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term_b:{LABEL_TERM})
        WHERE ta.id IN $shared_table_ids AND term_b.id <> $term_id
              {term_b_filter}
        RETURN DISTINCT term_b.id AS id, term_b.name AS name,
                        term_b.description AS description
        UNION
        MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->(:{LABEL_COLUMN_ATTRIBUTE})
              -[:{REL_PROPERTY_OF}]->(term_b:{LABEL_TERM})
        WHERE ta.id IN $shared_table_ids AND term_b.id <> $term_id
              {term_b_filter}
        RETURN DISTINCT term_b.id AS id, term_b.name AS name,
                        term_b.description AS description
        """,
        step2_params,
    )
    return [dict(r) for r in term_rows]


def fetch_related_terms_counts(
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return per-term related-term counts for all terms.

    Builds a ``term_id → set(table_id)`` map and a ``table_id → set(term_id)``
    reverse map from Neo4j, then computes for each term the number of distinct
    other terms that share at least one table with it.  A term is connected to
    a table via REPRESENTS, PROPERTY_OF (its own ColumnAttribute) **or**
    SEMANTIC_FK (a foreign-key column that references one of its
    ColumnAttributes) — the same three paths used by ``fetch_related_terms``.

    *zone_ids* is a hard authorization boundary, not a relevance filter. A
    Term can be represented by more than one Table (see ``merge_term``), so
    a term only contributes pairs — via **either** the REPRESENTS or the
    ColumnAttribute path — when **every** table that represents it (via
    REPRESENTS) is reachable through *zone_ids*.  This is the same
    all-or-nothing rule applied consistently to both paths — a term whose
    ColumnAttribute happens to live on an in-zone table doesn't get a free
    pass if it also REPRESENTS an out-of-zone table — so this function stays
    in agreement with ``fetch_all_terms`` (the ``/terms``
    list) and ``fetch_related_terms`` (the single-term related list): a
    term's count here always matches how many terms actually show up on its
    detail page.  Pass ``None`` to return counts for all terms (admin /
    internal callers).

    Each entry is ``{term_id: str, count: int}``.
    """
    term_tables, table_terms = build_term_table_maps(fetch_term_table_pairs(zone_ids))
    result: list[dict[str, Any]] = []
    for term_id, tables in term_tables.items():
        related: set[str] = set()
        for tab_id in tables:
            related.update(table_terms.get(tab_id, set()))
        related.discard(term_id)
        result.append({"term_id": term_id, "count": len(related)})
    return result


def fetch_term_table_pairs(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return ``{term_id, table_id}`` rows linking Terms to their tables.

    A term is linked to a table via REPRESENTS or via its ColumnAttribute
    (PROPERTY_OF, reached through HAS_ATTRIBUTE or SEMANTIC_FK) — the same
    three paths used by ``fetch_related_terms``. Shared by
    ``fetch_related_terms_counts`` and
    ``gsf.dal.exploration.fetch_semantic_exploration_graph`` so per-term
    related counts and the Exploration graph's term↔term edges are always
    computed from one definition of "related". *zone_ids* applies the same
    all-or-nothing scoping as ``fetch_related_terms_counts``. Pass a
    pre-resolved *data_ids_by_zone* (see ``resolve_accessible_catalog_ids``)
    when the caller already resolved *zone_ids* for this request, to skip a
    repeat Neo4j round trip.
    """
    conn = get_neo4j_conn()
    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    if resolved is None:
        filter_clause = ""
        params: dict[str, Any] = {"source": SEMANTIC_SOURCE}
    else:
        table_ids = list(resolved["table_ids"])
        filter_clause = (
            f"WHERE ta.id IN $table_ids"
            f" AND NOT EXISTS {{"
            f" (other:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)"
            f" WHERE NOT other.id IN $table_ids"
            f" }}"
        )
        params = {"source": SEMANTIC_SOURCE, "table_ids": table_ids}

    return conn.query_read(
        f"""
        MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{source: $source}})
        {filter_clause}
        RETURN term.id AS term_id, ta.id AS table_id
        UNION
        MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->(:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        {filter_clause}
        RETURN term.id AS term_id, ta.id AS table_id
        """,
        params,
    )


def build_term_table_maps(
    pairs: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Split ``{term_id, table_id}`` rows into term→tables and table→terms maps."""
    term_tables: dict[str, set[str]] = {}
    table_terms: dict[str, set[str]] = {}
    for row in pairs:
        tid = row.get("term_id")
        tab = row.get("table_id")
        if tid and tab:
            term_tables.setdefault(tid, set()).add(tab)
            table_terms.setdefault(tab, set()).add(tid)
    return term_tables, table_terms
