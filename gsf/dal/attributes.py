# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j read/write for ColumnAttribute and SemanticFK entities.

Also contains find_join_path, which traverses SEMANTIC_FK / HAS_ATTRIBUTE /
CONTAINS edges to resolve multi-hop join routes at retrieval time, and the
shared find_shortest_labeled_path helper underneath it — reused by
gsf.dal.terms.find_term_link_path for the same traversal one level up
(Term-to-Term instead of Column-to-Column) for the Exploration graph.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_SEMANTIC_FK,
    SEMANTIC_SOURCE,
)
from gsf.dal.datasources import fetch_col_table_contexts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ColumnAttribute CRUD
# ---------------------------------------------------------------------------


def merge_column_attribute(
    *,
    term_name: str,
    table_id: str,
    source_column: str,
    attr_name: str,
    datatype: str,
    description: str | None,
) -> str | None:
    """Merge the ColumnAttribute node and return its persistent ``id`` (UUID)."""
    rows = get_neo4j_conn().query_write(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN} {{name: $source_column}})
        MATCH (term:{LABEL_TERM} {{name: $term_name, source: $source}})
        MERGE (attr:{LABEL_COLUMN_ATTRIBUTE} {{
            name: $attr_name,
            source_column: $source_column,
            term_name: $term_name,
            table_id: $table_id,
            source: $source
        }})
        ON CREATE SET attr.id = randomUUID()
        SET attr.datatype = $datatype,
            attr.description = coalesce($description, attr.description)
        MERGE (col)-[:{REL_HAS_ATTRIBUTE}]->(attr)
        MERGE (attr)-[:{REL_PROPERTY_OF}]->(term)
        RETURN attr.id AS id
        """,
        {
            "table_id": table_id,
            "source_column": source_column,
            "term_name": term_name,
            "attr_name": attr_name,
            "datatype": datatype,
            "description": description,
            "source": SEMANTIC_SOURCE,
        },
    )
    return rows[0]["id"] if rows else None


def update_column_attribute(
    attr_id: str,
    term_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    certified: bool | None = None,
) -> dict[str, Any] | None:
    """Update ColumnAttribute metadata and return its embedding context."""
    rows = get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: $attr_id}})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM} {{id: $term_id}})
        SET attr.name = coalesce($name, attr.name),
            attr.description = coalesce($description, attr.description),
            attr.certified = coalesce($certified, attr.certified)
        WITH attr, term
        OPTIONAL MATCH (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->(attr)
        OPTIONAL MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (table:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col)
        RETURN attr.id AS id,
               attr.name AS name,
               attr.description AS description,
               attr.term_name AS term_name,
               attr.source_column AS source_column,
               col.id AS column_id,
               col.sample_values AS sample_values,
               col.is_unique AS is_unique,
               table.id AS table_id,
               table.name AS table_name,
               sch.name AS schema_name,
               term.id AS term_id,
               term.synonyms AS term_synonyms,
               head(collect(DISTINCT db.name)) AS database_name,
               coalesce(attr.certified, false) AS certified
        """,
        {
            "attr_id": attr_id,
            "term_id": term_id,
            "name": name,
            "description": description,
            "certified": certified,
        },
    )
    return dict(rows[0]) if rows else None


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


def fetch_attr_column_contexts(
    attr_ids: list[str],
    *,
    database_name: str | None,
) -> dict[str, dict]:
    """Fetch Column + Table + Schema + Term context for ColumnAttribute IDs.

    Prefers the attribute's own defining column (HAS_ATTRIBUTE) over a
    referencing FK column (SEMANTIC_FK) when both exist for the same
    attribute — picking the wrong one here (as the previous unordered
    OPTIONAL MATCH could) resolves the attribute to the anchor's own FK
    column instead of the real hub table it points at, producing a spurious
    "1 hop" join path that's actually just an intra-table hop.

    *database_name* is required (pass ``None`` explicitly for "don't scope")
    so every call site has to make a conscious choice instead of silently
    inheriting an unscoped default — this is exactly the kind of query a
    caller can forget to scope and not notice until a shared ColumnAttribute
    or Term pulls in a wrong-database column. When given, only attributes
    whose defining/referencing column belongs to that database are returned;
    an attribute with no resolvable column at all (``col`` is null) still
    passes through, since there's nothing to scope.

    Returns a mapping of attr_id -> {attr_name, attr_description, col_id,
    col_name, table_id, table_name, schema_name, database_name, term_name,
    datatype}.
    """
    if not attr_ids:
        return {}
    query = """
    UNWIND $attr_ids AS attr_id
    MATCH (attr:ColumnAttribute {id: attr_id})
    OPTIONAL MATCH (definingCol:Column)-[:HAS_ATTRIBUTE]->(attr)
    WITH attr, collect(definingCol)[0] AS definingCol
    OPTIONAL MATCH (refCol:Column)-[:SEMANTIC_FK]->(attr)
    WITH attr, definingCol, collect(refCol)[0] AS refCol
    WITH attr, coalesce(definingCol, refCol) AS col
    OPTIONAL MATCH (col)<-[:CONTAINS]-(tbl:Table)<-[:CONTAINS]-(sch:Schema)
          <-[:CONTAINS]-(db:Database)
    WHERE $database_name IS NULL OR col IS NULL OR db.name = $database_name
    OPTIONAL MATCH (attr)-[:PROPERTY_OF]->(term:Term)
    RETURN attr.id AS attr_id, attr.name AS attr_name,
           attr.description AS attr_description, attr.datatype AS datatype,
           col.id AS col_id, col.name AS col_name,
           tbl.id AS table_id, tbl.name AS table_name, sch.name AS schema_name,
           db.name AS database_name, term.name AS term_name
    """
    try:
        rows = get_neo4j_conn().query_read(
            query,
            {"attr_ids": attr_ids, "database_name": database_name},
        )
    except Exception:
        logger.warning("fetch_attr_column_contexts: Neo4j query failed", exc_info=True)
        return {}
    result: dict[str, dict] = {}
    for row in rows:
        aid = row.get("attr_id")
        if not aid:
            continue
        result[aid] = {
            "attr_name": row.get("attr_name") or "",
            "attr_description": row.get("attr_description") or "",
            "col_id": row.get("col_id"),
            "col_name": row.get("col_name") or "",
            "table_id": row.get("table_id"),
            "table_name": row.get("table_name") or "",
            "schema_name": row.get("schema_name") or "",
            "database_name": row.get("database_name") or "",
            "term_name": row.get("term_name") or "",
            # Only present on attrs re-ingested since this field was added —
            # older rows fall back to "" here (rendered as no tag downstream).
            "datatype": row.get("datatype") or "",
        }
    return result


# ---------------------------------------------------------------------------
# Column lookup by name (for verifying a join predicate already written in SQL)
# ---------------------------------------------------------------------------


def find_column_id_by_table_and_name(
    table_name: str,
    column_name: str,
    database_name: str | None = None,
) -> str | None:
    """Resolve a ``table.column`` reference from generated SQL to its Column id.

    Case-insensitive on both table and column name, since the SQL came from an
    LLM and may not match the graph's stored casing exactly. When
    *database_name* is given, scopes the match to that database only — the
    same table/column name can exist in multiple co-resident databases
    (see :func:`find_unlinked_fk_columns`), and an unscoped match could
    silently resolve to the wrong database's column. Returns ``None`` (not an
    exception) on no match or an ambiguous multi-database match without
    *database_name*, so callers can treat "can't verify" the same as "no
    known edge" rather than crash.
    """
    if not table_name or not column_name:
        return None
    if database_name:
        rows = get_neo4j_conn().query_read(
            f"""
            MATCH (d:{Labels.DB} {{name: $database_name}})-[:{Edges.CONTAINS}]->
                  (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
                  (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
            WHERE toLower(t.name) = toLower($table_name)
              AND toLower(col.name) = toLower($column_name)
            RETURN col.id AS id
            LIMIT 1
            """,
            {
                "database_name": database_name,
                "table_name": table_name,
                "column_name": column_name,
            },
        )
    else:
        rows = get_neo4j_conn().query_read(
            f"""
            MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
            WHERE toLower(t.name) = toLower($table_name)
              AND toLower(col.name) = toLower($column_name)
            RETURN col.id AS id
            LIMIT 2
            """,
            {"table_name": table_name, "column_name": column_name},
        )
        if len(rows) > 1:
            logger.info(
                "find_column_id_by_table_and_name: ambiguous match for %s.%s "
                "with no database_name given — treating as unresolved",
                table_name,
                column_name,
            )
            return None
    return rows[0]["id"] if rows else None


def find_table_id_by_name(
    table_name: str, database_name: str | None = None
) -> str | None:
    """Resolve a bare table name (from a join-path hop) to its Table id.

    Same database-scoping rationale as :func:`find_column_id_by_table_and_name`
    — an unscoped lookup (e.g. ``datasources.fetch_table_by_name``) risks
    matching a same-named table in a different co-resident database.
    """
    if not table_name:
        return None
    if database_name:
        rows = get_neo4j_conn().query_read(
            f"""
            MATCH (d:{Labels.DB} {{name: $database_name}})-[:{Edges.CONTAINS}]->
                  (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
                  (t:{Labels.TABLE})
            WHERE toLower(t.name) = toLower($table_name)
            RETURN t.id AS id
            LIMIT 1
            """,
            {"database_name": database_name, "table_name": table_name},
        )
    else:
        rows = get_neo4j_conn().query_read(
            f"""
            MATCH (t:{Labels.TABLE})
            WHERE toLower(t.name) = toLower($table_name)
            RETURN t.id AS id
            LIMIT 2
            """,
            {"table_name": table_name},
        )
        if len(rows) > 1:
            logger.info(
                "find_table_id_by_name: ambiguous match for %s with no "
                "database_name given — treating as unresolved",
                table_name,
            )
            return None
    return rows[0]["id"] if rows else None


def find_table_key_columns(
    table_name: str, database_name: str | None = None
) -> dict[str, list[str]]:
    """Return ``{"pk": [...], "unique": [...]}`` column names (lowercased) for
    a bare table name, used to detect a vacuous ``GROUP BY``/``PARTITION BY``
    (grouping by a column already unique per row makes the aggregate a no-op).

    Same database-scoping rationale as :func:`find_table_id_by_name` — scope
    to *database_name* when given, since an unscoped lookup risks matching a
    same-named table in a different co-resident database. ``pk`` comes
    from the Table node's ``pk`` property (set at ingestion from the DDL);
    ``unique`` comes from ``Column.is_unique`` (set from observed-data
    profiling — see :func:`gsf.dal.datasources.store_column_uniqueness`), so
    it also catches a unique-in-practice column with no declared constraint.
    Returns ``{"pk": [], "unique": []}`` (not ``None``) when no match is found.
    """
    empty: dict[str, list[str]] = {"pk": [], "unique": []}
    if not table_name:
        return empty
    db_scope = (
        f"MATCH (d:{Labels.DB} {{name: $database_name}})-[:{Edges.CONTAINS}]->"
        f"      (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)\n"
        if database_name
        else ""
    )
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})
        WHERE toLower(t.name) = toLower($table_name)
        {db_scope}
        OPTIONAL MATCH (t)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
        WHERE c.is_unique = true
        RETURN t.pk AS pk, collect(DISTINCT toLower(c.name)) AS unique_cols
        LIMIT 1
        """,
        {"table_name": table_name, "database_name": database_name},
    )
    if not rows:
        return empty
    pk = [str(c).lower() for c in (rows[0].get("pk") or [])]
    unique_cols = [c for c in (rows[0].get("unique_cols") or []) if c]
    return {"pk": pk, "unique": unique_cols}


def column_participates_in_semantic_fk(col_id: str) -> bool:
    """Whether *col_id* is already known to the FK graph, on either side.

    True if the column is itself an FK-holder (outgoing ``SEMANTIC_FK``) or is
    the referenced/identity side of one (its own ``ColumnAttribute``, via
    ``HAS_ATTRIBUTE``, is the target of some other column's ``SEMANTIC_FK``).
    Used to scope the join-path check to columns ingestion already treats as
    FK-shaped, rather than flagging arbitrary equality joins (date ranges,
    status matches, business logic) the graph was never meant to model.
    """
    if not col_id:
        return False
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (col:{Labels.COLUMN} {{id: $col_id}})
        RETURN
            EXISTS {{ (col)-[:{REL_SEMANTIC_FK}]->() }}
            OR EXISTS {{
                (col)-[:{REL_HAS_ATTRIBUTE}]->(:{LABEL_COLUMN_ATTRIBUTE})
                    <-[:{REL_SEMANTIC_FK}]-()
            }} AS participates
        """,
        {"col_id": col_id},
    )
    return bool(rows and rows[0].get("participates"))


# ---------------------------------------------------------------------------
# SemanticFK
# ---------------------------------------------------------------------------


def find_unlinked_fk_columns(
    database_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return Column nodes with no SEMANTIC_FK and no HAS_ATTRIBUTE edge.

    These are FK columns that have not yet been linked to a ColumnAttribute.

    When *database_name* is provided, only columns belonging to that database
    are returned. Multiple databases can be co-resident in the same Neo4j
    graph, so scoping keeps each compile pass' FK-resolution isolated to a
    single database. When omitted, every unlinked
    FK column in the graph is returned.
    """
    if database_name is not None:
        result = get_neo4j_conn().query_read(
            f"""
            MATCH (d:{Labels.DB} {{name: $database_name}})-[:{Edges.CONTAINS}]->
                  (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
                  (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
            WHERE NOT (col)-[:{REL_SEMANTIC_FK}]->()
              AND NOT (col)-[:{REL_HAS_ATTRIBUTE}]->()
            OPTIONAL MATCH (col)-[:{Edges.FOREIGN_KEY}]->(tgt:{Labels.COLUMN})
            OPTIONAL MATCH (tgt_table:{Labels.TABLE})-[:{Edges.CONTAINS}]->(tgt)
            RETURN col.id          AS id,
                   col.name        AS name,
                   col.description AS description,
                   col.sample_values AS sample_values,
                   t.id            AS table_id,
                   t.name          AS table_name,
                   tgt.id          AS fk_target_col_id,
                   tgt_table.id    AS fk_target_table_id
            """,
            {"database_name": database_name},
        )
    else:
        result = get_neo4j_conn().query_read(
            f"""
            MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
            WHERE NOT (col)-[:{REL_SEMANTIC_FK}]->()
              AND NOT (col)-[:{REL_HAS_ATTRIBUTE}]->()
            OPTIONAL MATCH (col)-[:{Edges.FOREIGN_KEY}]->(tgt:{Labels.COLUMN})
            OPTIONAL MATCH (tgt_table:{Labels.TABLE})-[:{Edges.CONTAINS}]->(tgt)
            RETURN col.id          AS id,
                   col.name        AS name,
                   col.description AS description,
                   col.sample_values AS sample_values,
                   t.id            AS table_id,
                   t.name          AS table_name,
                   tgt.id          AS fk_target_col_id,
                   tgt_table.id    AS fk_target_table_id
            """
        )
    return result


_COLUMN_PATH_RETURN = (
    "col.id AS id, col.name AS column_name, "
    "t.id AS table_id, t.name AS table_name, "
    "sch.id AS schema_id, db.id AS db_id"
)


def _column_path_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Project a Neo4j column-path row into the API column-ref shape."""
    return {
        "id": row["id"],
        "column_name": row["column_name"],
        "table_id": row["table_id"],
        "table_name": row["table_name"],
        "schema_id": row["schema_id"],
        "db_id": row["db_id"],
    }


def fetch_column_attribute_columns_map(
    attr_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Return primary/referenced columns keyed by ColumnAttribute id.

    Each value is ``{primary_column, referenced_columns}``. Missing
    attributes are omitted; callers should default to
    ``primary_column=None`` / ``referenced_columns=[]``.

    Each column dict includes catalog path ids (``db_id``, ``schema_id``,
    ``table_id``, ``id``) plus display names so the UI can navigate to
    ``/data?focus=db|schema|table|column``.
    """
    if not attr_ids:
        return {}

    conn = get_neo4j_conn()
    primary_rows = conn.query_read(
        f"""
        UNWIND $attr_ids AS attr_id
        MATCH (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: attr_id}})
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(sch:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col)
        RETURN attr.id AS attr_id, {_COLUMN_PATH_RETURN}
        """,
        {"attr_ids": attr_ids},
    )
    referenced_rows = conn.query_read(
        f"""
        UNWIND $attr_ids AS attr_id
        MATCH (col:{Labels.COLUMN})-[:{REL_SEMANTIC_FK}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: attr_id}})
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(sch:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col)
        RETURN attr.id AS attr_id, {_COLUMN_PATH_RETURN}
        ORDER BY t.name, col.name
        """,
        {"attr_ids": attr_ids},
    )

    result: dict[str, dict[str, Any]] = {
        attr_id: {"primary_column": None, "referenced_columns": []}
        for attr_id in attr_ids
    }
    for row in primary_rows:
        attr_id = row["attr_id"]
        if attr_id in result and result[attr_id]["primary_column"] is None:
            result[attr_id]["primary_column"] = _column_path_dict(row)
    for row in referenced_rows:
        attr_id = row["attr_id"]
        if attr_id in result:
            result[attr_id]["referenced_columns"].append(_column_path_dict(row))
    return result


def merge_semantic_fk(src_column_id: str, tgt_attr_id: str) -> None:
    """Create a SEMANTIC_FK edge from a source Column to a target ColumnAttribute."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (src:{Labels.COLUMN} {{id: $src_id}})
        MATCH (tgt:{LABEL_COLUMN_ATTRIBUTE} {{id: $tgt_id}})
        MERGE (src)-[:{REL_SEMANTIC_FK}]->(tgt)
        """,
        {"src_id": src_column_id, "tgt_id": tgt_attr_id},
    )


# ---------------------------------------------------------------------------
# Join path traversal
# ---------------------------------------------------------------------------


def _extract_fk_hops(
    path_nodes: list[dict], rel_types: list[str]
) -> list[tuple[dict, dict]]:
    """Reconstruct real FK-mediated crossings from a raw expandConfig path.

    A traversal path can interleave CONTAINS edges (Table<->Column, purely
    structural containment — one table has many columns) with the actual
    semantic crossings (SEMANTIC_FK / HAS_ATTRIBUTE, Column<->ColumnAttribute).
    Filtering the path down to Column-labeled nodes and pairing them by
    position (0&1, 2&3, ...) silently discards what actually connects each
    pair — so two columns that merely share a table (Column <-CONTAINS-
    Table -CONTAINS-> Column, no FK between them at all) get asserted as a
    join. Confirmed live: therapy_details/medchg share a table with zero FK
    relationship, and the old pairing logic reported them as a "1 hop" join.

    This walks the path in order and only emits a hop where a Column reaches
    a ColumnAttribute via HAS_ATTRIBUTE/SEMANTIC_FK, and that same attribute
    is reached by another Column the same way — i.e. an actual FK crossing,
    not co-location. CONTAINS is still needed and still used, just never as
    the crossing itself: it's how the path steps from a crossing's landing
    column, through its table, to a *different* column that continues the
    next crossing (e.g. bridging two hops through a shared pivot table).

    Returns a list of (source_column_node, target_column_node) pairs, in
    path order — the actual columns real crossings.
    """
    hops: list[tuple[dict, dict]] = []
    pending_src: dict | None = None
    for i in range(len(rel_types)):
        cur, rel, nxt = path_nodes[i], rel_types[i], path_nodes[i + 1]
        if (
            cur.get("label") == "Column"
            and rel in ("HAS_ATTRIBUTE", "SEMANTIC_FK")
            and nxt.get("label") == "ColumnAttribute"
        ):
            pending_src = cur
        elif (
            pending_src is not None
            and cur.get("label") == "ColumnAttribute"
            and rel in ("HAS_ATTRIBUTE", "SEMANTIC_FK")
            and nxt.get("label") == "Column"
        ):
            hops.append((pending_src, nxt))
            pending_src = None
    return hops


def _database_names(col_ctx: dict[str, dict]) -> set[str]:
    """Distinct non-empty ``database_name`` values across a col_ctx mapping."""
    return {
        context.get("database_name")
        for context in col_ctx.values()
        if context.get("database_name")
    }


def _spans_multiple_databases(col_ctx: dict[str, dict]) -> bool:
    """Whether the columns in *col_ctx* belong to more than one database.

    A path/bridge that's structurally valid in Neo4j can still cross two
    ingested databases — e.g. two tables that independently FK into a
    same-named column in an unrelated database. That's not a real join:
    the generated SQL would reference a table that doesn't exist in the
    query's target database. Shared by :func:`find_join_path` and
    :func:`find_table_bridge`, which both build hops from a Neo4j path and
    need the same guard before handing one back to a caller.
    """
    return len(_database_names(col_ctx)) > 1


def find_shortest_labeled_path(
    anchor_id: str,
    dest_id: str,
    *,
    node_label: str,
    relationship_filter: str,
    label_filter: str,
    max_level: int,
    log_label: str,
) -> tuple[list[dict], list[str]]:
    """Run `apoc.path.expandConfig` for the shortest path between two same-labeled nodes.

    The actual Neo4j traversal shared by `find_join_path` below
    (Column-to-Column) and `gsf.dal.terms.find_term_link_path`
    (Term-to-Term) — apoc.path.expandConfig is used instead of a plain
    Cypher variable-length pattern because a variable-length pattern
    applies a single direction to every relationship type, whereas both
    callers need one relationship type (SEMANTIC_FK) to behave differently
    from the others (see each function's own docstring for *why* it picks
    the direction it does). Everything else — which relationship types are
    even walkable, which labels the path may pass through, how far it's
    allowed to search, and what the raw node/relationship chain gets
    turned into afterwards — differs enough between the two callers that
    only this innermost "run the query, hand back the raw
    nodes/relationship types" part is actually shared. Deliberately kept
    here (rather than moved alongside `find_term_link_path` into
    `gsf.dal.terms`) so `find_join_path` doesn't have to reach into that
    module for it — `gsf.dal.terms` already imports from here for other
    helpers (e.g. `fetch_column_attribute_columns_map`), so the dependency
    only has to run one way. `bfs: true` + `limit: 1` yields the shortest
    path; `uniqueness: 'NODE_GLOBAL'` keeps the search from revisiting a
    node.

    Returns `(path_nodes, path_rel_types)` — both empty when the two ids
    are equal, either endpoint doesn't exist, no such path exists, or the
    query itself fails (logged as a warning tagged with *log_label*).
    """
    if not anchor_id or not dest_id or anchor_id == dest_id:
        return [], []

    path_query = f"""
    MATCH (anchor:{node_label} {{id: $anchor_id}})
    MATCH (dest:{node_label} {{id: $dest_id}})
    CALL apoc.path.expandConfig(anchor, {{
        relationshipFilter: '{relationship_filter}',
        labelFilter: '{label_filter}',
        terminatorNodes: [dest],
        bfs: true,
        uniqueness: 'NODE_GLOBAL',
        minLevel: 1,
        maxLevel: {max_level},
        limit: 1
    }}) YIELD path
    RETURN [n IN nodes(path) | {{id: n.id, name: n.name, label: labels(n)[0]}}] AS path_nodes,
           [r IN relationships(path) | type(r)] AS path_rel_types
    """
    try:
        rows = get_neo4j_conn().query_read(
            path_query, {"anchor_id": anchor_id, "dest_id": dest_id}
        )
    except Exception:
        logger.warning(
            "%s: Neo4j query failed for %s -> %s",
            log_label,
            anchor_id,
            dest_id,
            exc_info=True,
        )
        return [], []

    if not rows:
        return [], []
    return rows[0].get("path_nodes") or [], rows[0].get("path_rel_types") or []


def find_join_path(anchor_col_id: str, dest_col_id: str) -> list[dict]:
    """Find the shortest semantic join path between two Column nodes.

    SEMANTIC_FK is directional (Column -> ColumnAttribute) and is followed
    only in that outgoing direction: an FK column points at the attribute it
    references. Traversing it undirected would hop from one FK column up to a
    shared target attribute and back down a *different* FK column, fabricating
    a join between two unrelated columns that merely reference the same target
    (e.g. two person-id columns). HAS_ATTRIBUTE and CONTAINS stay undirected.

    Returns a list of hop dicts:
        [{source_database, source_schema, source_table, source_column,
          target_database, target_schema, target_table, target_column}, ...]
    Returns [] when anchor == dest or no path exists.
    """
    # find_shortest_labeled_path (shared with gsf.dal.terms.find_term_link_path)
    # runs the raw apoc.path.expandConfig traversal; _extract_fk_hops below
    # then validates the raw path for real FK crossings instead of pairing
    # nodes positionally, which conflates co-location (columns that merely
    # share a table) with an actual join — see _extract_fk_hops's docstring
    # for the confirmed wrong-join failures that positional pairing caused.
    path_nodes, rel_types = find_shortest_labeled_path(
        anchor_col_id,
        dest_col_id,
        node_label=Labels.COLUMN,
        relationship_filter=f"{REL_SEMANTIC_FK}>|{REL_HAS_ATTRIBUTE}|{Edges.CONTAINS}",
        label_filter=f"-{Labels.SCHEMA}",
        max_level=30,
        log_label="find_join_path",
    )
    if not path_nodes:
        return []

    if len(path_nodes) > 2:
        # A path was found (more than the two endpoint columns) but
        # _extract_fk_hops below may still discard it as pure co-location —
        # log so that case is distinguishable from "no path exists at all"
        # in the "N hop(s)" summary the caller logs.
        table_names = [n.get("name") for n in path_nodes if n.get("label") == "Table"]
        logger.debug(
            "find_join_path: raw path %s -> %s spans table(s) %s, "
            "validating for real FK crossings",
            anchor_col_id,
            dest_col_id,
            table_names,
        )
    crossings = _extract_fk_hops(path_nodes, rel_types)
    if not crossings:
        if len(path_nodes) > 2:
            logger.info(
                "find_join_path: discarding path %s -> %s — nodes were "
                "reachable but only via co-location (no real FK crossing), "
                "reporting no join path instead of a fabricated one",
                anchor_col_id,
                dest_col_id,
            )
        return []

    col_ids = [c["id"] for pair in crossings for c in pair if c.get("id")]
    col_ctx = fetch_col_table_contexts(col_ids)
    if _spans_multiple_databases(col_ctx):
        logger.warning(
            "find_join_path: rejected cross-database path %s -> %s (%s)",
            anchor_col_id,
            dest_col_id,
            ", ".join(sorted(_database_names(col_ctx))),
        )
        return []

    hops: list[dict] = []
    for src, tgt in crossings:
        src_ctx = col_ctx.get(src.get("id") or "", {})
        tgt_ctx = col_ctx.get(tgt.get("id") or "", {})
        hops.append(
            {
                "source_database": src_ctx.get("database_name", ""),
                "source_schema": src_ctx.get("schema_name", ""),
                "source_table": src_ctx.get("table_name", ""),
                "source_column": src.get("name", ""),
                "target_database": tgt_ctx.get("database_name", ""),
                "target_schema": tgt_ctx.get("schema_name", ""),
                "target_table": tgt_ctx.get("table_name", ""),
                "target_column": tgt.get("name", ""),
            }
        )
    return hops


def find_shared_hub_bridge(col_a_id: str, col_b_id: str) -> dict:
    """Find a genuine shared-identity hub connecting two FK columns that
    :func:`find_join_path` cannot reach.

    ``find_join_path``'s forward-only ``SEMANTIC_FK`` traversal is a
    deliberate guard: two unrelated FK columns that merely reference the
    same target (e.g. two ``person_id`` columns for different roles) must
    never be reported as directly joined by walking the edge backwards.
    But that guard also blocks a *legitimate* case with the identical shape
    — two spokes of the same real identity hub (e.g. ``robot_details`` and
    ``actuation_data`` both referencing ``robot_record`` via its own primary
    key) — from ever being discovered, since neither spoke has anything
    pointing *out* toward the other.

    This is intentionally much narrower than lifting the forward-only guard
    in general: it only reports a bridge when BOTH columns independently
    hold a *forward* ``SEMANTIC_FK`` edge to the exact same
    ``ColumnAttribute``, AND that attribute's defining column is the real,
    declared primary key of its own table (``hubCol.name IN hubTable.pk``)
    — the same discipline :func:`find_anchor_hub_siblings` already applies.
    Requiring the shared target to be a genuine identity column (not just
    any attribute two FK columns happen to share) is what keeps this from
    reopening the "two unrelated FK columns, coincidentally shared target"
    fabrication risk the forward-only design exists to prevent.

    Returns ``{"hub_table", "hub_column"}`` (schema-unqualified — callers
    already have the written table/column names and schema from the SQL
    itself, so this only needs to supply the hub side), or ``{}`` if no such
    shared, PK-anchored hub exists.
    """
    if col_a_id == col_b_id:
        return {}

    query = """
    MATCH (ca:Column {id: $col_a_id})-[:SEMANTIC_FK]->(hubAttr:ColumnAttribute)
          <-[:SEMANTIC_FK]-(cb:Column {id: $col_b_id})
    MATCH (hubCol:Column)-[:HAS_ATTRIBUTE]->(hubAttr)
    MATCH (hubTable:Table)-[:CONTAINS]->(hubCol)
    WHERE hubCol.name IN coalesce(hubTable.pk, [])
    RETURN hubCol.name AS hub_column, hubTable.name AS hub_table
    LIMIT 1
    """
    try:
        rows = get_neo4j_conn().query_read(
            query, {"col_a_id": col_a_id, "col_b_id": col_b_id}
        )
    except Exception:
        logger.warning(
            "find_shared_hub_bridge: Neo4j query failed for %s / %s",
            col_a_id,
            col_b_id,
            exc_info=True,
        )
        return {}
    if not rows:
        return {}
    return {
        "hub_table": rows[0].get("hub_table") or "",
        "hub_column": rows[0].get("hub_column") or "",
    }


def find_anchor_hub_siblings(
    anchor_table_id: str, max_siblings: int | None = 6
) -> tuple[list[dict], int]:
    """Find tables that share a hub table with *anchor_table_id* via FK.

    Looser than :func:`find_join_path` on purpose: it takes ONE step backward
    through a hub table's real, DB-declared primary key (verified against
    ``Table.pk``, not just any ColumnAttribute) — starting only from the
    anchor table's own outgoing SEMANTIC_FK edges, never from every
    candidate table's PK. This finds sibling tables (e.g. two tables that
    both reference the same parent) that :func:`find_join_path`'s
    forward-only traversal structurally cannot reach, without risking the
    false-positive fan-out find_join_path guards against.

    Includes the hub table itself (not just its siblings) — a question can
    need the hub as its own base/pivot table (e.g. to count every row even
    when a sibling has no matching data), and the hub is cheap and safe to
    add: exactly one verified, single-hop forward FK per distinct hub, not
    proportional to how many siblings reference it.

    Sibling count is capped at *max_siblings* per hub, applied here in
    Neo4j-return order (arbitrary — no relevance signal). *max_siblings=None*
    skips capping entirely and returns every sibling; callers that want a
    relevance-ranked cap (e.g. embedding similarity to the question) should
    pass ``None`` here and rank+truncate themselves — see
    ``CandidatePreparationAgent._rank_and_cap_hub_siblings`` for the ranked
    version used in production, which replaced a blind order-based cap after
    an audit found it was silently dropping the one relevant sibling in
    ~30% of truncation events. This function's own cap stays as a safety-net
    default for any other/future caller that doesn't rank — measured against
    the live graph, most hubs have few siblings (median 2), but some are
    genuine mega-hubs (up to 20), and an uncapped expansion would dump a
    large, low-precision batch of tables into the candidate set for those.

    This is a discovery/relevance signal, not a verified join path — it
    does NOT claim the anchor and sibling should be joined directly (they
    usually should each join the shared hub instead). Callers must not feed
    this into SQL-generation join instructions; it exists only to stop the
    relevance filter from dropping a structurally-connected table it has no
    other way to recognize.

    Returns ``(results, truncated_count)``. Each result dict has
    ``{source_table, target_table, id, hub_table, is_hub}`` — the first two
    keys match :func:`find_join_path`'s hop shape for prompt rendering: for
    hub entries ``target_table`` is the hub name, for sibling entries it's
    the sibling name. ``truncated_count`` is how many sibling tables were
    dropped by the cap (0 if none were).
    """
    query = """
    MATCH (anchorTable:Table {id: $anchor_table_id})-[:CONTAINS]->(fkCol:Column)
          -[:SEMANTIC_FK]->(hubAttr:ColumnAttribute)
    MATCH (hubCol:Column)-[:HAS_ATTRIBUTE]->(hubAttr)
    MATCH (hubTable:Table)-[:CONTAINS]->(hubCol)
    WHERE hubCol.name IN coalesce(hubTable.pk, [])
    MATCH (siblingCol:Column)-[:SEMANTIC_FK]->(hubAttr)
    MATCH (siblingTable:Table)-[:CONTAINS]->(siblingCol)
    WHERE siblingTable.id <> anchorTable.id
    RETURN DISTINCT siblingTable.id AS sibling_id, siblingTable.name AS sibling_table,
           hubTable.id AS hub_id, hubTable.name AS hub_table,
           anchorTable.name AS anchor_table
    """
    try:
        rows = get_neo4j_conn().query_read(query, {"anchor_table_id": anchor_table_id})
    except Exception:
        logger.warning(
            "find_anchor_hub_siblings: Neo4j query failed for table %s",
            anchor_table_id,
            exc_info=True,
        )
        return [], 0

    anchor_name = ""
    hubs: dict[str, str] = {}  # hub_id -> hub_name
    siblings_by_hub: dict[str, list[dict]] = {}
    for row in rows:
        anchor_name = row.get("anchor_table") or anchor_name
        hub_id, hub_name = row.get("hub_id"), row.get("hub_table")
        if not hub_id:
            continue
        hubs[hub_id] = hub_name or ""
        sib_id, sib_name = row.get("sibling_id"), row.get("sibling_table")
        if sib_id:
            siblings_by_hub.setdefault(hub_id, []).append(
                {"id": sib_id, "name": sib_name or ""}
            )

    results: list[dict] = []
    truncated = 0
    for hub_id, hub_name in hubs.items():
        results.append(
            {
                "source_table": anchor_name,
                "target_table": hub_name,
                "id": hub_id,
                "hub_table": hub_name,
                "is_hub": True,
            }
        )
        sibs = siblings_by_hub.get(hub_id, [])
        if max_siblings is not None and len(sibs) > max_siblings:
            truncated += len(sibs) - max_siblings
            sibs = sibs[:max_siblings]
        for s in sibs:
            results.append(
                {
                    "source_table": anchor_name,
                    "target_table": s["name"],
                    "id": s["id"],
                    "hub_table": hub_name,
                    "is_hub": False,
                }
            )
    return results, truncated


def find_table_bridge(
    table_a_id: str,
    table_b_id: str,
    allowed_table_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Find bridge tables AND their join hops (if any) needed to connect two tables.

    Runs the same forward-only traversal as :func:`find_join_path`, but
    starts from table_a as a whole rather than a single representative
    column — a table's own primary key is usually a pure identity column
    with no outgoing FK of its own (the real FK, e.g. ``member_fan_pivot``,
    is a *different* column on the same table), so anchoring on the PK
    alone would miss the real connection or find an unrelated coincidental
    one. Starting from the Table node and letting CONTAINS (undirected, same
    as find_join_path) walk into every one of its own columns finds the real
    path regardless of which specific column carries the FK. Tried in both
    directions since SEMANTIC_FK is forward-only.

    IMPORTANT: an unrestricted traversal can find a real but coincidental
    forward path through a table that was never presented to (or judged by)
    the relevance filter at all — e.g. two tables that both happen to FK
    into an unrelated "preferences" table for reasons having nothing to do
    with the actual question. That's not "the filter missed a bridge it
    should have kept", it's introducing information the filter never had a
    chance to accept or reject. So when *allowed_table_ids* is given, a
    found path is only returned if every intermediate table on it is a
    member of that set (typically: the tables that were candidates *before*
    the relevance filter ran) — this keeps the function to "restore a
    bridge the filter had and dropped", never "discover a new one".

    Returns ``(bridge_tables, hops)``:
      - ``bridge_tables``: ``{id, name}`` dicts for every distinct table
        strictly between table_a and table_b on the shortest path found —
        excludes the two endpoint tables themselves.
      - ``hops``: join hop dicts in the same shape :func:`find_join_path`
        produces (``{source_schema, source_table, source_column,
        target_schema, target_table, target_column}``), describing the
        actual FK columns connecting table_a -> ... -> table_b. Without
        this, callers only learn a bridge table's *name*, not how to join
        it — earlier versions of this function returned table names alone
        and downstream SQL-gen had to guess the join, which produced a
        fabricated join between two unrelated PK columns.

    Both empty if no path exists in either direction, or the only path
    found steps outside *allowed_table_ids*.
    """
    query = """
    MATCH (start:Table {id: $start_id})
    MATCH (end:Table {id: $end_id})
    CALL apoc.path.expandConfig(start, {
        relationshipFilter: 'SEMANTIC_FK>|HAS_ATTRIBUTE|CONTAINS',
        labelFilter: '-Schema',
        terminatorNodes: [end],
        bfs: true, uniqueness: 'NODE_GLOBAL', minLevel: 1, maxLevel: 30, limit: 1
    }) YIELD path
    RETURN [n IN nodes(path) | {id: n.id, name: n.name, label: labels(n)[0]}] AS path_nodes,
    [r IN relationships(path) | type(r)] AS rel_types
    """
    for src, dst in ((table_a_id, table_b_id), (table_b_id, table_a_id)):
        try:
            rows = get_neo4j_conn().query_read(query, {"start_id": src, "end_id": dst})
        except Exception:
            logger.warning(
                "find_table_bridge: Neo4j query failed for %s -> %s",
                src,
                dst,
                exc_info=True,
            )
            continue
        if not rows:
            continue
        path_nodes: list[dict] = rows[0].get("path_nodes") or []
        rel_types: list[str] = rows[0].get("rel_types") or []
        tables = [n for n in path_nodes if n.get("label") == "Table"]
        bridge = {
            t["id"]: t for t in tables if t.get("id") not in (table_a_id, table_b_id)
        }
        if not bridge:
            continue
        if allowed_table_ids is not None and not set(bridge).issubset(
            allowed_table_ids
        ):
            logger.info(
                "find_table_bridge: discarding path %s -> %s — bridge table(s) "
                "%s were never a candidate, not just restoring a dropped one",
                src,
                dst,
                [
                    t["name"]
                    for t in bridge.values()
                    if t["id"] not in allowed_table_ids
                ],
            )
            continue

        # Rebuild the actual join hops: only real SEMANTIC_FK/HAS_ATTRIBUTE
        # crossings count, never a pair of columns that merely share a table
        # (see _extract_fk_hops). A bridge table with no real crossing is a
        # structural coincidence, not a joinable path — reject it rather
        # than hand back a table we can't actually explain how to join.
        crossings = _extract_fk_hops(path_nodes, rel_types)
        if not crossings:
            logger.warning(
                "find_table_bridge: path %s -> %s found bridge table(s) %s "
                "but no real FK crossing — discarding (co-location, not a "
                "join)",
                src,
                dst,
                [t["name"] for t in bridge.values()],
            )
            continue

        col_ids = [c["id"] for pair in crossings for c in pair if c.get("id")]
        col_ctx = fetch_col_table_contexts(col_ids)
        if _spans_multiple_databases(col_ctx):
            logger.warning(
                "find_table_bridge: rejected cross-database path %s -> %s (%s)",
                src,
                dst,
                ", ".join(sorted(_database_names(col_ctx))),
            )
            continue

        hops: list[dict] = []
        for c_src, c_tgt in crossings:
            src_ctx = col_ctx.get(c_src.get("id") or "", {})
            tgt_ctx = col_ctx.get(c_tgt.get("id") or "", {})
            hops.append(
                {
                    "source_schema": src_ctx.get("schema_name", ""),
                    "source_table": src_ctx.get("table_name", ""),
                    "source_column": c_src.get("name", ""),
                    "target_schema": tgt_ctx.get("schema_name", ""),
                    "target_table": tgt_ctx.get("table_name", ""),
                    "target_column": c_tgt.get("name", ""),
                }
            )
        return list(bridge.values()), hops
    return [], []


def find_kept_table_bridges(
    table_ids: list[str],
    allowed_table_ids: set[str] | None = None,
    max_bridge_tables: int = 5,
) -> tuple[list[dict], list[list[dict]], int]:
    """Find bridge tables AND their join hops needed to connect pairs of
    already-kept tables.

    Runs *after* the relevance filter has already narrowed candidates down
    — kept sets are consistently small in practice (2-4 tables), so
    checking every pair is cheap by construction. Stops early once
    *max_bridge_tables* distinct bridge tables have been found regardless,
    as a hard safety cap for the rare case a kept set is larger than usual.

    *allowed_table_ids*, if given, is passed through to
    :func:`find_table_bridge` to restrict results to tables that were
    already candidates before the relevance filter ran — see its docstring
    for why this matters.

    If the LLM already kept a genuine bridge table C alongside A and B, the
    A-C and C-B pairs would otherwise "rediscover" C as if it were new,
    burning cap budget on a table that was never missing — so any table
    already present in *table_ids* is excluded from counting as a found
    bridge (it needs no restoring; it's already there). Join hops for such
    a pair are still collected — even an already-kept table needs its join
    condition surfaced, or SQL-gen has the table but not the join, same
    failure this whole function exists to prevent.

    Returns ``(bridge_tables, bridge_paths, skipped_pairs)``:
      - ``bridge_tables``: ``{id, name}`` dicts (deduped across all pairs,
        excluding tables already in *table_ids*).
      - ``bridge_paths``: one hop-list per pair that found a bridge (each
        hop shaped like :func:`find_join_path`'s output) — meant to be
        appended to ``attribute_join_paths`` as ``{"path": hops}`` entries
        so SQL-gen actually learns how to join the bridge table in, not
        just that it exists.
      - ``skipped_pairs``: how many remaining pairs were never checked
        because the cap was already hit.
    """
    if len(table_ids) < 2:
        return [], [], 0

    already_kept = set(table_ids)
    pairs = list(itertools.combinations(table_ids, 2))
    found: dict[str, dict] = {}
    bridge_paths: list[list[dict]] = []
    checked = 0
    for a_id, b_id in pairs:
        if len(found) >= max_bridge_tables:
            break
        checked += 1
        bridge_tables, hops = find_table_bridge(a_id, b_id, allowed_table_ids)
        for t in bridge_tables:
            if t["id"] in already_kept:
                continue
            found.setdefault(t["id"], t)
        if bridge_tables and hops:
            bridge_paths.append(hops)

    skipped = len(pairs) - checked
    if skipped:
        logger.info(
            "find_kept_table_bridges: cap of %d bridge table(s) hit — "
            "skipped %d/%d remaining pair(s)",
            max_bridge_tables,
            skipped,
            len(pairs),
        )
    return list(found.values()), bridge_paths, skipped
