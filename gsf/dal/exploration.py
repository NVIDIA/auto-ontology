# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for the data- and semantic-layer Exploration graphs.

Contains only functions that call ``get_neo4j_conn()`` directly (aside from
the internal calls each graph builder makes to its own sibling functions
below). All read functions use the ``fetch_*`` prefix.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.dal.cypher_fragments import paging_clause, table_description_expr
from gsf.dal.datasources import TABLE_COUNTS_SUBQUERY
from gsf.dal.sql_attributes import fetch_sql_attribute_counts
from gsf.dal.terms import (
    build_term_table_maps,
    fetch_all_terms,
    fetch_column_attribute_counts,
    fetch_related_term_ids,
    fetch_related_terms_counts,
    fetch_term_table_pairs,
    fetch_terms_by_ids,
    find_term_link_path,
    term_is_in_scope,
)
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

# Hard ceiling on nodes returned by an Exploration graph endpoint, regardless
# of what a caller requests — keeps the response bounded on large catalogs.
MAX_EXPLORATION_GRAPH_NODES = 500

# A Sql node with no query text is not a data-layer edge. Every read that
# derives table-to-table relationships has to apply this, or the same pair of
# tables is connected in one view and unconnected in another. Expects `sql`.
_NON_EMPTY_SQL = (
    "sql.sql_full_query IS NOT NULL AND trim(toString(sql.sql_full_query)) <> ''"
)


def fetch_data_exploration_edges(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return table pairs connected by a shared SQL query or a foreign key.

    A query becomes an exploration edge when its ``Sql`` node references at
    least two visible tables — each such edge includes the SQL text shown
    when the user selects that connection. A foreign key between two
    tables' columns also becomes an edge (flagged ``via_foreign_key``), even
    when no stored SQL query ever referenced both tables together; that
    edge carries an empty ``queries`` list unless a shared SQL query also
    connects the same pair, in which case the two are merged into one edge.
    Every edge also carries a ``foreign_keys`` list — one entry per FK
    column pair between the two tables (``source_column``,
    ``target_column``, and each column's ``sample_values``) — so the
    client can show which columns join the pair even when there's no
    stored SQL query to display. Empty for edges that are SQL-only.
    ``relationship_types`` names the underlying Neo4j relationship
    type(s) backing the edge (``SQL`` and/or ``FOREIGN_KEY``, from
    ``Edges``) so a client can label the connection the same way it would
    read in the graph itself.
    Pass a pre-resolved *data_ids_by_zone* (see
    ``resolve_accessible_catalog_ids``) when the caller already resolved
    *zone_ids* for this request, to skip a repeat Neo4j round trip.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    if data_ids_by_zone is not None:
        table_ids = list(data_ids_by_zone["table_ids"])
        if not table_ids:
            return []
        table_filter = "source.id IN $table_ids AND target.id IN $table_ids AND "
        params: dict[str, Any] = {"table_ids": table_ids}
    else:
        table_filter = ""
        params = {}

    conn = get_neo4j_conn()
    sql_rows = conn.query_read(
        f"""
        MATCH (source:{Labels.TABLE})<-[:{Edges.SQL}]-(sql:{Labels.SQL})
              -[:{Edges.SQL}]->(target:{Labels.TABLE})
        WHERE {table_filter}source.id < target.id AND {_NON_EMPTY_SQL}
        RETURN source.id AS source,
               target.id AS target,
               collect(DISTINCT sql.sql_full_query) AS queries
        """,
        params,
    )
    fk_rows = conn.query_read(
        f"""
        MATCH (source:{Labels.TABLE})-[:{Edges.CONTAINS}]->(src_col:{Labels.COLUMN})
              -[:{Edges.FOREIGN_KEY}]->(tgt_col:{Labels.COLUMN})<-[:{Edges.CONTAINS}]-
              (target:{Labels.TABLE})
        WHERE {table_filter}source.id <> target.id
        RETURN DISTINCT source.id AS source, target.id AS target,
               src_col.name AS source_column, tgt_col.name AS target_column,
               src_col.sample_values AS source_sample_values,
               tgt_col.sample_values AS target_sample_values
        """,
        params,
    )

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sql_rows:
        source, target = row.get("source"), row.get("target")
        if not source or not target:
            continue
        edges[(source, target)] = {
            "source": source,
            "target": target,
            "queries": list(row.get("queries") or []),
            "via_foreign_key": False,
            "foreign_keys": [],
            "relationship_types": [Edges.SQL],
        }
    for row in fk_rows:
        a, b = row.get("source"), row.get("target")
        if not a or not b:
            continue
        # Edge keys are normalized to (min id, max id) above; when a FK row's
        # (source, target) came in reversed relative to that order, the
        # column pair it carries must be swapped along with it so
        # `foreign_keys[].source_column` always names a column on
        # `edge["source"]`, never on `edge["target"]`.
        flipped = a > b
        key = (b, a) if flipped else (a, b)
        fk_detail = {
            "source_column": row.get("target_column" if flipped else "source_column"),
            "target_column": row.get("source_column" if flipped else "target_column"),
            "source_sample_values": row.get(
                "target_sample_values" if flipped else "source_sample_values"
            ),
            "target_sample_values": row.get(
                "source_sample_values" if flipped else "target_sample_values"
            ),
        }
        edge = edges.get(key)
        if edge is None:
            edge = {
                "source": key[0],
                "target": key[1],
                "queries": [],
                "via_foreign_key": True,
                "foreign_keys": [],
                "relationship_types": [Edges.FOREIGN_KEY],
            }
            edges[key] = edge
        else:
            edge["via_foreign_key"] = True
            if Edges.FOREIGN_KEY not in edge["relationship_types"]:
                edge["relationship_types"].append(Edges.FOREIGN_KEY)
        if fk_detail not in edge["foreign_keys"]:
            edge["foreign_keys"].append(fk_detail)

    return sorted(edges.values(), key=lambda edge: (edge["source"], edge["target"]))


def fetch_table_exploration_details(
    table_id: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return SQL queries and one ordered page of Terms for a visible Table.

    Each term carries ``relationship_types`` — the real Neo4j relationship
    type(s) reaching it from the table (see the query below) — so a client
    can label the connection the same way it would read in the graph itself.
    A Table's own ColumnAttributes are deliberately *not* included here —
    they're two hops out (``Table-CONTAINS->Column-HAS_ATTRIBUTE->
    ColumnAttribute``), one more than a Table's own graph expansion should
    reveal in one double-click; a Column's own expansion
    (``fetch_column_exploration_details``, one real hop from the Column
    itself) is what surfaces those instead.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if data_ids_by_zone is not None and table_id not in data_ids_by_zone["table_ids"]:
        return {"queries": [], "terms": [], "terms_total": 0}

    conn = get_neo4j_conn()
    queries = conn.query_read(
        f"""
        MATCH (sql:{Labels.SQL})-[:{Edges.SQL}]->
              (t:{Labels.TABLE} {{id: $table_id}})
        RETURN DISTINCT sql.id AS id, sql.sql_full_query AS sql
        ORDER BY id
        """,
        {"table_id": table_id},
    )
    term_params: dict[str, Any] = {"table_id": table_id, "source": SEMANTIC_SOURCE}
    # `rel_type` names the real Neo4j relationship reaching this term from
    # the table — `REPRESENTS` directly, or the `HAS_ATTRIBUTE`/`SEMANTIC_FK`
    # edge off whichever column carries it — mirroring the `path` a term↔term
    # link's `relationship_types` is built from in `fetch_semantic_exploration_graph`.
    # A term reachable both ways collects both types once grouped below.
    # Both branches require `source: SEMANTIC_SOURCE` on the ColumnAttribute
    # and Term nodes, matching `fetch_term_table_pairs`'s own definition of a
    # term↔table link — otherwise a Table's expansion here could surface a
    # non-semantic Term that never appears on the semantic graph, in its own
    # related-term count, or anywhere else "related" is computed from.
    linked_terms_subquery = f"""
        CALL () {{
            MATCH (t:{Labels.TABLE} {{id: $table_id}})
                  -[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{source: $source}})
            RETURN term.id AS id, term.name AS name, term.description AS description,
                   '{REL_REPRESENTS}' AS rel_type
            UNION
            MATCH (t:{Labels.TABLE} {{id: $table_id}})
                  -[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
                  -[attr_rel:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->
                  (:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})-[:{REL_PROPERTY_OF}]->
                  (term:{LABEL_TERM} {{source: $source}})
            RETURN term.id AS id, term.name AS name, term.description AS description,
                   type(attr_rel) AS rel_type
        }}
    """
    total_rows = conn.query_read(
        f"""
        {linked_terms_subquery}
        RETURN count(DISTINCT id) AS total
        """,
        dict(term_params),
    )
    total = total_rows[0]["total"] if total_rows else 0
    if not total:
        return {
            "queries": [
                {"id": row.get("id") or "", "sql": row.get("sql") or ""}
                for row in queries
                if row.get("sql")
            ],
            "terms": [],
            "terms_total": 0,
        }

    paging = paging_clause(skip, limit, term_params)
    terms = conn.query_read(
        f"""
        {linked_terms_subquery}
        WITH id, name, description, collect(DISTINCT rel_type) AS relationship_types
        ORDER BY name, id
        {paging}
        RETURN id, name, description, relationship_types
        """,
        term_params,
    )
    return {
        "queries": [
            {"id": row.get("id") or "", "sql": row.get("sql") or ""}
            for row in queries
            if row.get("sql")
        ],
        "terms": [dict(row) for row in terms if row.get("id")],
        "terms_total": total,
    }


def fetch_term_exploration_details(
    term_id: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return one ordered page of Tables linked to a visible Term.

    The reverse of ``fetch_table_exploration_details``'s Term list: each
    table row carries ``relationship_types`` — the real Neo4j relationship
    type(s) reaching it from the term (``REPRESENTS`` directly, or the
    ``HAS_ATTRIBUTE``/``SEMANTIC_FK`` edge off one of its columns) — mirroring
    how that function labels the reverse direction, so a client can label an
    expansion edge from either end the same way.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if not term_is_in_scope(term_id, zone_ids, data_ids_by_zone):
        return {"tables": [], "tables_total": 0}

    table_filter = ""
    params: dict[str, Any] = {"term_id": term_id, "source": SEMANTIC_SOURCE}
    if data_ids_by_zone is not None:
        table_filter = "AND ta.id IN $table_ids"
        params["table_ids"] = list(data_ids_by_zone["table_ids"])

    conn = get_neo4j_conn()
    # `rel_type` mirrors `fetch_table_exploration_details`'s own `rel_type`,
    # just walked in the opposite direction (from the term to its tables
    # rather than from a table to its terms). The second branch's
    # ColumnAttribute requires `source: SEMANTIC_SOURCE`, matching
    # `fetch_term_table_pairs`'s own definition of a term↔table link —
    # `term_id` is already a known, scope-checked Term (see
    # `term_is_in_scope` above), so the Term match itself needs no such
    # filter, unlike the ColumnAttribute it's reached through.
    linked_tables_subquery = f"""
        CALL () {{
            MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->
                  (term:{LABEL_TERM} {{id: $term_id}})
            WHERE true {table_filter}
            RETURN ta.id AS id, '{REL_REPRESENTS}' AS rel_type
            UNION
            MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
                  -[attr_rel:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->
                  (:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})-[:{REL_PROPERTY_OF}]->
                  (term:{LABEL_TERM} {{id: $term_id}})
            WHERE true {table_filter}
            RETURN ta.id AS id, type(attr_rel) AS rel_type
        }}
    """
    total_rows = conn.query_read(
        f"""
        {linked_tables_subquery}
        RETURN count(DISTINCT id) AS total
        """,
        dict(params),
    )
    total = total_rows[0]["total"] if total_rows else 0
    if not total:
        return {"tables": [], "tables_total": 0}

    paging = paging_clause(skip, limit, params)
    tables = conn.query_read(
        f"""
        {linked_tables_subquery}
        WITH id, collect(DISTINCT rel_type) AS relationship_types
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE} {{id: id}})
        WITH t, db, s, relationship_types
        ORDER BY t.name, t.id
        {paging}
        RETURN t.id AS id, t.name AS name, t.table_type AS table_type,
               db.id AS database_id, db.name AS database_name,
               s.id AS schema_id, s.name AS schema_name,
               relationship_types
        """,
        params,
    )
    return {
        "tables": [dict(row) for row in tables if row.get("id")],
        "tables_total": total,
    }


def fetch_column_exploration_details(
    column_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return the ColumnAttribute, outgoing FOREIGN_KEY Column and Sql
    queries (if any) linked to a visible Column.

    Closes the loop `Table-CONTAINS->Column-HAS_ATTRIBUTE/SEMANTIC_FK->
    ColumnAttribute-PROPERTY_OF->Term` from the Column's own end:
    `fetch_table_exploration_details` links a Table to its Terms and
    `fetch_term_exploration_details` links a Term to its Tables, and this is
    the same idea one hop in from either — used by the Exploration graph's
    Column expansion (see `expandColumnNode` in `ExplorationView.tsx`) to
    graft the ColumnAttribute node/edge on alongside the Column's
    already-known owning Table. Both `HAS_ATTRIBUTE` and `SEMANTIC_FK` are
    checked — same as every other Column->ColumnAttribute traversal in the
    codebase (see e.g. `fetch_table_exploration_details` above) — since a
    foreign-key-shaped column (like a `user_id`) is linked via `SEMANTIC_FK`
    rather than `HAS_ATTRIBUTE`. `column_attribute.relationship_type` names
    whichever of the two actually backs the edge, so the client can label
    it accordingly (see `expandColumnNode`). `column_attribute` is `None`
    when the column has neither (most columns don't).

    Separately (and independently — a Column can have either, both, or
    neither), also follows the Column's own *physical* `FOREIGN_KEY` edge
    (see `fetch_data_exploration_edges`'s identical traversal, the same
    relationship a raw FK constraint in the source database produces) to
    the Column it references, enriched with that target Column's own
    owning Table/Schema/Database ids/names — same shape as
    `_enrich_catalog_path_nodes` gives a link-path hop — so the client can
    graft it on as a fully expandable Column node, not just a bare label.
    `foreign_key_column` is `None` when the column has no outgoing FK.

    Also follows that same `FOREIGN_KEY` edge in reverse — every visible
    Column whose own outgoing FK points *at* this one (typically this
    Column is a table's primary key and the others are foreign keys into
    it) — into `referencing_columns`, same enrichment as
    `foreign_key_column` above. Without this, expanding an FK *target*
    Column (the common case) would show nothing for the referencing side,
    even though it's a real, direct edge one hop away. Capped at 50 rather
    than paginated, like every other field here.

    Also follows the Column's own incoming `SQL` edges — the reverse of
    `fetch_sql_exploration_details`'s own `Sql-[SQL]->Column` traversal,
    the same real relationship the ingestion pipeline draws for every
    column a parsed query references (`parse_query_slim` in
    `nemo_retriever`) — to every visible `Sql` node that referenced this
    column, so the client can graft each on as a fully expandable Sql node
    (see `expandSqlNode`). `sql_queries` is empty when no stored query
    ever referenced this column directly.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    table_filter = ""
    params: dict[str, Any] = {"column_id": column_id, "source": SEMANTIC_SOURCE}
    if data_ids_by_zone is not None:
        table_filter = "AND t.id IN $table_ids"
        params["table_ids"] = list(data_ids_by_zone["table_ids"])

    attr_rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN} {{id: $column_id}})
        WHERE true {table_filter}
        MATCH (col)-[attr_rel:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        RETURN attr.id AS id, attr.name AS name, attr.description AS description,
               type(attr_rel) AS relationship_type
        LIMIT 1
        """,
        params,
    )

    fk_table_filter = ""
    if data_ids_by_zone is not None:
        fk_table_filter = "AND fk_t.id IN $table_ids"
    fk_rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN} {{id: $column_id}})
        WHERE true {table_filter}
        MATCH (col)-[:{Edges.FOREIGN_KEY}]->(fk_col:{Labels.COLUMN})
              <-[:{Edges.CONTAINS}]-(fk_t:{Labels.TABLE})
              <-[:{Edges.CONTAINS}]-(fk_s:{Labels.SCHEMA})
              <-[:{Edges.CONTAINS}]-(fk_db:{Labels.DB})
        WHERE true {fk_table_filter}
        RETURN fk_col.id AS id, fk_col.name AS name,
               fk_col.description AS description,
               fk_col.data_type AS data_type,
               fk_t.id AS table_id, fk_t.name AS table_name,
               fk_db.id AS database_id, fk_db.name AS database_name,
               fk_s.id AS schema_id, fk_s.name AS schema_name
        LIMIT 1
        """,
        params,
    )

    # The reverse of `fk_rows` above — every visible Column whose own
    # outgoing FOREIGN_KEY points *at* this Column (e.g. this is the
    # primary-key side of the relationship), enriched the same way. Without
    # this, double-clicking an FK *target* Column (the common case for a
    # table's primary key) shows nothing for the referencing side, even
    # though `fetch_data_exploration_edges`/the Relationships modal already
    # count it as a real edge in both directions. Reuses `fk_table_filter`
    # (identical `t.id IN $table_ids` shape, just against `src_t` instead of
    # `fk_t`) and `params`, same as `fk_rows`. Capped rather than paginated,
    # like every other field here — a shared lookup-style column (e.g. a
    # common `id`) could otherwise be referenced by very many source columns.
    src_table_filter = fk_table_filter.replace("fk_t.id", "src_t.id")
    referencing_rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN} {{id: $column_id}})
        WHERE true {table_filter}
        MATCH (col)<-[:{Edges.FOREIGN_KEY}]-(src_col:{Labels.COLUMN})
              <-[:{Edges.CONTAINS}]-(src_t:{Labels.TABLE})
              <-[:{Edges.CONTAINS}]-(src_s:{Labels.SCHEMA})
              <-[:{Edges.CONTAINS}]-(src_db:{Labels.DB})
        WHERE true {src_table_filter}
        RETURN DISTINCT src_col.id AS id, src_col.name AS name,
               src_col.description AS description,
               src_col.data_type AS data_type,
               src_t.id AS table_id, src_t.name AS table_name,
               src_db.id AS database_id, src_db.name AS database_name,
               src_s.id AS schema_id, src_s.name AS schema_name
        ORDER BY src_t.name, src_col.name, src_col.id
        LIMIT 50
        """,
        params,
    )

    # `params` already carries `table_ids` (when zone-restricted) from the
    # `table_filter` above — reused as-is since a Sql node's own visibility
    # rule (every table it references must be in scope) needs the exact
    # same accessible-table set, not a fresh resolution of it.
    sql_table_filter = ""
    if data_ids_by_zone is not None:
        sql_table_filter = f"""
            AND NOT EXISTS {{
                (sql)-[:{Edges.SQL}]->(tbl:{Labels.TABLE})
                WHERE NOT tbl.id IN $table_ids
            }}
        """
    sql_rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN} {{id: $column_id}})
        WHERE true {table_filter}
        MATCH (sql:{Labels.SQL})-[:{Edges.SQL}]->(col)
        WHERE true {sql_table_filter}
        RETURN DISTINCT sql.id AS id, sql.sql_full_query AS sql
        ORDER BY sql.id
        """,
        params,
    )

    return {
        "column_attribute": dict(attr_rows[0]) if attr_rows else None,
        "foreign_key_column": dict(fk_rows[0]) if fk_rows else None,
        "referencing_columns": [dict(row) for row in referencing_rows if row.get("id")],
        "sql_queries": [
            dict(row) for row in sql_rows if row.get("id") and row.get("sql")
        ],
    }


def fetch_column_attribute_exploration_details(
    attr_id: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return the Term owning a visible ColumnAttribute, and one ordered
    page of every Column linked to it.

    Closes the loop the other exploration ``fetch_*_details`` functions
    draw from a Column's own end: a Term's own expansion (`expandTermNode`
    in `ExplorationView.tsx`) already grafts a ColumnAttribute on via its
    `PROPERTY_OF` edge, and a Column's own expansion (`expandColumnNode`)
    grafts one on via its `HAS_ATTRIBUTE`/`SEMANTIC_FK` edge — either way,
    double-clicking that ColumnAttribute node in turn (see
    `expandColumnAttributeNode`) should reconnect it to its owning Term as
    well as *every* Column that shares it, not just the single "primary"
    one whichever expansion grafted the attribute on already knew about —
    a common attribute (e.g. a `user_id`-shaped one) is typically
    `HAS_ATTRIBUTE`/`SEMANTIC_FK`-linked from many Columns across many
    Tables at once, same as Neo4j Browser would show expanding the node
    directly.
    `term` is `None` when the attribute has no owning Term, or that Term
    falls outside *zone_ids* — mirrors ``term_is_in_scope``'s use in
    ``fetch_term_exploration_details``.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    term_rows = get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: $attr_id}})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        RETURN term.id AS id, term.name AS name, term.description AS description
        LIMIT 1
        """,
        {"attr_id": attr_id},
    )
    term = (
        dict(term_rows[0])
        if term_rows
        and term_is_in_scope(term_rows[0]["id"], zone_ids, data_ids_by_zone)
        else None
    )

    conn = get_neo4j_conn()
    table_filter = ""
    params: dict[str, Any] = {"attr_id": attr_id}
    if data_ids_by_zone is not None:
        table_filter = "AND t.id IN $table_ids"
        params["table_ids"] = list(data_ids_by_zone["table_ids"])

    linked_columns_subquery = f"""
        CALL () {{
            MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
                  -[col_rel:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->
                  (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: $attr_id}})
            WHERE true {table_filter}
            RETURN col.id AS id, type(col_rel) AS rel_type
        }}
    """
    total_rows = conn.query_read(
        f"""
        {linked_columns_subquery}
        RETURN count(DISTINCT id) AS total
        """,
        dict(params),
    )
    total = total_rows[0]["total"] if total_rows else 0
    if not total:
        return {"term": term, "columns": [], "columns_total": 0}

    paging = paging_clause(skip, limit, params)
    columns = conn.query_read(
        f"""
        {linked_columns_subquery}
        WITH id, collect(DISTINCT rel_type) AS relationship_types
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN} {{id: id}})
        WITH col, t, db, s, relationship_types
        ORDER BY t.name, col.name, col.id
        {paging}
        RETURN col.id AS id, col.name AS name, col.description AS description,
               col.data_type AS data_type,
               t.id AS table_id, t.name AS table_name,
               db.id AS database_id, db.name AS database_name,
               s.id AS schema_id, s.name AS schema_name,
               relationship_types
        """,
        params,
    )
    return {
        "term": term,
        "columns": [dict(row) for row in columns if row.get("id")],
        "columns_total": total,
    }


def fetch_sql_attribute_exploration_details(
    attr_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return the Sql query and owning Term behind a visible SqlAttribute.

    Closes the loop the other exploration ``fetch_*_details`` functions
    above draw from a Table/Column's own end: a Term's own expansion
    (`expandTermNode` in `ExplorationView.tsx`) already grafts a
    SqlAttribute on via its `PROPERTY_OF` edge, and this fetches what that
    SqlAttribute itself connects to — its `Sql` query node
    (`SqlAttribute-[HAS_SQL]->Sql`) and its owning Term
    (`SqlAttribute-[PROPERTY_OF]->Term`) — so double-clicking the
    SqlAttribute node in turn (see `expandSqlAttributeNode`) can graft both
    on. Mirrors `_sql_attr_zone_filter` in `gsf/dal/sql_attributes.py`: the
    Sql query only comes back when every table it references is visible
    through *zone_ids* — `sql` is `None` otherwise even when the
    attribute/term themselves are in scope.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    sql_table_filter = ""
    sql_params: dict[str, Any] = {"attr_id": attr_id}
    if data_ids_by_zone is not None:
        sql_table_filter = f"""
            AND NOT EXISTS {{
                (sql)-[:{Edges.SQL}]->(tbl:{Labels.TABLE})
                WHERE NOT tbl.id IN $table_ids
            }}
        """
        sql_params["table_ids"] = list(data_ids_by_zone["table_ids"])

    conn = get_neo4j_conn()
    sql_rows = conn.query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $attr_id}})
              -[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        WHERE true {sql_table_filter}
        RETURN sql.id AS id, sql.sql_full_query AS sql
        ORDER BY sql.id
        LIMIT 1
        """,
        sql_params,
    )
    term_rows = conn.query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $attr_id}})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        RETURN term.id AS id, term.name AS name, term.description AS description
        LIMIT 1
        """,
        {"attr_id": attr_id},
    )
    return {
        "sql": dict(sql_rows[0]) if sql_rows else None,
        "term": dict(term_rows[0]) if term_rows else None,
    }


def fetch_sql_exploration_details(
    sql_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return the CustomAnalysis, Column and SqlAttribute nodes hanging off
    a visible Sql node.

    A SqlAttribute's Sql node (``_persist_attr_with_sql`` in
    ``gsf/server/sql_attributes/service.py``) is MERGE-identified purely by
    its SQL text (``sql_node.match_props = {"sql_full_query": sql}``,
    resolved via ``apoc.merge.node.eager``) — so a CustomAnalysis created
    from that exact same SQL text (see ``_EXPORT_CUSTOM_ANALYSES_QUERY`` in
    ``gsf/dal/model_interchange.py`` for the same MERGE pattern) shares that
    very node rather than getting its own. This closes that loop from the
    Sql node's own end, for the Exploration graph's ``sql`` node expansion
    (see ``expandSqlNode`` in ``ExplorationView.tsx``) — most Sql nodes have
    no CustomAnalysis to show here, since sharing only happens when the
    exact SQL text was independently saved both ways.

    Also returns every visible Column the Sql node's own ``SQL`` edges reach
    directly (``Sql-[SQL]->Column``) — the same real relationship the
    ingestion pipeline draws for every column a parsed query references
    (``parse_query_slim`` in ``nemo_retriever``, alongside the identical
    ``Sql-[SQL]->Table`` edge ``fetch_data_exploration_edges`` already reads
    elsewhere), enriched with each column's owning Table/Schema/Database ids
    and names — same shape as ``fetch_column_attribute_exploration_details``'s
    own Column rows — so a client can graft one onto the graph as a fully
    expandable Column node instead of the expansion doing nothing for the
    common case of a Sql node with no CustomAnalysis at all.

    Also returns every visible Table the Sql node's own ``SQL`` edges reach
    directly (``Sql-[SQL]->Table``) — the very edge Neo4j Browser itself
    shows when expanding a ``Sql`` node (e.g. ``query_...-[:SQL]->orders``),
    and the same relationship ``fetch_data_exploration_edges`` reads to
    connect two tables sharing a query — enriched with each table's owning
    Schema/Database ids and names so a client can graft it on as a fully
    expandable Table node, same as ``expandTermNode``'s own Table
    neighbours in ``ExplorationView.tsx``.

    Additionally returns every visible SqlAttribute that this same Sql node
    backs (``SqlAttribute-[HAS_SQL]->Sql``) — the reverse of
    ``fetch_sql_attribute_exploration_details``'s own traversal — each with
    its owning Term's id/name, so the client can graft it on as a fully
    expandable SqlAttribute node (see ``expandSqlAttributeNode``) even when
    the Sql node was reached some other way (e.g. via a Column) rather than
    from that very SqlAttribute. A SqlAttribute whose own owning Term falls
    outside *zone_ids* is dropped, mirroring ``term_is_in_scope``'s use
    elsewhere.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    table_filter = ""
    params: dict[str, Any] = {"sql_id": sql_id}
    if data_ids_by_zone is not None:
        table_filter = f"""
            AND NOT EXISTS {{
                (sql)-[:{Edges.SQL}]->(tbl:{Labels.TABLE})
                WHERE NOT tbl.id IN $table_ids
            }}
        """
        params["table_ids"] = list(data_ids_by_zone["table_ids"])

    conn = get_neo4j_conn()
    rows = conn.query_read(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS})-[:{Edges.HAS_SQL}]->
              (sql:{Labels.SQL} {{id: $sql_id}})
        WHERE true {table_filter}
        RETURN ca.id AS id, ca.name AS name, ca.description AS description
        ORDER BY ca.name
        """,
        params,
    )

    column_table_filter = ""
    column_params: dict[str, Any] = {"sql_id": sql_id}
    if data_ids_by_zone is not None:
        column_table_filter = "AND t.id IN $table_ids"
        column_params["table_ids"] = list(data_ids_by_zone["table_ids"])
    column_rows = conn.query_read(
        f"""
        MATCH (sql:{Labels.SQL} {{id: $sql_id}})-[:{Edges.SQL}]->(col:{Labels.COLUMN})
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col)
        WHERE true {column_table_filter}
        RETURN DISTINCT col.id AS id, col.name AS name, col.description AS description,
               col.data_type AS data_type,
               t.id AS table_id, t.name AS table_name,
               db.id AS database_id, db.name AS database_name,
               s.id AS schema_id, s.name AS schema_name
        ORDER BY t.name, col.name, col.id
        """,
        column_params,
    )

    # Reuses `column_table_filter`/`column_params` above — the same
    # zone-visibility rule (`t.id IN $table_ids`) restricting the Column
    # rows' own owning tables applies unchanged to a Table reached
    # directly, since both walk the identical `Sql-[SQL]->` edge onto `t`.
    table_rows = conn.query_read(
        f"""
        MATCH (sql:{Labels.SQL} {{id: $sql_id}})-[:{Edges.SQL}]->(t:{Labels.TABLE})
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
        WHERE true {column_table_filter}
        RETURN DISTINCT t.id AS id, t.name AS name, t.table_type AS table_type,
               db.id AS database_id, db.name AS database_name,
               s.id AS schema_id, s.name AS schema_name
        ORDER BY t.name, t.id
        """,
        column_params,
    )

    attr_rows = conn.query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE})-[:{Edges.HAS_SQL}]->
              (sql:{Labels.SQL} {{id: $sql_id}})
        OPTIONAL MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        RETURN DISTINCT attr.id AS id, attr.name AS name,
               attr.description AS description,
               term.id AS term_id, term.name AS term_name
        ORDER BY attr.name
        """,
        {"sql_id": sql_id},
    )
    sql_attributes = [
        dict(row)
        for row in attr_rows
        if row.get("id")
        and (
            row.get("term_id") is None
            or term_is_in_scope(row["term_id"], zone_ids, data_ids_by_zone)
        )
    ]

    return {
        "custom_analyses": [dict(row) for row in rows if row.get("id")],
        "columns": [dict(row) for row in column_rows if row.get("id")],
        "tables": [dict(row) for row in table_rows if row.get("id")],
        "sql_attributes": sql_attributes,
    }


def fetch_exploration_related_nodes(
    node_id: str,
    layer: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return one page of nodes related to an Exploration node.

    This deliberately resolves relationships from the uncapped graph
    definitions rather than from either Exploration graph endpoint. Data nodes
    share SQL or a foreign key; semantic nodes share a table through any of
    the three Term-to-table paths used by ``fetch_related_terms``.
    """
    if layer == "data":
        return _fetch_data_exploration_related_nodes(
            node_id, zone_ids=zone_ids, skip=skip, limit=limit
        )
    if layer == "semantic":
        return _fetch_semantic_exploration_related_nodes(
            node_id, zone_ids=zone_ids, skip=skip, limit=limit
        )
    raise ValueError(f"Unsupported Exploration layer: {layer}")


def _data_related_tables_match(other_filter: str) -> str:
    """Cypher matching every Table related to ``$node_id``, binding ``t``/``db``/``s``.

    Related means what ``fetch_data_exploration_edges`` means by it: a shared
    SQL query with text, or a foreign key in either direction — so a node's
    degree on the graph and the size of its related list are the same number.
    The catalog path is OPTIONAL for that reason: requiring it would drop a
    related table that has no Schema/Database above it from the page while
    the graph still counted it.
    """
    return f"""
        CALL () {{
            MATCH (:{Labels.TABLE} {{id: $node_id}})<-[:{Edges.SQL}]-(sql:{Labels.SQL})
                  -[:{Edges.SQL}]->(other:{Labels.TABLE})
            WHERE other.id <> $node_id AND {_NON_EMPTY_SQL} {other_filter}
            RETURN DISTINCT other.id AS related_id
            UNION
            MATCH (:{Labels.TABLE} {{id: $node_id}})-[:{Edges.CONTAINS}]->
                  (:{Labels.COLUMN})-[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})
                  <-[:{Edges.CONTAINS}]-(other:{Labels.TABLE})
            WHERE other.id <> $node_id {other_filter}
            RETURN DISTINCT other.id AS related_id
            UNION
            MATCH (other:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
                  -[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})
                  <-[:{Edges.CONTAINS}]-(:{Labels.TABLE} {{id: $node_id}})
            WHERE other.id <> $node_id {other_filter}
            RETURN DISTINCT other.id AS related_id
        }}
        WITH DISTINCT related_id
        MATCH (t:{Labels.TABLE} {{id: related_id}})
        OPTIONAL MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
    """


def _fetch_data_exploration_related_nodes(
    node_id: str,
    zone_ids: list[str] | None,
    *,
    skip: int,
    limit: int | None,
) -> dict[str, Any]:
    """Page Tables related to *node_id* through a shared SQL query or a foreign key.

    Every query here is scoped to ``$node_id`` (or, for
    ``_fetch_table_relationship_counts``, to the tables on the page just
    fetched) rather than building the entire data-layer edge set through
    ``fetch_data_exploration_edges`` and filtering it down to one node in
    Python.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if data_ids_by_zone is not None and node_id not in data_ids_by_zone["table_ids"]:
        return {"nodes": [], "total": 0}

    other_filter = ""
    zone_table_ids: list[str] | None = None
    params: dict[str, Any] = {"node_id": node_id}
    if data_ids_by_zone is not None:
        zone_table_ids = list(data_ids_by_zone["table_ids"])
        other_filter = "AND other.id IN $zone_table_ids"
        params["zone_table_ids"] = zone_table_ids

    conn = get_neo4j_conn()
    related_match = _data_related_tables_match(other_filter)
    total_rows = conn.query_read(
        f"""
        {related_match}
        RETURN count(DISTINCT related_id) AS total
        """,
        params,
    )
    total = total_rows[0]["total"] if total_rows else 0
    if not total:
        return {"nodes": [], "total": 0}

    paging = paging_clause(skip, limit, params)
    rows = conn.query_read(
        f"""
        {related_match}
        WITH related_id, t, db, s
        ORDER BY db.id, s.id
        WITH related_id     AS id,
             t.name         AS name,
             t.table_type   AS table_type,
             head(collect(db.id)) AS database_id,
             head(collect(s.id))  AS schema_id
        RETURN id, name, table_type, database_id, schema_id
        ORDER BY name, id
        {paging}
        """,
        params,
    )
    relationship_counts = _fetch_table_relationship_counts(
        [row["id"] for row in rows], zone_table_ids=zone_table_ids
    )
    return {
        "nodes": [
            {
                **dict(row),
                "relationship_count": relationship_counts.get(row["id"], 0),
            }
            for row in rows
        ],
        "total": total,
    }


def _fetch_table_relationship_counts(
    table_ids: list[str],
    zone_table_ids: list[str] | None,
) -> dict[str, int]:
    """Return ``{table_id: relationship_count}`` for a small, known set of tables.

    Companion to ``_fetch_data_exploration_related_nodes``: rendering its
    "Relationships" column needs each *returned* table's own total degree,
    under the same definition of an edge, which this re-derives with three
    ``UNWIND``-scoped scans bounded by the page just fetched.
    """
    if not table_ids:
        return {}
    conn = get_neo4j_conn()
    params: dict[str, Any] = {"table_ids": table_ids}
    other_filter = ""
    if zone_table_ids is not None:
        other_filter = "AND other.id IN $zone_table_ids"
        params["zone_table_ids"] = zone_table_ids

    sql_rows = conn.query_read(
        f"""
        UNWIND $table_ids AS tid
        MATCH (:{Labels.TABLE} {{id: tid}})<-[:{Edges.SQL}]-(sql:{Labels.SQL})
              -[:{Edges.SQL}]->(other:{Labels.TABLE})
        WHERE other.id <> tid AND {_NON_EMPTY_SQL} {other_filter}
        RETURN DISTINCT tid AS id, other.id AS other_id
        """,
        params,
    )
    fk_out_rows = conn.query_read(
        f"""
        UNWIND $table_ids AS tid
        MATCH (:{Labels.TABLE} {{id: tid}})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})<-[:{Edges.CONTAINS}]-
              (other:{Labels.TABLE})
        WHERE other.id <> tid {other_filter}
        RETURN DISTINCT tid AS id, other.id AS other_id
        """,
        params,
    )
    fk_in_rows = conn.query_read(
        f"""
        UNWIND $table_ids AS tid
        MATCH (other:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})<-[:{Edges.CONTAINS}]-
              (:{Labels.TABLE} {{id: tid}})
        WHERE other.id <> tid {other_filter}
        RETURN DISTINCT tid AS id, other.id AS other_id
        """,
        params,
    )
    neighbours: dict[str, set[str]] = {}
    for row in (*sql_rows, *fk_out_rows, *fk_in_rows):
        neighbours.setdefault(row["id"], set()).add(row["other_id"])
    return {tid: len(others) for tid, others in neighbours.items()}


def _fetch_semantic_exploration_related_nodes(
    node_id: str,
    zone_ids: list[str] | None,
    *,
    skip: int,
    limit: int | None,
) -> dict[str, Any]:
    """Page Terms related under the existing three-path, zone-safe semantics.

    ``term_is_in_scope`` checks just ``node_id`` rather than building the
    whole in-scope set via ``fetch_all_terms``, and the relationship
    counts are scoped to the page just fetched.

    Which terms are related has to be resolved in full — it is what ``total``
    counts — but only the page's own rows are read: ``fetch_related_term_ids``
    yields ids, and ``fetch_terms_by_ids`` orders and pages them in Neo4j
    under the same rule ``fetch_related_terms`` uses, so a term can't move
    between pages.

    *zone_ids* is resolved to accessible catalog ids exactly once here (see
    ``resolve_accessible_catalog_ids``) and threaded through the visibility
    check, the related-id pass and the counts — five separate resolutions of
    the same zones, otherwise, for one request.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if not term_is_in_scope(node_id, zone_ids, data_ids_by_zone):
        return {"nodes": [], "total": 0}

    related_ids = fetch_related_term_ids(node_id, zone_ids, data_ids_by_zone)
    if not related_ids:
        return {"nodes": [], "total": 0}

    page = fetch_terms_by_ids(related_ids, skip=skip, limit=limit)
    page_ids = [term["id"] for term in page]

    relationship_counts = _fetch_term_relationship_counts(
        page_ids, zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone
    )
    return {
        "nodes": [
            {
                "id": term["id"],
                "name": term.get("name") or "",
                "relationship_count": relationship_counts.get(term["id"], 0),
            }
            for term in page
        ],
        "total": len(related_ids),
    }


def _fetch_term_relationship_counts(
    term_ids: list[str],
    zone_ids: list[str] | None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> dict[str, int]:
    """Return ``{term_id: relationship_count}`` for a small, known set of terms.

    Companion to ``_fetch_semantic_exploration_related_nodes``: rendering its
    "Relationships" column needs each *returned* term's own count of related
    terms. Reuses the Terms list's own count so a term's number is the same
    wherever it is shown.
    """
    return {
        row["term_id"]: row["count"]
        for row in fetch_related_terms_counts(
            zone_ids, term_ids=term_ids, data_ids_by_zone=data_ids_by_zone
        )
    }


def fetch_table_zones_map(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
    table_ids: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{table_id: [zone, ...]}`` for every visible Table.

    Zones are resolved via ``Zone -[ZONE_OF]-> item`` where *item* is the
    table itself or one of its ancestors (Schema, Database), matching the
    resolution used by ``get_full_term_by_id`` / ``get_full_sql_attribute_by_id``.
    ``item`` is found by walking ``CONTAINS`` backwards from ``t`` (0..2 hops,
    0 meaning ``item`` is ``t`` itself) rather than matching ``t`` and ``item``
    independently and filtering afterwards — keeping every ``MATCH`` chained
    through a shared variable avoids a cartesian product between all tables
    and all zone/item pairs on large catalogs.
    Used to render Zone chips in the Exploration graph without a per-node
    request. When *zone_ids* is supplied, both the visible tables and the
    returned zone names/colors are restricted to that set. Disabled zones
    are included (with ``enabled: False``) so admins can see and manage
    them; they never grant access since *zone_ids* itself is computed from
    enabled zones only. Pass ``None`` to return zones for every table
    (admin / internal callers). Pass a pre-resolved *data_ids_by_zone*
    (see ``resolve_accessible_catalog_ids``) when the caller already
    resolved *zone_ids* for this request, to skip a repeat Neo4j round trip.
    Pass *table_ids* to further restrict the scan to a known set of tables
    (e.g. the owning tables of a batch of ColumnAttributes); it is
    intersected with the zone-accessible tables so it never widens access.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    params: dict[str, Any] = {}
    filter_ids: set[str] | None = None
    if data_ids_by_zone is not None:
        filter_ids = set(data_ids_by_zone["table_ids"])
    if table_ids is not None:
        filter_ids = (
            set(table_ids) if filter_ids is None else filter_ids & set(table_ids)
        )
    table_filter = ""
    if filter_ids is not None:
        if not filter_ids:
            return {}
        table_filter = "WHERE t.id IN $table_ids"
        params["table_ids"] = list(filter_ids)

    # A viewer never sees a disabled zone's chip, even if its id ended up in
    # zone_ids (e.g. access granted before the zone was disabled) — admins
    # (zone_ids=None) still see disabled zones so they can manage them.
    zone_filter = (
        ""
        if zone_ids is None
        else f"WHERE z.id IN $zone_ids AND NOT z:{LABEL_ZONE_DISABLED}"
    )
    if zone_ids is not None:
        params["zone_ids"] = zone_ids

    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})
        {table_filter}
        MATCH (item)-[:{Edges.CONTAINS}*0..2]->(t)
        MATCH (z:{ZONE_LABEL_PATTERN})-[:{REL_ZONE_OF}]->(item)
        {zone_filter}
        RETURN DISTINCT t.id   AS table_id,
                        z.id    AS id,
                        z.name  AS name,
                        z.color AS color,
                        NOT z:{LABEL_ZONE_DISABLED} AS enabled
        ORDER BY t.id, z.name
        """,
        params,
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(row["table_id"], []).append(
            {
                "id": row["id"],
                "name": row["name"],
                "color": row["color"],
                "enabled": row["enabled"],
            }
        )
    return result


def fetch_data_exploration_graph(
    zone_ids: list[str] | None = None,
    limit: int = MAX_EXPLORATION_GRAPH_NODES,
) -> dict[str, list[dict[str, Any]]]:
    """Return the whole data-layer Exploration graph in one payload.

    Builds ``{"nodes": [...], "links": [...]}`` server-side so the client
    renders the data graph from a single request instead of walking the
    catalog tree (databases → schemas → tables) with one request per level.

    Each node is a visible Table with its column / SQL / Term counts, its
    relationship count, owning Database and Schema ids and names, and
    resolved Zone chips. Links are the SQL- and foreign-key-backed table
    connections from ``fetch_data_exploration_edges``. When *zone_ids* is
    supplied both nodes and links are restricted to tables reachable through
    those zones.

    *limit* caps the number of nodes returned — always clamped to
    ``MAX_EXPLORATION_GRAPH_NODES`` regardless of what's passed in, so the
    response stays bounded on catalogs with many tables. Nodes are ordered
    by name before truncating for a deterministic result, and links are
    filtered to only pairs where both ends are among the returned nodes.
    Relationship counts are computed before that truncation, so a table's
    number matches the related-nodes endpoint rather than the subset drawn.

    *zone_ids* is resolved to accessible catalog ids exactly once (see
    ``resolve_accessible_catalog_ids``) and threaded through the node query,
    ``fetch_table_zones_map`` and ``fetch_data_exploration_edges``, each of
    which would otherwise re-resolve it for itself.
    """
    limit = max(1, min(limit, MAX_EXPLORATION_GRAPH_NODES))
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    where_clause, params = resolve_table_filter(
        zone_ids, "t.id", data_ids_by_zone=data_ids_by_zone
    )
    nodes = get_neo4j_conn().query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE})
        {where_clause}
        {TABLE_COUNTS_SUBQUERY}
        RETURN t.id AS id,
               t.name AS name,
               t.table_type AS table_type,
               db.id AS database_id,
               db.name AS database_name,
               s.id AS schema_id,
               s.name AS schema_name,
               {table_description_expr("t")} AS description,
               columns_count,
               sql_count,
               size(unique_term_ids) AS terms_count
        ORDER BY name
        LIMIT $limit
        """,
        {**params, "limit": limit},
    )
    node_ids = {row["id"] for row in nodes}
    zones_by_table = fetch_table_zones_map(zone_ids, data_ids_by_zone=data_ids_by_zone)
    edges = fetch_data_exploration_edges(
        zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone
    )
    # Degree over the whole accessible edge set, not just the links returned:
    # `limit` can leave a neighbour out of the payload, and this has to agree
    # with what ``fetch_exploration_related_nodes`` reports for the same table.
    relationship_counts: Counter[str] = Counter()
    for edge in edges:
        relationship_counts[edge["source"]] += 1
        relationship_counts[edge["target"]] += 1
    return {
        "nodes": [
            {
                **dict(row),
                "zones": zones_by_table.get(row["id"], []),
                "relationship_count": relationship_counts[row["id"]],
            }
            for row in nodes
        ],
        "links": [
            edge
            for edge in edges
            if edge["source"] in node_ids and edge["target"] in node_ids
        ],
    }


def fetch_semantic_exploration_graph(
    zone_ids: list[str] | None = None,
    limit: int = MAX_EXPLORATION_GRAPH_NODES,
) -> dict[str, list[dict[str, Any]]]:
    """Return the whole semantic-layer Exploration graph in one payload.

    Builds ``{"nodes": [...], "links": [...]}`` server-side so the client
    renders the semantic graph from a single request instead of fetching
    related terms once per node (an N+1 over every term).

    Each node is a Term with its resolved Zone chips, related-term count and
    ColumnAttribute / SqlAttribute counts. Links are the undirected
    term↔term relationships (two terms sharing at least one table) between
    visible terms, each carrying ``relationship_types`` — the underlying
    Neo4j relationship type(s) (``REPRESENTS``, ``HAS_ATTRIBUTE``,
    ``SEMANTIC_FK``) connecting either term to a table they share, from
    ``fetch_term_table_pairs`` — so a client can label the connection the
    same way it would read in the graph itself. All counts reuse the same
    helpers as the ``/terms`` list and the per-term counts endpoints, so
    numbers match across pages. When *zone_ids* is supplied nodes, counts
    and links are all zone-scoped.

    *limit* caps the number of nodes returned — always clamped to
    ``MAX_EXPLORATION_GRAPH_NODES`` regardless of what's passed in, so the
    response stays bounded on catalogs with many terms. Relationship counts
    are computed from the full (untruncated) graph so numbers stay accurate
    even when a related term itself falls outside the returned set; nodes
    are then ordered by name and truncated, and links are filtered to only
    pairs where both ends are among the returned nodes.

    *zone_ids* is resolved to accessible catalog ids exactly once (see
    ``resolve_accessible_catalog_ids``) and threaded through every sub-query
    below, each of which would otherwise re-resolve it for itself.
    """
    limit = max(1, min(limit, MAX_EXPLORATION_GRAPH_NODES))
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)

    terms = fetch_all_terms(zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone)
    column_counts = {
        row["term_id"]: row["count"]
        for row in fetch_column_attribute_counts(
            zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone
        )
    }
    sql_counts = {
        row["term_id"]: row["count"]
        for row in fetch_sql_attribute_counts(
            zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone
        )
    }

    term_table_pairs = fetch_term_table_pairs(
        zone_ids, data_ids_by_zone=data_ids_by_zone
    )
    term_tables, table_terms = build_term_table_maps(term_table_pairs)
    # `path` is which Neo4j relationship type connected this particular
    # (term, table) pair — kept per-pair (rather than folded into
    # `term_tables`/`table_terms` above) so a term↔term edge below can
    # report *why* each side reaches their shared table.
    path_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in term_table_pairs:
        term_id, table_id, path = (
            row.get("term_id"),
            row.get("table_id"),
            row.get("path"),
        )
        if term_id and table_id and path:
            path_by_pair.setdefault((term_id, table_id), set()).add(path)

    visible_term_ids = {term["id"] for term in terms}
    relationship_counts: dict[str, int] = {}
    link_keys: set[tuple[str, str]] = set()
    link_relationship_types: dict[tuple[str, str], set[str]] = {}
    for term_id, tables in term_tables.items():
        related: set[str] = set()
        for table_id in tables:
            related.update(table_terms.get(table_id, set()))
        related.discard(term_id)
        relationship_counts[term_id] = len(related)
        if term_id not in visible_term_ids:
            continue
        for other_id in related:
            if other_id not in visible_term_ids:
                continue
            key = tuple(sorted((term_id, other_id)))
            link_keys.add(key)
            types = link_relationship_types.setdefault(key, set())
            for table_id in tables & term_tables.get(other_id, set()):
                types.update(path_by_pair.get((term_id, table_id), ()))
                types.update(path_by_pair.get((other_id, table_id), ()))

    nodes = sorted(
        (
            {
                "id": term["id"],
                "name": term["name"],
                "description": term.get("description"),
                "synonyms": term.get("synonyms") or [],
                "zones": term.get("zones") or [],
                "relationship_count": relationship_counts.get(term["id"], 0),
                "column_attributes_count": column_counts.get(term["id"], 0),
                "sql_attributes_count": sql_counts.get(term["id"], 0),
            }
            for term in terms
        ),
        key=lambda node: node["name"],
    )[:limit]
    kept_ids = {node["id"] for node in nodes}
    links = [
        {
            "source": source,
            "target": target,
            "relationship_types": sorted(
                link_relationship_types.get((source, target), set())
            ),
        }
        for source, target in sorted(link_keys)
        if source in kept_ids and target in kept_ids
    ]
    return {"nodes": nodes, "links": links}


# Maps a path node's raw Neo4j label (see `find_term_link_path`) to the
# lowerCamelCase `type` the Exploration graph's node kinds already use on
# the client (`GraphCanvas.tsx`'s own `NodeType`), so a client can graft/
# style a path node exactly like any other node of that type without its
# own label → type lookup.
_LINK_PATH_NODE_TYPE = {
    LABEL_TERM: "term",
    Labels.TABLE: "table",
    Labels.COLUMN: "column",
    LABEL_COLUMN_ATTRIBUTE: "columnAttribute",
}


def _link_path_node(node: dict[str, Any]) -> dict[str, Any]:
    """Project one `find_term_link_path` path node into the API node-ref shape."""
    label = node.get("label")
    result = {
        "id": node.get("id"),
        "name": node.get("name"),
        "type": _LINK_PATH_NODE_TYPE.get(label, (label or "").lower()),
    }
    if label in (Labels.TABLE, Labels.COLUMN):
        # Only Table/Column nodes carry these (see
        # `_enrich_catalog_path_nodes` in `gsf/dal/attributes.py`) — lets a
        # client expand either further the same way any other Table/Column
        # node's expansion does, instead of staying a dead end just because
        # it came from a link-path graft.
        result["database_id"] = node.get("database_id")
        result["database_name"] = node.get("database_name")
        result["schema_id"] = node.get("schema_id")
        result["schema_name"] = node.get("schema_name")
    if label == Labels.COLUMN:
        # Only a Column node carries these — its own owning Table, which a
        # Table node itself obviously doesn't need.
        result["table_id"] = node.get("table_id")
        result["table_name"] = node.get("table_name")
    return result


def fetch_semantic_link_path(
    source_term_id: str,
    target_term_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return the real hop chain behind one term↔term Exploration graph edge.

    ``fetch_semantic_exploration_graph`` collapses every shared table (and
    every path type reaching it from either term) into one ``' / '``-joined
    ``relationship_types`` label — accurate, but it can't show *which*
    Table/Column/ColumnAttribute actually connects the two terms. This
    instead returns ``find_term_link_path``'s real ordered node chain
    (e.g. Term1 <-REPRESENTS- Table -CONTAINS-> Column -SEMANTIC_FK->
    ColumnAttribute -PROPERTY_OF-> Term2), so a client can graft those
    nodes onto the graph and highlight the actual path instead of just the
    collapsed label.

    Every Table/Column node the path passes through must fall inside
    *zone_ids* (mirroring ``term_is_in_scope``'s use elsewhere) — the path
    is dropped entirely rather than partially shown when one doesn't,
    since a partial chain that silently skips an out-of-scope hop would
    misrepresent how the two terms actually connect. Checking `Column`
    nodes here too (not just `Table`) matters because `SEMANTIC_FK`/
    `HAS_ATTRIBUTE` are traversed undirected (see `find_term_link_path`'s
    own doc comment) — a path can reach a Column from a ColumnAttribute on
    *both* sides (e.g. `Attr1 -[HAS_ATTRIBUTE]- Column -[HAS_ATTRIBUTE]-
    Attr2`) without ever stepping through its own `CONTAINS` edge, so its
    owning Table never appears as a hop node of its own to catch.
    """
    hops = find_term_link_path(source_term_id, target_term_id)
    if not hops:
        return {"hops": []}

    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if data_ids_by_zone is not None:
        table_ids = data_ids_by_zone["table_ids"]
        for hop in hops:
            for side in (hop["source"], hop["target"]):
                label = side.get("label")
                if label == Labels.TABLE and side.get("id") not in table_ids:
                    return {"hops": []}
                # `table_id` is already populated on every Column node by
                # `find_term_link_path`'s own `_enrich_catalog_path_nodes`
                # call, so this needs no extra Neo4j round trip.
                if label == Labels.COLUMN and side.get("table_id") not in table_ids:
                    return {"hops": []}

    return {
        "hops": [
            {
                "relationship": hop["relationship"],
                "source": _link_path_node(hop["source"]),
                "target": _link_path_node(hop["target"]),
            }
            for hop in hops
        ]
    }
