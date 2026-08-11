# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot every DAL read against the graph fixture, for the Postgres port.

The refactor replaces ~5,000 lines of Cypher with SQL while keeping every
``gsf.dal`` signature identical. Greenfield cutover means there is no production
data to diff against, so these captures are the **only** fidelity oracle: Phases
5-10 are graded by replaying them against the Postgres implementation and
requiring identical output.

**Ids are normalised.** Node ids are ``randomUUID()``, so a raw capture would
differ on every run and compare equal to nothing. Each id is replaced by a
stable token derived from what the node *is* — ``<table:pagila.public.film>``,
``<term:Film>`` — which also makes a failing diff readable, and means the
Postgres implementation is free to generate different uuids as long as the
graph shape matches.

Usage::

    uv run --no-sync python -m dev_tools.seed_graph_fixture --reset
    uv run --no-sync python -m dev_tools.capture_dal_golden
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("dev_tools.capture_dal_golden")

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "gsf" / "dal" / "tests" / "golden"

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


# ---------------------------------------------------------------------------
# Id normalisation
# ---------------------------------------------------------------------------


def build_id_map() -> dict[str, str]:
    """Map every node id to a stable token describing what the node is."""
    from gsf.dal.neo4j_tx import graph

    id_map: dict[str, str] = {}

    def add(rows: list[dict[str, Any]], template: str) -> None:
        for row in rows:
            node_id = row.get("id")
            if node_id:
                id_map[node_id] = template.format(**row)

    add(
        graph().query_read("MATCH (d:Database) RETURN d.id AS id, d.name AS name"),
        "<db:{name}>",
    )
    add(
        graph().query_read(
            "MATCH (d:Database)-[:CONTAINS]->(s:Schema) "
            "RETURN s.id AS id, d.name AS db, s.name AS name"
        ),
        "<schema:{db}.{name}>",
    )
    add(
        graph().query_read(
            "MATCH (d:Database)-[:CONTAINS]->(s:Schema)-[:CONTAINS]->(t:Table) "
            "RETURN t.id AS id, d.name AS db, s.name AS schema, t.name AS name"
        ),
        "<table:{db}.{schema}.{name}>",
    )
    add(
        graph().query_read(
            "MATCH (d:Database)-[:CONTAINS]->(s:Schema)-[:CONTAINS]->(t:Table)"
            "-[:CONTAINS]->(c:Column) "
            "RETURN c.id AS id, d.name AS db, s.name AS schema, "
            "t.name AS table, c.name AS name"
        ),
        "<col:{db}.{schema}.{table}.{name}>",
    )
    add(
        graph().query_read("MATCH (t:Term) RETURN t.id AS id, t.name AS name"),
        "<term:{name}>",
    )
    add(
        graph().query_read(
            "MATCH (a:ColumnAttribute) RETURN a.id AS id, a.name AS name, "
            "a.term_name AS term, a.source_column AS col"
        ),
        "<attr:{term}/{name}/{col}>",
    )
    add(
        graph().query_read("MATCH (a:SqlAttribute) RETURN a.id AS id, a.name AS name"),
        "<sqlattr:{name}>",
    )
    add(
        graph().query_read(
            "MATCH (a:CustomAnalysis) RETURN a.id AS id, a.name AS name"
        ),
        "<analysis:{name}>",
    )
    add(
        graph().query_read("MATCH (p:PqlAnalysis) RETURN p.id AS id, p.name AS name"),
        "<pql:{name}>",
    )
    add(
        graph().query_read(
            "MATCH (z:Zone|disableZone) RETURN z.id AS id, z.name AS name"
        ),
        "<zone:{name}>",
    )

    # Sql nodes carry no name; key them by their statement so the token is
    # stable across runs regardless of insertion order.
    for row in graph().query_read(
        "MATCH (s:Sql) RETURN s.id AS id, s.sql_full_query AS q ORDER BY s.sql_full_query"
    ):
        if row.get("id"):
            digest = re.sub(r"\s+", " ", (row.get("q") or "")).strip()[:60]
            id_map[row["id"]] = f"<sql:{digest}>"

    return id_map


# Keys whose value is wall-clock and therefore differs on every ingest. Redacted
# rather than dropped, so a port that stops populating them still fails.
VOLATILE_KEYS = frozenset({"created", "updated", "pulled", "last_seen", "timestamp"})


def _orient_edge(edge: dict[str, Any]) -> dict[str, Any]:
    """Put an undirected edge's ends in a stable order.

    Exploration edges are undirected, and the DAL already canonicalises them —
    ``fetch_data_exploration_edges`` selects ``WHERE source.id < target.id``,
    and the semantic graph stores ``tuple(sorted((a, b)))``. Both compare
    **uuids**, so which end lands in ``source`` is stable for a given database
    but flips whenever the fixture is rebuilt with fresh ids.

    Re-orienting by token restores determinism without hiding anything, but the
    swap has to be total: an edge also carries ``source_column`` /
    ``target_column`` and their sample values, and swapping the ends while
    leaving those put would describe an edge that does not exist.
    """
    if str(edge["source"]) <= str(edge["target"]):
        return edge

    def flip(value: Any) -> Any:
        if isinstance(value, list):
            return [flip(item) for item in value]
        if not isinstance(value, dict):
            return value
        swapped: dict[str, Any] = {}
        for key, val in value.items():
            if key.startswith("source"):
                swapped["target" + key[len("source") :]] = flip(val)
            elif key.startswith("target"):
                swapped["source" + key[len("target") :]] = flip(val)
            else:
                swapped[key] = flip(val)
        return swapped

    return flip(edge)


def normalise(value: Any, id_map: dict[str, str]) -> Any:
    """Make a DAL result comparable across runs.

    Four transformations, each for a specific source of noise:

    * **ids → tokens.** Node ids are ``randomUUID()``. An unmapped uuid becomes
      ``<unmapped-uuid>`` rather than being left alone, since leaving it would
      fail the next run for a reason unrelated to the code under test.
    * **sets → sorted lists.** Several reads return ``set`` objects, which have
      no stable iteration order and would otherwise be serialised via ``str()``
      *after* normalisation — leaving raw uuids inside a string.
    * **volatile keys redacted.** Ingest timestamps differ every run.
    * **lists sorted canonically.** See the caveat below.

    .. warning::
       Sorting lists means **these goldens do not cover result ordering**. That
       is deliberate: most DAL queries lack a total ``ORDER BY`` (see
       ``fetch_sorted_tables``, which orders only by ``query_count DESC`` while
       nearly every table ties at 0), so freezing an arbitrary observed order
       would fail the Postgres port for behaviour Neo4j never guaranteed.
       Ordering that *is* contractual — paging stability across ``skip``/
       ``limit`` — needs its own tests, per PLAN.md's invariants suite.
    """
    # DataFrames must be unpacked before anything else. A few reads return them,
    # and falling through to ``default=str`` at serialisation would capture a
    # *truncated repr* -- ids inside it were never seen by normalise, and the
    # "..." elides most columns entirely.
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        return normalise(value.to_dict(orient="records"), id_map)

    if isinstance(value, dict):
        # Keys are normalised too: several reads return maps *keyed by node id*
        # (``fetch_table_zones_map``, ``fetch_col_table_contexts``), so leaving
        # keys alone would leave raw uuids in the golden.
        normalised = {
            normalise(k, id_map): (
                "<redacted>" if k in VOLATILE_KEYS else normalise(v, id_map)
            )
            for k, v in value.items()
        }
        if {"source", "target"} <= set(normalised):
            normalised = _orient_edge(normalised)
        return dict(sorted(normalised.items(), key=lambda kv: str(kv[0])))
    if isinstance(value, (set, frozenset)):
        return sorted(
            (normalise(v, id_map) for v in value),
            key=lambda item: json.dumps(item, default=str),
        )
    if isinstance(value, (list, tuple)):
        return sorted(
            (normalise(v, id_map) for v in value),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if isinstance(value, str):
        if value in id_map:
            return id_map[value]
        if UUID_RE.search(value):
            return UUID_RE.sub(
                lambda m: id_map.get(m.group(0), "<unmapped-uuid>"), value
            )
        return value
    return value


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class Capture:
    """Collects ``name -> normalised result`` for one golden file."""

    def __init__(self, id_map: dict[str, str]) -> None:
        self._id_map = id_map
        self.results: dict[str, Any] = {}
        self.errors: dict[str, str] = {}

    def run(self, name: str, fn: Callable[[], Any]) -> None:
        try:
            self.results[name] = normalise(fn(), self._id_map)
        except Exception as exc:  # noqa: BLE001 — a raising read is itself a fact
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            logger.warning("  %s raised %s: %s", name, type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# What to capture
# ---------------------------------------------------------------------------

# Zero-argument reads. ``reset.delete_*`` and ``neo4j_tx.*`` are deliberately
# absent: the first three would wipe the fixture mid-capture, and the last two
# are plumbing rather than reads.
ZERO_ARG_READS: tuple[tuple[str, str], ...] = (
    ("attributes", "find_unlinked_fk_columns"),
    ("connections", "list_connections"),
    ("custom_analyses", "fetch_custom_analyses"),
    ("datasources", "fetch_all_schema_ids"),
    ("datasources", "fetch_all_tables_without_term"),
    ("datasources", "fetch_join_edges"),
    ("datasources", "fetch_sorted_tables"),
    ("pql_analyses", "list_pql_analyses"),
    ("sql_attributes", "list_sql_attributes"),
    ("terms", "fetch_terms_with_sqls"),
    ("terms", "semantic_layer_calculated"),
    ("zones", "list_zones"),
)

# Reads whose only parameter is zone scoping, captured under every zone mode.
ZONE_SCOPED_READS: tuple[tuple[str, str], ...] = (
    ("custom_analyses", "list_custom_analyses"),
    ("datasources", "fetch_databases"),
    ("exploration", "fetch_data_exploration_edges"),
    ("exploration", "fetch_data_exploration_graph"),
    ("exploration", "fetch_semantic_exploration_graph"),
    ("exploration", "fetch_table_zones_map"),
    ("sql_attributes", "fetch_sql_attribute_counts"),
    ("terms", "count_terms"),
    ("terms", "fetch_all_terms"),
    ("terms", "fetch_all_terms_and_attributes"),
    ("terms", "fetch_column_attribute_counts"),
    ("terms", "fetch_related_terms_counts"),
    ("terms", "fetch_term_table_pairs"),
    ("terms", "fetch_term_zones_map"),
)


def _zone_modes(ids: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """The three zone-scoping modes every scoped read is captured under.

    ``admin`` (``None``) sees everything, ``scoped`` sees one zone, ``none``
    (``[]``) sees nothing. A golden covering only the admin path would miss
    precisely the access-control regression that matters most.
    """
    return (("admin", None), ("scoped", [ids["zone_film"]]), ("none", []))


def _fixture_ids() -> dict[str, Any]:
    """Resolve the fixture's entities by name, so captures read declaratively."""
    from gsf.dal.neo4j_tx import graph

    def one(query: str, **params: Any) -> str | None:
        rows = graph().query_read(query, params)
        return rows[0]["id"] if rows else None

    ids: dict[str, Any] = {
        "db_pagila": one("MATCH (d:Database {name:'pagila'}) RETURN d.id AS id"),
        "schema_public": one(
            "MATCH (:Database {name:'pagila'})-[:CONTAINS]->(s:Schema {name:'public'}) "
            "RETURN s.id AS id"
        ),
        "attr_film_id": one(
            "MATCH (a:ColumnAttribute {name:'film id'}) RETURN a.id AS id"
        ),
        "sqlattr_revenue": one(
            "MATCH (a:SqlAttribute {name:'total revenue'}) RETURN a.id AS id"
        ),
        "analysis_top_films": one(
            "MATCH (a:CustomAnalysis {name:'Top rented films'}) RETURN a.id AS id"
        ),
        "pql_churn": one("MATCH (p:PqlAnalysis) RETURN p.id AS id"),
        "zone_film": one("MATCH (z:Zone {name:'Film domain'}) RETURN z.id AS id"),
        "zone_cross": one("MATCH (z:Zone {name:'Cross database'}) RETURN z.id AS id"),
        "zone_retired": one(
            "MATCH (z:disableZone {name:'Retired zone'}) RETURN z.id AS id"
        ),
    }

    for table in ("film", "rental", "customer", "payment", "inventory", "category"):
        ids[f"table_{table}"] = one(
            "MATCH (:Database {name:'pagila'})-[:CONTAINS]->(:Schema {name:'public'})"
            "-[:CONTAINS]->(t:Table {name:$n}) RETURN t.id AS id",
            n=table,
        )
    ids["table_track"] = one(
        "MATCH (:Database {name:'chinook'})-[:CONTAINS]->(:Schema {name:'main'})"
        "-[:CONTAINS]->(t:Table {name:'Track'}) RETURN t.id AS id"
    )
    for table, column in (
        ("film", "film_id"),
        ("film", "title"),
        ("rental", "customer_id"),
        ("customer", "customer_id"),
        ("inventory", "film_id"),
    ):
        ids[f"col_{table}_{column}"] = one(
            "MATCH (:Database {name:'pagila'})-[:CONTAINS]->(:Schema {name:'public'})"
            "-[:CONTAINS]->(:Table {name:$t})-[:CONTAINS]->(c:Column {name:$c}) "
            "RETURN c.id AS id",
            t=table,
            c=column,
        )
    for term in ("Film", "Customer", "Payment", "Track"):
        ids[f"term_{term}"] = one("MATCH (t:Term {name:$n}) RETURN t.id AS id", n=term)

    missing = sorted(k for k, v in ids.items() if v is None)
    if missing:
        raise SystemExit(
            f"fixture incomplete, ids not found: {missing}\n"
            f"Run: uv run --no-sync python -m dev_tools.seed_graph_fixture --reset"
        )
    return ids


def capture_arg_reads(cap: Capture, ids: dict[str, Any]) -> None:
    """Reads that take arguments, driven off the fixture's known entities."""
    from gsf.dal import (
        attributes,
        custom_analyses,
        datasources,
        exploration,
        pql_analyses,
        sql_attributes,
        terms,
        users,
        zones,
    )

    film = ids["table_film"]
    rental = ids["table_rental"]
    term_film = ids["term_Film"]
    term_customer = ids["term_Customer"]
    attr = ids["attr_film_id"]
    sqlattr = ids["sqlattr_revenue"]
    analysis = ids["analysis_top_films"]
    run = cap.run

    # -- datasources -------------------------------------------------------
    run(
        "datasources.fetch_schemas_for_database",
        lambda: datasources.fetch_schemas_for_database(ids["db_pagila"]),
    )
    run(
        "datasources.fetch_tables_for_schema",
        lambda: datasources.fetch_tables_for_schema(ids["schema_public"]),
    )
    run(
        "datasources.fetch_columns_for_table",
        lambda: datasources.fetch_columns_for_table(film),
    )
    run(
        "datasources.count_columns_for_table",
        lambda: datasources.count_columns_for_table(film),
    )
    run("datasources.fetch_table_by_id", lambda: datasources.fetch_table_by_id(film))
    run(
        "datasources.fetch_table_by_name",
        lambda: datasources.fetch_table_by_name("film"),
    )
    run(
        "datasources.fetch_table_context", lambda: datasources.fetch_table_context(film)
    )
    run(
        "datasources.fetch_tables_by_ids",
        lambda: datasources.fetch_tables_by_ids([film, rental]),
    )
    run(
        "datasources.fetch_join_neighbors",
        lambda: datasources.fetch_join_neighbors(film),
    )
    run(
        "datasources.fetch_schema_ids_for_database",
        lambda: datasources.fetch_schema_ids_for_database("pagila"),
    )
    run(
        "datasources.fetch_bridge_table_candidates",
        lambda: datasources.fetch_bridge_table_candidates("pagila"),
    )
    run(
        "datasources.fetch_col_table_contexts",
        lambda: datasources.fetch_col_table_contexts(
            [ids["col_film_title"], ids["col_rental_customer_id"]]
        ),
    )
    run(
        "datasources.fetch_parent_table_id_for_column",
        lambda: datasources.fetch_parent_table_id_for_column(ids["col_film_title"]),
    )
    run(
        "datasources.fetch_item_by_id",
        lambda: datasources.fetch_item_by_id(film, "Table"),
    )
    run(
        "datasources.fetch_node_properties_by_id",
        lambda: datasources.fetch_node_properties_by_id(film, "Table"),
    )
    run(
        "datasources.fetch_tables_and_columns_by_node_ids",
        lambda: datasources.fetch_tables_and_columns_by_node_ids(
            [film, ids["col_film_title"]]
        ),
    )
    # Unknown ids must stay empty rather than raise -- Neo4j returned [] where
    # a careless SQL port might return None or blow up.
    run(
        "datasources.fetch_table_by_id__unknown",
        lambda: datasources.fetch_table_by_id("no-such-id"),
    )
    run(
        "datasources.fetch_columns_for_table__unknown",
        lambda: datasources.fetch_columns_for_table("no-such-id"),
    )

    # -- terms -------------------------------------------------------------
    run("terms.get_full_term_by_id", lambda: terms.get_full_term_by_id(term_film))
    run("terms.get_slim_term_by_id", lambda: terms.get_slim_term_by_id(term_film))
    run("terms.get_term_certification", lambda: terms.get_term_certification(term_film))
    run(
        "terms.get_term_record_for_table", lambda: terms.get_term_record_for_table(film)
    )
    run(
        "terms.fetch_terms_by_ids",
        lambda: terms.fetch_terms_by_ids([term_film, term_customer]),
    )
    run(
        "terms.fetch_column_attributes_by_term_id",
        lambda: terms.fetch_column_attributes_by_term_id(term_film),
    )
    run(
        "terms.count_column_attributes_by_term_id",
        lambda: terms.count_column_attributes_by_term_id(term_film),
    )
    run("terms.fetch_related_terms", lambda: terms.fetch_related_terms(term_film))
    run("terms.fetch_related_term_ids", lambda: terms.fetch_related_term_ids(term_film))
    run(
        "terms.fetch_terms_and_attributes_for_table",
        lambda: terms.fetch_terms_and_attributes_for_table(film),
    )
    run("terms.fetch_table_schema_map", lambda: terms.fetch_table_schema_map("pagila"))
    run(
        "terms.fetch_term_and_column_attributes_for_embedding",
        lambda: terms.fetch_term_and_column_attributes_for_embedding(term_film),
    )
    run("terms.fetch_term_synonyms", lambda: terms.fetch_term_synonyms([attr]))
    run(
        "terms.find_column_attribute_by_column_id",
        lambda: terms.find_column_attribute_by_column_id(ids["col_film_film_id"]),
    )
    run(
        "terms.fetch_column_attribute_embedding_contexts_by_column_id",
        lambda: terms.fetch_column_attribute_embedding_contexts_by_column_id(
            ids["col_film_film_id"]
        ),
    )
    run(
        "terms.get_full_term_by_id__unknown",
        lambda: terms.get_full_term_by_id("no-such-id"),
    )

    # -- attributes --------------------------------------------------------
    run(
        "attributes.fetch_column_attribute_columns_map",
        lambda: attributes.fetch_column_attribute_columns_map([attr]),
    )
    run(
        "attributes.fetch_attr_column_contexts",
        lambda: attributes.fetch_attr_column_contexts([attr], database_name="pagila"),
    )
    run(
        "attributes.find_column_attribute_by_column_id",
        lambda: attributes.find_column_attribute_by_column_id(ids["col_film_film_id"]),
    )
    # find_join_path is the traversal with no mechanical SQL translation; these
    # four cases are the ones the recursive-CTE/BFS port has to reproduce.
    run(
        "attributes.find_join_path__rental_to_customer",
        lambda: attributes.find_join_path(
            ids["col_rental_customer_id"], ids["col_customer_customer_id"]
        ),
    )
    run(
        "attributes.find_join_path__inventory_to_film",
        lambda: attributes.find_join_path(
            ids["col_inventory_film_id"], ids["col_film_film_id"]
        ),
    )
    run(
        "attributes.find_join_path__same_column",
        lambda: attributes.find_join_path(
            ids["col_film_film_id"], ids["col_film_film_id"]
        ),
    )
    run(
        "attributes.find_join_path__unknown",
        lambda: attributes.find_join_path("no-such-id", ids["col_film_film_id"]),
    )

    # -- sql_attributes ----------------------------------------------------
    run(
        "sql_attributes.get_sql_attribute_by_id",
        lambda: sql_attributes.get_sql_attribute_by_id(sqlattr),
    )
    run(
        "sql_attributes.get_full_sql_attribute_by_id",
        lambda: sql_attributes.get_full_sql_attribute_by_id(sqlattr),
    )
    run(
        "sql_attributes.fetch_sql_attributes_by_term_id",
        lambda: sql_attributes.fetch_sql_attributes_by_term_id(term_film),
    )
    run(
        "sql_attributes.count_sql_attributes_by_term_id",
        lambda: sql_attributes.count_sql_attributes_by_term_id(term_film),
    )
    run(
        "sql_attributes.fetch_sql_attribute_docs",
        lambda: sql_attributes.fetch_sql_attribute_docs(sqlattr),
    )
    run(
        "sql_attributes.fetch_sql_attributes_with_sql",
        lambda: sql_attributes.fetch_sql_attributes_with_sql([sqlattr]),
    )
    run(
        "sql_attributes.fetch_tables_from_sql_attributes",
        lambda: sql_attributes.fetch_tables_from_sql_attributes([sqlattr]),
    )
    run(
        "sql_attributes.find_attr_by_name",
        lambda: sql_attributes.find_attr_by_name("total revenue", None),
    )
    run(
        "sql_attributes.find_attr_by_name__absent",
        lambda: sql_attributes.find_attr_by_name("no such attribute", None),
    )

    # -- custom / pql analyses --------------------------------------------
    run(
        "custom_analyses.get_custom_analysis_by_id",
        lambda: custom_analyses.get_custom_analysis_by_id(analysis),
    )
    run(
        "custom_analyses.fetch_custom_analyses_with_sql",
        lambda: custom_analyses.fetch_custom_analyses_with_sql([analysis]),
    )
    run(
        "custom_analyses.fetch_tables_from_custom_analyses",
        lambda: custom_analyses.fetch_tables_from_custom_analyses([analysis]),
    )
    run(
        "custom_analyses.find_analysis_by_name",
        lambda: custom_analyses.find_analysis_by_name("Top rented films", None),
    )
    run(
        "pql_analyses.get_pql_analysis_by_id",
        lambda: pql_analyses.get_pql_analysis_by_id(ids["pql_churn"]),
    )
    run(
        "pql_analyses.fetch_pql_analyses_by_ids",
        lambda: pql_analyses.fetch_pql_analyses_by_ids([ids["pql_churn"]]),
    )
    run(
        "pql_analyses.find_pql_analysis_by_name",
        lambda: pql_analyses.find_pql_analysis_by_name("Customer churn", None),
    )

    # -- exploration -------------------------------------------------------
    run(
        "exploration.fetch_table_exploration_details",
        lambda: exploration.fetch_table_exploration_details(film),
    )
    run(
        "exploration.fetch_exploration_related_nodes__data",
        lambda: exploration.fetch_exploration_related_nodes(film, "data"),
    )
    run(
        "exploration.fetch_exploration_related_nodes__semantic",
        lambda: exploration.fetch_exploration_related_nodes(term_film, "semantic"),
    )

    # -- zones / users -----------------------------------------------------
    run("zones.get_zone_by_id__enabled", lambda: zones.get_zone_by_id(ids["zone_film"]))
    run(
        "zones.get_zone_by_id__disabled",
        lambda: zones.get_zone_by_id(ids["zone_retired"]),
    )
    for label, zone_ids in _zone_modes(ids):
        run(
            f"users.resolve_accessible_catalog_ids[{label}]",
            lambda z=zone_ids: users.resolve_accessible_catalog_ids(z),
        )
        run(
            f"users.get_accessible_catalog_ids_for_zones[{label}]",
            lambda z=zone_ids: users.get_accessible_catalog_ids_for_zones(z or []),
        )


def capture_all() -> Capture:
    """Run every capture against the live graph. Used by the replay test too."""
    import importlib

    id_map = build_id_map()
    logger.info("  %d ids mapped", len(id_map))

    ids = _fixture_ids()
    cap = Capture(id_map)

    for module_name, fn_name in ZERO_ARG_READS:
        module = importlib.import_module(f"gsf.dal.{module_name}")
        cap.run(f"{module_name}.{fn_name}", getattr(module, fn_name))

    for module_name, fn_name in ZONE_SCOPED_READS:
        module = importlib.import_module(f"gsf.dal.{module_name}")
        fn = getattr(module, fn_name)
        for label, zone_ids in _zone_modes(ids):
            cap.run(
                f"{module_name}.{fn_name}[{label}]",
                lambda f=fn, z=zone_ids: f(zone_ids=z),
            )

    capture_arg_reads(cap, ids)
    return cap


def load_golden() -> dict[str, Any]:
    """The recorded golden, or an empty payload if it has not been captured."""
    path = GOLDEN_DIR / "dal_reads.json"
    if not path.is_file():
        return {"results": {}, "errors": {}}
    return json.loads(path.read_text())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    logger.info("capturing...")
    cap = capture_all()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLDEN_DIR / "dal_reads.json"
    out.write_text(
        json.dumps(
            {"results": cap.results, "errors": cap.errors},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    logger.info("wrote %s", out)
    logger.info("  captured: %d", len(cap.results))
    logger.info("  raised:   %d", len(cap.errors))
    for name, err in sorted(cap.errors.items()):
        logger.info("    %s -> %s", name, err)


if __name__ == "__main__":
    main()
