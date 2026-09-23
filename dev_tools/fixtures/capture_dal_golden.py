# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot every DAL read against the graph fixture, for the Postgres port.

The refactor replaces ~5,000 lines of SQL with SQL while keeping every
``auto_ontology.dal`` signature identical. Greenfield cutover means there is no production
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

    uv run --no-sync python -m dev_tools.fixtures.seed_graph_fixture --reset
    uv run --no-sync python -m dev_tools.fixtures.capture_dal_golden
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("dev_tools.fixtures.capture_dal_golden")

GOLDEN_DIR = (
    Path(__file__).resolve().parents[2] / "auto_ontology" / "dal" / "tests" / "golden"
)

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


# ---------------------------------------------------------------------------
# Id normalisation
# ---------------------------------------------------------------------------


def build_id_map() -> dict[str, str]:
    """Map every id to a stable token describing what it names.

    Read through the DAL, not the store — like ``_fixture_ids``, this was
    hardcoded SQL and is why the replay could not grade Postgres. A token has
    to describe the *same* entity on both backends or the normalised goldens
    compare nothing.
    """
    from auto_ontology.dal.custom_analyses import list_custom_analyses
    from auto_ontology.dal.datasources import (
        fetch_databases,
        fetch_schemas_by_ids,
        fetch_schemas_for_database,
    )
    from auto_ontology.dal.pql_analyses import list_pql_analyses
    from auto_ontology.dal.sql_attributes import list_sql_attributes
    from auto_ontology.dal.terms import (
        fetch_all_terms,
        fetch_column_attributes_by_term_id,
    )
    from auto_ontology.dal.zones import list_zones

    id_map: dict[str, str] = {}

    for database in fetch_databases():
        id_map[database["id"]] = f"<db:{database['name']}>"
        found = fetch_schemas_for_database(database["id"]) or {}
        for schema in found.get("schemas", []):
            id_map[schema["id"]] = (
                f"<schema:{database['name']}.{schema['schema_name']}>"
            )

    for row in fetch_schemas_by_ids():
        path = f"{row['database_name']}.{row['table_schema']}.{row['table_name']}"
        id_map[row["table_id"]] = f"<table:{path}>"
        id_map[row["column_id"]] = f"<col:{path}.{row['column_name']}>"

    for term in fetch_all_terms():
        id_map[term["id"]] = f"<term:{term['name']}>"
        for attribute in fetch_column_attributes_by_term_id(term["id"]):
            id_map[attribute["id"]] = (
                f"<attr:{attribute.get('term_name')}/{attribute['name']}"
                f"/{attribute.get('source_column')}>"
            )

    for attribute in list_sql_attributes():
        id_map[attribute["id"]] = f"<sqlattr:{attribute['name']}>"
    for analysis in list_custom_analyses():
        id_map[analysis["id"]] = f"<analysis:{analysis['name']}>"
    for analysis in list_pql_analyses():
        id_map[analysis["id"]] = f"<pql:{analysis['name']}>"
    for zone in list_zones():
        id_map[zone["id"]] = f"<zone:{zone['name']}>"

    # Statements carry no name, so they are keyed by their text -- stable across
    # runs regardless of insertion order, which an id is not.
    for statement in _all_statements():
        digest = re.sub(r"\s+", " ", statement.get("sql") or "").strip()[:60]
        if statement.get("id"):
            id_map[statement["id"]] = f"<sql:{digest}>"

    return id_map


def _all_statements() -> list[dict[str, Any]]:
    """Every stored statement, reached through the reads that expose them.

    No DAL function lists statements outright — they are always reached from a
    table, an attribute or an analysis — so this unions those routes. A
    statement no read can reach is one no golden can contain either, so nothing
    is lost by not finding it.
    """
    from auto_ontology.dal.datasources import fetch_schemas_by_ids
    from auto_ontology.dal.exploration import fetch_table_exploration_details

    seen: dict[str, dict[str, Any]] = {}
    for table_id in {row["table_id"] for row in fetch_schemas_by_ids()}:
        for query in fetch_table_exploration_details(table_id).get("queries", []):
            if query.get("id"):
                seen[query["id"]] = query
    return sorted(seen.values(), key=lambda row: row.get("sql") or "")


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
       would fail the Postgres port for behaviour the store never guaranteed.
       Ordering that *is* contractual — paging stability across ``skip``/
       ``limit`` — needs its own tests.
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

# Zero-argument reads. ``reset.delete_*`` is deliberately
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
    """Resolve the fixture's entities by name, through the DAL.

    **Backend-agnostic on purpose.** This used to be hardcoded SQL, which is
    why the replay could not grade Postgres at all: under ``AUTO_ONTOLOGY_STORE=postgres``
    it handed the store ids to the Postgres DAL and almost every read came back
    empty, so 49 of 128 comparisons failed for a reason that had nothing to do
    with the DAL. Resolving by name through the same facade the captures use
    means the two backends are asked the same questions about the same fixture.
    """
    from auto_ontology.dal.custom_analyses import list_custom_analyses
    from auto_ontology.dal.datasources import (
        fetch_databases,
        fetch_schemas_by_ids,
        fetch_schemas_for_database,
    )
    from auto_ontology.dal.pql_analyses import list_pql_analyses
    from auto_ontology.dal.sql_attributes import list_sql_attributes
    from auto_ontology.dal.terms import fetch_all_terms
    from auto_ontology.dal.zones import list_zones

    databases = {row["name"]: row["id"] for row in fetch_databases()}
    columns = fetch_schemas_by_ids()

    def table(database: str, schema: str, name: str) -> str | None:
        for row in columns:
            if (
                row["database_name"] == database
                and row["table_schema"] == schema
                and row["table_name"] == name
            ):
                return row["table_id"]
        return None

    def column(database: str, schema: str, table_name: str, name: str) -> str | None:
        for row in columns:
            if (
                row["database_name"] == database
                and row["table_schema"] == schema
                and row["table_name"] == table_name
                and row["column_name"] == name
            ):
                return row["column_id"]
        return None

    def first(rows: list[dict[str, Any]], **match: str) -> str | None:
        for row in rows:
            if all(row.get(key) == value for key, value in match.items()):
                return row["id"]
        return None

    schema_public = None
    if "pagila" in databases:
        found = fetch_schemas_for_database(databases["pagila"]) or {}
        for schema in found.get("schemas", []):
            if schema["schema_name"] == "public":
                schema_public = schema["id"]

    terms = fetch_all_terms()
    attributes = list_sql_attributes()
    analyses = list_custom_analyses()
    pql = list_pql_analyses()
    zones = list_zones()

    ids: dict[str, Any] = {
        "db_pagila": databases.get("pagila"),
        "schema_public": schema_public,
        "attr_film_id": _column_attribute_id("film id"),
        "sqlattr_revenue": first(attributes, name="total revenue"),
        "analysis_top_films": first(analyses, name="Top rented films"),
        "pql_churn": pql[0]["id"] if pql else None,
        "zone_film": first(zones, name="Film domain"),
        "zone_cross": first(zones, name="Cross database"),
        "zone_retired": first(zones, name="Retired zone"),
    }

    for name in ("film", "rental", "customer", "payment", "inventory", "category"):
        ids[f"table_{name}"] = table("pagila", "public", name)
    ids["table_track"] = table("chinook", "main", "Track")

    for table_name, column_name in (
        ("film", "film_id"),
        ("film", "title"),
        ("rental", "customer_id"),
        ("customer", "customer_id"),
        ("inventory", "film_id"),
    ):
        ids[f"col_{table_name}_{column_name}"] = column(
            "pagila", "public", table_name, column_name
        )

    for name in ("Film", "Customer", "Payment", "Track"):
        ids[f"term_{name}"] = first(terms, name=name)

    missing = sorted(key for key, value in ids.items() if value is None)
    if missing:
        raise SystemExit(
            f"fixture incomplete, ids not found: {missing}\n"
            "Run: uv run --no-sync python -m "
            "dev_tools.fixtures.seed_graph_fixture --reset"
        )
    return ids


def _column_attribute_id(name: str) -> str | None:
    """A ColumnAttribute id by name.

    No DAL read lists attributes by name alone — they are always reached through
    a Term — so this walks the terms the fixture seeded. Slower than a lookup
    and irrelevant at fixture scale.
    """
    from auto_ontology.dal.terms import (
        fetch_all_terms,
        fetch_column_attributes_by_term_id,
    )

    for term in fetch_all_terms():
        for attribute in fetch_column_attributes_by_term_id(term["id"]):
            if attribute.get("name") == name:
                return attribute["id"]
    return None


def capture_arg_reads(cap: Capture, ids: dict[str, Any]) -> None:
    """Reads that take arguments, driven off the fixture's known entities."""
    from auto_ontology.dal import (
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
    # Unknown ids must stay empty rather than raise -- the store returned [] where
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
        module = importlib.import_module(f"auto_ontology.dal.{module_name}")
        cap.run(f"{module_name}.{fn_name}", getattr(module, fn_name))

    for module_name, fn_name in ZONE_SCOPED_READS:
        module = importlib.import_module(f"auto_ontology.dal.{module_name}")
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
