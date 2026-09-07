# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the canonical fixture: catalog + a hand-authored semantic layer.

This is the input to the golden capture in ``gsf/dal/tests/test_golden.py``. It
must be **deterministic** — same fixture databases in, same catalog out —
because the goldens are compared byte for byte after id normalisation.

Two deliberate choices:

* **The semantic layer is hand-authored, not compiled.** ``/semantic/compile``
  drives an LLM, so it needs credentials and returns something slightly
  different every run. Neither is acceptable for a fidelity oracle. Writing the
  same artifacts directly through ``gsf.dal`` is reproducible and exercises the
  same write paths the DAL port has to preserve.
* **Embedding is stubbed out.** The service-layer creates embed into pgvector
  after writing the graph. Goldens capture graph reads, so the embed call is
  replaced with a no-op rather than requiring an embedding endpoint to build a
  fixture.

Usage::

    export POSTGRES_HOST=... POSTGRES_PORT=... POSTGRES_USER=... POSTGRES_PASSWORD=...
    export CONNECTION_STRINGS="postgresql://.../pagila,sqlite:////abs/path/chinook.sqlite"
    uv run --no-sync python -m dev_tools.fixtures.seed_graph_fixture --reset
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

logger = logging.getLogger("dev_tools.fixtures.seed_graph_fixture")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def ingest_catalog() -> None:
    """Write both fixture databases' catalogs into the graph."""
    from gsf.catalog import ingest_catalog as write_catalog
    from gsf.connectors.registry import create_connector

    raw = os.environ.get("CONNECTION_STRINGS", "")
    connection_strings = [cs.strip() for cs in raw.split(",") if cs.strip()]
    if not connection_strings:
        raise SystemExit("CONNECTION_STRINGS is empty; nothing to ingest")

    for connection_string in connection_strings:
        connector = create_connector(connection_string)
        try:
            logger.info("ingesting %s (%s)", connector.database_name, connector.dialect)
            write_catalog(connector)
        finally:
            connector.close()


# ---------------------------------------------------------------------------
# Catalog lookups
# ---------------------------------------------------------------------------
#
# Resolved through the DAL rather than by querying the store directly, so the
# fixture is built the same way the application reads it.


def _catalog_index() -> dict[str, dict[tuple[str, ...], str]]:
    """One pass over the catalog, indexed by name.

    Cached for the life of the process: the seeder resolves a few dozen names
    and the catalog does not change under it.
    """
    global _CATALOG_INDEX
    if _CATALOG_INDEX is not None:
        return _CATALOG_INDEX

    from gsf.dal.datasources import (
        fetch_databases,
        fetch_schemas_by_ids,
        fetch_schemas_for_database,
    )

    schemas: dict[tuple[str, ...], str] = {}
    for database in fetch_databases():
        found = fetch_schemas_for_database(database["id"])
        for schema in (found or {}).get("schemas", []):
            schemas[(database["name"], schema["schema_name"])] = schema["id"]

    tables: dict[tuple[str, ...], str] = {}
    columns: dict[tuple[str, ...], str] = {}
    for row in fetch_schemas_by_ids():
        key = (row["database_name"], row["table_schema"], row["table_name"])
        tables[key] = row["table_id"]
        columns[(*key, row["column_name"])] = row["column_id"]

    _CATALOG_INDEX = {"schemas": schemas, "tables": tables, "columns": columns}
    return _CATALOG_INDEX


_CATALOG_INDEX: dict[str, dict[tuple[str, ...], str]] | None = None


def _lookup(kind: str, key: tuple[str, ...]) -> str:
    found = _catalog_index()[kind].get(key)
    if found is None:
        raise LookupError(f"{kind[:-1]} not found: {'.'.join(key)}")
    return found


def _table_id(database: str, schema: str, table: str) -> str:
    return _lookup("tables", (database, schema, table))


def _column_id(database: str, schema: str, table: str, column: str) -> str:
    return _lookup("columns", (database, schema, table, column))


# ---------------------------------------------------------------------------
# Semantic layer
# ---------------------------------------------------------------------------

# (term name, description, owning table, synonyms)
TERMS: tuple[tuple[str, str, tuple[str, str, str], list[str]], ...] = (
    (
        "Film",
        "A motion picture available for rental.",
        ("pagila", "public", "film"),
        ["movie", "picture", "title"],
    ),
    (
        "Customer",
        "A person who rents films from a store.",
        ("pagila", "public", "customer"),
        ["renter", "patron"],
    ),
    (
        "Rental",
        "A single film rental transaction.",
        ("pagila", "public", "rental"),
        ["hire", "loan"],
    ),
    (
        "Payment",
        "Money received for a rental. Owning table is partitioned.",
        ("pagila", "public", "payment"),
        ["transaction"],
    ),
    (
        "Track",
        "A single piece of recorded music.",
        ("chinook", "main", "Track"),
        ["song"],
    ),
)

# (term name, table, source column, attribute name, datatype, description)
COLUMN_ATTRIBUTES: tuple[tuple[str, tuple[str, str, str], str, str, str, str], ...] = (
    (
        "Film",
        ("pagila", "public", "film"),
        "title",
        "film title",
        "text",
        "The film's title.",
    ),
    (
        "Film",
        ("pagila", "public", "film"),
        "rating",
        "film rating",
        "mpaa_rating",
        "MPAA rating; an enum.",
    ),
    (
        "Film",
        ("pagila", "public", "film"),
        "film_id",
        "film id",
        "integer",
        "Primary key of film.",
    ),
    (
        "Customer",
        ("pagila", "public", "customer"),
        "customer_id",
        "customer id",
        "integer",
        "Primary key of customer.",
    ),
    (
        "Customer",
        ("pagila", "public", "customer"),
        "email",
        "customer email",
        "character varying",
        "Contact address.",
    ),
    (
        "Rental",
        ("pagila", "public", "rental"),
        "rental_date",
        "rental date",
        "timestamp with time zone",
        "When the rental started.",
    ),
    (
        "Payment",
        ("pagila", "public", "payment"),
        "amount",
        "payment amount",
        "numeric",
        "Amount paid.",
    ),
    (
        "Track",
        ("chinook", "main", "Track"),
        "Name",
        "track name",
        "NVARCHAR(200)",
        "The track's title.",
    ),
)

# Column -> ColumnAttribute the column semantically references.
# rental.customer_id points at Customer's "customer id" attribute.
SEMANTIC_FKS: tuple[tuple[tuple[str, str, str, str], tuple[str, str]], ...] = (
    (("pagila", "public", "rental", "customer_id"), ("Customer", "customer id")),
    (("pagila", "public", "payment", "customer_id"), ("Customer", "customer id")),
    (("pagila", "public", "inventory", "film_id"), ("Film", "film id")),
)

# (name, description, expression, term name, source)
#
# Every statement must reference at least one column by name. ``parse_query_single``
# resolves tables *through column references*, so a bare ``SELECT count(*) FROM film``
# is rejected as "doesn't reference any table known to the catalog" even when the
# table is catalogued.
SQL_ATTRIBUTES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "total revenue",
        "Sum of all payment amounts.",
        "SELECT sum(amount) FROM payment",
        "Payment",
        "manual",
    ),
    (
        "film count",
        "Number of films in the catalog.",
        "SELECT count(film_id) FROM film",
        "Film",
        "sql",
    ),
    (
        "films per category",
        "Film count grouped by category, via the bridge table.",
        "SELECT c.name, count(fc.film_id) FROM film_category fc "
        "JOIN category c ON c.category_id = fc.category_id GROUP BY c.name",
        "Film",
        "bridgeTable",
    ),
    (
        "customer lifetime value",
        "Total paid by each customer.",
        "SELECT customer_id, sum(amount) FROM payment GROUP BY customer_id",
        "Customer",
        "table",
    ),
)

CUSTOM_ANALYSES: tuple[tuple[str, str, str], ...] = (
    (
        "Top rented films",
        "Films ordered by rental count.",
        "SELECT f.title, count(r.rental_id) AS rentals FROM rental r "
        "JOIN inventory i ON i.inventory_id = r.inventory_id "
        "JOIN film f ON f.film_id = i.film_id "
        "GROUP BY f.title ORDER BY rentals DESC",
    ),
    (
        "Revenue by staff",
        "Payment totals per staff member.",
        "SELECT staff_id, sum(amount) AS revenue FROM payment GROUP BY staff_id",
    ),
)

PQL_ANALYSES: tuple[tuple[str, str, str, str], ...] = (
    (
        "pql-churn",
        "Customer churn",
        "Predict which customers stop renting.",
        "PREDICT COUNT(rental.*, 30, days) = 0 FOR EACH customer.customer_id",
    ),
)


def seed_terms() -> dict[str, str]:
    from gsf.dal.terms import merge_term, update_term

    term_ids: dict[str, str] = {}
    for name, description, (db, schema, table), synonyms in TERMS:
        term_id = merge_term(
            name=name,
            description=description,
            table_id=_table_id(db, schema, table),
            synonyms=synonyms,
        )
        if term_id is None:
            raise RuntimeError(f"merge_term returned None for {name!r}")
        term_ids[name] = term_id

    # Certification flags must not be uniform, or a golden cannot tell a
    # dropped flag from a defaulted one.
    update_term(term_ids["Film"], name_certified=True, description_certified=True)
    update_term(term_ids["Customer"], name_certified=True)
    update_term(term_ids["Rental"], description_certified=True)
    return term_ids


def seed_column_attributes() -> dict[tuple[str, str], str]:
    from gsf.dal.attributes import merge_column_attribute

    attr_ids: dict[tuple[str, str], str] = {}
    for term_name, (
        db,
        schema,
        table,
    ), column, attr_name, datatype, desc in COLUMN_ATTRIBUTES:
        attr_id = merge_column_attribute(
            term_name=term_name,
            table_id=_table_id(db, schema, table),
            source_column=column,
            attr_name=attr_name,
            datatype=datatype,
            description=desc,
        )
        if attr_id is None:
            raise RuntimeError(
                f"merge_column_attribute returned None for {attr_name!r}"
            )
        attr_ids[(term_name, attr_name)] = attr_id
    return attr_ids


def seed_semantic_fks(attr_ids: dict[tuple[str, str], str]) -> None:
    from gsf.dal.attributes import merge_semantic_fk

    for (db, schema, table, column), key in SEMANTIC_FKS:
        merge_semantic_fk(_column_id(db, schema, table, column), attr_ids[key])


def seed_sql_attributes(term_ids: dict[str, str]) -> list[str]:
    from gsf.server.sql_attributes import service

    created: list[str] = []
    for name, description, expression, term_name, source in SQL_ATTRIBUTES:
        row = service.create_sql_attribute(
            name=name,
            description=description,
            expression=expression,
            term_id=term_ids[term_name],
            source=source,
        )
        created.append(row["id"])
    return created


def seed_custom_analyses() -> list[str]:
    from gsf.server.custom_analyses import service

    created: list[str] = []
    for name, description, sql in CUSTOM_ANALYSES:
        row = service.create_custom_analysis(
            name=name, description=description, sql=sql
        )
        created.append(row["id"])
    return created


def seed_pql_analyses() -> None:
    from gsf.dal.pql_analyses import upsert_pql_analysis_node

    for analysis_id, name, description, pql in PQL_ANALYSES:
        upsert_pql_analysis_node(analysis_id, name, description, pql)


def seed_zones() -> list[dict[str, Any]]:
    """Three zones: one scoped to a schema, one spanning databases, one disabled.

    Zone scoping is where a silent access-control regression would hide, so the
    fixture has to make the three interesting cases distinguishable.
    """
    from gsf.dal.zones import create_zone, set_zone_enabled

    film_domain = create_zone(
        name="Film domain",
        description="Tables describing films and their inventory.",
        color="#4f46e5",
        item_ids=[
            _table_id("pagila", "public", "film"),
            _table_id("pagila", "public", "inventory"),
            _table_id("pagila", "public", "category"),
        ],
    )
    # Spans both databases *and* targets a Schema rather than a Table, which is
    # the polymorphic ZONE_OF case.
    cross_database = create_zone(
        name="Cross database",
        description="A schema in one database plus a table in another.",
        color="#059669",
        item_ids=[
            _schema_id("pagila", "analytics"),
            _table_id("chinook", "main", "Track"),
        ],
    )
    retired = create_zone(
        name="Retired zone",
        description="Disabled; grants no access but stays visible to admins.",
        color="#dc2626",
        item_ids=[_table_id("pagila", "public", "staff")],
    )
    set_zone_enabled(retired["id"], False)
    return [film_domain, cross_database, retired]


def _schema_id(database: str, schema: str) -> str:
    return _lookup("schemas", (database, schema))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _stub_embedding() -> None:
    """Replace embed calls with no-ops.

    Building the fixture must not require an embedding endpoint: the goldens
    capture graph reads, and pgvector contents are not part of any DAL read
    result.
    """
    from gsf.server.custom_analyses import service as ca_service
    from gsf.server.sql_attributes import service as sa_service

    sa_service._embed_sql_attribute = lambda *a, **k: None  # type: ignore[assignment]
    sa_service._reembed_sql_attribute = lambda *a, **k: None  # type: ignore[assignment]
    ca_service.embed_custom_analyses = lambda *a, **k: None  # type: ignore[assignment]


def reset_graph() -> None:
    """Empty the store, whichever one is selected.

    ``delete_all_data(None)`` is the DAL's own definition of a full wipe, so the
    fixture is built from the same empty state a reset produces — **plus the
    zones, which that reset does not touch.** Zones are not part of either
    layer: the SQL's semantic labels never included them, so
    ``delete_all_data`` left them standing on both backends. The old
    ``MATCH (n) DETACH DELETE n`` swept them up incidentally, and a fixture
    built on top of leftover zones is not reproducible.
    """
    from gsf.dal.reset import delete_all_data
    from gsf.dal.zones import delete_zone, list_zones

    global _CATALOG_INDEX
    _CATALOG_INDEX = None
    delete_all_data(None)
    for zone in list_zones():
        delete_zone(zone["id"])
    logger.info("store cleared")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete every node first (the fixture must be built from empty)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _stub_embedding()

    if args.reset:
        reset_graph()

    logger.info("ingesting catalogs...")
    ingest_catalog()

    logger.info("seeding terms...")
    term_ids = seed_terms()

    logger.info("seeding column attributes...")
    attr_ids = seed_column_attributes()

    logger.info("seeding semantic FKs...")
    seed_semantic_fks(attr_ids)

    logger.info("seeding SQL attributes...")
    seed_sql_attributes(term_ids)

    logger.info("seeding custom analyses...")
    seed_custom_analyses()

    logger.info("seeding PQL analyses...")
    seed_pql_analyses()

    logger.info("seeding zones...")
    zones = seed_zones()

    _log_summary(zones)


def _log_summary(zones: list[dict[str, Any]]) -> None:
    """Print what was built, counted through the DAL rather than the store.

    A summary is worth having — it is how you notice a fixture that seeded
    without error and produced nothing — but it must not reintroduce a
    store-specific query, so it counts what the DAL can see.
    """
    from gsf.dal.datasources import fetch_databases, fetch_schemas_by_ids
    from gsf.dal.custom_analyses import list_custom_analyses
    from gsf.dal.pql_analyses import list_pql_analyses
    from gsf.dal.sql_attributes import list_sql_attributes
    from gsf.dal.terms import count_terms

    columns = fetch_schemas_by_ids()
    counts = {
        "databases": len(fetch_databases()),
        "tables": len({row["table_id"] for row in columns}),
        "columns": len(columns),
        "terms": count_terms(),
        "sql_attributes": len(list_sql_attributes()),
        "custom_analyses": len(list_custom_analyses()),
        "pql_analyses": len(list_pql_analyses()),
    }
    logger.info("--- fixture ---")
    for name, total in counts.items():
        logger.info("  %-16s %s", name, total)
    logger.info("  %-16s %s", "zones", [z["name"] for z in zones])


if __name__ == "__main__":
    main()
