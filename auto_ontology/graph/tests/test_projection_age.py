# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync and read the graph on a real Postgres with Apache AGE.

Seeds one small catalog with every vertex label and every edge type, syncs it
into a graph of its own, and reads it back through the same functions the CLI
and the REST route use. The graph is dropped afterwards.

Needs a migrated database on a server with the AGE extension available (for
example the ``apache/age`` image) and skips without one.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

pytest.importorskip("sqlalchemy")

from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import get_engine, store  # noqa: E402
from auto_ontology.graph import age, projection, queries  # noqa: E402
from auto_ontology.graph.mapping import edge_labels  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_age() -> None:
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM tag_target LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")
    with get_engine().connect() as conn:
        available = conn.exec_driver_sql(
            "SELECT 1 FROM pg_available_extensions WHERE name = 'age'"
        ).first()
    if available is None:
        pytest.skip("Apache AGE extension not available on this server")


@pytest.fixture(scope="module")
def seeded() -> Iterator[dict[str, str]]:
    """One catalog that exercises every label and edge type."""
    p = f"g{uuid.uuid4().hex[:8]}"
    ids = {
        k: f"{p}-{k}"
        for k in (
            "db",
            "schema",
            "orders",
            "customers",
            "o_cust",
            "c_id",
            "o_total",
            "term",
            "cust_term",
            "attr",
            "sql_attr",
            "tag",
            "zone",
        )
    }
    rows = [
        (
            s.catalog_database,
            {
                "id": ids["db"],
                "name": f"{p}-db",
                "dialect": "postgres",
                "connection": {"password": "never-projected"},
            },
        ),
        (
            s.catalog_schema,
            {"id": ids["schema"], "database_id": ids["db"], "name": "public"},
        ),
        (
            s.catalog_table,
            {"id": ids["orders"], "schema_id": ids["schema"], "name": "orders"},
        ),
        (
            s.catalog_table,
            {"id": ids["customers"], "schema_id": ids["schema"], "name": "customers"},
        ),
        (
            s.catalog_column,
            {
                "id": ids["o_cust"],
                "table_id": ids["orders"],
                "name": "customer_id",
                "sample_values": "never-projected",
            },
        ),
        (
            s.catalog_column,
            {"id": ids["o_total"], "table_id": ids["orders"], "name": "total"},
        ),
        (
            s.catalog_column,
            {"id": ids["c_id"], "table_id": ids["customers"], "name": "id"},
        ),
        (
            s.term,
            {
                "id": ids["term"],
                "name": f"{p} Order",
                "synonyms": ["purchase"],
                "description": "quote ' and $$ survive",
            },
        ),
        (s.term, {"id": ids["cust_term"], "name": f"{p} Customer"}),
        (
            s.column_attribute,
            {
                "id": ids["attr"],
                "name": "Order Total",
                "source_column": "total",
                "term_name": f"{p} Order",
                "table_id": ids["orders"],
            },
        ),
        (
            s.sql_attribute,
            {"id": ids["sql_attr"], "name": f"{p} Revenue", "expression": "SUM(total)"},
        ),
        (s.tag, {"id": ids["tag"], "name": f"{p}-finance"}),
        (s.zone, {"id": ids["zone"], "name": f"{p}-zone"}),
        (
            s.column__foreign_key,
            {"source_column_id": ids["o_cust"], "target_column_id": ids["c_id"]},
        ),
        (
            s.table__join,
            {
                "source_table_id": ids["orders"],
                "target_table_id": ids["customers"],
                "join_columns": [["customer_id", "id"]],
            },
        ),
        (
            s.column__join,
            {
                "source_column_id": ids["o_cust"],
                "target_column_id": ids["c_id"],
                "refs": ["q1", "q2"],
            },
        ),
        (
            s.column__union,
            {
                "source_column_id": ids["o_total"],
                "target_column_id": ids["o_total"],
                "refs": ["q3"],
            },
        ),
        (s.table__term, {"table_id": ids["orders"], "term_id": ids["term"]}),
        (s.table__term, {"table_id": ids["customers"], "term_id": ids["cust_term"]}),
        (
            s.column__has_attribute,
            {"column_id": ids["o_total"], "attribute_id": ids["attr"]},
        ),
        (
            s.column__semantic_fk,
            {"column_id": ids["o_cust"], "attribute_id": ids["attr"]},
        ),
        (
            s.column_attribute__term,
            {"attribute_id": ids["attr"], "term_id": ids["term"]},
        ),
        (
            s.sql_attribute__term,
            {"attribute_id": ids["sql_attr"], "term_id": ids["term"]},
        ),
        (s.tag_target, {"tag_id": ids["tag"], "term_id": ids["term"]}),
        (s.zone_target, {"zone_id": ids["zone"], "table_id": ids["orders"]}),
    ]
    with get_engine().begin() as conn:
        for table, values in rows:
            conn.execute(table.insert().values(**values))
    yield ids
    with get_engine().begin() as conn:
        for table in (s.zone, s.tag, s.sql_attribute, s.term, s.catalog_database):
            conn.execute(table.delete().where(table.c.id.like(f"{p}-%")))


@pytest.fixture(scope="module")
def graph(seeded: dict[str, str]) -> Iterator[str]:
    name = f"ao_test_{uuid.uuid4().hex[:8]}"
    projection.sync_graph(graph=name, batch_size=2)  # tiny batches exercise batching
    yield name
    with get_engine().begin() as conn:
        age.prepare(conn)
        conn.exec_driver_sql("SELECT drop_graph(%s, true)", (name,))


def _edges(conn_graph: str, label: str) -> set[tuple[str, str]]:
    rows = queries.run_cypher(
        f"MATCH (a)-[:{label}]->(b) RETURN a.id, b.id",
        columns=("a", "b"),
        graph=conn_graph,
    )
    return {(a, b) for a, b in rows}


def test_every_edge_type_is_projected(graph: str, seeded: dict[str, str]) -> None:
    i = seeded
    expected = {
        "CONTAINS": {
            (i["db"], i["schema"]),
            (i["schema"], i["orders"]),
            (i["orders"], i["o_cust"]),
        },
        "FOREIGN_KEY": {(i["o_cust"], i["c_id"])},
        "JOIN": {(i["orders"], i["customers"]), (i["o_cust"], i["c_id"])},
        "UNION": {(i["o_total"], i["o_total"])},
        "REPRESENTS": {(i["orders"], i["term"])},
        "HAS_ATTRIBUTE": {(i["o_total"], i["attr"])},
        "SEMANTIC_FK": {(i["o_cust"], i["attr"])},
        "PROPERTY_OF": {(i["attr"], i["term"]), (i["sql_attr"], i["term"])},
        "TAGGED": {(i["tag"], i["term"])},
        "ZONE_OF": {(i["zone"], i["orders"])},
    }
    assert set(expected) == set(edge_labels())
    for label, pairs in expected.items():
        assert pairs <= _edges(graph, label), label


def test_properties_survive_and_secrets_do_not(
    graph: str, seeded: dict[str, str]
) -> None:
    ((term,),) = queries.run_cypher(
        "MATCH (t:Term {id: $id}) RETURN properties(t)",
        {"id": seeded["term"]},
        graph=graph,
    )
    assert term["description"] == "quote ' and $$ survive"
    assert term["synonyms"] == ["purchase"]

    ((db,),) = queries.run_cypher(
        "MATCH (d:Database {id: $id}) RETURN properties(d)",
        {"id": seeded["db"]},
        graph=graph,
    )
    ((col,),) = queries.run_cypher(
        "MATCH (c:Column {id: $id}) RETURN properties(c)",
        {"id": seeded["o_cust"]},
        graph=graph,
    )
    assert "connection" not in db
    assert "sample_values" not in col

    ((join,),) = queries.run_cypher(
        "MATCH (:Table {id: $a})-[j:JOIN]->(:Table) RETURN properties(j)",
        {"a": seeded["orders"]},
        graph=graph,
    )
    assert join["join_columns"] == [["customer_id", "id"]]


def test_neighborhood_walks_from_a_term(graph: str, seeded: dict[str, str]) -> None:
    one = queries.term_neighborhood(seeded["term"], depth=1, graph=graph)
    assert one is not None
    ids = {n["id"] for n in one["nodes"]}
    assert ids == {seeded[k] for k in ("term", "orders", "attr", "sql_attr", "tag")}
    assert {
        "type": "REPRESENTS",
        "source": seeded["orders"],
        "target": seeded["term"],
    } in [{k: e[k] for k in ("type", "source", "target")} for e in one["edges"]]
    assert one["truncated"] is False

    # Two terms linked through a table join, the hop chain the exploration API
    # describes: Term <-REPRESENTS- Table -JOIN-> Table -REPRESENTS-> Term.
    three = queries.term_neighborhood(seeded["term"], depth=3, graph=graph)
    assert seeded["cust_term"] in {n["id"] for n in three["nodes"]}

    capped = queries.term_neighborhood(
        seeded["term"], depth=3, max_nodes=3, graph=graph
    )
    assert len(capped["nodes"]) == 3
    assert capped["truncated"] is True


def test_unknown_term_is_none(graph: str) -> None:
    assert queries.term_neighborhood("no-such-term", graph=graph) is None


def test_reads_cannot_write(graph: str) -> None:
    with pytest.raises(Exception, match="read-only"):
        queries.run_cypher("CREATE (n:Term {id: 'x'}) RETURN n", graph=graph)
    assert not queries.run_cypher("MATCH (n:Term {id: 'x'}) RETURN n", graph=graph)


def test_resync_replaces_rather_than_duplicates(
    graph: str, seeded: dict[str, str]
) -> None:
    before = queries.graph_status(graph=graph)
    projection.sync_graph(graph=graph)
    after = queries.graph_status(graph=graph)
    assert after["synced_at"] and after["synced_at"] >= before["synced_at"]
    assert {**after, "synced_at": None} == {**before, "synced_at": None}
    rows = queries.run_cypher(
        "MATCH (t:Term {id: $id}) RETURN count(t)", {"id": seeded["term"]}, graph=graph
    )
    assert rows == [(1,)]


def test_missing_graph_is_reported(seeded: dict[str, str]) -> None:
    status = queries.graph_status(graph="ao_never_built")
    assert status["built"] is False
    with pytest.raises(age.GraphNotBuilt):
        queries.term_neighborhood(seeded["term"], graph="ao_never_built")
