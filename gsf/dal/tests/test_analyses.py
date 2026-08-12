# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CustomAnalysis, PqlAnalysis, and UI-managed connections.

The zone rule on ``list_custom_analyses`` is the sharpest thing here. It is
all-or-nothing for a more concrete reason than the Term or SqlAttribute rules:
the analysis's **SQL text is returned to the caller**, so showing one that
touches an out-of-zone table discloses that table's name and columns even though
no rows are ever read.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from gsf.dal import connections as conn  # noqa: E402
from gsf.dal import custom_analyses as ca  # noqa: E402
from gsf.dal import pql_analyses as pa  # noqa: E402
from gsf.dal import schema as s  # noqa: E402
from gsf.dal.session import store  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read(f"SELECT 1 FROM {s.SCHEMA}.custom_analysis LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


class World:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.database = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.database, name="public")
        self.tables: dict[str, str] = {}
        self.analyses: dict[str, str] = {}
        self.zones: dict[str, str] = {}
        self.queries: list[str] = []

    def table(self, name: str, columns: tuple[str, ...] = ("id",)) -> str:
        tid = _add(s.catalog_table, schema_id=self.schema, name=name)
        self.tables[name] = tid
        for position, column in enumerate(columns, start=1):
            _add(
                s.catalog_column,
                table_id=tid,
                name=column,
                data_type="integer",
                ordinal_position=position,
            )
        return tid

    def analysis(
        self,
        name: str,
        *,
        sql: str | None = None,
        tables: tuple[str, ...] = (),
        description: str | None = None,
    ) -> str:
        aid = _add(
            s.custom_analysis, name=f"{self.prefix}-{name}", description=description
        )
        self.analyses[name] = aid
        if sql is not None:
            qid = _add(s.sql_query, sql_full_query=sql)
            self.queries.append(qid)
            _link(s.custom_analysis_sql, analysis_id=aid, sql_query_id=qid)
            for table in tables:
                _link(s.sql_query_table, sql_query_id=qid, table_id=self.tables[table])
        return aid

    def zone(self, name: str, *, table: str) -> str:
        zid = _add(s.zone, name=f"{self.prefix}-{name}")
        self.zones[name] = zid
        _link(s.zone_target, zone_id=zid, table_id=self.tables[table])
        return zid


@pytest.fixture
def world():
    w = World(f"c-{uuid.uuid4().hex[:8]}")
    yield w
    for zone_id in w.zones.values():
        store().query_write(s.zone.delete().where(s.zone.c.id == zone_id))
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == w.database)
    )
    for table in (s.custom_analysis, s.pql_analysis):
        store().query_write(table.delete().where(table.c.name.like(f"{w.prefix}%")))
    for query_id in w.queries:
        store().query_write(s.sql_query.delete().where(s.sql_query.c.id == query_id))


def _mine(rows, world):
    return [r for r in rows if str(r.get("name", "")).startswith(world.prefix)]


# --------------------------------------------------------------------------
# CustomAnalysis — zone scoping
# --------------------------------------------------------------------------


@pytest.fixture
def scoped(world):
    world.table("orders")
    world.table("secrets")
    world.analysis("inside", sql="SELECT 1", tables=("orders",))
    world.analysis("outside", sql="SELECT 2", tables=("secrets",))
    world.analysis("both", sql="SELECT 3", tables=("orders", "secrets"))
    world.zone_id = world.zone("Sales", table="orders")
    return world


def test_an_analysis_touching_one_out_of_zone_table_is_hidden(scoped) -> None:
    """The SQL text is the payload, so partial visibility leaks a schema."""
    visible = {r["name"] for r in ca.list_custom_analyses(zone_ids=[scoped.zone_id])}
    assert visible == {f"{scoped.prefix}-inside"}


def test_no_zone_ids_means_unscoped_not_empty(scoped) -> None:
    assert len(_mine(ca.list_custom_analyses(None), scoped)) == 3
    assert ca.list_custom_analyses(zone_ids=[]) == []


# --------------------------------------------------------------------------
# CustomAnalysis — reads
# --------------------------------------------------------------------------


def test_an_analysis_with_no_sql_is_invisible_to_the_list(world) -> None:
    world.analysis("orphan")
    world.analysis("real", sql="SELECT 1")
    assert {r["name"] for r in _mine(ca.list_custom_analyses(), world)} == {
        f"{world.prefix}-real"
    }


def test_but_fetch_with_sql_still_returns_it(world) -> None:
    """The one place the statement join is optional.

    The caller asked for specific ids, and a silently absent entry is harder to
    notice than one with an empty ``sql``.
    """
    orphan = world.analysis("orphan")
    rows = ca.fetch_custom_analyses_with_sql([orphan])
    assert rows == [
        {
            "id": orphan,
            "name": f"{world.prefix}-orphan",
            "description": "",
            "sql": "",
        }
    ]


def test_domain_rules_fold_the_sql_into_the_description(world) -> None:
    """The `description` here is a rendered blob, not the column."""
    world.analysis("revenue", sql="SELECT 1", description="how revenue is counted")

    rule = next(
        r for r in ca.fetch_custom_analyses() if r["name"].startswith(world.prefix)
    )
    assert rule == {
        "name": f"{world.prefix}-revenue",
        "description": "how revenue is counted\nSQL: SELECT 1",
    }


def test_a_rule_with_no_description_is_still_a_rule(world) -> None:
    world.analysis("bare", sql="SELECT 1")
    rule = next(
        r for r in ca.fetch_custom_analyses() if r["name"].startswith(world.prefix)
    )
    assert rule["description"] == "SQL: SELECT 1"


def test_find_by_name_excludes_the_analysis_being_edited(world) -> None:
    analysis = world.analysis("revenue", sql="SELECT 1")
    assert ca.find_analysis_by_name(f"{world.prefix}-revenue", analysis) is None
    assert ca.find_analysis_by_name(f"{world.prefix}-revenue", None) == {
        "id": analysis,
        "name": f"{world.prefix}-revenue",
    }


def test_find_by_sql_matches_exact_text_only(world) -> None:
    """Unlike find_attr_by_expression, which normalises. Inherited, not chosen.

    Loosening it here would start rejecting saves that succeed today.
    """
    analysis = world.analysis("revenue", sql="SELECT  1")
    assert ca.find_analysis_by_sql("SELECT  1", None) == {
        "id": analysis,
        "name": f"{world.prefix}-revenue",
    }
    assert ca.find_analysis_by_sql("select 1", None) is None


def test_get_by_id_is_an_existence_check(world) -> None:
    analysis = world.analysis("revenue", sql="SELECT 1")
    assert ca.get_custom_analysis_by_id(analysis) == analysis
    assert ca.get_custom_analysis_by_id("no-such-analysis") is None


def test_tables_from_analyses_nests_columns(world) -> None:
    world.table("orders", columns=("id", "amount"))
    first = world.analysis("one", sql="SELECT 1", tables=("orders",))
    second = world.analysis("two", sql="SELECT 2", tables=("orders",))

    tables = ca.fetch_tables_from_custom_analyses([first, second])
    assert len(tables) == 1
    assert [c["name"] for c in tables[0]["columns"]] == ["id", "amount"]


def test_empty_id_lists_short_circuit() -> None:
    assert ca.fetch_custom_analyses_with_sql([]) == []
    assert ca.fetch_tables_from_custom_analyses([]) == []
    assert pa.fetch_pql_analyses_by_ids([]) == {}


# --------------------------------------------------------------------------
# CustomAnalysis — writes
# --------------------------------------------------------------------------


def test_detaching_sql_leaves_the_statement_alone(world) -> None:
    analysis = world.analysis("revenue", sql="SELECT 1")
    query_id = world.queries[-1]

    ca.detach_existing_sql_edges(analysis)

    assert ca.fetch_custom_analyses_with_sql([analysis])[0]["sql"] == ""
    assert store().query_read(
        select(s.sql_query.c.id).where(s.sql_query.c.id == query_id)
    )


def test_delete_removes_the_analysis_and_its_statement(world) -> None:
    """`DETACH DELETE ca, sql` deleted both, and that is preserved.

    An analysis's statement is not shared — it is parsed from text typed into
    this analysis — so leaving it behind accumulates unreachable rows that still
    surface in query-history reads.
    """
    analysis = world.analysis("revenue", sql="SELECT 1")
    query_id = world.queries[-1]

    ca.delete_custom_analysis_node(analysis)

    assert ca.get_custom_analysis_by_id(analysis) is None
    assert (
        store().query_read(select(s.sql_query.c.id).where(s.sql_query.c.id == query_id))
        == []
    )


def test_delete_cascades_the_link_row(world) -> None:
    analysis = world.analysis("revenue", sql="SELECT 1")
    ca.delete_custom_analysis_node(analysis)
    assert (
        store().query_read(
            select(s.custom_analysis_sql.c.sql_query_id).where(
                s.custom_analysis_sql.c.analysis_id == analysis
            )
        )
        == []
    )


# --------------------------------------------------------------------------
# CustomAnalysis — embedding docs
# --------------------------------------------------------------------------


def test_analysis_doc_text_is_reproduced_exactly(world) -> None:
    """Changing a separator silently invalidates every stored vector."""
    analysis = world.analysis("revenue", sql="SELECT 1", description="total revenue")

    assert ca._custom_analysis_docs(analysis) == [
        {
            "text": (
                f"custom_analysis: {world.prefix}-revenue, "
                f"description: total revenue, sql: SELECT 1"
            ),
            "name": f"{world.prefix}-revenue",
            "label": "CustomAnalysis",
            "id": analysis,
        }
    ]


def test_a_blank_analysis_description_is_omitted(world) -> None:
    analysis = world.analysis("revenue", sql="SELECT 1", description="   ")
    text = ca._custom_analysis_docs(analysis)[0]["text"]
    assert "description:" not in text


# --------------------------------------------------------------------------
# PqlAnalysis
# --------------------------------------------------------------------------


@pytest.fixture
def pql(world):
    pa.upsert_pql_analysis_node(
        str(uuid.uuid4()), f"{world.prefix}-churn", "who will churn", "PREDICT x"
    )
    world.pql_id = pa.find_pql_analysis_by_name(f"{world.prefix}-churn", None)["id"]
    return world


def test_pql_upsert_is_by_id_and_overwrites(pql) -> None:
    """A PUT: every field is assigned, because the conflict checks already ran."""
    pa.upsert_pql_analysis_node(
        pql.pql_id, f"{pql.prefix}-churn", "revised", "PREDICT y"
    )
    rows = [r for r in pa.list_pql_analyses() if r["name"].startswith(pql.prefix)]
    assert rows == [
        {
            "id": pql.pql_id,
            "name": f"{pql.prefix}-churn",
            "description": "revised",
            "pql": "PREDICT y",
        }
    ]


def test_pql_conflict_lookups_exclude_the_row_being_edited(pql) -> None:
    assert pa.find_pql_analysis_by_name(f"{pql.prefix}-churn", pql.pql_id) is None
    assert pa.find_pql_analysis_by_pql("PREDICT x", pql.pql_id) is None
    assert pa.find_pql_analysis_by_pql("PREDICT x", None)["id"] == pql.pql_id


def test_pql_fetch_by_ids_strips_and_defaults(pql) -> None:
    """These go into a prompt, where None would render as the word "None"."""
    pa.upsert_pql_analysis_node(pql.pql_id, f"{pql.prefix}-churn", "  ", " PREDICT x ")
    row = pa.fetch_pql_analyses_by_ids([pql.pql_id])[pql.pql_id]
    assert row["description"] == ""
    assert row["pql"] == "PREDICT x"


def test_pql_delete(pql) -> None:
    pa.delete_pql_analysis_node(pql.pql_id)
    assert pa.get_pql_analysis_by_id(pql.pql_id) is None


def test_pql_docs_embed_the_question_not_the_pql(pql) -> None:
    """Retrieval here is question-to-question; the PQL is payload.

    Embedding the query body would pull the vector toward syntax and away from
    the question a user actually types.
    """
    docs = pa._pql_analysis_docs(pql.pql_id)
    assert docs == [
        {
            "text": f"{pql.prefix}-churn: who will churn",
            "name": f"{pql.prefix}-churn",
            "id": pql.pql_id,
        }
    ]
    assert "PREDICT" not in docs[0]["text"]


def test_a_blank_pql_description_is_omitted(pql) -> None:
    pa.upsert_pql_analysis_node(pql.pql_id, f"{pql.prefix}-churn", "  ", "PREDICT x")
    assert pa._pql_analysis_docs(pql.pql_id)[0]["text"] == f"{pql.prefix}-churn"


# --------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------


def test_insert_connection_creates_the_database_if_absent(world) -> None:
    """A connection is normally configured *before* anything is ingested."""
    name = f"{world.prefix}-fresh"
    try:
        result = conn.insert_connection(
            connection='{"host": "db.example", "port": 5432}', database_name=name
        )
        assert result["name"] == name
        assert result["connection"] == {"host": "db.example", "port": 5432}
    finally:
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.name == name)
        )


def test_insert_connection_updates_an_existing_database(world) -> None:
    conn.insert_connection(connection='{"host": "first"}', database_name=world.prefix)
    conn.insert_connection(connection='{"host": "second"}', database_name=world.prefix)
    row = store().query_read(
        select(s.catalog_database.c.connection).where(
            s.catalog_database.c.id == world.database
        )
    )[0]
    assert row["connection"] == {"host": "second"}


def test_databases_without_a_connection_are_skipped(world, monkeypatch) -> None:
    """This is "connections", not "databases" — an ingested catalog with no UI
    connection has nothing to return."""
    monkeypatch.setattr(conn, "read_secret", lambda name: None)
    before = conn.list_connections()

    conn.insert_connection(connection='{"host": "db"}', database_name=world.prefix)
    after = conn.list_connections()

    assert {"host": "db"} in after
    assert len(after) == len(before) + 1


def test_a_vault_secret_wins_over_the_stored_connection(world, monkeypatch) -> None:
    conn.insert_connection(connection='{"host": "stored"}', database_name=world.prefix)
    monkeypatch.setattr(
        conn,
        "read_secret",
        lambda name: {"host": "vault"} if name == world.prefix else None,
    )
    assert {"host": "vault"} in conn.list_connections()
    assert {"host": "stored"} not in conn.list_connections()


def test_verify_connectivity_succeeds_against_a_live_store() -> None:
    conn.verify_connectivity()
