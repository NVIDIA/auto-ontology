# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes behind ambiguity resolution, against a real store."""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("sqlalchemy")

from auto_ontology.dal import ambiguity as a  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402
from auto_ontology.dal.tests.test_descriptions import (  # noqa: E402, F401
    World,
    _one,
    require_schema,
)


@pytest.fixture
def world():
    w = World(f"amb-{uuid.uuid4().hex[:8]}")
    yield w
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == w.db)
    )
    store().query_write(s.term.delete().where(s.term.c.id == w.term))
    store().query_write(
        s.column_attribute.delete().where(s.column_attribute.c.id.in_(w.attributes))
    )


def test_fetch_lists_each_attribute_with_its_column_and_table(world) -> None:
    world.table("standings", description="Standings after each race.")
    cid = world.column(
        "standings",
        "points",
        description="Points.",
        value_description="cumulative",
        description_certified=True,
    )
    aid = world.attribute("standings", "points", description="Season points.")

    rows = a.fetch_attributes_for_ambiguity(_db_name(world))

    assert [r["id"] for r in rows] == [aid]
    row = rows[0]
    assert row["column_id"] == cid
    assert row["column_name"] == "points"
    assert row["column_description"] == "Points."
    assert row["column_certified"] is True
    assert row["table_name"] == "standings"
    assert row["table_description"] == "Standings after each race."
    assert row["description"] == "Season points."
    assert row["value_description"] == "cumulative"


def test_write_resets_the_others_and_fetch_returns_non_empty(world) -> None:
    standings = world.table("standings")
    results = world.table("results")
    world.column("standings", "points")
    world.column("results", "points")
    first = world.attribute("standings", "points")
    second = world.attribute("results", "points")
    name = _db_name(world)

    assert a.write_ambiguity_resolvers(name, {first: "old", second: "Per race."}) == 2
    assert a.write_ambiguity_resolvers(name, {second: "Per race."}) == 1

    assert _one(s.column_attribute, first)["ambiguity_resolver"] == ""
    assert _one(s.column_attribute, second)["ambiguity_resolver"] == "Per race."
    assert a.fetch_ambiguity_resolvers([standings, results]) == {
        (results, "points"): "Per race."
    }
    assert a.fetch_ambiguity_resolvers([]) == {}


def _db_name(w: World) -> str:
    return _one(s.catalog_database, w.db)["name"]
