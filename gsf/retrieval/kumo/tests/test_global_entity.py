# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predicting for a table that names no entity.

``weekly_sales(week_date, sales)`` holds one row per period and names nothing to
predict for each of. Its only candidate key is the column that orders it, and
KumoRFM will not let one column be both the identity and the time, so every
``FOR EACH`` against it is refused. Making the period the entity does not help:
each period would own a single row, leaving no history to learn from.

Strandedness is judged from the built graph rather than from the frames, because
the graph is what KumoRFM enforces. These cover the shapes where the two disagree.
"""

import pandas as pd
import pytest

from gsf.retrieval.kumo.predictor import (
    GLOBAL_ENTITY_KEY,
    GLOBAL_ENTITY_TABLE,
    _declare_primary_keys,
    _link_global_entity,
    _stranded_tables,
    _supply_global_entity,
)

rfm = pytest.importorskip("kumorfm.rfm", reason="KumoRFM SDK is not installed")


def _weekly(**extra) -> pd.DataFrame:
    weeks = pd.date_range("2023-01-02", periods=60, freq="W-MON")
    return pd.DataFrame(
        {
            "week_date": weeks,
            "sales": [4000 + (i * 37 % 900) for i in range(len(weeks))],
            **extra,
        }
    )


def _built(frames: dict[str, pd.DataFrame], catalog_keys=None):
    graph = rfm.Graph.from_data(frames, edges=[], infer_metadata=True, verbose=False)
    _declare_primary_keys(graph, catalog_keys or {})
    return graph


def test_a_bare_series_is_stranded() -> None:
    assert _stranded_tables(_built({"weekly_sales": _weekly()})) == ["weekly_sales"]


def test_a_key_the_graph_could_not_hold_does_not_count() -> None:
    """The catalog naming the time column as the key is not a key.

    ``primary_key`` refuses the time column, ``_declare_primary_keys`` swallows
    that, and the table is left with no identity. Trusting the catalog's intent
    rather than the graph left it unpredictable.
    """
    graph = _built({"weekly_sales": _weekly()}, {"weekly_sales": ["week_date"]})

    assert graph["weekly_sales"].primary_key is None
    assert _stranded_tables(graph) == ["weekly_sales"]


def test_a_period_label_is_not_an_identity() -> None:
    """``week_no`` is a label on a periodic aggregate, not something to predict for."""
    graph = _built({"weekly_sales": _weekly(week_no=range(60))})

    assert _stranded_tables(graph) == ["weekly_sales"]


def test_a_table_with_two_timestamps_is_still_a_series() -> None:
    weeks = pd.date_range("2023-01-02", periods=60, freq="W-MON")
    frames = {
        "weekly_sales": pd.DataFrame(
            {
                "period_start": weeks,
                "period_end": weeks,
                "sales": [1] * 60,
            }
        )
    }

    assert _stranded_tables(_built(frames)) == ["weekly_sales"]


def test_a_keyed_table_is_left_alone() -> None:
    frames = {"stores": pd.DataFrame({"store_id": [1, 2, 3], "city": [*"abc"]})}

    assert _stranded_tables(_built(frames, {"stores": ["store_id"]})) == []


def test_a_keyed_table_elsewhere_does_not_strand_an_unrelated_series() -> None:
    """Retrieval returns a mix; a keyed table says nothing about a series that
    carries none of its columns and so can never link to it."""
    frames = {
        "stores": pd.DataFrame({"store_id": [1, 2, 3], "city": [*"abc"]}),
        "weekly_sales": _weekly(),
    }

    assert _stranded_tables(_built(frames, {"stores": ["store_id"]})) == [
        "weekly_sales"
    ]


def test_a_series_that_reaches_an_entity_is_left_alone() -> None:
    frames = {
        "stores": pd.DataFrame({"store_id": [1, 2, 3], "city": [*"abc"]}),
        "weekly_sales": _weekly(store_id=[1, 2, 3] * 20),
    }
    graph = _built(frames, {"stores": ["store_id"]})
    graph.link(src_table="weekly_sales", fkey="store_id", dst_table="stores")

    assert _stranded_tables(graph) == []


def test_the_entity_is_keyed_and_each_series_points_at_it() -> None:
    frames = {
        "weekly_sales": _weekly(),
        "daily_ops": pd.DataFrame(
            {
                "d": pd.date_range("2024-01-01", periods=50),
                "n": range(50),
            }
        ),
    }
    stranded = _stranded_tables(_built(frames))
    _supply_global_entity(frames, stranded)
    graph = _built(frames)

    _link_global_entity(graph, stranded)

    assert graph[GLOBAL_ENTITY_TABLE].primary_key.name == GLOBAL_ENTITY_KEY
    assert sorted((e.src_table, e.dst_table) for e in graph.edges) == [
        ("daily_ops", GLOBAL_ENTITY_TABLE),
        ("weekly_sales", GLOBAL_ENTITY_TABLE),
    ]


def test_the_query_the_engine_refused_now_validates() -> None:
    from kumorfm.rfm.query_parser import parse_query_locally

    frames = {"weekly_sales": _weekly()}
    stranded = _stranded_tables(_built(frames))
    _supply_global_entity(frames, stranded)
    graph = _built(frames)
    _link_global_entity(graph, stranded)

    validated = parse_query_locally(
        f"PREDICT SUM(weekly_sales.sales, 0, 1, weeks) "
        f"FOR EACH {GLOBAL_ENTITY_TABLE}.{GLOBAL_ENTITY_KEY}",
        graph._to_api_graph_definition(),
    )

    assert validated.entity_column == f"{GLOBAL_ENTITY_TABLE}.{GLOBAL_ENTITY_KEY}"


def test_keying_the_entity_never_fails_the_request() -> None:
    """It is supplemental, so a graph that cannot take it is left as it was."""
    frames = {"weekly_sales": _weekly()}
    graph = _built(frames)

    _link_global_entity(graph, ["weekly_sales"])

    assert GLOBAL_ENTITY_TABLE not in graph.tables


@pytest.mark.parametrize(
    "table",
    [GLOBAL_ENTITY_TABLE, GLOBAL_ENTITY_TABLE.upper()],
    ids=["as-written", "recased"],
)
@pytest.mark.parametrize(
    "ids", [None, {GLOBAL_ENTITY_TABLE: [1]}], ids=["absent", "given"]
)
def test_no_query_is_run_against_the_supplied_entity(table: str, ids) -> None:
    """It exists only in memory, so any query naming it would fail.

    The resolver answers from the entity itself rather than trusting the model to
    avoid writing one, and does so whether or not the caller passed ids.
    """
    from gsf.retrieval.kumo.pql_gen import _resolve_indices

    class _Warehouse:
        def execute(self, sql: str):
            raise AssertionError(f"queried the warehouse: {sql[:60]}")

    indices = _resolve_indices(
        f"PREDICT SUM(weekly_sales.sales, 0, 1, weeks) "
        f"FOR EACH {table}.{GLOBAL_ENTITY_KEY}",
        f"SELECT {GLOBAL_ENTITY_KEY} FROM {table}",
        _Warehouse(),
        100,
        None,
        ids,
    )

    assert indices == [1]
