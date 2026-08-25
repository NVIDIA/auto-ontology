# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predicting for a table that names no entity.

``weekly_sales(week_date, sales)`` holds one row per period and names nothing to
predict for each of. Its only candidate key is the column that orders it, and
KumoRFM will not let one column be both the identity and the time, so every
``FOR EACH`` against it is refused. Making the period the entity does not help:
each period would own a single row, leaving no history to learn from.
"""

import pandas as pd
import pytest

from gsf.retrieval.kumo.predictor import (
    GLOBAL_ENTITY_KEY,
    GLOBAL_ENTITY_TABLE,
    _add_global_entity,
    _link_global_entity,
)

rfm = pytest.importorskip("kumorfm.rfm", reason="KumoRFM SDK is not installed")


def _weekly() -> pd.DataFrame:
    weeks = pd.date_range("2023-01-02", periods=60, freq="W-MON")
    return pd.DataFrame(
        {
            "week_date": weeks,
            "sales": [4000 + (i * 37 % 900) for i in range(len(weeks))],
        }
    )


def test_a_bare_time_series_is_given_an_entity() -> None:
    frames = {"weekly_sales": _weekly()}

    assert _add_global_entity(frames, {}) is True
    assert GLOBAL_ENTITY_TABLE in frames
    assert len(frames[GLOBAL_ENTITY_TABLE]) == 1
    assert GLOBAL_ENTITY_KEY in frames["weekly_sales"].columns


def test_a_real_entity_is_never_displaced() -> None:
    """An invented entity is a last resort, not a preference."""
    frames = {
        "weekly_sales": _weekly(),
        "stores": pd.DataFrame({"store_id": [1, 2, 3], "city": ["a", "b", "c"]}),
    }

    assert _add_global_entity(frames, {}) is False
    assert GLOBAL_ENTITY_TABLE not in frames


def test_a_declared_key_is_never_displaced() -> None:
    frames = {"weekly_sales": _weekly()}

    assert _add_global_entity(frames, {"weekly_sales": ["week_date"]}) is False
    assert GLOBAL_ENTITY_TABLE not in frames


def test_a_table_with_no_time_column_is_left_alone() -> None:
    """Without a time column there is no series to observe over time."""
    frames = {"lookup": pd.DataFrame({"code": ["a", "b"], "label": ["x", "y"]})}

    assert _add_global_entity(frames, {}) is False


def test_a_unique_measure_is_not_mistaken_for_an_identity() -> None:
    """The regression this guard exists for.

    ``sales`` happens to hold a distinct value on every row, but a measure is
    distinct whenever no two periods coincide. Reading that as an identity would
    leave the series without an entity again.
    """
    frame = _weekly()
    assert frame["sales"].is_unique, "fixture must have a unique measure"

    assert _add_global_entity({"weekly_sales": frame}, {}) is True


def test_the_entity_is_keyed_and_the_series_points_at_it() -> None:
    frames = {"weekly_sales": _weekly()}
    _add_global_entity(frames, {})
    graph = rfm.Graph.from_data(frames, edges=[], infer_metadata=True, verbose=False)
    graph["weekly_sales"].time_column = "week_date"

    _link_global_entity(graph)

    assert graph[GLOBAL_ENTITY_TABLE].primary_key.name == GLOBAL_ENTITY_KEY
    assert [(e.src_table, e.fkey, e.dst_table) for e in graph.edges] == [
        ("weekly_sales", GLOBAL_ENTITY_KEY, GLOBAL_ENTITY_TABLE)
    ]


def test_the_query_the_engine_refused_now_validates() -> None:
    from kumorfm.rfm.query_parser import parse_query_locally

    frames = {"weekly_sales": _weekly()}
    _add_global_entity(frames, {})
    graph = rfm.Graph.from_data(frames, edges=[], infer_metadata=True, verbose=False)
    graph["weekly_sales"].time_column = "week_date"
    _link_global_entity(graph)

    query = (
        f"PREDICT SUM(weekly_sales.sales, 0, 1, weeks) "
        f"FOR EACH {GLOBAL_ENTITY_TABLE}.{GLOBAL_ENTITY_KEY}"
    )
    validated = parse_query_locally(query, graph._to_api_graph_definition())

    assert validated.entity_column == f"{GLOBAL_ENTITY_TABLE}.{GLOBAL_ENTITY_KEY}"
