# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite and spaced-name primary keys, against the real KumoRFM SDK.

A warehouse table keyed on several columns (``PEOPLE`` on
``('Customer ID', 'REGION')``) is only usable once GSF declares that key itself:
metadata inference picks no key for such a table, and KumoRFM then answers a
declared composite key with a single surrogate ``__kumo_key_*`` column that stands
in for the tuple on both ends of every edge. These exercise the SDK rather than a
stand-in because that substitution is the thing being handled.
"""

import pandas as pd
import pytest

from gsf.retrieval.kumo.kumo_model import KumoModel, build_graph_context, key_columns
from gsf.retrieval.kumo.predictor import (
    _apply_join_paths,
    _catalog_key_columns,
    _declare_primary_keys,
    _entity_ids,
)

rfm = pytest.importorskip(
    "kumo_relational_client.relational", reason="Relational engine is not installed"
)


PEOPLE = pd.DataFrame(
    {
        "Customer ID": ["A", "B", "C", "A"],
        "REGION": ["West", "East", "West", "East"],
        "Segment": ["Consumer", "Corporate", "Consumer", "Home Office"],
    }
)
ORDERS = pd.DataFrame(
    {
        "Order ID": [1, 2, 3, 4],
        "Customer ID": ["A", "B", "C", "A"],
        "REGION": ["West", "East", "West", "East"],
        "Line Total": [10.0, 20.0, 30.0, 40.0],
        "Order Date": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        ),
    }
)

# Only the Customer ID side of the identity, as a catalog that records one
# foreign key per column reports it.
CUSTOMER_ID_JOIN = [
    {
        "path": [
            {
                "source_table": "ORDERS",
                "source_column": "Customer ID",
                "target_table": "PEOPLE",
                "target_column": "Customer ID",
            }
        ]
    }
]


def _frames() -> dict[str, pd.DataFrame]:
    return {"PEOPLE": PEOPLE.copy(), "ORDERS": ORDERS.copy()}


def _graph(declare: bool = True):
    graph = rfm.Graph.from_data(_frames(), edges=[], infer_metadata=True, verbose=False)
    if declare:
        _declare_primary_keys(
            graph, {"PEOPLE": ["Customer ID", "REGION"], "ORDERS": ["Order ID"]}
        )
    return graph


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"pk": ["Customer ID", "REGION"]}, ["Customer ID", "REGION"]),
        ({"pk": ["Order ID"]}, ["Order ID"]),
        ({"pk": "Order ID"}, ["Order ID"]),
        ({"pk": [" Customer ID ", "", None]}, ["Customer ID"]),
        ({}, []),
        ({"pk": None}, []),
    ],
)
def test_catalog_key_columns_reads_the_catalog_spelling(entry, expected) -> None:
    assert _catalog_key_columns(entry) == expected


def test_inference_alone_finds_no_key() -> None:
    """The premise: without an explicit declaration these tables have no identity."""
    graph = _graph(declare=False)

    assert key_columns(graph["PEOPLE"]) == []
    assert key_columns(graph["ORDERS"]) == []


def test_declare_primary_keys_sets_composite_and_single_keys() -> None:
    graph = _graph(declare=False)

    assert (
        _declare_primary_keys(
            graph, {"PEOPLE": ["Customer ID", "REGION"], "ORDERS": ["Order ID"]}
        )
        == 2
    )
    assert key_columns(graph["PEOPLE"]) == ["Customer ID", "REGION"]
    assert key_columns(graph["ORDERS"]) == ["Order ID"]


def test_declare_primary_keys_leaves_a_key_absent_from_the_frame() -> None:
    graph = _graph(declare=False)

    assert _declare_primary_keys(graph, {"PEOPLE": ["Nope"]}) == 0
    assert key_columns(graph["PEOPLE"]) == []


def test_apply_join_paths_completes_a_partly_described_identity() -> None:
    """REGION is named by no join path, so it is taken from ORDERS' own column."""
    graph = _graph()

    assert _apply_join_paths(graph, CUSTOMER_ID_JOIN) == 1
    assert [(e.src_table, e.dst_table) for e in graph.edges] == [("ORDERS", "PEOPLE")]


def test_apply_join_paths_skips_an_identity_it_cannot_complete() -> None:
    graph = rfm.Graph.from_data(
        {"PEOPLE": PEOPLE.copy(), "ORDERS": ORDERS.drop(columns=["REGION"])},
        edges=[],
        infer_metadata=True,
        verbose=False,
    )
    _declare_primary_keys(graph, {"PEOPLE": ["Customer ID", "REGION"]})

    assert _apply_join_paths(graph, CUSTOMER_ID_JOIN) == 0
    assert graph.edges == []


def test_entity_ids_are_tuples_for_a_composite_key() -> None:
    graph = _graph()

    ids = _entity_ids(graph, _frames())

    assert ids["people"] == [
        ("A", "West"),
        ("B", "East"),
        ("C", "West"),
        ("A", "East"),
    ]
    assert ids["orders"] == [1, 2, 3, 4]


def _surrogate(graph, table: str) -> str:
    return next(c.name for c in graph[table].columns if c.name.startswith("__kumo_key"))


@pytest.mark.parametrize(
    "entity",
    ["PEOPLE.`Customer ID`", "PEOPLE.REGION"],
    ids=["named-part", "other-part"],
)
def test_validate_accepts_any_column_of_a_composite_identity(entity: str) -> None:
    """The rule the service applies, applied by the parser bundled in the wheel.

    Until kumorfm 2.28.0 the identity reached the parser through an argument only
    ``KumoRFM`` passed, so a caller holding the graph definition -- which is all a
    pre-check has -- rejected the spelling the service accepts, and had to swap in
    the surrogate to get past its own gate.
    """
    graph = _graph()
    _apply_join_paths(graph, CUSTOMER_ID_JOIN)
    model = KumoModel(model=None, graph=graph)

    validated = model.validate_pql(
        f"PREDICT COUNT(ORDERS.*, 0, 30, days) > 0 FOR EACH {entity}"
    )

    assert validated.entity_column == f"PEOPLE.{_surrogate(graph, 'PEOPLE')}"


def test_validate_still_rejects_a_column_that_is_not_the_key() -> None:
    graph = _graph()
    _apply_join_paths(graph, CUSTOMER_ID_JOIN)
    model = KumoModel(model=None, graph=graph)

    with pytest.raises(Exception, match="(?i)primary key"):
        model.validate_pql(
            "PREDICT COUNT(ORDERS.*, 0, 30, days) > 0 FOR EACH PEOPLE.Segment"
        )


def test_graph_context_quotes_spaced_names_and_hides_the_surrogate() -> None:
    graph = _graph()
    _apply_join_paths(graph, CUSTOMER_ID_JOIN)

    ddl, edges, col_stypes, _time_columns = build_graph_context(graph)

    assert "__kumo_key" not in ddl
    assert "PRIMARY KEY (`Customer ID`, REGION)" in ddl
    assert "FOREIGN KEY ORDERS.(`Customer ID`, REGION) -> PEOPLE.<pk>" in ddl
    # The lint reads columns by their own name, so those stay unquoted.
    assert "customer id" in col_stypes["people"]
    assert not any("__kumo_key" in c for cols in col_stypes.values() for c in cols)
    assert [(src, dst) for src, _fkey, dst in edges] == [("ORDERS", "PEOPLE")]
