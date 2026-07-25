"""Tests for bridge-table SqlAttribute discovery and compilation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gsf.dal import datasources as neo4j_datasources
from gsf.dal.sql_attributes import SqlAttributeNameConflict
from gsf.semantic import bridge_tables
from gsf.semantic.bridge_tables import (
    _BridgeSqlAttributeProposal,
    _junction_description,
    _table_name_to_term_name,
    build_bridge_tables_sql_attributes,
)
from gsf.semantic.constants import SQL_ATTR_SOURCE_BRIDGE


def test_table_name_to_term_name() -> None:
    assert _table_name_to_term_name("order_items") == "Order Items"


def test_junction_description_lists_related_tables() -> None:
    desc = _junction_description(
        "order_items",
        [
            {"target_table": "orders"},
            {"target_table": "products"},
        ],
    )
    assert "orders" in desc
    assert "products" in desc


def test_junction_description_self_referential() -> None:
    desc = _junction_description(
        "also_buy",
        [
            {"target_table": "product"},
            {"target_table": "product"},
        ],
    )
    assert "product" in desc
    assert "Self-referential" in desc


@patch("gsf.dal.datasources.get_neo4j_conn")
def test_fetch_bridge_table_candidates_query_uses_bridge_source(
    mock_conn: MagicMock,
) -> None:
    mock_conn.return_value.query_read.return_value = [
        {
            "table_id": "t-bridge",
            "table_name": "order_items",
            "schema_name": "sales",
            "description": "",
            "fk_pairs": [
                {
                    "source_column": "order_id",
                    "target_table": "orders",
                    "target_schema": "sales",
                    "target_column": "id",
                    "target_table_id": "t-orders",
                },
                {
                    "source_column": "product_id",
                    "target_table": "products",
                    "target_schema": "sales",
                    "target_column": "id",
                    "target_table_id": "t-products",
                },
            ],
        }
    ]

    rows = neo4j_datasources.fetch_bridge_table_candidates("shop")
    assert len(rows) == 1
    assert rows[0]["table_name"] == "order_items"
    assert len(rows[0]["fk_pairs"]) == 2

    call = mock_conn.return_value.query_read.call_args
    params = call[0][1]
    assert params["database_name"] == "shop"
    assert params["bridge_source"] == SQL_ATTR_SOURCE_BRIDGE
    query = call[0][0]
    assert "HAS_ATTRIBUTE" in query


@patch("gsf.dal.datasources.get_neo4j_conn")
def test_fetch_bridge_table_candidates_returns_empty(mock_conn: MagicMock) -> None:
    mock_conn.return_value.query_read.return_value = []
    assert neo4j_datasources.fetch_bridge_table_candidates("shop") == []


@patch("gsf.semantic.bridge_tables.create_sql_attribute")
@patch("gsf.semantic.bridge_tables._generate_bridge_sql_attribute")
@patch("gsf.semantic.bridge_tables._ensure_bridge_term_id")
@patch("gsf.semantic.bridge_tables.fetch_bridge_table_candidates")
def test_build_bridge_tables_persists_with_bridge_source(
    mock_fetch: MagicMock,
    mock_term: MagicMock,
    mock_llm: MagicMock,
    mock_create: MagicMock,
) -> None:
    mock_fetch.return_value = [
        {
            "table_id": "t-bridge",
            "table_name": "order_items",
            "schema_name": "sales",
            "description": "",
            "fk_pairs": [],
        }
    ]
    mock_term.return_value = "term-bridge"
    mock_llm.return_value = _BridgeSqlAttributeProposal(
        name="Order Products",
        description="Links orders to products.",
        expression=(
            "SELECT * FROM sales.orders o "
            "JOIN sales.order_items oi ON o.id = oi.order_id "
            "JOIN sales.products p ON p.id = oi.product_id"
        ),
    )
    mock_create.return_value = {"id": "attr-1"}

    count = build_bridge_tables_sql_attributes("shop")

    assert count == 1
    mock_create.assert_called_once_with(
        name="Order Products",
        description="Links orders to products.",
        expression=mock_llm.return_value.expression,
        term_id="term-bridge",
        connector="shop",
        source=SQL_ATTR_SOURCE_BRIDGE,
    )


@patch("gsf.semantic.bridge_tables.create_sql_attribute")
@patch("gsf.semantic.bridge_tables._generate_bridge_sql_attribute")
@patch("gsf.semantic.bridge_tables._ensure_bridge_term_id")
@patch("gsf.semantic.bridge_tables.fetch_bridge_table_candidates")
def test_build_bridge_tables_skips_name_conflict(
    mock_fetch: MagicMock,
    mock_term: MagicMock,
    mock_llm: MagicMock,
    mock_create: MagicMock,
) -> None:
    mock_fetch.return_value = [
        {
            "table_id": "t-bridge",
            "table_name": "order_items",
            "schema_name": "sales",
            "fk_pairs": [],
        }
    ]
    mock_term.return_value = "term-bridge"
    mock_llm.return_value = _BridgeSqlAttributeProposal(
        name="Order Products",
        description="Links orders to products.",
        expression="SELECT * FROM sales.order_items",
    )
    mock_create.side_effect = SqlAttributeNameConflict("exists")

    assert build_bridge_tables_sql_attributes("shop") == 0


@patch("gsf.semantic.bridge_tables.merge_term")
@patch("gsf.semantic.bridge_tables.get_term_id_for_table")
def test_ensure_bridge_term_creates_when_missing(
    mock_get_term: MagicMock,
    mock_merge: MagicMock,
) -> None:
    mock_get_term.return_value = None
    mock_merge.return_value = "term-new"

    term_id = bridge_tables._ensure_bridge_term_id(
        {
            "table_id": "t1",
            "table_name": "order_items",
            "fk_pairs": [
                {"target_table": "orders"},
                {"target_table": "products"},
            ],
        }
    )

    assert term_id == "term-new"
    mock_merge.assert_called_once()
    assert mock_merge.call_args[0][0] == "Order Items"


@patch("gsf.semantic.bridge_tables.get_term_id_for_table")
def test_ensure_bridge_term_reuses_existing(mock_get_term: MagicMock) -> None:
    mock_get_term.return_value = "term-existing"
    term_id = bridge_tables._ensure_bridge_term_id(
        {"table_id": "t1", "table_name": "order_items", "fk_pairs": []}
    )
    assert term_id == "term-existing"
