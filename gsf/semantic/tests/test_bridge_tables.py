"""Tests for bridge-table SqlAttribute discovery and compilation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gsf.dal.neo4j import datasources as neo4j_datasources
from gsf.dal.sql_attributes import SqlAttributeNameConflict
from gsf.semantic.bridge_tables import (
    _BridgeSqlAttributeProposal,
    _pick_owner_term_id,
    _target_terms_for_bridge,
    build_bridge_tables_sql_attributes,
)
from gsf.semantic.constants import SQL_ATTR_SOURCE_BRIDGE


@patch("gsf.dal.neo4j.datasources.graph")
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


@patch("gsf.dal.neo4j.datasources.graph")
def test_fetch_bridge_table_candidates_returns_empty(mock_conn: MagicMock) -> None:
    mock_conn.return_value.query_read.return_value = []
    assert neo4j_datasources.fetch_bridge_table_candidates("shop") == []


@patch("gsf.semantic.bridge_tables.get_term_record_for_table")
def test_target_terms_for_bridge_dedupes_self_referential(
    mock_get_term: MagicMock,
) -> None:
    mock_get_term.return_value = {
        "id": "term-product",
        "name": "Product",
        "description": "A sellable item.",
    }
    terms = _target_terms_for_bridge(
        {
            "fk_pairs": [
                {"target_table_id": "t-product", "target_table": "product"},
                {"target_table_id": "t-product", "target_table": "product"},
            ]
        }
    )
    assert len(terms) == 1
    assert terms[0]["id"] == "term-product"
    assert mock_get_term.call_count == 1


def test_pick_owner_term_single_candidate_skips_rerank() -> None:
    term_id = _pick_owner_term_id(
        [{"id": "term-a", "name": "Orders", "description": "Customer orders"}],
        "Links orders to products.",
    )
    assert term_id == "term-a"


@patch("gsf.semantic.bridge_tables.rerank_hits")
def test_pick_owner_term_reranks_two_candidates(mock_rerank: MagicMock) -> None:
    mock_rerank.return_value = [
        {
            "id": "term-products",
            "name": "Products",
            "text": "Products. Sellable goods.",
            "_rerank_score": 0.9,
        }
    ]
    term_id = _pick_owner_term_id(
        [
            {"id": "term-orders", "name": "Orders", "description": "Customer orders"},
            {
                "id": "term-products",
                "name": "Products",
                "description": "Sellable goods.",
            },
        ],
        "Links orders to products via a junction table.",
    )
    assert term_id == "term-products"
    mock_rerank.assert_called_once()
    args, kwargs = mock_rerank.call_args
    assert args[0] == "Links orders to products via a junction table."
    assert {h["id"] for h in args[1]} == {"term-orders", "term-products"}


@patch("gsf.semantic.bridge_tables.create_sql_attribute")
@patch("gsf.semantic.bridge_tables._pick_owner_term_id")
@patch("gsf.semantic.bridge_tables._generate_bridge_sql_attribute")
@patch("gsf.semantic.bridge_tables._target_terms_for_bridge")
@patch("gsf.semantic.bridge_tables.fetch_bridge_table_candidates")
def test_build_bridge_tables_persists_with_bridge_source(
    mock_fetch: MagicMock,
    mock_terms: MagicMock,
    mock_llm: MagicMock,
    mock_pick: MagicMock,
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
    mock_terms.return_value = [
        {"id": "term-orders", "name": "Orders", "description": "Orders"},
        {"id": "term-products", "name": "Products", "description": "Products"},
    ]
    mock_llm.return_value = _BridgeSqlAttributeProposal(
        name="Order Products",
        description="Links orders to products.",
        expression=(
            "SELECT * FROM sales.orders o "
            "JOIN sales.order_items oi ON o.id = oi.order_id "
            "JOIN sales.products p ON p.id = oi.product_id"
        ),
    )
    mock_pick.return_value = "term-products"
    mock_create.return_value = {"id": "attr-1"}

    count = build_bridge_tables_sql_attributes("shop")

    assert count == 1
    mock_pick.assert_called_once_with(
        mock_terms.return_value, "Links orders to products."
    )
    mock_create.assert_called_once_with(
        name="Order Products",
        description="Links orders to products.",
        expression=mock_llm.return_value.expression,
        term_id="term-products",
        connector="shop",
        source=SQL_ATTR_SOURCE_BRIDGE,
    )


@patch("gsf.semantic.bridge_tables.create_sql_attribute")
@patch("gsf.semantic.bridge_tables._pick_owner_term_id")
@patch("gsf.semantic.bridge_tables._generate_bridge_sql_attribute")
@patch("gsf.semantic.bridge_tables._target_terms_for_bridge")
@patch("gsf.semantic.bridge_tables.fetch_bridge_table_candidates")
def test_build_bridge_tables_skips_name_conflict(
    mock_fetch: MagicMock,
    mock_terms: MagicMock,
    mock_llm: MagicMock,
    mock_pick: MagicMock,
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
    mock_terms.return_value = [
        {"id": "term-orders", "name": "Orders", "description": "Orders"}
    ]
    mock_llm.return_value = _BridgeSqlAttributeProposal(
        name="Order Products",
        description="Links orders to products.",
        expression="SELECT * FROM sales.order_items",
    )
    mock_pick.return_value = "term-orders"
    mock_create.side_effect = SqlAttributeNameConflict("exists")

    assert build_bridge_tables_sql_attributes("shop") == 0


@patch("gsf.semantic.bridge_tables.create_sql_attribute")
@patch("gsf.semantic.bridge_tables._generate_bridge_sql_attribute")
@patch("gsf.semantic.bridge_tables._target_terms_for_bridge")
@patch("gsf.semantic.bridge_tables.fetch_bridge_table_candidates")
def test_build_bridge_tables_skips_when_no_target_terms(
    mock_fetch: MagicMock,
    mock_terms: MagicMock,
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
    mock_terms.return_value = []

    assert build_bridge_tables_sql_attributes("shop") == 0
    mock_llm.assert_not_called()
    mock_create.assert_not_called()
