"""Tests for Neo4j DAL merge key behavior (mocked driver)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gsf.dal import terms as neo4j_terms
from gsf.dal import datasources as neo4j_datasources
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Props


@patch("gsf.dal.terms.get_neo4j_conn")
def test_merge_term_uses_name_and_source(mock_conn: MagicMock) -> None:
    mock_conn.return_value = MagicMock()
    neo4j_terms.merge_term("Orders", "Order entity", "table-1")
    call = mock_conn.return_value.query_write.call_args
    query, params = call[0][0], call[0][1]
    assert "MERGE (term:Term" in query or "MERGE (term:" in query
    assert params["name"] == "Orders"
    assert params["source"] == Props.SEMANTIC_SOURCE


@patch("gsf.dal.datasources.get_neo4j_conn")
def test_store_column_sample_values_skips_empty(mock_conn: MagicMock) -> None:
    mock_conn.return_value = MagicMock()
    neo4j_datasources.store_column_sample_values("table-1", {})
    mock_conn.return_value.query_write.assert_not_called()


@patch("gsf.dal.datasources.get_neo4j_conn")
def test_store_column_sample_values_writes_json(mock_conn: MagicMock) -> None:
    import json

    mock_conn.return_value = MagicMock()
    neo4j_datasources.store_column_sample_values("table-1", {"amount": [10, 20, 30]})
    call = mock_conn.return_value.query_write.call_args
    params = call[0][1]
    assert params["table_id"] == "table-1"
    entries = params["entries"]
    assert len(entries) == 1
    assert entries[0]["column_name"] == "amount"
    assert json.loads(entries[0]["sample_values"]) == [10, 20, 30]
