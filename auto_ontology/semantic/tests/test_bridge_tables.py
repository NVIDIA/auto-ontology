"""Tests for bridge-table semantic orchestration."""

from unittest.mock import patch

from auto_ontology.semantic.bridge_tables import build_bridge_tables_sql_attributes


def test_structural_bridge_is_marked_before_downstream_skip() -> None:
    candidate = {
        "table_id": "bridge-id",
        "table_name": "order_product",
        "schema_name": "public",
        "fk_pairs": [],
    }

    with (
        patch(
            "auto_ontology.semantic.bridge_tables.fetch_bridge_table_candidates",
            return_value=[candidate],
        ),
        patch(
            "auto_ontology.semantic.bridge_tables.mark_table_as_junction"
        ) as mock_mark,
        patch(
            "auto_ontology.semantic.bridge_tables._target_terms_for_bridge",
            return_value=[],
        ),
        patch(
            "auto_ontology.semantic.bridge_tables._generate_bridge_sql_attribute"
        ) as generate,
    ):
        created = build_bridge_tables_sql_attributes("warehouse")

    assert created == 0
    mock_mark.assert_called_once_with("bridge-id")
    generate.assert_not_called()
