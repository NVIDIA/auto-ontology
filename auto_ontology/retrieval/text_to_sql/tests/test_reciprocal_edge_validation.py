# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    SQLValidationAgent,
)
from auto_ontology.retrieval.text_to_sql.reciprocal_edge_validation import (
    find_reciprocal_edge_violations,
)


def _tables(*, verified: bool = True) -> list[dict]:
    return [
        {
            "name": "relationships",
            "schema_name": "main",
            "relationship_semantics": [
                {
                    "relationship_type": "reciprocal_undirected_edge",
                    "endpoint_columns": ["left_ref", "right_ref"],
                    "edge_key_columns": ["edge_key"],
                    "target_table": "entities",
                    "target_column": "entity_id",
                    "reciprocal_rows_verified": verified,
                }
            ],
        }
    ]


def _violations(sql: str, question: str) -> list[str]:
    return find_reciprocal_edge_violations(sql, question, _tables(), dialect="sqlite")


def test_listing_entities_from_one_endpoint_is_valid() -> None:
    sql = "SELECT DISTINCT r.left_ref FROM main.relationships AS r"
    assert _violations(sql, "List all participating entities.") == []


def test_counting_unique_entities_from_one_endpoint_is_valid() -> None:
    sql = "SELECT COUNT(DISTINCT r.left_ref) FROM main.relationships AS r"
    assert _violations(sql, "How many participating entities are there?") == []


def test_counting_relationships_by_edge_key_is_valid() -> None:
    sql = "SELECT COUNT(DISTINCT r.edge_key) FROM main.relationships AS r"
    assert _violations(sql, "How many relationships are there?") == []


def test_neighbor_traversal_is_valid() -> None:
    sql = (
        "SELECT DISTINCT target.label "
        "FROM main.relationships AS r "
        "JOIN main.entities AS source ON source.entity_id = r.left_ref "
        "JOIN main.entities AS target ON target.entity_id = r.right_ref "
        "WHERE source.label = 'selected'"
    )
    assert _violations(sql, "List the neighbors of the selected entity.") == []


def test_canonical_unordered_pair_is_valid() -> None:
    sql = (
        "SELECT DISTINCT LEAST(r.left_ref, r.right_ref), "
        "GREATEST(r.left_ref, r.right_ref) "
        "FROM main.relationships AS r"
    )
    assert _violations(sql, "List each unordered relationship pair.") == []


def test_explicit_request_for_both_endpoint_roles_is_unchanged() -> None:
    sql = "SELECT r.left_ref, r.right_ref FROM main.relationships AS r"
    assert _violations(sql, "Return both endpoint roles for each relationship.") == []


def test_directed_table_without_verified_fact_is_unchanged() -> None:
    sql = "SELECT r.left_ref, r.right_ref FROM main.relationships AS r"
    assert (
        find_reciprocal_edge_violations(
            sql,
            "List each source and target.",
            _tables(verified=False),
            dialect="sqlite",
        )
        == []
    )


def test_two_endpoint_outputs_for_one_concept_are_detected() -> None:
    sql = "SELECT r.left_ref, r.right_ref FROM main.relationships AS r"
    issues = _violations(sql, "List all participating entities.")
    assert any("Both reciprocal endpoint columns" in issue for issue in issues)


def test_same_target_attribute_through_both_aliases_is_detected() -> None:
    sql = (
        "SELECT first_entity.label, second_entity.label "
        "FROM main.relationships AS r "
        "JOIN main.entities AS first_entity "
        "ON first_entity.entity_id = r.left_ref "
        "JOIN main.entities AS second_entity "
        "ON second_entity.entity_id = r.right_ref"
    )
    issues = _violations(sql, "List all participating entity labels.")
    assert any("same target attribute" in issue for issue in issues)


def test_physical_reciprocal_row_count_is_detected() -> None:
    sql = "SELECT COUNT(*) FROM main.relationships AS r"
    issues = _violations(sql, "How many relationships are there?")
    assert any("counts physical reciprocal rows" in issue for issue in issues)


def test_endpoint_or_count_without_deduplication_is_detected() -> None:
    sql = (
        "SELECT COUNT(*) FROM main.relationships AS r "
        "WHERE r.left_ref = 10 OR r.right_ref = 10"
    )
    issues = _violations(sql, "How many neighbors does the entity have?")
    assert any("counts physical reciprocal rows" in issue for issue in issues)


def test_unchanged_reconstruction_is_rejected() -> None:
    sql = "SELECT r.left_ref, r.right_ref FROM main.relationships AS r"
    issues = find_reciprocal_edge_violations(
        sql,
        "List all participating entities.",
        _tables(),
        dialect="sqlite",
        previously_rejected_sql=(
            " SELECT r.left_ref, r.right_ref FROM main.relationships AS r "
        ),
    )
    assert any("same SQL previously rejected" in issue for issue in issues)


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.get_schemas_by_ids",
    return_value={},
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.fetch_all_schema_ids",
    return_value=[],
)
@patch.object(
    SQLValidationAgent,
    "_sql_parse_validation",
    return_value={"success": True},
)
def test_unified_validation_routes_certain_violation_to_reconstruction(
    _parse_validation,
    _fetch_schema_ids,
    _get_schemas,
) -> None:
    sql = "SELECT r.left_ref, r.right_ref FROM main.relationships AS r"
    state = {
        "llm": object(),
        "initial_question": "List all participating entities.",
        "connectors": [],
        "path_state": {
            "relevant_tables": _tables(),
            "sql_generation_result": SimpleNamespace(sql_code=sql),
        },
    }

    result = SQLValidationAgent().execute(state)

    assert result["decision"] == "invalid_sql"
    path_state = result["path_state"]
    assert "correction is mandatory" in path_state["error"]
    assert path_state["error_known_fixable"] is True
    assert path_state["reciprocal_edge_rejected_sql"] == sql
