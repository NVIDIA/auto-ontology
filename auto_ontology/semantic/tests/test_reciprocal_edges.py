# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_ontology.connectors.sqlite import SQLiteDatabase
from auto_ontology.semantic.reciprocal_edges import (
    _is_verified,
    _verification_sql,
    detect_reciprocal_edges,
)


def _roles() -> list[dict]:
    common = {
        "source_table_id": "relationship-table",
        "target_table_id": "entity-table",
        "target_table_name": "entities",
        "target_column_id": "entity-id-column",
        "target_column_name": "entity_id",
    }
    return [
        {
            **common,
            "source_column_id": "left-column",
            "source_column_name": "left_ref",
        },
        {
            **common,
            "source_column_id": "right-column",
            "source_column_name": "right_ref",
        },
    ]


def _connector() -> MagicMock:
    connector = MagicMock()
    connector.database_name = "db"
    connector.dialect = "sqlite"
    connector.qualify.return_value = '"main"."relationships"'
    return connector


def test_small_accidental_symmetric_sample_is_not_sufficient() -> None:
    assert not _is_verified(
        {
            "orientation_count": 4,
            "missing_reciprocal_count": 0,
            "malformed_edge_count": 0,
        }
    )


def test_verification_query_accepts_full_symmetry_and_rejects_partial(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "relationships.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE relationships "
        "(edge_key INTEGER, left_ref INTEGER, right_ref INTEGER)"
    )
    connection.executemany(
        "INSERT INTO relationships VALUES (?, ?, ?)",
        [
            (1, 10, 11),
            (1, 11, 10),
            (2, 12, 13),
            (2, 13, 12),
            (3, 14, 15),
            (3, 15, 14),
        ],
    )
    connection.commit()
    connector = SQLiteDatabase(str(database_path))
    sql = _verification_sql(
        connector,
        None,
        "relationships",
        "left_ref",
        "right_ref",
        ["edge_key"],
    )

    verified = connector.execute(sql).iloc[0].to_dict()
    assert _is_verified(verified)

    connection.execute("DELETE FROM relationships WHERE edge_key = 3 AND left_ref = 15")
    connection.commit()
    partial = connector.execute(sql).iloc[0].to_dict()
    assert not _is_verified(partial)
    connector.close()
    connection.close()


@patch("auto_ontology.semantic.reciprocal_edges.replace_relationship_semantics")
@patch("auto_ontology.semantic.reciprocal_edges.fetch_semantic_fk_join_groups")
@patch("auto_ontology.semantic.reciprocal_edges.fetch_table_headers_for_database")
@patch("auto_ontology.semantic.reciprocal_edges.get_connectors")
@patch("auto_ontology.semantic.reciprocal_edges.ProbeExecutor")
def test_detects_fully_verified_reciprocal_rows(
    probe_cls,
    get_connectors,
    fetch_tables,
    fetch_roles,
    replace_facts,
) -> None:
    connector = _connector()
    get_connectors.return_value = [connector]
    fetch_tables.return_value = [
        {
            "id": "relationship-table",
            "name": "relationships",
            "schema_name": "main",
            "pk": ["edge_key", "left_ref", "right_ref"],
        }
    ]
    fetch_roles.return_value = _roles()
    probe = probe_cls.return_value.__enter__.return_value
    probe.run.return_value = {
        "ok": True,
        "rows": [
            {
                "orientation_count": 8,
                "missing_reciprocal_count": 0,
                "malformed_edge_count": 0,
            }
        ],
    }

    assert detect_reciprocal_edges("db") == 1

    facts = replace_facts.call_args.args[1]["relationship-table"]
    assert facts == [
        {
            "relationship_type": "reciprocal_undirected_edge",
            "table": "relationships",
            "endpoint_columns": ["left_ref", "right_ref"],
            "edge_key_columns": ["edge_key"],
            "target_table": "entities",
            "target_column": "entity_id",
            "reciprocal_rows_verified": True,
        }
    ]
    verification_sql = probe.run.call_args.args[0]
    assert "NOT EXISTS" in verification_sql
    assert '"left_ref"' in verification_sql
    assert '"right_ref"' in verification_sql


@patch("auto_ontology.semantic.reciprocal_edges.replace_relationship_semantics")
@patch("auto_ontology.semantic.reciprocal_edges.fetch_semantic_fk_join_groups")
@patch("auto_ontology.semantic.reciprocal_edges.fetch_table_headers_for_database")
@patch("auto_ontology.semantic.reciprocal_edges.get_connectors")
@patch("auto_ontology.semantic.reciprocal_edges.ProbeExecutor")
def test_rejects_partial_or_accidental_symmetry_and_clears_stale_facts(
    probe_cls,
    get_connectors,
    fetch_tables,
    fetch_roles,
    replace_facts,
) -> None:
    get_connectors.return_value = [_connector()]
    fetch_tables.return_value = [
        {
            "id": "relationship-table",
            "name": "relationships",
            "schema_name": "main",
            "pk": ["edge_key", "left_ref", "right_ref"],
        }
    ]
    fetch_roles.return_value = _roles()
    probe = probe_cls.return_value.__enter__.return_value
    probe.run.return_value = {
        "ok": True,
        "rows": [
            {
                "orientation_count": 4,
                "missing_reciprocal_count": 1,
                "malformed_edge_count": 1,
            }
        ],
    }

    assert detect_reciprocal_edges("db") == 0
    replace_facts.assert_called_once_with("db", {})


@patch("auto_ontology.semantic.reciprocal_edges.replace_relationship_semantics")
@patch("auto_ontology.semantic.reciprocal_edges.fetch_semantic_fk_join_groups")
@patch("auto_ontology.semantic.reciprocal_edges.fetch_table_headers_for_database")
@patch("auto_ontology.semantic.reciprocal_edges.get_connectors")
@patch("auto_ontology.semantic.reciprocal_edges.ProbeExecutor")
def test_requires_two_roles_to_the_same_target_and_an_edge_key(
    probe_cls,
    get_connectors,
    fetch_tables,
    fetch_roles,
    replace_facts,
) -> None:
    get_connectors.return_value = [_connector()]
    fetch_tables.return_value = [
        {
            "id": "relationship-table",
            "name": "relationships",
            "schema_name": "main",
            "pk": ["left_ref", "right_ref"],
        }
    ]
    roles = _roles()
    roles[1]["target_table_id"] = "other-entity-table"
    fetch_roles.return_value = roles

    assert detect_reciprocal_edges("db") == 0
    probe_cls.return_value.__enter__.return_value.run.assert_not_called()
    replace_facts.assert_called_once_with("db", {})
