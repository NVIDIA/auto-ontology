# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from auto_ontology.retrieval.text_to_sql.semantic_scope_validation import (
    parse_sql_join_signatures,
    validate_semantic_join_scope,
)


def _record(
    fingerprint: str,
    hops: list[tuple[str, str, str, str]],
    *,
    destination: str,
    confidence: str = "high",
    trust: str = "certified",
    scope: str = "direct client samples",
    grain: str = "one row per client",
) -> dict[str, Any]:
    return {
        "destination_id": destination,
        "destination": {"table_name": destination},
        "fingerprint": fingerprint,
        "recommendation": {"confidence": confidence, "rationale": "ranked"},
        "candidate": {
            "hops": [
                {
                    "source_schema": "chem",
                    "source_table": source_table,
                    "source_column": source_column,
                    "target_schema": "chem",
                    "target_table": target_table,
                    "target_column": target_column,
                    "trust": trust,
                }
                for source_table, source_column, target_table, target_column in hops
            ],
            "meaning": {
                "qualification_scope": scope,
                "resulting_grain": grain,
            },
        },
    }


def test_matches_aliased_and_qualified_join() -> None:
    candidate = _record(
        "direct",
        [("clients", "id", "samples", "client_id")],
        destination="samples",
    )

    result = validate_semantic_join_scope(
        "SELECT c.id FROM chem.clients c JOIN chem.samples s ON c.id = s.client_id",
        [candidate],
    )

    assert result.status == "matched"
    assert result.selected_fingerprints == ["direct"]


def test_equal_hop_matches_remain_ambiguous() -> None:
    first = _record(
        "first",
        [("clients", "id", "samples", "client_id")],
        destination="samples",
    )
    second = {**first, "fingerprint": "second"}

    result = validate_semantic_join_scope(
        "SELECT * FROM clients c JOIN samples s ON c.id = s.client_id",
        [first, second],
    )

    assert result.status == "ambiguous"
    assert set(result.selected_fingerprints) == {"first", "second"}
    assert not result.enforce_rejection


def test_longer_route_wins_over_subset_candidate() -> None:
    short = _record(
        "short",
        [("clients", "id", "assays", "client_id")],
        destination="assays",
    )
    long = _record(
        "long",
        [
            ("clients", "id", "assays", "client_id"),
            ("assays", "id", "samples", "assay_id"),
        ],
        destination="assays",
    )

    result = validate_semantic_join_scope(
        "SELECT * FROM clients c JOIN assays a ON c.id = a.client_id "
        "JOIN samples s ON a.id = s.assay_id",
        [short, long],
    )

    assert result.status == "matched"
    assert result.selected_fingerprints == ["long"]


def test_unresolved_unqualified_columns_fail_open() -> None:
    candidate = _record(
        "direct",
        [("clients", "id", "samples", "client_id")],
        destination="samples",
    )

    result = validate_semantic_join_scope(
        "SELECT * FROM clients JOIN samples ON id = client_id",
        [candidate],
    )

    assert result.status == "unknown_metadata"
    assert not result.enforce_rejection


def test_scope_wrong_route_rejected_only_with_trusted_high_alternative() -> None:
    shared_project = _record(
        "shared-project",
        [
            ("clients", "project_id", "projects", "id"),
            ("projects", "id", "samples", "project_id"),
        ],
        destination="samples",
        confidence="low",
        scope="samples in any project shared with the client",
        grain="client-sample pairs",
    )
    direct = _record(
        "direct-client",
        [("clients", "id", "samples", "client_id")],
        destination="samples",
        confidence="high",
        scope="samples directly owned by the client",
    )

    result = validate_semantic_join_scope(
        "SELECT s.id FROM clients c "
        "JOIN projects p ON c.project_id = p.id "
        "JOIN samples s ON p.id = s.project_id",
        [direct, shared_project],
    )

    assert result.status == "semantic_mismatch"
    assert result.enforce_rejection
    assert result.auto_repair_fingerprint == "direct-client"
    assert "samples directly owned by the client" in result.message
    assert "chem.clients.id = chem.samples.client_id" in result.message


def test_suggested_alternative_is_warning_only() -> None:
    shared_project = _record(
        "shared-project",
        [
            ("clients", "project_id", "projects", "id"),
            ("projects", "id", "samples", "project_id"),
        ],
        destination="samples",
        confidence="low",
        trust="suggested",
        scope="shared project",
    )
    suggestion = _record(
        "suggested-direct",
        [("clients", "id", "samples", "client_id")],
        destination="samples",
        confidence="high",
        trust="suggested",
    )

    result = validate_semantic_join_scope(
        "SELECT s.id FROM clients c "
        "JOIN projects p ON c.project_id = p.id "
        "JOIN samples s ON p.id = s.project_id",
        [suggestion, shared_project],
    )

    assert result.status == "unknown_metadata"
    assert not result.enforce_rejection
    assert result.warnings


def test_parser_reports_unresolved_join_without_rejecting() -> None:
    signatures, resolved = parse_sql_join_signatures(
        "SELECT * FROM clients c JOIN samples s ON lower(c.id) = s.client_id"
    )

    assert not signatures
    assert not resolved
