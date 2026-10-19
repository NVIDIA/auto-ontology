# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from auto_ontology.retrieval.text_to_sql.models import (
    SemanticScopeValidationResult,
)
from auto_ontology.retrieval.text_to_sql.semantic_path_observability import (
    TRACE_KEY,
    record_ranked_paths,
    record_validator_outcome,
)


def _record(fingerprint: str, confidence: str, trust: str) -> dict:
    return {
        "destination_id": "district",
        "fingerprint": fingerprint,
        "recommendation": {
            "confidence": confidence,
            "rationale": "direct client district intent",
        },
        "candidate": {
            "hops": [
                {
                    "fingerprint": f"hop-{fingerprint}",
                    "source_schema": "bank",
                    "source_table": "client",
                    "source_column": "district_id",
                    "target_schema": "bank",
                    "target_table": "district",
                    "target_column": "id",
                    "direction": "forward",
                    "relationship_kind": "location",
                    "trust": trust,
                    "interpretation_source": (
                        "curated" if trust == "curated" else "suggestion"
                    ),
                    "relationship_metadata": {
                        "version": 1,
                        "curated": {
                            "generator": "manual",
                            "generator_version": "2",
                        },
                    },
                }
            ],
            "meaning": {
                "qualification_scope": "client district",
                "resulting_grain": "one row per client",
                "fan_out": False,
            },
        },
    }


def test_trace_is_json_serializable_and_captures_every_path_signal() -> None:
    path_state: dict = {}
    records = [
        _record("direct-client", "high", "curated"),
        _record("account-disposition", "low", "suggested"),
    ]

    record_ranked_paths(path_state, records)
    trace = path_state[TRACE_KEY]

    json.dumps(path_state)
    assert [path["fingerprint"] for path in trace["paths"]] == [
        "direct-client",
        "account-disposition",
    ]
    assert trace["paths"][0]["hops"][0]["structural_source"] == "semantic_fk"
    assert trace["paths"][0]["hops"][0]["metadata"] == {
        "source": "curated",
        "version": 1,
        "generator": "manual",
        "generator_version": "2",
    }
    assert trace["paths"][0]["scope"] == "client district"
    assert trace["paths"][0]["grain"] == "one row per client"
    assert trace["paths"][0]["fan_out"] is False
    assert trace["paths"][0]["ranking"]["rank"] == 1
    assert trace["paths"][1]["ranking"]["rank"] == 2


def test_mismatch_records_enforcement_and_intervention() -> None:
    path_state: dict = {}
    record_ranked_paths(
        path_state,
        [
            _record("direct-client", "high", "curated"),
            _record("account-disposition", "low", "suggested"),
        ],
    )
    mismatch = SemanticScopeValidationResult(
        status="semantic_mismatch",
        selected_fingerprints=["account-disposition"],
        alternative_fingerprints=["direct-client"],
        enforce_rejection=True,
    )

    record_validator_outcome(
        path_state,
        "SELECT * FROM client JOIN district ON client.district_id = district.id",
        mismatch,
    )
    trace = path_state[TRACE_KEY]
    assert trace["sql_selections"][-1]["would_reject"] is True
    assert trace["sql_selections"][-1]["enforced"] is True
    assert trace["counters"]["interventions"] == 1
    assert trace["counters"]["selected_rank"] == 2
    assert trace["counters"]["repair_result"] == "pending"
