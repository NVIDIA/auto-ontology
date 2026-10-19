# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest

from auto_ontology.retrieval.text_to_sql.agents import candidates_preparation
from auto_ontology.retrieval.text_to_sql.agents.candidates_preparation import (
    CandidatePreparationAgent,
    _semantic_path_table_ids,
    _trusted_path_table_ids,
)
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_semantic_context,
)
from auto_ontology.retrieval.text_to_sql.models import (
    JoinPathRankItem,
    JoinPathRankingModel,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState
from auto_ontology.semantic.join_paths import SemanticJoinPathCandidate


def _candidate(suffix: str, trust: str = "curated") -> SemanticJoinPathCandidate:
    return SemanticJoinPathCandidate.model_validate(
        {
            "hops": [
                {
                    "source_table_id": "accounts",
                    "source_column_id": f"account-{suffix}",
                    "target_table_id": f"events-{suffix}",
                    "target_column_id": f"event-{suffix}",
                    "relationship_source_column_id": f"account-{suffix}",
                    "relationship_target_attribute_id": f"attr-{suffix}",
                    "relationship_id": f"edge-{suffix}",
                    "direction": "forward",
                    "trust": trust,
                    "source_schema": "public",
                    "source_table": "accounts",
                    "source_column": "id",
                    "target_schema": "public",
                    "target_table": f"events_{suffix}",
                    "target_column": "account_id",
                }
            ],
            "meaning": {
                "pattern": "direct",
                "relationship_trust": [trust],
                "description": "Account directly has qualifying events.",
                "qualification_scope": "direct relationship",
                "resulting_grain": "preserves accounts row grain",
                "fan_out": False,
                "duplication_risk": "none",
                "contrast": "No shared container.",
            },
            "is_shortest": True,
        }
    )


def _chemistry_candidate(scope: str) -> SemanticJoinPathCandidate:
    incidence = scope == "incidence"
    middle = "connected" if incidence else "molecule"
    kind = "incidence" if incidence else "membership"
    intent_phrases = (
        ["bonded connected directly participating"]
        if incidence
        else ["molecules containing all bonds sulfur iodine"]
    )
    hops = []
    for index, (source, target, direction) in enumerate(
        [
            ("atom", middle, "reverse" if incidence else "forward"),
            (middle, "bond", "forward" if incidence else "reverse"),
        ]
    ):
        hops.append(
            {
                "source_table_id": source,
                "source_column_id": f"{source}-column-{index}",
                "target_table_id": target,
                "target_column_id": f"{target}-column-{index}",
                "relationship_source_column_id": f"{scope}-source-{index}",
                "relationship_target_attribute_id": f"{scope}-attr-{index}",
                "relationship_id": f"{scope}-edge-{index}",
                "direction": direction,
                "trust": "certified",
                "source_table": source,
                "source_column": "id",
                "target_table": target,
                "target_column": "id",
                "relationship_kind": kind,
                "relationship_metadata": {
                    "version": 1,
                    "curated": {
                        "kind": kind,
                        "intent_phrases": intent_phrases,
                        "review": "certified",
                    },
                },
            }
        )
    return SemanticJoinPathCandidate.model_validate(
        {
            "hops": hops,
            "meaning": {
                "pattern": ("shared_child_junction" if incidence else "shared_parent"),
                "relationship_trust": ["certified", "certified"],
                "description": f"Atom to bond by {scope}.",
                "qualification_scope": (
                    "direct incidence through connected"
                    if incidence
                    else "shared-parent membership through molecule"
                ),
                "resulting_grain": "bond rows",
                "fan_out": True,
                "duplication_risk": "possible",
                "contrast": "Direct incidence differs from shared membership.",
            },
            "is_shortest": True,
        }
    )


def _financial_candidate(scope: str) -> SemanticJoinPathCandidate:
    direct = scope == "client"
    route = (
        [("client", "district")]
        if direct
        else [("client", "disp"), ("disp", "account"), ("account", "district")]
    )
    phrase = (
        "client district residence home" if direct else "account branch district opened"
    )
    hops = [
        {
            "source_table_id": source,
            "source_column_id": f"{scope}-source-{index}",
            "target_table_id": target,
            "target_column_id": f"{scope}-target-{index}",
            "relationship_source_column_id": f"{scope}-fk-{index}",
            "relationship_target_attribute_id": f"{scope}-attr-{index}",
            "relationship_id": f"{scope}-edge-{index}",
            "direction": "forward",
            "trust": "certified",
            "source_table": source,
            "source_column": "id",
            "target_table": target,
            "target_column": "id",
            "relationship_kind": "location" if target == "district" else "membership",
            "relationship_metadata": {
                "version": 1,
                "curated": {
                    "kind": "location" if target == "district" else "membership",
                    "intent_phrases": [phrase],
                    "review": "certified",
                },
            },
        }
        for index, (source, target) in enumerate(route)
    ]
    return SemanticJoinPathCandidate.model_validate(
        {
            "hops": hops,
            "meaning": {
                "pattern": "direct" if direct else "chain",
                "relationship_trust": ["certified"] * len(hops),
                "description": f"District in {scope} scope.",
                "qualification_scope": (
                    "direct client location"
                    if direct
                    else "account or branch location through disposition"
                ),
                "resulting_grain": "client rows",
                "fan_out": not direct,
                "duplication_risk": "none" if direct else "possible",
                "contrast": "Client and account districts are distinct roles.",
            },
            "is_shortest": direct,
        }
    )


def test_ambiguous_invalid_ranking_retains_every_path(monkeypatch) -> None:
    candidates = [_candidate("a"), _candidate("b")]
    calls: list[Any] = []

    def invalid_ranking(*args: Any, **kwargs: Any) -> JoinPathRankingModel:
        calls.append((args, kwargs))
        return JoinPathRankingModel(
            rankings=[
                JoinPathRankItem(
                    fingerprint="invented",
                    confidence="high",
                    rationale="invalid",
                )
            ]
        )

    monkeypatch.setattr(
        candidates_preparation, "invoke_with_structured_output", invalid_ranking
    )
    ranked = CandidatePreparationAgent()._rank_join_path_group(
        cast(AgentState, {"llm": object()}),
        "accounts with qualifying events",
        "destination",
        {"table_name": "events", "attr_name": "events"},
        candidates,
        [],
    )

    assert len(calls) == 1
    assert {item["fingerprint"] for item in ranked} == {
        candidate.fingerprint for candidate in candidates
    }
    assert all(item["recommendation"]["rationale"] for item in ranked)


def test_only_trusted_recommended_paths_protect_tables() -> None:
    curated = _candidate("curated", trust="curated")
    suggested = _candidate("suggested", trust="suggested")
    ranked = [
        {
            "fingerprint": curated.fingerprint,
            "recommendation": {"confidence": "medium"},
        },
        {
            "fingerprint": suggested.fingerprint,
            "recommendation": {"confidence": "high"},
        },
    ]

    assert _trusted_path_table_ids(ranked, [curated, suggested]) == {
        "accounts",
        "events-curated",
    }


def test_relationship_hits_and_all_path_options_seed_tables() -> None:
    candidate = _candidate("indirect")
    ranked = [{"candidate": candidate.model_dump(mode="json")}]

    assert _semantic_path_table_ids(
        [
            {
                "source_table_id": "clients",
                "target_table_id": "districts",
            }
        ],
        ranked,
    ) == {"clients", "districts", "accounts", "events-indirect"}


@pytest.mark.parametrize(
    ("question", "expected_scope"),
    [
        ("atoms directly connected to single bonds", "incidence"),
        ("all bonds in molecules containing iodine", "membership"),
        ("molecules containing sulfur and also containing a double bond", "membership"),
    ],
)
def test_general_chemistry_intents_rank_the_correct_scope(
    question: str,
    expected_scope: str,
) -> None:
    incidence = _chemistry_candidate("incidence")
    membership = _chemistry_candidate("membership")

    ranked = CandidatePreparationAgent()._rank_join_path_group(
        cast(AgentState, {"llm": object()}),
        question,
        "bond-attribute",
        {"table_name": "bond", "attr_name": "bond"},
        [incidence, membership],
        [],
    )

    expected = incidence if expected_scope == "incidence" else membership
    assert ranked[0]["fingerprint"] == expected.fingerprint
    assert ranked[0]["recommendation"]["confidence"] == "high"


@pytest.mark.parametrize(
    ("question", "expected_scope"),
    [
        ("show the client's district of residence", "client"),
        ("show the district where the account was opened", "account"),
    ],
)
def test_financial_role_intents_distinguish_district_scope(
    question: str,
    expected_scope: str,
) -> None:
    direct = _financial_candidate("client")
    account = _financial_candidate("account")

    ranked = CandidatePreparationAgent()._rank_join_path_group(
        cast(AgentState, {"llm": object()}),
        question,
        "district-attribute",
        {"table_name": "district", "attr_name": "district"},
        [direct, account],
        [],
    )

    expected = direct if expected_scope == "client" else account
    assert ranked[0]["fingerprint"] == expected.fingerprint
    assert ranked[0]["recommendation"]["confidence"] == "high"


def test_semantic_context_groups_options_and_renders_exact_joins() -> None:
    candidate = _candidate("incidence")
    record = {
        "destination_id": "dest",
        "destination": {"attr_name": "events", "table_name": "events_incidence"},
        "fingerprint": candidate.fingerprint,
        "candidate": candidate.model_dump(mode="json"),
        "recommendation": {"confidence": "high", "rationale": "Intent match."},
    }

    rendered = format_semantic_context(
        {
            "attr_name": "Account",
            "col_name": "id",
            "table_name": "accounts",
            "schema_name": "public",
        },
        [
            {
                "id": "dest",
                "attr_name": "events",
                "table_name": "events_incidence",
                "col_name": "account_id",
                "path": candidate.join_path(),
            }
        ],
        join_path_candidates=[record],
    )

    assert candidate.fingerprint in rendered
    assert "Recommendation: high" in rendered
    assert "resulting grain:" in rendered
    assert "Fan-out: False" in rendered
    assert "public.accounts.id = public.events_incidence.account_id" in rendered
    assert "'bonded to' or 'connected to'" in rendered
    assert "client's own district" in rendered
    assert (
        rendered.count("public.accounts.id = public.events_incidence.account_id") == 1
    )
