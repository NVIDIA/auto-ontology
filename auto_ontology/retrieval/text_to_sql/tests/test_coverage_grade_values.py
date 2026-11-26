# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A named value that the data holds counts as covered (#461)."""

from typing import Any
from unittest.mock import patch

from auto_ontology.retrieval.entity_coverage.agents import coverage
from auto_ontology.retrieval.entity_coverage.agents.coverage import (
    CoverageGradeAgent,
)

_REPO_VALUES = {"attr-repo": ["astrolabe", "chorecraft", "infoFlows"]}


def _grade(
    entities: list[str],
    hits: list[dict[str, Any]],
    values: dict[str, list[str]],
) -> dict[str, Any]:
    path_state = {
        "entities": entities,
        "retrieved_column_attributes": hits,
        "return_uncovered_entities": True,
    }

    def fake_values(attr_ids: list[str]) -> dict[str, list[str]]:
        return {i: values[i] for i in attr_ids if i in values}

    with (
        patch.object(coverage, "fetch_attr_sample_values", side_effect=fake_values),
        patch.object(coverage, "fetch_attr_column_contexts", return_value={}),
    ):
        out = CoverageGradeAgent().execute({"path_state": path_state})  # type: ignore[arg-type]
    return out["path_state"]["final_response"]


def _hits() -> list[dict[str, Any]]:
    return [
        # The concept phrase lands within the 0.75 cutoff.
        {"id": "attr-version", "score": 0.4, "query_entity": "harness version"},
        # The repo column is retrieved for the value, but beyond the cutoff.
        {"id": "attr-repo", "score": 0.9, "query_entity": "astrolabe"},
    ]


def test_value_named_in_the_data_is_covered() -> None:
    response = _grade(["astrolabe", "harness version"], _hits(), _REPO_VALUES)

    assert response["coverage"] == 1.0
    assert response["uncovered_entities"] == []


def test_value_match_ignores_case_and_spacing() -> None:
    response = _grade(["  ASTROLABE "], _hits(), _REPO_VALUES)

    assert response["coverage"] == 1.0


def test_value_absent_from_the_data_stays_uncovered() -> None:
    response = _grade(["nonesuch", "harness version"], _hits(), _REPO_VALUES)

    assert response["coverage"] == 0.5
    assert response["uncovered_entities"] == ["nonesuch"]


def test_phrase_merely_containing_a_value_stays_uncovered() -> None:
    values = {"attr-status": ["open", "closed"]}
    hits = [{"id": "attr-status", "score": 0.9, "query_entity": "open ticket"}]

    response = _grade(["open ticket"], hits, values)

    assert response["coverage"] == 0.0
    assert response["uncovered_entities"] == ["open ticket"]


def test_values_of_columns_that_were_not_retrieved_are_ignored() -> None:
    response = _grade(["astrolabe"], [], _REPO_VALUES)

    assert response["coverage"] == 0.0
