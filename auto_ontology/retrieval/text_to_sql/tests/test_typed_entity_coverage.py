# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage identity stays tied to the user's original typed-entity text."""

from auto_ontology.retrieval.entity_coverage.agents.coverage import (
    _compute_coverage,
    _entity_original_texts,
    _uncovered_entities,
)


def test_typed_entity_coverage_uses_original_text() -> None:
    entities = _entity_original_texts(
        [
            {
                "type": "field",
                "original_text": "artist",
                "meaning": "card artist",
            },
            {
                "type": "exact_value",
                "original_text": "Angel of Mercy",
                "meaning": "card name",
            },
        ]
    )
    hits = [{"query_entities": ["artist"]}]

    assert entities == ["artist", "Angel of Mercy"]
    assert _compute_coverage(entities, hits) == 0.5
    assert _uncovered_entities(entities, hits) == ["Angel of Mercy"]
