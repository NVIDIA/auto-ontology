# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.data_access.semantic_search import _hits_to_semantic_rows
from auto_ontology.semantic.constants import LABEL_CONCEPT, LABEL_SQL_ATTRIBUTE


def _hits() -> list[dict]:
    return [
        {"text": "Concept: Order.", "metadata": {"id": "c", "label": LABEL_CONCEPT}},
        {"text": "Order id.", "metadata": {"id": "a", "label": LABEL_SQL_ATTRIBUTE}},
    ]


def test_concepts_are_not_returned_by_an_unfiltered_search() -> None:
    assert [r["id"] for r in _hits_to_semantic_rows(_hits())] == ["a"]


def test_concepts_are_returned_when_asked_for_by_label() -> None:
    rows = _hits_to_semantic_rows(_hits(), label_filter={LABEL_CONCEPT})

    assert [r["id"] for r in rows] == ["c"]
