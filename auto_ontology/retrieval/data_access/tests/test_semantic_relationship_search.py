# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.data_access.semantic_search import (
    _hits_to_semantic_rows,
)


def test_relationship_search_preserves_path_seed_metadata() -> None:
    rows = _hits_to_semantic_rows(
        [
            {
                "text": "Client belongs to a district.",
                "_distance": 0.1,
                "metadata": {
                    "id": "client-district",
                    "edge_id": "client-district",
                    "label": "SemanticRelationship",
                    "database_name": "financial",
                    "source_table_id": "client-table",
                    "target_table_id": "district-table",
                    "kind": "location",
                    "intent_phrases": ["client's district"],
                    "trust": "certified",
                    "version": 1,
                },
            }
        ],
        label_filter={"SemanticRelationship"},
        per_label_k=3,
    )

    assert rows == [
        {
            "text": "Client belongs to a district.",
            "id": "client-district",
            "label": "SemanticRelationship",
            "score": 0.1,
            "database_name": "financial",
            "edge_id": "client-district",
            "source_table_id": "client-table",
            "target_table_id": "district-table",
            "kind": "location",
            "intent_phrases": ["client's district"],
            "trust": "certified",
            "version": 1,
        }
    ]
