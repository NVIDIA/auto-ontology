# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for optional semantic relationship enrichment."""

from __future__ import annotations

from unittest.mock import MagicMock

from auto_ontology.semantic import relationship_enrichment as enrichment
from auto_ontology.semantic.relationship_metadata import RelationshipKind


def _context() -> dict:
    return {
        "column_id": "source-column",
        "attribute_id": "target-attribute",
        "relationship_metadata": {
            "curated": {"kind": "ownership", "review": "certified"}
        },
        "source_column_name": "customer_id",
        "source_table_id": "orders",
        "source_table_name": "orders",
        "target_column_name": "id",
        "target_table_id": "customers",
        "target_table_name": "customers",
    }


def _enable(monkeypatch) -> MagicMock:
    monkeypatch.setattr(enrichment, "resolve", lambda *_args: "configured")
    client = MagicMock()
    monkeypatch.setattr(enrichment, "get_llm_client", lambda **_kwargs: client)
    monkeypatch.setattr(
        enrichment, "fetch_semantic_fk_contexts", lambda _db: [_context()]
    )
    return client


def test_enrichment_persists_abstention_as_resumable_suggestion(monkeypatch) -> None:
    _enable(monkeypatch)
    merge = MagicMock(return_value=True)
    monkeypatch.setattr(enrichment, "merge_semantic_fk", merge)
    abstain = enrichment.RelationshipEnrichmentOutput(
        source_column_id="source-column",
        target_attribute_id="target-attribute",
        abstain=True,
    )
    monkeypatch.setattr(
        enrichment,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: abstain,
    )

    assert enrichment.enrich_semantic_fk_relationships("db") == 1
    suggestion = merge.call_args.args[2].suggestion
    assert suggestion is not None
    assert suggestion.abstained is True
    assert suggestion.kind is None


def test_enrichment_rejects_invalid_ids(monkeypatch) -> None:
    _enable(monkeypatch)
    merge = MagicMock(return_value=True)
    monkeypatch.setattr(enrichment, "merge_semantic_fk", merge)
    invalid = enrichment.RelationshipEnrichmentOutput(
        source_column_id="invented-column",
        target_attribute_id="target-attribute",
        abstain=False,
        kind=RelationshipKind.REFERENCE,
        confidence=0.8,
    )
    monkeypatch.setattr(
        enrichment,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: invalid,
    )

    assert enrichment.enrich_semantic_fk_relationships("db") == 0
    merge.assert_not_called()


def test_enrichment_writes_suggestion_only_and_preserves_curated(monkeypatch) -> None:
    _enable(monkeypatch)
    result = enrichment.RelationshipEnrichmentOutput(
        source_column_id="source-column",
        target_attribute_id="target-attribute",
        abstain=False,
        kind=RelationshipKind.REFERENCE,
        role_name="customer",
        description="Order references its customer.",
        source_role="order",
        target_role="customer",
        intent_phrases=["orders by customer"],
        confidence=0.9,
    )
    monkeypatch.setattr(
        enrichment,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: result,
    )
    merge = MagicMock(return_value=True)
    monkeypatch.setattr(enrichment, "merge_semantic_fk", merge)

    assert enrichment.enrich_semantic_fk_relationships("db") == 1
    metadata = merge.call_args.args[2]
    assert metadata.suggestion is not None
    assert metadata.suggestion.kind is RelationshipKind.REFERENCE
    assert metadata.curated is None
    assert metadata.facts is None


def test_enrichment_is_noop_without_llm(monkeypatch) -> None:
    contexts = MagicMock()
    monkeypatch.setattr(enrichment, "fetch_semantic_fk_contexts", contexts)
    monkeypatch.setattr(enrichment, "resolve", lambda *_args: "")
    assert enrichment.enrich_semantic_fk_relationships("db") == 0
    contexts.assert_not_called()


def test_current_generator_receipt_is_resumable(monkeypatch) -> None:
    _enable(monkeypatch)
    context = _context()
    context["relationship_metadata"]["suggestion"] = {
        "generator": "semantic_relationship_enrichment",
        "generator_version": "1",
        "abstained": True,
    }
    monkeypatch.setattr(enrichment, "fetch_semantic_fk_contexts", lambda _db: [context])
    invoke = MagicMock()
    monkeypatch.setattr(enrichment, "invoke_with_structured_output", invoke)

    assert enrichment.enrich_semantic_fk_relationships("db") == 0
    invoke.assert_not_called()


def test_transient_llm_failure_leaves_edge_pending(monkeypatch) -> None:
    _enable(monkeypatch)
    merge = MagicMock()
    monkeypatch.setattr(enrichment, "merge_semantic_fk", merge)
    monkeypatch.setattr(
        enrichment,
        "invoke_with_structured_output",
        MagicMock(side_effect=RuntimeError("temporary failure")),
    )

    assert enrichment.enrich_semantic_fk_relationships("db") == 0
    merge.assert_not_called()
