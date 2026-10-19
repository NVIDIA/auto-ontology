# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional LLM enrichment of resolved semantic relationships."""

from __future__ import annotations

import json
import logging
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from auto_ontology.dal.attributes import (
    fetch_semantic_fk_contexts,
    merge_semantic_fk,
)
from auto_ontology.semantic.relationship_metadata import (
    RelationshipKind,
    RelationshipMetadata,
    RelationshipProvenance,
    RelationshipReview,
    RelationshipSuggestion,
)
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.model_config import resolve
from auto_ontology.utils.sample_values import stringify_sample_values

logger = logging.getLogger(__name__)

_GENERATOR = "semantic_relationship_enrichment"
_GENERATOR_VERSION = "1"
_MAX_COMPETING_EDGES = 8

_IntentPhrase = Annotated[str, Field(min_length=1, max_length=160)]


class RelationshipEnrichmentOutput(BaseModel):
    """Strict exact-ID response for one relationship."""

    model_config = ConfigDict(extra="forbid")

    source_column_id: str = Field(min_length=1, max_length=128)
    target_attribute_id: str = Field(min_length=1, max_length=128)
    abstain: bool
    kind: RelationshipKind | None = None
    role_name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=1000)
    source_role: str | None = Field(default=None, max_length=128)
    target_role: str | None = Field(default=None, max_length=128)
    intent_phrases: list[_IntentPhrase] = Field(default_factory=list, max_length=12)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


_SYSTEM_PROMPT = """\
Interpret one already-resolved database relationship. The relationship target is
fixed: never choose or invent IDs. Use the competing edges to disambiguate roles.
Return the exact source_column_id and target_attribute_id supplied. Abstain when
the semantic role is uncertain. Do not infer physical facts or review/certify data.
"""


def enrich_semantic_fk_relationships(database_name: str) -> int:
    """Write resumable model suggestions when an LLM is configured."""
    if not resolve("REASONING", "API_KEY"):
        logger.info("Relationship enrichment skipped: no reasoning LLM configured")
        return 0

    contexts = fetch_semantic_fk_contexts(database_name)
    pending = [
        row
        for row in contexts
        if not _already_enriched(row.get("relationship_metadata") or {})
    ]
    if not pending:
        return 0

    try:
        client = get_llm_client(max_tokens=2048)
    except EnvironmentError:
        logger.info("Relationship enrichment skipped: no reasoning LLM configured")
        return 0

    written = 0
    for row in pending:
        competitors = [
            _prompt_context(other)
            for other in contexts
            if other["column_id"] != row["column_id"]
            and (
                other["source_table_id"] == row["source_table_id"]
                or other["target_table_id"] == row["target_table_id"]
            )
        ][:_MAX_COMPETING_EDGES]
        payload = {
            "relationship": _prompt_context(row),
            "competing_edges": competitors,
        }
        try:
            result = invoke_with_structured_output(
                client,
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(payload, sort_keys=True)),
                ],
                RelationshipEnrichmentOutput,
            )
        except Exception:
            logger.warning(
                "Relationship enrichment failed for %s -> %s; continuing",
                row["column_id"],
                row["attribute_id"],
                exc_info=True,
            )
            continue
        if not _valid_result(row, result):
            continue
        suggestion = RelationshipSuggestion(
            kind=None if result.abstain else result.kind,
            role_name=None if result.abstain else result.role_name,
            description=None if result.abstain else result.description,
            source_role=None if result.abstain else result.source_role,
            target_role=None if result.abstain else result.target_role,
            intent_phrases=[] if result.abstain else result.intent_phrases,
            generator=_GENERATOR,
            generator_version=_GENERATOR_VERSION,
            provenance=RelationshipProvenance.MODEL,
            confidence=None if result.abstain else result.confidence,
            review=RelationshipReview.UNREVIEWED,
            abstained=result.abstain,
        )
        if merge_semantic_fk(
            row["column_id"],
            row["attribute_id"],
            RelationshipMetadata(suggestion=suggestion),
        ):
            written += 1
    return written


def _already_enriched(metadata: dict) -> bool:
    suggestion = metadata.get("suggestion") or {}
    return (
        suggestion.get("generator") == _GENERATOR
        and suggestion.get("generator_version") == _GENERATOR_VERSION
    )


def _prompt_context(row: dict) -> dict:
    source_samples = stringify_sample_values(
        row.get("source_sample_values"),
        max_len=30,
    )
    target_samples = stringify_sample_values(
        row.get("target_sample_values"),
        max_len=30,
    )
    return {
        "source_column_id": row["column_id"],
        "target_attribute_id": row["attribute_id"],
        "source_table": row["source_table_name"],
        "source_table_description": row.get("source_table_description") or "",
        "source_column": row["source_column_name"],
        "source_column_description": row.get("source_column_description") or "",
        "source_sample_values": (source_samples or [])[:5],
        "target_table": row["target_table_name"],
        "target_table_description": row.get("target_table_description") or "",
        "target_column": row["target_column_name"],
        "target_column_description": row.get("target_column_description") or "",
        "target_sample_values": (target_samples or [])[:5],
        "target_attribute": row.get("target_attribute_name") or "",
        "target_attribute_description": row.get("target_attribute_description") or "",
        "target_term": row.get("target_term_name") or "",
        "facts": (row.get("relationship_metadata") or {}).get("facts") or {},
    }


def _valid_result(
    row: dict,
    result: RelationshipEnrichmentOutput | None,
) -> bool:
    if result is None:
        return False
    if (
        result.source_column_id != row["column_id"]
        or result.target_attribute_id != row["attribute_id"]
    ):
        logger.warning("Relationship enrichment returned mismatched edge IDs")
        return False
    return result.abstain or (result.kind is not None and result.confidence is not None)


__all__ = ["RelationshipEnrichmentOutput", "enrich_semantic_fk_relationships"]
