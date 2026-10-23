# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the entity-coverage pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from auto_ontology.utils.llm_invoke import StrictLLMOutputModel


class ExtractedEntity(StrictLLMOutputModel):
    """One user-text span and the schema concept used to retrieve it."""

    type: Literal["exact_value", "field"]
    original_text: str = Field(
        ...,
        description="Exact text from the sanitized question represented by this entity.",
    )
    meaning: str = Field(
        ...,
        description=(
            "Schema-oriented meaning to use for semantic retrieval. For an exact "
            "value, name the kind of field likely to contain it, not the literal."
        ),
    )


class QuestionExtractionLiteModel(StrictLLMOutputModel):
    """Sanitize + entity extraction only (no subject / used_glossary_names)."""

    sanitized_question: str = Field(
        ...,
        description=(
            "Concise, SQL-ready rewrite of the user's request. "
            "Preserve factual constraints and remove narrative fluff."
        ),
    )
    required_entity_name: list[ExtractedEntity] = Field(
        ...,
        description=(
            "Typed field concepts and literal values present in the completed "
            "sanitized question. Search uses meaning; value probing uses "
            "original_text."
        ),
    )


class QuestionExtractionModel(QuestionExtractionLiteModel):
    """Full sanitize + entity extraction structured output (text-to-sql)."""

    subject: str = Field(
        ...,
        description=(
            "Main subject of the question - the single thing the user is asking "
            "about, resolved through the glossary when the question uses a "
            "shortcut. Excludes filters, aggregations, and date qualifiers."
        ),
    )
    used_glossary_names: list[str] = Field(
        ...,
        description=(
            "Names of glossary entries used to interpret the question. "
            "Return each name exactly as it appears in the glossary, or an "
            "empty list when no glossary definition applies."
        ),
    )


class EntityCoverageResponse(BaseModel):
    """Final response for the entity-coverage pipeline."""

    model_config = ConfigDict(extra="forbid")

    coverage: float = Field(..., ge=0.0, le=1.0)
    candidates: list[dict[str, Any]]
    uncovered_entities: list[str] | None = Field(
        default=None,
        description=(
            "Extracted entities with no covering candidate. Only present when "
            "``return_uncovered_entities=True`` on the import payload."
        ),
    )


__all__ = [
    "ExtractedEntity",
    "QuestionExtractionLiteModel",
    "QuestionExtractionModel",
    "EntityCoverageResponse",
]
