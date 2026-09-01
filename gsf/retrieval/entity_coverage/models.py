# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the entity-coverage pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gsf.utils.llm_invoke import StrictLLMOutputModel


class QuestionExtractionLiteModel(StrictLLMOutputModel):
    """Sanitize + entity extraction only (no subject / used_glossary_names)."""

    sanitized_question: str = Field(
        ...,
        description=(
            "Concise, SQL-ready rewrite of the user's request. "
            "Preserve factual constraints and remove narrative fluff."
        ),
    )
    required_entity_name: list[str] = Field(
        ...,
        description=(
            "Concepts explicitly mentioned in the question that refer to "
            "database entities. Only extract what the question actually says. "
            "When words describe a single filterable item, keep them in one "
            "phrase instead of splitting. Keep brand names, product names, and "
            "descriptive named constants; omit numeric literals, dates, and "
            "number-based thresholds."
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
    "QuestionExtractionLiteModel",
    "QuestionExtractionModel",
    "EntityCoverageResponse",
]
