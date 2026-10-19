# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed metadata carried by a semantic relationship.

The three sections deliberately have different owners. Deterministic catalog
inspection may refresh ``facts`` and automated inference may refresh
``suggestion``; neither needs to read or rewrite the human-owned ``curated``
section.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class RelationshipKind(StrEnum):
    """Controlled semantic meaning of a relationship."""

    INCIDENCE = "incidence"
    MEMBERSHIP = "membership"
    CONTAINMENT = "containment"
    OWNERSHIP = "ownership"
    LOCATION = "location"
    IDENTITY = "identity"
    ASSOCIATION = "association"
    REFERENCE = "reference"
    OTHER = "other"


class RelationshipProvenance(StrEnum):
    """Controlled origin of a relationship assertion."""

    CATALOG = "catalog"
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    IMPORT = "import"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class RelationshipReview(StrEnum):
    """Review state used to choose suggestion/curated precedence."""

    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    CERTIFIED = "certified"
    REJECTED = "rejected"


class RelationshipResolutionSource(StrEnum):
    """Evidence that selected the target of a semantic FK."""

    DECLARED = "declared"
    VDB_LLM = "vdb_llm"
    SQL_SAMPLE = "sql_sample"
    IMPORT = "import"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class RelationshipCardinality(StrEnum):
    """Deterministic source-to-target relationship cardinality."""

    ONE_TO_ONE = "one_to_one"
    MANY_TO_ONE = "many_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"
    UNKNOWN = "unknown"


class RelationshipOptionality(StrEnum):
    """Whether the source side may omit its target."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    UNKNOWN = "unknown"


class _StrictMetadataModel(BaseModel):
    # YAML and JSON carry enum members as strings, so strictness here means a
    # closed shape rather than rejecting their wire representation.
    model_config = ConfigDict(extra="forbid")


class RelationshipFacts(_StrictMetadataModel):
    """Refreshable facts derived from catalog structure."""

    creator: RelationshipProvenance | None = None
    resolution_source: RelationshipResolutionSource | None = None
    resolution_generator: str | None = Field(default=None, max_length=128)
    resolution_version: str | None = Field(default=None, max_length=64)
    physical_foreign_key: bool | None = None
    declared_foreign_key: bool | None = None
    source_nullable: bool | None = None
    source_declared_unique: bool | None = None
    source_sampled_unique: bool | None = None
    target_nullable: bool | None = None
    target_declared_unique: bool | None = None
    target_sampled_unique: bool | None = None
    cardinality: RelationshipCardinality | None = None
    optionality: RelationshipOptionality | None = None
    source_to_target_fan_out: bool | None = None
    target_to_source_fan_out: bool | None = None
    source_junction: bool | None = None
    target_junction: bool | None = None


_IntentPhrase = Annotated[str, Field(min_length=1, max_length=160)]


class _RelationshipInterpretation(_StrictMetadataModel):
    kind: RelationshipKind | None = None
    role_name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=1000)
    source_role: str | None = Field(default=None, max_length=128)
    target_role: str | None = Field(default=None, max_length=128)
    intent_phrases: list[_IntentPhrase] = Field(default_factory=list, max_length=12)
    generator: str | None = Field(default=None, max_length=128)
    generator_version: str | None = Field(default=None, max_length=64)
    provenance: RelationshipProvenance | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    review: RelationshipReview | None = None
    rationale: str | None = None


class RelationshipSuggestion(_RelationshipInterpretation):
    """Machine-owned interpretation, never a substitute for curation."""

    abstained: bool = False


class CuratedRelationship(_RelationshipInterpretation):
    """Human-owned interpretation and review decision."""


class RelationshipMetadataV1(_StrictMetadataModel):
    """Version-one semantic relationship metadata envelope."""

    version: Literal[1] = 1
    facts: RelationshipFacts | None = None
    suggestion: RelationshipSuggestion | None = None
    curated: CuratedRelationship | None = None

    def effective_interpretation(
        self,
    ) -> CuratedRelationship | RelationshipSuggestion | None:
        """Apply reviewed/certified curation precedence over suggestions."""
        if self.curated is None or self.curated.review is RelationshipReview.UNREVIEWED:
            return self.suggestion
        if self.curated.review is RelationshipReview.REJECTED:
            return None
        return self.curated


RelationshipMetadata = RelationshipMetadataV1


__all__ = [
    "CuratedRelationship",
    "RelationshipFacts",
    "RelationshipCardinality",
    "RelationshipKind",
    "RelationshipMetadata",
    "RelationshipMetadataV1",
    "RelationshipOptionality",
    "RelationshipProvenance",
    "RelationshipResolutionSource",
    "RelationshipReview",
    "RelationshipSuggestion",
]
