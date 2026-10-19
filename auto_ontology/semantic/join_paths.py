# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, deterministic semantic join-path results."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from auto_ontology.semantic.relationship_metadata import (
    RelationshipCardinality,
    RelationshipKind,
    RelationshipMetadata,
)


class TraversalDirection(StrEnum):
    """Direction relative to the stored FK-column -> owner-attribute edge."""

    FORWARD = "forward"
    REVERSE = "reverse"


class InterpretationSource(StrEnum):
    """The metadata section supplying a hop's effective interpretation."""

    CURATED = "curated"
    SUGGESTION = "suggestion"
    UNKNOWN = "unknown"


class RelationshipTrust(StrEnum):
    """Trust level of the interpretation selected for a hop."""

    CERTIFIED = "certified"
    CURATED = "curated"
    SUGGESTED = "suggested"
    UNKNOWN = "unknown"


class JoinPathPattern(StrEnum):
    """Structural meaning of a complete path."""

    DIRECT = "direct"
    SHARED_CHILD_JUNCTION = "shared_child_junction"
    SHARED_PARENT = "shared_parent"
    CHAIN = "chain"
    UNKNOWN = "unknown"


class DuplicationRisk(StrEnum):
    """Risk that the path expands endpoint rows."""

    NONE = "none"
    POSSIBLE = "possible"
    HIGH = "high"
    UNKNOWN = "unknown"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _fingerprint(prefix: str, value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


class SemanticJoinHop(_FrozenModel):
    """One oriented table crossing plus the semantic relationship behind it."""

    source_database: str = ""
    source_schema: str = ""
    source_table: str = ""
    source_column: str = ""
    target_database: str = ""
    target_schema: str = ""
    target_table: str = ""
    target_column: str = ""
    source_table_id: str
    source_column_id: str
    target_table_id: str
    target_column_id: str
    relationship_source_column_id: str
    relationship_target_attribute_id: str
    relationship_id: str
    direction: TraversalDirection
    raw_relationship_metadata: dict[str, Any] = Field(default_factory=dict)
    relationship_metadata: RelationshipMetadata | None = None
    interpretation_source: InterpretationSource = InterpretationSource.UNKNOWN
    trust: RelationshipTrust = RelationshipTrust.UNKNOWN
    relationship_kind: RelationshipKind | None = None
    source_role: str | None = None
    target_role: str | None = None
    cardinality: RelationshipCardinality = RelationshipCardinality.UNKNOWN
    fan_out: bool | None = None
    source_junction: bool | None = None
    target_junction: bool | None = None
    description: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        """Stable structural identity, unaffected by descriptions or row order."""
        return _fingerprint(
            "sjh",
            {
                "relationship": [
                    self.relationship_source_column_id,
                    self.relationship_target_attribute_id,
                ],
                "direction": self.direction,
                "source_column": self.source_column_id,
                "target_column": self.target_column_id,
            },
        )

    def join_dict(self) -> dict[str, str]:
        """Return the structural hop fields used by SQL prompt formatting."""
        return {
            "source_database": self.source_database,
            "source_schema": self.source_schema,
            "source_table": self.source_table,
            "source_column": self.source_column,
            "target_database": self.target_database,
            "target_schema": self.target_schema,
            "target_table": self.target_table,
            "target_column": self.target_column,
        }


class JoinPathMeaning(_FrozenModel):
    """Composed interpretation and row-grain consequences of a path."""

    pattern: JoinPathPattern = JoinPathPattern.UNKNOWN
    relationship_kinds: tuple[RelationshipKind, ...] = ()
    relationship_trust: tuple[RelationshipTrust, ...] = ()
    source_endpoint_meaning: str = "unknown"
    target_endpoint_meaning: str = "unknown"
    qualification_scope: str = "unknown"
    resulting_grain: str = "unknown"
    fan_out: bool | None = None
    duplication_risk: DuplicationRisk = DuplicationRisk.UNKNOWN
    description: str = ""
    contrast: str = ""


class SemanticJoinPathCandidate(_FrozenModel):
    """A simple join path and its deterministic semantic composition."""

    hops: tuple[SemanticJoinHop, ...]
    meaning: JoinPathMeaning
    is_shortest: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        """Stable identity across query order and mutable semantic prose."""
        return _fingerprint("sjp", [hop.fingerprint for hop in self.hops])

    @property
    def hop_count(self) -> int:
        return len(self.hops)

    def join_path(self) -> list[dict[str, str]]:
        """Return this candidate in the structural join-path shape."""
        return [hop.join_dict() for hop in self.hops]


# Concise aliases for callers that do not need the semantic namespace prefix.
JoinHopCandidate = SemanticJoinHop
JoinPathCandidate = SemanticJoinPathCandidate


__all__ = [
    "DuplicationRisk",
    "InterpretationSource",
    "JoinHopCandidate",
    "JoinPathCandidate",
    "JoinPathMeaning",
    "JoinPathPattern",
    "RelationshipTrust",
    "SemanticJoinHop",
    "SemanticJoinPathCandidate",
    "TraversalDirection",
]
