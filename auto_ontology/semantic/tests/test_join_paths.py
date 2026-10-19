# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for typed semantic join path results."""

from __future__ import annotations

from auto_ontology.dal.attributes import _compose_path_meaning
from auto_ontology.semantic.join_paths import (
    InterpretationSource,
    JoinPathPattern,
    SemanticJoinHop,
    SemanticJoinPathCandidate,
    TraversalDirection,
)
from auto_ontology.semantic.relationship_metadata import RelationshipKind


def _hop(
    source: str,
    target: str,
    *,
    direction: TraversalDirection,
    relationship: str,
    junction: bool = False,
    kind: RelationshipKind | None = None,
    fan_out: bool | None = False,
) -> SemanticJoinHop:
    source_column = f"{source}-column"
    target_column = f"{target}-column"
    return SemanticJoinHop(
        source_table=source,
        source_column="id",
        target_table=target,
        target_column="id",
        source_table_id=source,
        source_column_id=source_column,
        target_table_id=target,
        target_column_id=target_column,
        relationship_source_column_id=f"{relationship}-source",
        relationship_target_attribute_id=f"{relationship}-attribute",
        relationship_id=relationship,
        direction=direction,
        interpretation_source=InterpretationSource.CURATED,
        relationship_kind=kind,
        fan_out=fan_out,
        source_junction=junction,
    )


def test_fingerprints_are_stable_and_structural_projection_is_exact() -> None:
    hop = _hop(
        "clients",
        "districts",
        direction=TraversalDirection.FORWARD,
        relationship="client-district",
    )
    changed_prose = hop.model_copy(update={"description": "New mutable prose."})
    candidate = SemanticJoinPathCandidate(
        hops=(hop,),
        meaning=_compose_path_meaning((hop,)),
        is_shortest=True,
    )

    assert hop.fingerprint == changed_prose.fingerprint
    assert (
        candidate.fingerprint
        == SemanticJoinPathCandidate(
            hops=(changed_prose,),
            meaning=_compose_path_meaning((changed_prose,)),
            is_shortest=True,
        ).fingerprint
    )
    assert set(hop.join_dict()) == {
        "source_database",
        "source_schema",
        "source_table",
        "source_column",
        "target_database",
        "target_schema",
        "target_table",
        "target_column",
    }


def test_chemistry_incidence_differs_from_shared_molecule_membership() -> None:
    incidence = (
        _hop(
            "chemicals",
            "chemical_molecule",
            direction=TraversalDirection.REVERSE,
            relationship="chemical-incidence",
            junction=True,
            kind=RelationshipKind.INCIDENCE,
        ),
        _hop(
            "chemical_molecule",
            "molecules",
            direction=TraversalDirection.FORWARD,
            relationship="molecule-incidence",
            junction=True,
            kind=RelationshipKind.ASSOCIATION,
        ),
    )
    membership = (
        _hop(
            "chemical_samples",
            "molecules",
            direction=TraversalDirection.FORWARD,
            relationship="sample-molecule",
            kind=RelationshipKind.MEMBERSHIP,
        ),
        _hop(
            "molecules",
            "assay_samples",
            direction=TraversalDirection.REVERSE,
            relationship="assay-molecule",
            kind=RelationshipKind.MEMBERSHIP,
        ),
    )

    incidence_meaning = _compose_path_meaning(incidence)
    membership_meaning = _compose_path_meaning(membership)
    assert incidence_meaning.pattern is JoinPathPattern.SHARED_CHILD_JUNCTION
    assert "directly associated" in incidence_meaning.description
    assert membership_meaning.pattern is JoinPathPattern.SHARED_PARENT
    assert "membership" in membership_meaning.description
    assert incidence_meaning.contrast != membership_meaning.contrast
