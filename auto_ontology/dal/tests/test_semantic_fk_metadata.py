# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema and migration contract for semantic-FK metadata."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from sqlalchemy.dialects.postgresql import JSONB

from auto_ontology.dal import schema as s
from auto_ontology.semantic.relationship_metadata import (
    CuratedRelationship,
    RelationshipKind,
    RelationshipMetadata,
    RelationshipReview,
    RelationshipSuggestion,
)


def test_schema_declares_non_null_jsonb_metadata_with_an_empty_default() -> None:
    column = s.column__semantic_fk.c.relationship_metadata

    assert isinstance(column.type, JSONB)
    assert column.nullable is False
    assert str(column.server_default.arg) == "'{}'::jsonb"


def test_migration_adds_and_removes_the_metadata_column(monkeypatch) -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "alembic"
        / "versions"
        / "91f6c7d4a2b8_add_semantic_fk_relationship_metadata.py"
    )
    spec = importlib.util.spec_from_file_location(
        "semantic_fk_metadata_migration", path
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    added: list[tuple[str, object]] = []
    dropped: list[tuple[str, str]] = []
    monkeypatch.setattr(
        migration.op,
        "add_column",
        lambda table, column: added.append((table, column)),
    )
    monkeypatch.setattr(
        migration.op,
        "drop_column",
        lambda table, column: dropped.append((table, column)),
    )

    migration.upgrade()
    migration.downgrade()

    table, column = added[0]
    assert table == "column__semantic_fk"
    assert column.name == "relationship_metadata"
    assert isinstance(column.type, JSONB)
    assert column.nullable is False
    assert str(column.server_default.arg) == "'{}'::jsonb"
    assert dropped == [("column__semantic_fk", "relationship_metadata")]


def test_controlled_kinds_and_reviewed_curated_precedence() -> None:
    assert {
        "incidence",
        "membership",
        "containment",
        "ownership",
        "location",
        "identity",
        "association",
        "reference",
        "other",
    }.issubset({kind.value for kind in RelationshipKind})

    suggestion = RelationshipSuggestion(kind=RelationshipKind.REFERENCE)
    unreviewed = RelationshipMetadata(
        suggestion=suggestion,
        curated=CuratedRelationship(
            kind=RelationshipKind.OWNERSHIP,
            review=RelationshipReview.UNREVIEWED,
        ),
    )
    certified = RelationshipMetadata(
        suggestion=suggestion,
        curated=CuratedRelationship(
            kind=RelationshipKind.OWNERSHIP,
            review=RelationshipReview.CERTIFIED,
        ),
    )

    assert unreviewed.effective_interpretation() is suggestion
    assert certified.effective_interpretation() is certified.curated
