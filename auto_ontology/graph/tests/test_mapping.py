# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The projection mapping agrees with the schema it reads.

These run without a database. They pin the two ways a mapping goes stale
quietly: a column renamed in ``schema.py`` (the sync would fail at runtime,
long after the change), and a new foreign key between projected tables that
nobody added as an edge (the sync would succeed and the graph would simply be
missing it).
"""

from __future__ import annotations

from auto_ontology.dal import schema as s
from auto_ontology.graph import mapping
from auto_ontology.graph.mapping import EDGES, EXCLUDED_COLUMNS, NODES

#: Foreign keys between projected tables that are not edges, and why.
NOT_EDGES = {
    # Provenance of a tag label (which rule applied it), not a relationship
    # between catalog objects.
    ("tag_target", "rule_id"),
    # Owning table of an attribute, already reachable as
    # Table -CONTAINS-> Column -HAS_ATTRIBUTE-> ColumnAttribute.
    ("column_attribute", "table_id"),
}


def test_every_projected_column_exists() -> None:
    for spec in NODES:
        assert "id" in spec.table.c, spec.label
        for name in spec.properties:
            assert name in spec.table.c, f"{spec.label}.{name}"
    for spec in EDGES:
        for name in (spec.source_column, spec.target_column, *spec.properties):
            assert name in spec.table.c, f"{spec.label} on {spec.table.name}.{name}"


def test_sensitive_columns_are_never_projected() -> None:
    projected = {(spec.table.name, name) for spec in NODES for name in spec.properties}
    projected |= {(spec.table.name, name) for spec in EDGES for name in spec.properties}
    assert not projected & EXCLUDED_COLUMNS


def test_edges_connect_projected_labels() -> None:
    labels = set(mapping.vertex_labels())
    for spec in EDGES:
        assert spec.source_label in labels, spec
        assert spec.target_label in labels, spec


def test_labels_are_unique() -> None:
    labels = mapping.vertex_labels()
    assert len(labels) == len(set(labels))
    assert not set(labels) & set(mapping.edge_labels())


def test_every_foreign_key_between_projected_tables_is_an_edge() -> None:
    projected_tables = {spec.table.name for spec in NODES}
    covered = {(spec.table.name, spec.source_column) for spec in EDGES}
    covered |= {(spec.table.name, spec.target_column) for spec in EDGES}

    missing = []
    for table in s.METADATA.tables.values():
        for fk in table.foreign_keys:
            if fk.column.table.name not in projected_tables:
                continue
            key = (table.name, fk.parent.name)
            if key not in covered and key not in NOT_EDGES:
                missing.append(key)
    # Association tables whose other end is not projected (statement tier,
    # user groups, owners) are out of scope by design; only links whose *both*
    # ends are projected must appear.
    missing = [
        (t, c)
        for t, c in missing
        if t in projected_tables
        or all(
            fk.column.table.name in projected_tables
            for fk in s.METADATA.tables[t].foreign_keys
        )
    ]
    assert not missing, f"foreign keys with no edge spec: {sorted(missing)}"
