# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which Postgres rows become which graph vertices and edges.

The store is relational; this module is the one place that says how its rows
read as a property graph. The vocabulary is not new: every label and edge type
below is the name the codebase already uses for the same thing
(:mod:`auto_ontology.catalog.constants`, :mod:`auto_ontology.semantic.constants`,
:mod:`auto_ontology.server.zones.constants`), so a Cypher query reads like the
comments in :mod:`auto_ontology.dal.schema` and the exploration API's
``relationship`` values.

Every edge is one of two shapes, and :class:`EdgeSpec` covers both with the same
four fields:

* an **association table** (``column__foreign_key``): source and target are its
  two id columns;
* a **parent foreign key** (``catalog_table.schema_id``): the table is the
  child, the source column is the parent FK and the target column is ``id``.

Polymorphic tables (``tag_target``, ``zone_target``) carry one nullable FK per
target kind, so each kind is its own :class:`EdgeSpec` and rows whose target
column is NULL are skipped. Their direction runs from the tag or zone to the
object it labels or scopes.

**What is deliberately not projected.** ``catalog_database.connection`` holds
connection settings and ``catalog_column.sample_values`` holds raw values from
the customer's data; neither is metadata an agent needs to traverse, and a
second copy is a second place to protect. Statement-tier rows (``sql_query`` and
its links) are usage logs that grow with every ingested query and are left out
for the same reason the MCP surface leaves them out.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Table

from auto_ontology.catalog.constants import Edges, Labels
from auto_ontology.dal import schema as s
from auto_ontology.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_REPRESENTS,
    REL_SEMANTIC_FK,
)
from auto_ontology.server.zones.constants import LABEL_ZONE, REL_ZONE_OF

LABEL_TAG = "Tag"
REL_TAGGED = "TAGGED"


@dataclass(frozen=True)
class NodeSpec:
    """One vertex label and the table its rows come from."""

    label: str
    table: Table
    #: Columns copied onto the vertex as properties. ``id`` is always copied
    #: and is the key edges are matched on.
    properties: tuple[str, ...]


@dataclass(frozen=True)
class EdgeSpec:
    """One edge type between two labels, and the table that stores it."""

    label: str
    source_label: str
    target_label: str
    table: Table
    #: Column holding the source vertex's ``id``.
    source_column: str
    #: Column holding the target vertex's ``id``.
    target_column: str
    #: Columns copied onto the edge as properties.
    properties: tuple[str, ...] = ()


NODES: tuple[NodeSpec, ...] = (
    NodeSpec(Labels.DB, s.catalog_database, ("name", "dialect", "description")),
    NodeSpec(Labels.SCHEMA, s.catalog_schema, ("name", "description")),
    NodeSpec(
        Labels.TABLE,
        s.catalog_table,
        ("name", "description", "table_type", "is_junction_table", "pk"),
    ),
    NodeSpec(
        Labels.COLUMN,
        s.catalog_column,
        (
            "name",
            "description",
            "data_type",
            "format",
            "is_unique",
            "is_nullable",
            "ordinal_position",
        ),
    ),
    NodeSpec(LABEL_TERM, s.term, ("name", "description", "source", "synonyms")),
    NodeSpec(
        LABEL_COLUMN_ATTRIBUTE,
        s.column_attribute,
        ("name", "description", "datatype", "source", "certified"),
    ),
    NodeSpec(
        LABEL_SQL_ATTRIBUTE,
        s.sql_attribute,
        ("name", "description", "expression", "source", "certified"),
    ),
    NodeSpec(LABEL_TAG, s.tag, ("name",)),
    NodeSpec(LABEL_ZONE, s.zone, ("name", "description", "enabled")),
)


def _parent(child: Table, parent_label: str, child_label: str, fk: str) -> EdgeSpec:
    return EdgeSpec(Edges.CONTAINS, parent_label, child_label, child, fk, "id")


EDGES: tuple[EdgeSpec, ...] = (
    # Catalog hierarchy: parent FK on the child row.
    _parent(s.catalog_schema, Labels.DB, Labels.SCHEMA, "database_id"),
    _parent(s.catalog_table, Labels.SCHEMA, Labels.TABLE, "schema_id"),
    _parent(s.catalog_column, Labels.TABLE, Labels.COLUMN, "table_id"),
    # Physical relationships between columns and tables.
    EdgeSpec(
        Edges.FOREIGN_KEY,
        Labels.COLUMN,
        Labels.COLUMN,
        s.column__foreign_key,
        "source_column_id",
        "target_column_id",
    ),
    EdgeSpec(
        Edges.JOIN,
        Labels.TABLE,
        Labels.TABLE,
        s.table__join,
        "source_table_id",
        "target_table_id",
        ("join_columns",),
    ),
    EdgeSpec(
        Edges.JOIN,
        Labels.COLUMN,
        Labels.COLUMN,
        s.column__join,
        "source_column_id",
        "target_column_id",
        ("refs",),
    ),
    EdgeSpec(
        Edges.UNION,
        Labels.COLUMN,
        Labels.COLUMN,
        s.column__union,
        "source_column_id",
        "target_column_id",
        ("refs",),
    ),
    # Semantic layer.
    EdgeSpec(
        REL_REPRESENTS, Labels.TABLE, LABEL_TERM, s.table__term, "table_id", "term_id"
    ),
    EdgeSpec(
        REL_HAS_ATTRIBUTE,
        Labels.COLUMN,
        LABEL_COLUMN_ATTRIBUTE,
        s.column__has_attribute,
        "column_id",
        "attribute_id",
    ),
    EdgeSpec(
        REL_SEMANTIC_FK,
        Labels.COLUMN,
        LABEL_COLUMN_ATTRIBUTE,
        s.column__semantic_fk,
        "column_id",
        "attribute_id",
        ("use_in_join_paths",),
    ),
    EdgeSpec(
        REL_PROPERTY_OF,
        LABEL_COLUMN_ATTRIBUTE,
        LABEL_TERM,
        s.column_attribute__term,
        "attribute_id",
        "term_id",
    ),
    EdgeSpec(
        REL_PROPERTY_OF,
        LABEL_SQL_ATTRIBUTE,
        LABEL_TERM,
        s.sql_attribute__term,
        "attribute_id",
        "term_id",
    ),
    # Governance: tags and zones, one spec per target kind.
    *(
        EdgeSpec(REL_TAGGED, LABEL_TAG, target, s.tag_target, "tag_id", column)
        for target, column in (
            (LABEL_TERM, "term_id"),
            (Labels.TABLE, "table_id"),
            (Labels.COLUMN, "column_id"),
            (LABEL_COLUMN_ATTRIBUTE, "column_attribute_id"),
            (LABEL_SQL_ATTRIBUTE, "sql_attribute_id"),
        )
    ),
    *(
        EdgeSpec(REL_ZONE_OF, LABEL_ZONE, target, s.zone_target, "zone_id", column)
        for target, column in (
            (Labels.DB, "database_id"),
            (Labels.SCHEMA, "schema_id"),
            (Labels.TABLE, "table_id"),
        )
    ),
)

#: Columns that must never reach the graph, checked by the tests so a later
#: edit to :data:`NODES` cannot copy them by accident.
EXCLUDED_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {
        ("catalog_database", "connection"),
        ("catalog_column", "sample_values"),
    }
)


def vertex_labels() -> tuple[str, ...]:
    return tuple(spec.label for spec in NODES)


def edge_labels() -> tuple[str, ...]:
    """Distinct edge types, in first-seen order."""
    return tuple(dict.fromkeys(spec.label for spec in EDGES))
