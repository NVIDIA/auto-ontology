# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label → table mapping for the label-generic write path.

The catalog writers were built against a property graph: they hand
:mod:`gsf.catalog.store.edges` a ``CatalogNode`` carrying a *label* and a bag of
properties, and expect the store to work out where it goes. Relationally there
is no such thing as "a node" — so this module is the translation, and every
generic function in ``store/pg/`` resolves through it.

Three things it has to encode, because none survives the move on its own:

**Which table a label lives in**, and which of a node's properties are real
columns. A property with no column is dropped rather than erroring: the Cypher
happily set arbitrary keys, and refusing them here would fail ingests that work
today.

**How identity is decided.** ``match_props`` is whatever the parser chose to
match on, and it differs per label — ``Table`` and ``Column`` match on a
pre-generated ``id``, ``Schema`` matches on ``(database_name, name)`` where
``database_name`` is not even a column on the row, and ``Database`` matches on
``name``. Each needs its own predicate.

**That ``CONTAINS`` is not an edge.** It is the child's parent foreign key, so
writing one is an ``UPDATE`` of the child rather than an ``INSERT`` of a row.
This is the single place the schema decision that collapsed ``reset.py``'s APOC
cascade into ``ON DELETE CASCADE`` has to be paid back, and getting it wrong
would silently orphan every table in the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Table, func

from gsf.catalog.constants import Edges, Labels
from gsf.dal.pg import schema as s


@dataclass(frozen=True)
class LabelSpec:
    """Where a node label lives, and how one is identified."""

    label: str
    table: Table
    #: Property names that map to real columns. Anything else is dropped.
    columns: frozenset[str]
    #: Column holding the parent's id, for the ``CONTAINS`` hierarchy.
    parent_column: str | None = None
    #: Label of the parent this row hangs off.
    parent_label: str | None = None
    #: Natural key used when the parser matches on something other than ``id``.
    #: ``("name",)`` means "unique by name"; ``("parent", "name")`` means
    #: "unique by name within its parent", which is how ``Schema``, ``Table``
    #: and ``Column`` are actually unique.
    natural_key: tuple[str, ...] = field(default_factory=tuple)
    #: ``ON CONFLICT`` target, when the unique index is on an *expression*
    #: rather than the columns themselves. ``sql_query`` is unique on
    #: ``md5(sql_full_query)``, because statement text can exceed the btree row
    #: limit — so the conflict target has to name the same expression or
    #: Postgres cannot match it to an index.
    conflict_expression: object | None = None


def _cols(table: Table) -> frozenset[str]:
    return frozenset(c.name for c in table.columns)


LABELS: dict[str, LabelSpec] = {
    Labels.DB: LabelSpec(
        label=Labels.DB,
        table=s.catalog_database,
        columns=_cols(s.catalog_database),
        natural_key=("name",),
    ),
    Labels.SCHEMA: LabelSpec(
        label=Labels.SCHEMA,
        table=s.catalog_schema,
        columns=_cols(s.catalog_schema),
        parent_column="database_id",
        parent_label=Labels.DB,
        natural_key=("parent", "name"),
    ),
    Labels.TABLE: LabelSpec(
        label=Labels.TABLE,
        table=s.catalog_table,
        columns=_cols(s.catalog_table),
        parent_column="schema_id",
        parent_label=Labels.SCHEMA,
        natural_key=("parent", "name"),
    ),
    Labels.COLUMN: LabelSpec(
        label=Labels.COLUMN,
        table=s.catalog_column,
        columns=_cols(s.catalog_column),
        parent_column="table_id",
        parent_label=Labels.TABLE,
        natural_key=("parent", "name"),
    ),
    Labels.SQL: LabelSpec(
        label=Labels.SQL,
        table=s.sql_query,
        columns=_cols(s.sql_query),
        natural_key=("sql_full_query",),
        conflict_expression=func.md5(s.sql_query.c.sql_full_query),
    ),
    Labels.CUSTOM_ANALYSIS: LabelSpec(
        label=Labels.CUSTOM_ANALYSIS,
        table=s.custom_analysis,
        columns=_cols(s.custom_analysis),
        natural_key=("name",),
    ),
}


@dataclass(frozen=True)
class EdgeSpec:
    """How one relationship type is stored.

    ``table is None`` means the relationship is a parent foreign key on the
    child row, not a row of its own — see the module docstring.
    """

    edge_label: str
    table: Table | None
    source_column: str | None = None
    target_column: str | None = None
    #: Edge properties that map to real columns on the edge table.
    property_columns: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_parent_link(self) -> bool:
        return self.table is None


EDGES: dict[tuple[str, str, str], EdgeSpec] = {
    # CONTAINS: the child's parent FK, at all three levels.
    (Edges.CONTAINS, Labels.DB, Labels.SCHEMA): EdgeSpec(Edges.CONTAINS, None),
    (Edges.CONTAINS, Labels.SCHEMA, Labels.TABLE): EdgeSpec(Edges.CONTAINS, None),
    (Edges.CONTAINS, Labels.TABLE, Labels.COLUMN): EdgeSpec(Edges.CONTAINS, None),
    (Edges.FOREIGN_KEY, Labels.COLUMN, Labels.COLUMN): EdgeSpec(
        Edges.FOREIGN_KEY,
        s.column_foreign_key,
        "source_column_id",
        "target_column_id",
    ),
    (Edges.JOIN, Labels.TABLE, Labels.TABLE): EdgeSpec(
        Edges.JOIN,
        s.table_join,
        "source_table_id",
        "target_table_id",
        frozenset({"join_columns"}),
    ),
    (Edges.SQL, Labels.SQL, Labels.TABLE): EdgeSpec(
        Edges.SQL, s.sql_query_table, "sql_query_id", "table_id"
    ),
    (Edges.HAS_SQL, Labels.CUSTOM_ANALYSIS, Labels.SQL): EdgeSpec(
        Edges.HAS_SQL, s.custom_analysis_sql, "analysis_id", "sql_query_id"
    ),
}


class UnknownLabel(KeyError):
    """A node label with nowhere to go.

    Raised rather than ignored: a silently dropped node is a missing catalog
    entry that only surfaces as an empty UI page much later.
    """


class UnknownEdge(KeyError):
    """A relationship shape the schema has no place for."""


def label_spec(label: str) -> LabelSpec:
    try:
        return LABELS[label]
    except KeyError:
        raise UnknownLabel(
            f"no table for node label {label!r}; known labels: {sorted(LABELS)}"
        ) from None


def edge_spec(edge_label: str, from_label: str, to_label: str) -> EdgeSpec:
    try:
        return EDGES[(edge_label, from_label, to_label)]
    except KeyError:
        raise UnknownEdge(
            f"no mapping for {from_label} -[{edge_label}]-> {to_label}; "
            f"known: {sorted(EDGES)}"
        ) from None


def projected(label: str, properties: dict) -> dict:
    """Keep only the properties that are real columns on *label*'s table.

    Dropping unknown keys rather than raising is deliberate. The Cypher accepted
    any property, and several are set opportunistically by callers that do not
    know or care what the schema holds; rejecting them would fail ingests that
    work today, for no gain.
    """
    spec = label_spec(label)
    return {k: v for k, v in properties.items() if k in spec.columns}
