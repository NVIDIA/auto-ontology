# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translation from the callers' graph vocabulary into tables and columns.

This is a boundary module, and the asymmetry is the point. On the way in it
speaks the callers' language — a *label* string and a bag of properties, because
that is what ``write.py`` and the parsers still hand over. On the way out it
speaks only tables, columns and foreign keys. Nothing behind this module needs
to know a graph was ever involved.

The graph words that remain are inputs being translated, not descriptions of how
anything is stored: ``label`` is the legacy name arriving from a caller, and it
is the *key* of the lookup, not the thing being described.

Three things it has to encode, because none survives the move on its own:

**Which table an incoming label maps to**, and which of the supplied properties
are real columns. A property with no column is dropped rather than erroring: the
Callers set properties opportunistically, and refusing them would fail ingests that
work today.

**How identity is decided**, which differs per entity. ``Table`` and ``Column``
arrive with a pre-generated ``id``; ``Schema`` arrives with
``(database_name, name)`` where ``database_name`` is not a column on the row at
all; ``Database`` arrives with ``name``. Each needs its own predicate, and the
keys are parent-scoped — ``public`` exists once per database, not once.

**Which relationships are association tables and which are foreign key
columns.** ``CONTAINS`` is the latter: it is the child's parent FK, so writing
one is an ``UPDATE`` of the child rather than an ``INSERT``. This is where the
``ON DELETE CASCADE`` chain that makes ``reset.py``'s delete a single statement
is set up — and getting it wrong would silently orphan every table in the
catalog.

Note on naming: "link" rather than "relation" for an association, because in
relational vocabulary a *relation* is a table — the opposite of what is meant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Table, func

from gsf.catalog.constants import Edges, Labels
from gsf.dal import schema as s
from gsf.semantic.constants import LABEL_SQL_ATTRIBUTE


@dataclass(frozen=True)
class EntitySpec:
    """Which table an incoming label maps to, and how a row is identified."""

    #: The legacy label callers send. The lookup key, not a description of
    #: how the row is stored.
    label: str
    table: Table
    #: Supplied property names that are real columns. Anything else is dropped.
    columns: frozenset[str]
    #: Foreign key column holding the parent's id, for the catalog hierarchy.
    parent_column: str | None = None
    #: Legacy label of the parent this row hangs off.
    parent_label: str | None = None
    #: Natural key used when the caller matches on something other than ``id``.
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


ENTITIES: dict[str, EntitySpec] = {
    Labels.DB: EntitySpec(
        label=Labels.DB,
        table=s.catalog_database,
        columns=_cols(s.catalog_database),
        natural_key=("name",),
    ),
    Labels.SCHEMA: EntitySpec(
        label=Labels.SCHEMA,
        table=s.catalog_schema,
        columns=_cols(s.catalog_schema),
        parent_column="database_id",
        parent_label=Labels.DB,
        natural_key=("parent", "name"),
    ),
    Labels.TABLE: EntitySpec(
        label=Labels.TABLE,
        table=s.catalog_table,
        columns=_cols(s.catalog_table),
        parent_column="schema_id",
        parent_label=Labels.SCHEMA,
        natural_key=("parent", "name"),
    ),
    Labels.COLUMN: EntitySpec(
        label=Labels.COLUMN,
        table=s.catalog_column,
        columns=_cols(s.catalog_column),
        parent_column="table_id",
        parent_label=Labels.TABLE,
        natural_key=("parent", "name"),
    ),
    Labels.SQL: EntitySpec(
        label=Labels.SQL,
        table=s.sql_query,
        columns=_cols(s.sql_query),
        natural_key=("sql_full_query",),
        conflict_expression=func.md5(s.sql_query.c.sql_full_query),
    ),
    Labels.CUSTOM_ANALYSIS: EntitySpec(
        label=Labels.CUSTOM_ANALYSIS,
        table=s.custom_analysis,
        columns=_cols(s.custom_analysis),
        natural_key=("name",),
    ),
    # The semantic write path routes through here too: creating a SqlAttribute
    # persists its statement with `add_query`, exactly as an ingest does, so
    # the attribute lands with the same table and column links -- which is what
    # makes it appear in the exploration graph and pass the zone checks.
    LABEL_SQL_ATTRIBUTE: EntitySpec(
        label=LABEL_SQL_ATTRIBUTE,
        table=s.sql_attribute,
        columns=_cols(s.sql_attribute),
        natural_key=("name",),
    ),
}


@dataclass(frozen=True)
class LinkSpec:
    """How one relationship is stored: association table, or foreign key.

    ``table is None`` means it is a foreign key column on the child row rather
    than an association table — see the module docstring.
    """

    #: The legacy relationship name callers send.
    edge_label: str
    table: Table | None
    source_column: str | None = None
    target_column: str | None = None
    #: Supplied properties that are real columns on the association table.
    property_columns: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_parent_link(self) -> bool:
        return self.table is None


LINKS: dict[tuple[str, str, str], LinkSpec] = {
    # CONTAINS is a foreign key column on the child, at all three levels --
    # not an association table.
    (Edges.CONTAINS, Labels.DB, Labels.SCHEMA): LinkSpec(Edges.CONTAINS, None),
    (Edges.CONTAINS, Labels.SCHEMA, Labels.TABLE): LinkSpec(Edges.CONTAINS, None),
    (Edges.CONTAINS, Labels.TABLE, Labels.COLUMN): LinkSpec(Edges.CONTAINS, None),
    (Edges.FOREIGN_KEY, Labels.COLUMN, Labels.COLUMN): LinkSpec(
        Edges.FOREIGN_KEY,
        s.column_foreign_key,
        "source_column_id",
        "target_column_id",
    ),
    (Edges.JOIN, Labels.TABLE, Labels.TABLE): LinkSpec(
        Edges.JOIN,
        s.table_join,
        "source_table_id",
        "target_table_id",
        frozenset({"join_columns"}),
    ),
    # Column-level JOIN/UNION, written by query ingestion. `refs` accumulates
    # rather than overwrites -- see `_upsert_edge_row`.
    (Edges.JOIN, Labels.COLUMN, Labels.COLUMN): LinkSpec(
        Edges.JOIN,
        s.column_join,
        "source_column_id",
        "target_column_id",
        frozenset({"refs"}),
    ),
    (Edges.UNION, Labels.COLUMN, Labels.COLUMN): LinkSpec(
        Edges.UNION,
        s.column_union,
        "source_column_id",
        "target_column_id",
        frozenset({"refs"}),
    ),
    (Edges.SQL, Labels.SQL, Labels.TABLE): LinkSpec(
        Edges.SQL, s.sql_query_table, "sql_query_id", "table_id"
    ),
    # Same relationship type, different target: a statement points at both the
    # tables it reads and the leaf columns it selects.
    (Edges.SQL, Labels.SQL, Labels.COLUMN): LinkSpec(
        Edges.SQL, s.sql_query_column, "sql_query_id", "column_id"
    ),
    (Edges.HAS_SQL, Labels.CUSTOM_ANALYSIS, Labels.SQL): LinkSpec(
        Edges.HAS_SQL, s.custom_analysis_sql, "analysis_id", "sql_query_id"
    ),
    (Edges.HAS_SQL, LABEL_SQL_ATTRIBUTE, Labels.SQL): LinkSpec(
        Edges.HAS_SQL, s.sql_attribute_sql, "attribute_id", "sql_query_id"
    ),
}


class UnknownLabel(KeyError):
    """An incoming label with no table to map to.

    Raised rather than ignored: a silently dropped write is a missing catalog
    entry that only surfaces as an empty page much later.
    """


class UnknownLink(KeyError):
    """A relationship shape the schema has nowhere to put."""


def entity_spec(label: str | list[str]) -> EntitySpec:
    """Resolve a label to its table.

    Accepts a list as well as a string: a node in a property graph can carry
    several labels, so the write primitive takes a list and callers such
    as ``merge_schema_nodes`` pass one through unchanged. The first is used,
    matching ``labels(n)[0]`` everywhere else in the codebase — GSF never puts
    more than one label on a catalog node.
    """
    if isinstance(label, (list, tuple)):
        if not label:
            raise UnknownLabel("empty label list")
        label = label[0]
    try:
        return ENTITIES[label]
    except KeyError:
        raise UnknownLabel(
            f"no table for label {label!r}; known: {sorted(ENTITIES)}"
        ) from None


def link_spec(edge_label: str, from_label: str, to_label: str) -> LinkSpec:
    try:
        return LINKS[(edge_label, from_label, to_label)]
    except KeyError:
        raise UnknownLink(
            f"no mapping for {from_label} -[{edge_label}]-> {to_label}; "
            f"known: {sorted(LINKS)}"
        ) from None


def projected(label: str, properties: dict) -> dict:
    """Keep only the supplied properties that are real columns.

    Dropping unknown keys rather than raising is deliberate: several are set
    opportunistically by callers that do not know or care what the schema holds,
    and rejecting them would fail ingests that work today, for no gain.
    """
    spec = entity_spec(label)
    return {k: v for k, v in properties.items() if k in spec.columns}
