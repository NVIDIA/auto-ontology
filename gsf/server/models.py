# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Item models carried inside the response envelopes.

``gsf/server/responses.py`` owns the envelopes (``{"data": ...}``); this
module owns what goes *inside* them, so the generated OpenAPI spec names the
fields a caller actually receives instead of an opaque object.

The shapes are derived from the ``RETURN`` clauses of the Cypher queries in
``gsf/dal/`` plus whatever the DAL/service layer adds in Python afterwards.
Cypher names fields but does not type them, so the rule here is: a field is
required only when the row is anchored on it (an ``id`` matched by the query,
or a value the Python layer writes unconditionally). Everything else is
optional and nullable — a missing Neo4j property comes back as ``null``, and a
required-but-null field would turn a documentation change into a 500.

Item shapes are shared across routers on purpose: a Zone chip is rendered by
the zones, terms, sql-attributes and exploration endpoints alike, so it is
modelled once here rather than four times in four router packages.

Every model allows extra keys — see :class:`ApiModel`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ApiModel",
    "ColumnAttribute",
    "ColumnRef",
    "ColumnSummary",
    "CustomAnalysis",
    "DataExplorationGraph",
    "DataGraphNode",
    "DatabaseSummary",
    "EntityCoverageCandidate",
    "EntityCoverageResult",
    "ExplorationEdge",
    "ExplorationRelatedNode",
    "ExplorationRelatedNodes",
    "ForeignKeyRef",
    "GraphLink",
    "IdRef",
    "NodeUpdateResult",
    "PqlAnalysis",
    "PublicConnection",
    "SchemaSummary",
    "SemanticExplorationGraph",
    "SemanticGraphNode",
    "SqlAttribute",
    "SqlExpressionValidationResult",
    "SqlValidationResult",
    "SsoFederationState",
    "TableColumns",
    "TableExplorationDetails",
    "TableSqlQuery",
    "TableSummary",
    "Term",
    "TermCountEntry",
    "TermDetail",
    "TermListItem",
    "TermSummary",
    "TermTable",
    "Zone",
    "ZoneChip",
    "ZoneItem",
]


class ApiModel(BaseModel):
    """Base for every response model.

    ``extra="allow"`` is mandatory: FastAPI *drops* any field a handler
    returns that the declared ``response_model`` does not know about, so a
    strict model would silently truncate live responses instead of merely
    documenting them.
    """

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


class IdRef(ApiModel):
    """``{"id": ...}`` — echoed by the delete endpoints to confirm the target."""

    id: str


class NodeUpdateResult(ApiModel):
    """Echo of a catalog node patch: ``{id, ...the properties you sent}``.

    ``update_node_properties`` builds the body as ``{"id": ..., **{k: props[k]
    for k in patch}}``, so only the keys the request actually patched come
    back — every one of them is optional here for that reason, and each may be
    null when the node has no value for it. The route as a whole answers null
    when the patch was empty or the node does not exist.
    """

    id: str
    description: str | None = None
    sample_values: list[str] | None = None
    description_certified: bool | None = None


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------


class ZoneChip(ApiModel):
    """A zone as rendered next to something else it grants access to.

    ``enabled`` is false for a zone whose ``Zone`` label was swapped for
    ``disableZone``; those are returned to admins only.
    """

    id: str
    name: str | None = None
    color: str | None = None
    enabled: bool | None = None


class ZoneItem(ApiModel):
    """A catalog node a zone grants access to (a database, schema or table)."""

    id: str
    name: str | None = None
    label: str | None = None


class Zone(ApiModel):
    """A zone with its catalog data items.

    ``items`` is absent on the list endpoint, a list of ``ZoneItem`` on the
    detail/update endpoints, and a plain list of the ids that were linked on
    create.
    """

    id: str
    name: str | None = None
    description: str | None = None
    color: str | None = None
    label: str | None = None
    enabled: bool | None = None
    items: list[ZoneItem | str] | None = None


# ---------------------------------------------------------------------------
# Catalog (databases / schemas / tables / columns)
# ---------------------------------------------------------------------------


class SchemaSummary(ApiModel):
    """One schema of a database, with how many tables it holds."""

    id: str
    schema_name: str | None = None
    description: str | None = None
    tables_count: int | None = None


class DatabaseSummary(ApiModel):
    """One catalog database. ``schemas`` is always empty — the tree is lazy."""

    id: str
    name: str | None = None
    description: str | None = None
    num_of_schemas: int | None = None
    schemas: list[SchemaSummary] = Field(default_factory=list)


class TableSummary(ApiModel):
    """One table of a schema, with its column / SQL / Term counts."""

    id: str
    name: str | None = None
    table_type: str | None = None
    database_name: str | None = None
    schema_name: str | None = None
    description: str | None = None
    description_certified: bool | None = None
    columns_count: int | None = None
    sql_count: int | None = None
    terms_count: int | None = None


class ColumnSummary(ApiModel):
    """One column of a table, ordered by ``ordinal_position``."""

    id: str
    ordinal_position: int | None = None
    column_name: str | None = None
    data_type: str | None = None
    description: str | None = None
    description_certified: bool | None = None
    sample_values: list[str] | None = None


class TableColumns(ApiModel):
    """A table with one page of its columns.

    Every field but ``columns`` is absent from the empty fallback the route
    returns when the table (or its schema/database path) is missing.
    """

    table_name: str | None = None
    table_type: str | None = None
    schema_name: str | None = None
    database_name: str | None = None
    columns: list[ColumnSummary] = Field(default_factory=list)
    columns_count: int | None = None


# ---------------------------------------------------------------------------
# Terms and attributes
# ---------------------------------------------------------------------------


class TermSummary(ApiModel):
    """The three fields every Term projection carries."""

    id: str
    name: str | None = None
    description: str | None = None


class Term(TermSummary):
    """A Term with its certification rollup and resolved zone chips."""

    synonyms: list[str] | None = None
    name_certified: bool | None = None
    description_certified: bool | None = None
    certification: str | None = None
    zones: list[ZoneChip] | None = None


class TermTable(ApiModel):
    """A table that REPRESENTS a Term, with its catalog path ids."""

    id: str
    name: str | None = None
    schema_id: str | None = None
    db_id: str | None = None


class TermDetail(Term):
    """One Term with the tables that represent it and its related terms."""

    table_count: int | None = None
    tables: list[TermTable] = Field(default_factory=list)
    related_terms: list[TermSummary] = Field(default_factory=list)


class TermListItem(Term):
    """A Term as it appears in the paged ``/terms`` list."""

    schema_names: list[str] | None = None


class TermCountEntry(ApiModel):
    """``{term_id, count}`` — one per-term breakdown row. Zeroes are omitted."""

    term_id: str
    count: int


class ColumnRef(ApiModel):
    """A column with the catalog path ids needed to navigate to it."""

    id: str
    column_name: str | None = None
    table_id: str | None = None
    table_name: str | None = None
    schema_id: str | None = None
    db_id: str | None = None


class ColumnAttribute(ApiModel):
    """A ColumnAttribute of a Term.

    ``primary_column`` is the HAS_ATTRIBUTE owner and ``referenced_columns``
    the SEMANTIC_FK sources; both are absent from the patch response, which
    echoes only the fields it wrote.
    """

    id: str
    name: str | None = None
    description: str | None = None
    term_name: str | None = None
    source_column: str | None = None
    datatype: str | None = None
    table_id: str | None = None
    sample_values: list[str] | None = None
    certified: bool | None = None
    zones: list[ZoneChip] | None = None
    primary_column: ColumnRef | None = None
    referenced_columns: list[ColumnRef] | None = None


class SqlAttribute(ApiModel):
    """A SqlAttribute with its owning Term and the SQL text behind it.

    ``zones`` is resolved only by the single-attribute read; ``sql_id`` and
    ``database_name`` only by the create/update writes.
    """

    id: str
    name: str | None = None
    description: str | None = None
    description_suggestion: str | None = None
    expression: str | None = None
    source: str | None = None
    sql: str | None = None
    sql_id: str | None = None
    certified: bool | None = None
    term_id: str | None = None
    term_name: str | None = None
    database_name: str | None = None
    zones: list[ZoneChip] | None = None


class SqlExpressionValidationResult(ApiModel):
    """``/sql-attributes/validate`` — a parse failure is a 422, not ``valid: false``."""

    valid: bool
    expression: str


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


class CustomAnalysis(ApiModel):
    """A verified SQL example, joined with the text of its ``Sql`` node."""

    id: str
    name: str | None = None
    description: str | None = None
    sql: str | None = None


class SqlValidationResult(ApiModel):
    """``/custom-analyses/validate`` — a parse failure is a 422, not ``valid: false``."""

    valid: bool
    sql: str


class PqlAnalysis(ApiModel):
    """A verified PQL few-shot. The PQL text is a property, not a child node."""

    id: str
    name: str | None = None
    description: str | None = None
    pql: str | None = None


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------


class ForeignKeyRef(ApiModel):
    """One foreign-key column pair behind an exploration edge.

    ``source_column`` always names a column on the edge's ``source``. The
    sample values come straight off the Column node, unparsed — profiling
    stores them as a JSON string, a catalog PATCH stores a list.
    """

    source_column: str | None = None
    target_column: str | None = None
    source_sample_values: list[str] | str | None = None
    target_sample_values: list[str] | str | None = None


class ExplorationEdge(ApiModel):
    """Two tables connected by a shared SQL query and/or a foreign key.

    ``queries`` is empty for an edge that exists only because of a foreign
    key; ``foreign_keys`` is empty for a SQL-only edge.
    """

    source: str
    target: str
    queries: list[str] = Field(default_factory=list)
    via_foreign_key: bool = False
    foreign_keys: list[ForeignKeyRef] = Field(default_factory=list)


class DataGraphNode(ApiModel):
    """A table on the data-layer exploration graph.

    ``relationship_count`` is the table's degree over the whole accessible
    edge set, so it does not shrink when ``limit`` truncates the payload.
    """

    id: str
    name: str | None = None
    table_type: str | None = None
    database_id: str | None = None
    database_name: str | None = None
    schema_id: str | None = None
    schema_name: str | None = None
    description: str | None = None
    columns_count: int | None = None
    sql_count: int | None = None
    terms_count: int | None = None
    zones: list[ZoneChip] = Field(default_factory=list)
    relationship_count: int | None = None


class DataExplorationGraph(ApiModel):
    """``{nodes, links}`` for the data layer, in one response."""

    nodes: list[DataGraphNode] = Field(default_factory=list)
    links: list[ExplorationEdge] = Field(default_factory=list)


class GraphLink(ApiModel):
    """An undirected term↔term link (two terms sharing at least one table)."""

    source: str
    target: str


class SemanticGraphNode(ApiModel):
    """A term on the semantic-layer exploration graph."""

    id: str
    name: str | None = None
    description: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    zones: list[ZoneChip] = Field(default_factory=list)
    relationship_count: int | None = None
    column_attributes_count: int | None = None
    sql_attributes_count: int | None = None


class SemanticExplorationGraph(ApiModel):
    """``{nodes, links}`` for the semantic layer, in one response."""

    nodes: list[SemanticGraphNode] = Field(default_factory=list)
    links: list[GraphLink] = Field(default_factory=list)


class TableSqlQuery(ApiModel):
    """A stored SQL query that references the table.

    ``sql`` is nullable: unlike the other Sql reads in ``gsf/dal/exploration``
    this query has no ``_NON_EMPTY_SQL`` filter, so a Sql node with a null or
    blank ``sql_full_query`` reaches the response. Declaring it required would
    turn that row into a 500 for the whole table-detail modal.
    """

    id: str
    sql: str | None = None


class TableExplorationDetails(ApiModel):
    """The table-detail modal: its SQL queries and one page of its Terms."""

    queries: list[TableSqlQuery] = Field(default_factory=list)
    terms: list[TermSummary] = Field(default_factory=list)
    terms_total: int | None = None


class ExplorationRelatedNode(ApiModel):
    """One related node. The catalog path fields are data-layer only."""

    id: str
    name: str | None = None
    table_type: str | None = None
    database_id: str | None = None
    schema_id: str | None = None
    relationship_count: int | None = None


class ExplorationRelatedNodes(ApiModel):
    """One page of related nodes; ``total`` counts every related node."""

    nodes: list[ExplorationRelatedNode] = Field(default_factory=list)
    total: int | None = None


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


class PublicConnection(ApiModel):
    """A UI-managed connection with its credentials stripped.

    ``connection`` is absent from the delete response, which echoes only the
    database name it tore down; the list and create reads carry the full
    credential-free config.
    """

    database_name: str
    connection: dict[str, Any] | None = None


class SsoFederationState(ApiModel):
    """The one flag ``PATCH /connections/{database_name}/sso-federation`` sets.

    The route deliberately echoes just the flag rather than the connection, so
    a caller never has to re-send credentials to toggle it.
    """

    database_name: str
    sso_federation: bool


# ---------------------------------------------------------------------------
# Entity coverage
# ---------------------------------------------------------------------------


class EntityCoverageCandidate(ApiModel):
    """One retrieved semantic candidate, ranked by vector distance."""

    label: str | None = None
    attribute: str | None = None
    term: str | None = None
    id: str | None = None


class EntityCoverageResult(ApiModel):
    """A 0–1 coverage grade with the candidates it was graded from.

    ``uncovered_entities`` is only present when the flow was asked for it.
    """

    coverage: float
    candidates: list[EntityCoverageCandidate] = Field(default_factory=list)
    uncovered_entities: list[str] | None = None
