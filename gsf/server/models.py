# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Item models carried inside the response envelopes.

``gsf/server/responses.py`` owns the envelopes (``{"data": ...}``); this
module owns what goes *inside* them, so the generated OpenAPI spec names the
fields a caller actually receives instead of an opaque object.

The shapes mirror what the DAL reads return from
``gsf/dal/`` plus whatever the DAL/service layer adds in Python afterwards.
The DAL returns plain dicts, so the rule here is: a field is
required only when the row is anchored on it (an ``id`` matched by the query,
or a value the Python layer writes unconditionally). Everything else is
optional and nullable — a missing value comes back as ``null``, and a
required-but-null field would turn a documentation change into a 500.

Item shapes are shared across routers on purpose: a Zone chip is rendered by
the zones, terms, sql-attributes and exploration endpoints alike, so it is
modelled once here rather than four times in four router packages.

Every model allows extra keys — see :class:`ApiModel`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
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
    "GlobalSearchBreadcrumb",
    "GlobalSearchItem",
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
    "TableExplorationTerm",
    "TableSqlQuery",
    "TableSummary",
    "Tag",
    "TagChip",
    "TagDetail",
    "TagItem",
    "TagTargetType",
    "Term",
    "TermCountEntry",
    "TermDetail",
    "TermExplorationDetails",
    "TermExplorationTable",
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


class TagChip(ApiModel):
    """A tag as rendered beside the object it labels.

    Deliberately narrower than :class:`Tag`: a chip needs the name to show and
    the id to remove itself by, and the timestamps describe the tag rather than
    the labelling, so carrying them on every chip of every row would say
    nothing the tag's own page does not say better.

    Here rather than with the tag models below because every taggable thing
    carries it, and the catalog ones are modelled before the tags themselves.
    """

    id: str
    name: str


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


class GlobalSearchBreadcrumb(ApiModel):
    """One hop in a global-search hit's catalog/semantic path."""

    name: str
    type: str
    id: str | None = None


class GlobalSearchItem(ApiModel):
    """One global-search hit: identity plus optional certified/parent path.

    ``type`` is the entity's label (``Term``, ``Table``, ``ColumnAttribute``, …).
    ``table_type`` is set on ``Table`` nodes so the client can tell views apart.
    """

    id: str
    name: str | None = None
    type: str
    table_type: str | None = None
    description: str | None = None
    certified: bool | str | None = None
    parent_id: str | None = None
    breadcrumbs: list[GlobalSearchBreadcrumb] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)


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
    #: The catalog has no table detail read -- a table's page is built from the
    #: schema's list -- so its chips travel with the row.
    tags: list[TagChip] = Field(default_factory=list)


class ColumnSummary(ApiModel):
    """One column of a table, ordered by ``ordinal_position``."""

    id: str
    ordinal_position: int | None = None
    column_name: str | None = None
    data_type: str | None = None
    description: str | None = None
    description_certified: bool | None = None
    sample_values: list[str] | None = None
    tags: list[TagChip] = Field(default_factory=list)


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
# Tags
# ---------------------------------------------------------------------------


class Tag(ApiModel):
    """A tag, as both the list and the create endpoint return it.

    The four columns describing *when* are required, unlike most of this
    module: each is ``NOT NULL`` and ``gsf.dal.tags`` selects them on every
    path.

    ``created`` and ``modified`` are the first timestamps this module carries,
    so they set the convention: a timezone-aware ``datetime``, which Pydantic
    serialises as ISO 8601 and the client parses directly. They are read from
    the database rather than the application clock, so a client comparing two
    tags is comparing one clock.

    The two describing *who* are nullable, and each null says something
    different. ``created_by`` is null for a tag made by a caller that reached
    FastAPI without the gateway's identity header. ``modified_by`` is
    additionally null for a tag nobody has renamed, which is the same fact
    ``modified == created`` states.

    Both are opaque Better Auth user ids. Resolving one to a name is the
    gateway's job -- the accounts live in a schema this service does not own --
    so this API deliberately answers with the id it stored.
    """

    id: str
    name: str
    created: datetime
    modified: datetime
    created_by: str | None = None
    modified_by: str | None = None


class TagTargetType(StrEnum):
    """Which of the five things a tag points at.

    Used on both sides of tagging: it is the ``type`` of a :class:`TagItem` a
    tag's page reads, and the ``type`` the attach and detach routes validate
    against — which is what makes an unknown kind a 422 from FastAPI's own
    validation, listing the five it does accept.

    The five strings are also the keys ``gsf.dal.tags`` resolves to a
    ``tag_target`` column, and they are spelled out here rather than imported
    from it: this module is the shape of the API, and pulling a five-string
    vocabulary out of the DAL made every importer of it load SQLAlchemy and the
    whole table metadata too. ``gsf/server/tests/test_tag_target_type.py`` is
    what keeps the two spellings identical, since a kind the DAL has no column
    for would answer 500 rather than the 422 this enum exists to produce.
    """

    TERM = "term"
    TABLE = "table"
    COLUMN = "column"
    COLUMN_ATTRIBUTE = "column_attribute"
    SQL_ATTRIBUTE = "sql_attribute"


class TagItem(ApiModel):
    """One object carrying a tag, whichever of the five kinds it is.

    ``type`` is what tells them apart, as :class:`TagTargetType` spells the five.

    ``path`` says where the object sits — ``database.schema`` for a Table,
    ``database.schema.table`` for a Column, and the owning Term for either kind
    of attribute, which are properties of one in the same sense and so are
    described the same way. Null only for a Term, which is a glossary entry
    rather than a catalog object and sits under nothing.

    The four id fields are the same relationships as ids, which is what a link
    to the object's own page is built from: a Table and a Column are addressed
    by their whole catalog chain, an attribute by the Term whose page lists it,
    and a Term by its own id alone. Each kind fills only the ones it has.
    """

    id: str
    name: str
    type: TagTargetType
    path: str | None = None
    tagged: datetime
    database_id: str | None = None
    schema_id: str | None = None
    table_id: str | None = None
    term_id: str | None = None


class TagDetail(Tag):
    """A tag together with everything it labels.

    ``items`` is empty for a tag nothing carries — a freshly created tag, or one
    whose last object was untagged — which the page renders as its empty state
    rather than as a failed read.
    """

    items: list[TagItem] = Field(default_factory=list)


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
    """One Term with the tables that represent it and its related terms.

    ``tags`` is on the detail rather than on :class:`Term` because only this
    read resolves it: the paged list would need a second query per page to fill
    it, and its cards do not render chips.
    """

    table_count: int | None = None
    tables: list[TermTable] = Field(default_factory=list)
    related_terms: list[TermSummary] = Field(default_factory=list)
    tags: list[TagChip] = Field(default_factory=list)


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
    tags: list[TagChip] | None = None
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
    tags: list[TagChip] | None = None


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
    sample values are rendered through ``stringify_sample_values`` before
    reaching this model, so the types a column stores arrive here as a plain
    string list.
    """

    source_column: str | None = None
    target_column: str | None = None
    source_sample_values: list[str] | None = None
    target_sample_values: list[str] | None = None


class ExplorationEdge(ApiModel):
    """Two tables connected by a shared SQL query and/or a foreign key.

    ``queries`` is empty for an edge that exists only because of a foreign
    key; ``foreign_keys`` is empty for a SQL-only edge. ``relationship_types``
    names the relationship kind(s) behind the edge (``SQL`` and/or
    ``FOREIGN_KEY``), for labeling the connection in the graph.
    """

    source: str
    target: str
    queries: list[str] = Field(default_factory=list)
    via_foreign_key: bool = False
    foreign_keys: list[ForeignKeyRef] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)


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
    """An undirected term↔term link (two terms sharing at least one table).

    ``relationship_types`` names the relationship kind(s)
    (``REPRESENTS``, ``HAS_ATTRIBUTE``, ``SEMANTIC_FK``) connecting either
    term to a table they share, for labeling the connection in the graph.
    """

    source: str
    target: str
    relationship_types: list[str] = Field(default_factory=list)


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


class TableExplorationTerm(TermSummary):
    """A Term linked to a table, with the relationship kind(s) reaching it.

    ``relationship_types`` names how this term connects to the table —
    ``REPRESENTS`` directly, and/or ``HAS_ATTRIBUTE``/``SEMANTIC_FK`` via one
    of its columns — the same way ``ExplorationEdge``/``GraphLink`` name
    theirs, so a client can label the connection the same way it would read
    in the graph itself.
    """

    relationship_types: list[str] = Field(default_factory=list)


class TableExplorationDetails(ApiModel):
    """The table-detail modal: its SQL queries and one page of its Terms."""

    queries: list[TableSqlQuery] = Field(default_factory=list)
    terms: list[TableExplorationTerm] = Field(default_factory=list)
    terms_total: int | None = None


class TermExplorationTable(ApiModel):
    """A Table linked to a term, with the relationship kind(s) reaching it.

    The reverse of ``TableExplorationTerm``: ``relationship_types`` names how
    this table connects to the term — ``REPRESENTS`` directly, and/or
    ``HAS_ATTRIBUTE``/``SEMANTIC_FK`` via one of its columns — so a client can
    label an expansion edge from either end the same way.
    """

    id: str
    name: str | None = None
    table_type: str | None = None
    database_id: str | None = None
    database_name: str | None = None
    schema_id: str | None = None
    schema_name: str | None = None
    relationship_types: list[str] = Field(default_factory=list)


class TermExplorationDetails(ApiModel):
    """One ordered page of Tables linked to a Term, for expanding it on the graph."""

    tables: list[TermExplorationTable] = Field(default_factory=list)
    tables_total: int | None = None


class ColumnExplorationAttribute(ApiModel):
    """The ColumnAttribute HAS_ATTRIBUTE/SEMANTIC_FK-linked to a Column, if any.

    `relationship_type` names the relationship kind
    (``HAS_ATTRIBUTE`` or ``SEMANTIC_FK``) actually connecting the Column
    to this ColumnAttribute — same rationale as `relationship_types` on
    `TermExplorationTable`/`ExplorationLink` — so a client can label the
    edge the same way it would read in the graph itself, distinguishing a
    plain attribute from an FK-shaped one (like `user_id`).
    """

    id: str
    name: str | None = None
    description: str | None = None
    relationship_type: str | None = None


class ColumnExplorationForeignKeyColumn(ApiModel):
    """The Column a Column's own outgoing FOREIGN_KEY edge points at, if any.

    Carries the target Column's own owning Table/Schema/Database ids/names
    (like `ExplorationLinkPathNode` does for a link-path hop) so a client
    can graft it onto the graph as a fully expandable Column node — see
    `expandColumnNode` in `ExplorationView.tsx`.
    """

    id: str
    name: str | None = None
    description: str | None = None
    data_type: str | None = None
    table_id: str | None = None
    table_name: str | None = None
    database_id: str | None = None
    database_name: str | None = None
    schema_id: str | None = None
    schema_name: str | None = None


class ColumnExplorationDetails(ApiModel):
    """A Column's own ColumnAttribute, outgoing FOREIGN_KEY target Column,
    incoming FOREIGN_KEY source Columns, and referencing Sql queries, for
    expanding it on the graph.

    `column_attribute`/`foreign_key_column` are `None` for a column with no
    ColumnAttribute/no outgoing FK; `referencing_columns` is empty when no
    other Column's own FK points at this one (the reverse of
    `foreign_key_column`); `sql_queries` is empty for a column no stored
    query ever referenced directly — see `fetch_column_exploration_details`.
    """

    column_attribute: ColumnExplorationAttribute | None = None
    foreign_key_column: ColumnExplorationForeignKeyColumn | None = None
    referencing_columns: list[ColumnExplorationForeignKeyColumn] = Field(
        default_factory=list
    )
    sql_queries: list[TableSqlQuery] = Field(default_factory=list)


class ColumnAttributeExplorationColumn(ApiModel):
    """One Column HAS_ATTRIBUTE/SEMANTIC_FK-linked to a ColumnAttribute.

    Same shape as `ColumnExplorationForeignKeyColumn` — enough for a
    client to graft it on as a fully expandable Column node.
    `relationship_types` mirrors `TermExplorationTable.relationship_types`:
    a Column reaching the same attribute via more than one edge type
    collects both once grouped.
    """

    id: str
    name: str | None = None
    description: str | None = None
    data_type: str | None = None
    table_id: str | None = None
    table_name: str | None = None
    database_id: str | None = None
    database_name: str | None = None
    schema_id: str | None = None
    schema_name: str | None = None
    relationship_types: list[str] = Field(default_factory=list)


class ColumnAttributeExplorationDetails(ApiModel):
    """A ColumnAttribute's own owning Term and every linked Column, for
    expanding it on the graph.

    `term` is `None` only when the attribute has no owning Term (or it's
    out of scope). `columns` is one ordered page of *every* Column
    HAS_ATTRIBUTE/SEMANTIC_FK-linked to this attribute — not just the
    "primary" one whichever expansion grafted this attribute on already
    knew about — since a shared attribute (e.g. a common `user_id`-shaped
    one) can be linked from many Columns across many Tables at once. See
    `fetch_column_attribute_exploration_details`.
    """

    term: TermSummary | None = None
    columns: list[ColumnAttributeExplorationColumn] = Field(default_factory=list)
    columns_total: int = 0


class SqlAttributeExplorationSql(ApiModel):
    """The Sql query node HAS_SQL-linked to a SqlAttribute."""

    id: str
    sql: str | None = None


class SqlAttributeExplorationDetails(ApiModel):
    """A SqlAttribute's own Sql query and owning Term, for expanding it on the graph.

    Both are `None` only when the attribute itself doesn't exist (or is out
    of scope) — see `fetch_sql_attribute_exploration_details`. `sql` alone
    can be `None` when the attribute's SQL touches a table outside the
    caller's zones even though the attribute/term are visible.
    """

    sql: SqlAttributeExplorationSql | None = None
    term: TermSummary | None = None


class SqlExplorationCustomAnalysis(ApiModel):
    """A CustomAnalysis HAS_SQL-linked to the same Sql node as a SqlAttribute."""

    id: str
    name: str | None = None
    description: str | None = None


class SqlExplorationColumn(ApiModel):
    """One Column the Sql node's own `SQL` edge connects to directly.

    Same shape as `ColumnAttributeExplorationColumn` — enough for a client
    to graft it on as a fully expandable Column node.
    """

    id: str
    name: str | None = None
    description: str | None = None
    data_type: str | None = None
    table_id: str | None = None
    table_name: str | None = None
    database_id: str | None = None
    database_name: str | None = None
    schema_id: str | None = None
    schema_name: str | None = None


class SqlExplorationTable(ApiModel):
    """One Table the Sql node's own `SQL` edge connects to directly.

    One hop out from the statement along its own `SQL` edge — enough for a
    client to graft it on as a fully expandable Table node, same shape as
    `SqlExplorationColumn` minus the column-only fields.
    """

    id: str
    name: str | None = None
    table_type: str | None = None
    database_id: str | None = None
    database_name: str | None = None
    schema_id: str | None = None
    schema_name: str | None = None


class SqlExplorationSqlAttribute(ApiModel):
    """A SqlAttribute HAS_SQL-linked to the Sql node, with its owning Term.

    `term_id`/`term_name` are `None` only when the SqlAttribute has no
    owning Term at all — an out-of-scope owning Term instead drops the
    whole row, same as `fetch_sql_attribute_exploration_details`'s own
    `term`.
    """

    id: str
    name: str | None = None
    description: str | None = None
    term_id: str | None = None
    term_name: str | None = None


class SqlExplorationDetails(ApiModel):
    """The CustomAnalysis, Column, Table and SqlAttribute nodes hanging off a Sql node.

    `custom_analyses` is empty for the common case of a Sql node with no
    CustomAnalysis sharing it; `columns` is every visible Column the Sql
    node's own `SQL` edges reach directly; `tables` is every visible Table
    the Sql node's own `SQL` edges reach directly; `sql_attributes` is every
    visible SqlAttribute this same Sql node backs (the reverse of
    `SqlAttributeExplorationDetails.sql`) — see
    `fetch_sql_exploration_details`.
    """

    custom_analyses: list[SqlExplorationCustomAnalysis] = Field(default_factory=list)
    columns: list[SqlExplorationColumn] = Field(default_factory=list)
    tables: list[SqlExplorationTable] = Field(default_factory=list)
    sql_attributes: list[SqlExplorationSqlAttribute] = Field(default_factory=list)


class ExplorationLinkPathNode(ApiModel):
    """One Term/Table/Column/ColumnAttribute node along an `ExplorationLinkPathHop` chain.

    ``type`` matches the Exploration graph's own node kinds (``term``,
    ``table``, ``column``, ``columnAttribute``) so a client can graft/style
    this node exactly like any other of that type — see
    `fetch_semantic_link_path`. The catalog fields are only ever set for a
    ``table``/``column`` type node (see ``_enrich_catalog_path_nodes`` in
    ``gsf/dal/attributes.py``) — the same ids a client needs to expand
    either one further, exactly like a Table/Column node grafted anywhere
    else in the app, so neither is a dead end just because it came from a
    link path instead. ``table_id``/``table_name`` are a ``column`` node's
    own owning Table, additionally.
    """

    id: str
    name: str | None = None
    type: str
    database_id: str | None = None
    database_name: str | None = None
    schema_id: str | None = None
    schema_name: str | None = None
    table_id: str | None = None
    table_name: str | None = None


class ExplorationLinkPathHop(ApiModel):
    """One relationship traversed along a term↔term path.

    ``relationship`` is the underlying type (``REPRESENTS``, ``CONTAINS``,
    ``HAS_ATTRIBUTE``, ``SEMANTIC_FK`` or ``PROPERTY_OF``).
    """

    relationship: str
    source: ExplorationLinkPathNode
    target: ExplorationLinkPathNode


class ExplorationLinkPath(ApiModel):
    """The real ordered hop chain connecting two Terms, e.g. Term1
    <-REPRESENTS- Table -CONTAINS-> Column -SEMANTIC_FK-> ColumnAttribute
    -PROPERTY_OF-> Term2 — see `fetch_semantic_link_path`. ``hops`` is
    ordered from the first term to the second; empty only for a stale/
    hand-crafted request naming two terms with no path connecting them.
    """

    hops: list[ExplorationLinkPathHop] = Field(default_factory=list)


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
