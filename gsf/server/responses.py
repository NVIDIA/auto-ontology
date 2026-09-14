# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response envelopes shared by the API routers.

Every route declares a ``response_model`` so the generated OpenAPI spec
describes what callers actually receive, not just ``{}``. This module owns
the envelope (``data``/``count``/``total``); ``gsf/server/models.py`` owns
the item models that go inside it, and the aliases at the bottom of this file
bind the two together — a router imports one name per route.

Every model allows extra keys (see :class:`gsf.server.models.ApiModel`).
FastAPI would otherwise *drop* any field a handler returns that the model
does not declare, which would turn a documentation change into a breaking
API change.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from gsf.server.models import (
    ApiModel as _Payload,  # extra="allow" base — see the module docstring
)
from gsf.server.models import (
    ColumnAttribute,
    ColumnAttributeExplorationDetails,
    ColumnExplorationDetails,
    CustomAnalysis,
    DataExplorationGraph,
    DatabaseSummary,
    GlobalSearchItem,
    EntityCoverageResult,
    ExplorationEdge,
    ExplorationLinkPath,
    ExplorationRelatedNodes,
    IdRef,
    PqlAnalysis,
    PublicConnection,
    Rule,
    SchemaSummary,
    SemanticExplorationGraph,
    SqlAttribute,
    SqlAttributeExplorationDetails,
    SqlExplorationDetails,
    SqlExpressionValidationResult,
    SqlValidationResult,
    SsoFederationState,
    TableColumns,
    TableExplorationDetails,
    TableSummary,
    Tag,
    TagChip,
    TagItem,
    Term,
    TermCountEntry,
    TermDetail,
    TermExplorationDetails,
    TermListItem,
    Zone,
    ZoneChip,
)

T = TypeVar("T")

JsonObject = dict[str, Any]

__all__ = [
    "ChatCancelResponse",
    "ColumnAttributeExplorationDetailsResponse",
    "ColumnAttributePageResponse",
    "ColumnAttributePatchResponse",
    "ColumnAttributeResponse",
    "ColumnExplorationDetailsResponse",
    "ConnectionListResponse",
    "ConnectionResponse",
    "ConnectionTestResponse",
    "CustomAnalysisListResponse",
    "CustomAnalysisResponse",
    "DataExplorationGraphResponse",
    "DataResponse",
    "DatabaseListResponse",
    "DescriptionSuggestionResponse",
    "EntityCoverageResponse",
    "ExplorationEdgeListResponse",
    "ExplorationRelatedNodesResponse",
    "GlobalSearchCountResponse",
    "GlobalSearchListResponse",
    "HealthResponse",
    "IdResponse",
    "JsonObject",
    "ListResponse",
    "ModelImportResponse",
    "ObjectPageResponse",
    "PagedListResponse",
    "PqlAnalysisListResponse",
    "PqlAnalysisResponse",
    "RulePageResponse",
    "RuleResponse",
    "SchemasPayload",
    "SemanticExplorationGraphResponse",
    "SemanticRunningResponse",
    "SemanticStatusResponse",
    "SqlAttributeExplorationDetailsResponse",
    "SqlAttributeListResponse",
    "SqlAttributePageResponse",
    "SqlAttributePatchResponse",
    "SqlAttributeResponse",
    "SqlExplorationDetailsResponse",
    "SqlExpressionValidationResponse",
    "SqlValidationResponse",
    "SsoFederationResponse",
    "StatusResponse",
    "TableColumnsPageResponse",
    "TableExplorationDetailsResponse",
    "TableListResponse",
    "TableZonesResponse",
    "TagChipListResponse",
    "TagItemPageResponse",
    "TagPageResponse",
    "TagResponse",
    "TermDetailResponse",
    "TermExplorationDetailsResponse",
    "TermResponse",
    "TermsPageResponse",
    "ZoneListResponse",
    "ZoneResponse",
]


# ---------------------------------------------------------------------------
# Generic envelopes
# ---------------------------------------------------------------------------


class DataResponse(_Payload, Generic[T]):
    """``{"data": <item>}`` — the single-item envelope."""

    data: T


class ListResponse(_Payload, Generic[T]):
    """``{"data": [...], "count": n}`` — ``count`` is the length of ``data``."""

    data: list[T]
    count: int


class PagedListResponse(ListResponse[T], Generic[T]):
    """A list envelope whose ``total`` counts every match, not just this page."""

    total: int


class ObjectPageResponse(_Payload, Generic[T]):
    """A paged envelope whose ``data`` is an object rather than a list.

    ``/columns/{table_id}`` wraps ``{columns: [...]}``, so ``count`` is
    always 1 and ``total`` counts the table's columns.
    """

    data: T
    count: int
    total: int


# ---------------------------------------------------------------------------
# Status / acknowledgement responses
# ---------------------------------------------------------------------------


class StatusResponse(_Payload):
    """``{"status": "ok"}`` or ``{"status": "accepted"}`` for async triggers."""

    status: str


class HealthResponse(_Payload):
    """Readiness body — 200 when healthy, 503 when degraded.

    ``postgres`` reports whether the database answers; ``migrations`` whether it
    holds the revision this build expects. Both matter and they fail
    independently: a reachable database carrying no tables at all satisfies the
    first and not the second, and reporting only the first is how a health
    check comes to say ``ok`` about a server that 500s on every catalog call.

    Named ``migrations`` rather than the more obvious ``schema`` because that
    name shadows an attribute on Pydantic's ``BaseModel`` and the field would
    ship with a ``UserWarning`` on every import.
    """

    status: str
    postgres: dict[str, str]
    migrations: dict[str, str]


class SemanticStatusResponse(_Payload):
    calculated: bool
    running: bool
    last_success_at: str | None
    last_failure_at: str | None


class SemanticRunningResponse(_Payload):
    """The ingestion service's own status endpoint: whether a compilation pass
    is executing right now, when the last one finished successfully, and when
    it last failed (only set if more recent than the last success)."""

    running: bool
    last_success_at: str | None
    last_failure_at: str | None


class ChatCancelResponse(_Payload):
    cancelled: bool


# ---------------------------------------------------------------------------
# Route-specific shapes
# ---------------------------------------------------------------------------


class ConnectionTestResponse(_Payload):
    """``schemas`` is empty for connectors that cannot enumerate schemas."""

    success: bool
    schemas: list[str]


class ModelImportResponse(_Payload):
    success: bool
    summary: JsonObject


class SchemasPayload(_Payload):
    """``/schemas/{db_id}`` — unenveloped; null when the database is missing."""

    schemas_count: int
    schemas: list[SchemaSummary]


class DescriptionSuggestionResponse(_Payload):
    """``data`` is null when no description could be suggested."""

    data: str | None


class ColumnAttributePatchResponse(_Payload):
    """``term_certification`` is the owning Term's badge, recomputed on write.

    The patch returns it alongside ``data`` so the Terms list can update the
    badge without a second request, which is why this needs its own model
    rather than the plain ``DataResponse`` envelope.
    """

    data: ColumnAttribute
    term_certification: str | None


class SqlAttributePatchResponse(_Payload):
    """``term_certification`` is recomputed only when the patch names a term."""

    data: SqlAttribute
    term_certification: str | None


class TermsPageResponse(_Payload):
    """One page of the Terms list with its per-term count breakdowns.

    ``total`` counts every matching term; the count lists cover only the
    terms on this page when a limit was given.
    """

    terms: list[TermListItem]
    total: int
    column_attribute_counts: list[TermCountEntry]
    sql_attribute_counts: list[TermCountEntry]
    related_counts: list[TermCountEntry]


# ---------------------------------------------------------------------------
# Concrete envelopes — one alias per route
# ---------------------------------------------------------------------------

IdResponse = DataResponse[IdRef]

# Zones
ZoneResponse = DataResponse[Zone]
ZoneListResponse = ListResponse[Zone]

# Tags
TagResponse = DataResponse[Tag]
# Paged, but optionally: the settings list reads a window of the vocabulary and
# the tag picker reads all of it, so `total` is what tells the first when to stop
# asking and equals `count` for the second.
TagPageResponse = PagedListResponse[Tag]
# What a tag labels, paged: a tag applied by a rule can carry the whole catalog,
# so the objects are their own list rather than a field on the tag.
TagItemPageResponse = PagedListResponse[TagItem]
# The attach and detach routes answer with the *object's* tags after the change,
# which is what the page redraws from — not with the tag that was moved.
TagChipListResponse = ListResponse[TagChip]

# Rules
RuleResponse = DataResponse[Rule]
RulePageResponse = PagedListResponse[Rule]

# Catalog
DatabaseListResponse = ListResponse[DatabaseSummary]
TableListResponse = ListResponse[TableSummary]
TableColumnsPageResponse = ObjectPageResponse[TableColumns]
GlobalSearchListResponse = ListResponse[GlobalSearchItem]
GlobalSearchCountResponse = DataResponse[dict[str, int]]

# Analyses
CustomAnalysisResponse = DataResponse[CustomAnalysis]
CustomAnalysisListResponse = ListResponse[CustomAnalysis]
SqlValidationResponse = DataResponse[SqlValidationResult]
PqlAnalysisResponse = DataResponse[PqlAnalysis]
PqlAnalysisListResponse = ListResponse[PqlAnalysis]

# Terms and attributes
TermResponse = DataResponse[Term]
TermDetailResponse = DataResponse[TermDetail]
ColumnAttributeResponse = DataResponse[ColumnAttribute]
ColumnAttributePageResponse = PagedListResponse[ColumnAttribute]
SqlAttributeResponse = DataResponse[SqlAttribute]
SqlAttributeListResponse = ListResponse[SqlAttribute]
SqlAttributePageResponse = PagedListResponse[SqlAttribute]
SqlExpressionValidationResponse = DataResponse[SqlExpressionValidationResult]

# Exploration
ExplorationEdgeListResponse = ListResponse[ExplorationEdge]
DataExplorationGraphResponse = DataResponse[DataExplorationGraph]
SemanticExplorationGraphResponse = DataResponse[SemanticExplorationGraph]
TableExplorationDetailsResponse = DataResponse[TableExplorationDetails]
TermExplorationDetailsResponse = DataResponse[TermExplorationDetails]
ColumnExplorationDetailsResponse = DataResponse[ColumnExplorationDetails]
ColumnAttributeExplorationDetailsResponse = DataResponse[
    ColumnAttributeExplorationDetails
]
SqlAttributeExplorationDetailsResponse = DataResponse[SqlAttributeExplorationDetails]
SqlExplorationDetailsResponse = DataResponse[SqlExplorationDetails]
ExplorationLinkPathResponse = DataResponse[ExplorationLinkPath]
ExplorationRelatedNodesResponse = DataResponse[ExplorationRelatedNodes]
TableZonesResponse = DataResponse[dict[str, list[ZoneChip]]]

# Connections
ConnectionResponse = DataResponse[PublicConnection]
ConnectionListResponse = ListResponse[PublicConnection]
SsoFederationResponse = DataResponse[SsoFederationState]

# Metadata
EntityCoverageResponse = DataResponse[EntityCoverageResult]
