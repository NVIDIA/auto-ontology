# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The curated tool surface, selected from the public OpenAPI spec.

The spec describes 87 operations. Exposing all of them would be actively
harmful: agent tool-selection degrades well before that count, and most of the
surface — SSO providers, token management, custom prompts, compilation control
— has no business being agent-callable. So this module names an explicit
allow-list and everything else is excluded.

The allow-list is deliberately about *meaning* rather than storage. Nothing
here walks the physical catalog database by database: consumers should reach
the data through the semantic layer, which is the point of Auto Ontology, and a tool that
lists schemas or columns invites an agent to bypass it and reason about raw
tables instead. ``describe_table`` is the exception, because what it returns is
the terms and SQL attributes a table participates in.

Two things are deliberately overridden rather than taken from the spec:

* **Names.** Generated operation ids read ``get_api_terms_term_id_``, which
  tells a model nothing.
* **Descriptions.** The spec's text is written for a developer reading API
  docs and refers to the frontend client that calls each route
  ("termsApi.list — ..."). A tool description is a prompt; it has to say when
  to reach for the tool, in the vocabulary of the caller's problem.

Everything else — parameters, request bodies, and their per-field descriptions
— comes straight from the spec and stays correct because CI fails when the
spec drifts from the routes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from fastmcp.server.providers.openapi import MCPType, RouteMap


@dataclass(frozen=True)
class ToolSpec:
    """One published operation, with the identity an agent sees."""

    method: str
    path: str
    name: str
    description: str


# Ordered roughly as an agent would work: understand the vocabulary, then ask.
# Two tools are not here because neither maps to a single request/response
# operation: ``ask_question`` streams (``chat.py``), and ``check_readiness``
# folds three endpoints into one verdict (``readiness.py``). The latter is why
# no ``get_semantic_layer_status`` entry remains: it reported one of those three
# facts, and a compiled layer on its own never meant a question could be
# answered.
CURATED: tuple[ToolSpec, ...] = (
    ToolSpec(
        method="GET",
        path="/api/terms",
        name="search_terms",
        description=(
            "Search the business glossary. Terms are the human-meaningful "
            "concepts Auto Ontology has compiled over the connected databases — "
            '"Customer", "Open Order", "Net Revenue" — each mapped to real '
            "columns and SQL expressions. Use this first to learn what a "
            "business word actually means in this deployment before asking a "
            "question that depends on it."
        ),
    ),
    ToolSpec(
        method="GET",
        path="/api/terms/{term_id}",
        name="get_term",
        description=(
            "Fetch one glossary term: its description, synonyms, and the "
            "terms related to it. Use after search_terms to confirm a term "
            "means what you assumed."
        ),
    ),
    ToolSpec(
        method="GET",
        path="/api/terms/{term_id}/column-attributes",
        name="get_term_columns",
        description=(
            "List the physical table columns a glossary term maps to. This is "
            "how you get from a business concept to the data behind it."
        ),
    ),
    ToolSpec(
        method="GET",
        path="/api/terms/{term_id}/sql-attributes",
        name="get_term_sql_attributes",
        description=(
            "List the SQL attributes defined under a glossary term — reviewed "
            "SQL expressions such as a margin, a ratio, or a rolling total. "
            "Prefer reusing one of these over inventing the arithmetic "
            "yourself."
        ),
    ),
    ToolSpec(
        method="GET",
        path="/api/sql-attributes/{attr_id}",
        name="get_sql_attribute",
        description=(
            "Fetch one SQL attribute: its expression, the term it belongs to, "
            "and what it is for."
        ),
    ),
    ToolSpec(
        method="GET",
        path="/api/exploration/tables/{table_id}/details",
        name="describe_table",
        description=(
            "Describe one table in full: its columns, the glossary terms that "
            "reference it, and the SQL attributes defined over it. The richest "
            "single view of what a table is for."
        ),
    ),
    ToolSpec(
        method="POST",
        path="/api/question-entity-coverage",
        name="check_answerable",
        description=(
            "Grade how well the semantic layer covers the entities in a "
            "question, without running the full agent. A cheap pre-flight: "
            "use it to decide whether Auto Ontology can answer something before paying "
            "for ask_question, which is far slower and more expensive."
        ),
    ),
)


def _anchored(path: str) -> str:
    """Return an exact-match regex for an OpenAPI path template.

    Templates contain ``{}`` which are regex quantifiers, so the whole literal
    is escaped and anchored — without anchoring, ``/api/terms`` would also
    select ``/api/terms/{term_id}`` and quietly widen the surface.
    """
    return f"^{re.escape(path)}$"


def route_maps() -> list[RouteMap]:
    """Include exactly the curated operations; exclude everything else.

    The trailing catch-all is what makes this an allow-list: a new endpoint
    appearing in the spec is invisible to agents until it is named here.
    """
    maps = [
        RouteMap(
            methods=[spec.method],
            pattern=_anchored(spec.path),
            mcp_type=MCPType.TOOL,
        )
        for spec in CURATED
    ]
    maps.append(RouteMap(mcp_type=MCPType.EXCLUDE))
    return maps


def _by_route() -> dict[tuple[str, str], ToolSpec]:
    return {(spec.method.upper(), spec.path): spec for spec in CURATED}


def missing_from_spec(openapi_spec: dict[str, Any]) -> list[str]:
    """Return curated entries the spec does not define.

    A rename upstream would otherwise drop a tool silently: its route map
    matches nothing, and the server starts with a smaller surface than
    intended. Callers turn this into a startup failure instead.
    """
    published = {
        (method.upper(), path)
        for path, item in (openapi_spec.get("paths") or {}).items()
        for method in item
        if method.lower() in ("get", "post", "put", "patch", "delete")
    }
    return [
        f"{spec.method} {spec.path}"
        for spec in CURATED
        if (spec.method.upper(), spec.path) not in published
    ]


def names_by_operation_id(openapi_spec: dict[str, Any]) -> dict[str, str]:
    """Map each curated operation's id to the tool name agents should see.

    Operation ids are looked up from the spec rather than hardcoded, so a
    change to how the generator derives them does not silently un-map every
    name here.
    """
    curated = _by_route()
    names: dict[str, str] = {}
    for path, item in (openapi_spec.get("paths") or {}).items():
        for method, operation in item.items():
            spec = curated.get((method.upper(), path))
            if spec is None:
                continue
            operation_id = operation.get("operationId")
            if operation_id:
                names[operation_id] = spec.name
    return names


def apply_description(route: Any, component: Any) -> None:
    """Replace a generated component's description with the curated one.

    Shaped as a FastMCP ``ComponentFn``: called once per generated component
    and mutates it in place.
    """
    spec = _by_route().get(
        (str(getattr(route, "method", "")).upper(), getattr(route, "path", ""))
    )
    if spec is not None:
        component.description = spec.description


__all__ = [
    "CURATED",
    "ToolSpec",
    "apply_description",
    "missing_from_spec",
    "names_by_operation_id",
    "route_maps",
]
