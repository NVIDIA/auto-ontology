# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The curated allow-list stays honest about the API it claims to expose."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

from auto_ontology_mcp.config import DEFAULT_SPEC_PATH
from auto_ontology_mcp.tools import (
    CURATED,
    apply_description,
    missing_from_spec,
    names_by_operation_id,
    route_maps,
)

SPEC = json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))

# Anything that changes state. A read-only server is the whole safety argument
# for handing this to an autonomous agent, so it is asserted rather than assumed.
_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# The one exception: coverage grading is a POST because it takes a question in
# the body, but it only reads the semantic layer.
_READ_ONLY_POSTS = {"/api/question-entity-coverage"}


def test_every_curated_operation_exists_in_the_spec() -> None:
    # The guard that matters most: a renamed or removed endpoint would
    # otherwise silently drop its tool, leaving a server that starts fine but
    # is quietly missing capability.
    assert missing_from_spec(SPEC) == []


def test_nothing_walks_the_physical_catalog() -> None:
    # Consumers are meant to reach the data through the semantic layer. A tool
    # that lists databases, schemas, or raw columns invites an agent to reason
    # about tables directly instead, so these routes stay unpublished even
    # though the API offers them.
    physical = (
        "/api/datasources/dbs",
        "/api/schemas/",
        "/api/tables/",
        "/api/columns/",
    )
    published = [spec.name for spec in CURATED if spec.path.startswith(physical)]

    assert published == []


def test_no_curated_operation_mutates_state() -> None:
    mutating = [
        f"{spec.method} {spec.path}"
        for spec in CURATED
        if spec.method.upper() in _WRITE_METHODS and spec.path not in _READ_ONLY_POSTS
    ]

    assert mutating == []


def test_tool_names_are_unique() -> None:
    names = [spec.name for spec in CURATED]

    assert len(names) == len(set(names))


def test_tool_names_are_plain_identifiers() -> None:
    # Hosts vary in what they accept and some namespace tools by concatenation,
    # so stay well inside the safe set rather than at the edge of the spec.
    bad = [
        spec.name for spec in CURATED if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.name)
    ]

    assert bad == []


def test_descriptions_do_not_leak_frontend_client_names() -> None:
    # The spec's own text reads "termsApi.list — ..." because it documents the
    # route for whoever maintains the caller. That is noise to an agent, and
    # its presence here would mean an override silently stopped applying.
    leaked = [spec.name for spec in CURATED if "Api." in spec.description]

    assert leaked == []


def test_route_maps_end_with_a_catch_all_exclude() -> None:
    # Without this the allow-list is not one: every uncurated endpoint in the
    # spec would be published too.
    maps = route_maps()

    assert maps[-1].mcp_type.name == "EXCLUDE"
    assert maps[-1].pattern == ".*"
    assert len(maps) == len(CURATED) + 1


def test_collection_pattern_does_not_match_its_item_route() -> None:
    # '/api/terms' unanchored also matches '/api/terms/{term_id}', which would
    # give both routes the same tool identity.
    collection = next(m for m in route_maps() if m.pattern.endswith(r"terms$"))

    assert re.match(collection.pattern, "/api/terms")
    assert not re.match(collection.pattern, "/api/terms/{term_id}")


def test_operation_ids_map_to_curated_names() -> None:
    names = names_by_operation_id(SPEC)

    assert sorted(names.values()) == sorted(spec.name for spec in CURATED)


def test_operation_ids_are_read_from_the_spec_not_guessed() -> None:
    names = names_by_operation_id(SPEC)

    # Confirms the lookup is keyed on the spec's real ids; these are the
    # generated shapes the overrides exist to hide.
    assert names["get_api_terms"] == "search_terms"
    assert names["get_api_terms_term_id_"] == "get_term"


def test_apply_description_overrides_the_generated_text() -> None:
    component = SimpleNamespace(description="termsApi.list — one page of Terms.")

    apply_description(SimpleNamespace(method="GET", path="/api/terms"), component)

    assert component.description.startswith("Search the business glossary")


def test_apply_description_leaves_uncurated_routes_alone() -> None:
    component = SimpleNamespace(description="original")

    apply_description(SimpleNamespace(method="GET", path="/api/not-curated"), component)

    assert component.description == "original"
