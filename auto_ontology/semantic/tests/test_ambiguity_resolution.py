# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

from auto_ontology.semantic import ambiguity_resolution as ar
from auto_ontology.semantic.models import AmbiguityResolver, AmbiguityResolversResult


def _attr(aid: str, table: str, name: str, **values) -> dict:
    return {
        "id": aid,
        "name": name,
        "table_id": f"id-{table}",
        "table_name": table,
        "column_name": name,
        "column_id": f"col-{aid}",
        **values,
    }


def _run(attributes: list[dict], result: AmbiguityResolversResult | None) -> dict:
    with (
        patch.object(ar, "fetch_attributes_for_ambiguity", return_value=attributes),
        patch.object(ar, "get_llm_client"),
        patch.object(ar, "invoke_with_structured_output", return_value=result) as llm,
        patch.object(ar, "write_ambiguity_resolvers", return_value=2) as write,
        patch.object(
            ar, "write_attribute_descriptions", side_effect=lambda d: list(d)
        ) as attribute_fix,
        patch.object(
            ar, "update_node_properties", return_value={"id": "x"}
        ) as patch_node,
    ):
        ar.resolve_ambiguities("db")
    return {
        "llm": llm,
        "write": write,
        "attribute_fix": attribute_fix,
        "patch_node": patch_node,
    }


def test_normalized_name_ignores_case_and_word_separators() -> None:
    assert ar.normalized_name("pointsScored") == "points scored"
    assert ar.normalized_name("Points_Scored") == "points scored"
    assert ar.normalized_name("POINTS scored") == "points scored"
    assert ar.normalized_name("Charter Funding Type") != ar.normalized_name(
        "FundingType"
    )


def test_groups_need_the_same_name_in_different_tables() -> None:
    attributes = [
        _attr("a", "results", "points"),
        _attr("b", "driverStandings", "Points"),
        _attr("c", "results", "position"),
        _attr("d", "results", "Position"),
        _attr("e", "schools", "FundingType"),
        _attr("f", "frpm", "Charter Funding Type"),
    ]

    groups = ar.group_ambiguous(attributes)

    assert [[m["id"] for m in g] for g in groups] == [["a", "b"]]


def test_resolvers_map_to_members_and_the_rest_reset() -> None:
    attributes = [
        _attr("a", "results", "points"),
        _attr("b", "driverStandings", "points"),
        _attr("c", "races", "name"),
    ]
    result = AmbiguityResolversResult(
        members=[
            AmbiguityResolver(table_column="results.points", resolver="Per race."),
            AmbiguityResolver(
                table_column='"driverStandings"."points"', resolver="Running total."
            ),
            AmbiguityResolver(table_column="races.name", resolver="Not a member."),
        ]
    )
    mocks = _run(attributes, result)

    mocks["llm"].assert_called_once()
    prompt = mocks["llm"].call_args.args[1][1].content
    assert "results.points" in prompt and "driverStandings.points" in prompt
    assert "races.name" not in prompt
    mocks["write"].assert_called_once_with(
        "db", {"a": "Per race.", "b": "Running total."}
    )
    mocks["attribute_fix"].assert_called_once_with({})
    mocks["patch_node"].assert_not_called()


def test_prompt_shows_both_descriptions_only_when_they_differ() -> None:
    prompt = ar.build_group_prompt(
        [
            _attr("a", "results", "points", column_description="Race points."),
            _attr(
                "b",
                "standings",
                "points",
                column_description="Points.",
                description="Season total.",
            ),
            _attr("c", "other", "points", column_description="X.", description="X."),
        ]
    )
    assert "catalog description: Race points." in prompt
    assert "semantic description: Season total." in prompt
    assert "semantic description: X." not in prompt


def test_a_catalog_description_that_only_repeats_the_name_is_missing() -> None:
    bare = _attr("a", "results", "points", column_description="Points")
    empty = _attr("b", "standings", "points")
    written = _attr("c", "teams", "points", column_description="Points per race.")

    assert ar.is_bare_description(bare)
    assert ar.is_bare_description(empty)
    assert not ar.is_bare_description(written)
    prompt = ar.build_group_prompt([bare, empty, written])
    assert prompt.count("catalog description: (missing)") == 2
    assert "catalog description: Points per race." in prompt


def test_fixes_go_to_the_description_that_misleads() -> None:
    attributes = [
        _attr("a", "results", "points", column_description="Race points."),
        _attr(
            "b",
            "standings",
            "points",
            column_description="Points of each race.",
            description="Old gloss.",
        ),
        _attr(
            "c",
            "teams",
            "points",
            column_description="Points of each race.",
            column_certified=True,
        ),
    ]
    result = AmbiguityResolversResult(
        members=[
            AmbiguityResolver(
                table_column="results.points",
                resolver="Per race.",
                catalog_description_fix="Race points.",
            ),
            AmbiguityResolver(
                table_column="standings.points",
                resolver="Running total.",
                catalog_description_fix="Season total as of each race.",
                description_fix="Season total.",
            ),
            AmbiguityResolver(
                table_column="teams.points",
                resolver="Team total.",
                catalog_description_fix="Team season total.",
            ),
        ]
    )

    mocks = _run(attributes, result)

    mocks["attribute_fix"].assert_called_once_with({"b": "Season total."})
    mocks["patch_node"].assert_called_once_with(
        "col-b", {"description": "Season total as of each race."}
    )


def test_no_ambiguity_still_resets_the_database() -> None:
    mocks = _run([_attr("a", "results", "points")], None)

    mocks["llm"].assert_not_called()
    mocks["write"].assert_called_once_with("db", {})
