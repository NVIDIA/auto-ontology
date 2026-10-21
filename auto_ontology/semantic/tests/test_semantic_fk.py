# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-table semantic foreign-key resolution."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.semantic import semantic_fk
from auto_ontology.semantic.embed import _build_rows
from auto_ontology.semantic.models import SemanticFkJoinDefaultSelection


def _hit(
    attr_id: str,
    table_id: str,
    *,
    score: float = 0.5,
    table_name: str = "target_table",
    schema_name: str = "public",
    source_column: str = "id",
) -> dict:
    return {
        "metadata": {
            "id": attr_id,
            "table_id": table_id,
            "table_name": table_name,
            "schema_name": schema_name,
            "source_column": source_column,
            "is_unique": True,
        },
        "_distance": score,
        "text": f"ColumnAttribute {attr_id}",
    }


def test_format_sample_values_handles_legacy_json_string_and_native_list() -> None:
    # Legacy Column nodes still store sample_values as a JSON string.
    assert semantic_fk._format_sample_values('["1", "2"]') == "sample_values: 1, 2"
    # Current writers store a native list.
    assert semantic_fk._format_sample_values(["1", "2"]) == "sample_values: 1, 2"
    assert semantic_fk._format_sample_values(None) == ""
    assert semantic_fk._format_sample_values(["1", None, "2"]) == "sample_values: 1, 2"
    assert (
        semantic_fk._format_sample_values('["1", null, "2"]') == "sample_values: 1, 2"
    )


def test_distinct_samples_drops_none_and_preserves_order() -> None:
    assert semantic_fk._distinct_samples(["a", None, "b", "a"]) == ["a", "b"]
    assert semantic_fk._distinct_samples('["a", null, "b", "a"]') == ["a", "b"]


@patch("auto_ontology.semantic.semantic_fk._match_hit_by_sample_values")
@patch("auto_ontology.semantic.semantic_fk._llm_pick_hit", return_value="other-attr")
def test_vdb_resolution_excludes_same_table_candidates(
    mock_llm_pick: MagicMock,
    mock_sql_fallback: MagicMock,
) -> None:
    retriever = MagicMock()
    retriever.query.side_effect = [
        [_hit("same-attr", "table-1"), _hit("other-attr", "table-2")],
        [_hit("same-attr", "table-1")],
    ]
    column = {
        "id": "source-column",
        "name": "id",
        "table_id": "table-1",
        "table_name": "tags",
    }

    selected = semantic_fk._resolve_via_vdb(column, retriever, "database")

    assert selected == "other-attr"
    assert mock_llm_pick.call_args.args[1] == [_hit("other-attr", "table-2")]
    assert all(
        call.kwargs["vdb_kwargs"]["where"]["is_unique"] is True
        for call in retriever.query.call_args_list
    )
    mock_sql_fallback.assert_not_called()


@patch("auto_ontology.semantic.semantic_fk._llm_pick_hit")
def test_vdb_resolution_skips_llm_when_only_same_table_candidate_exists(
    mock_llm_pick: MagicMock,
) -> None:
    retriever = MagicMock()
    retriever.query.side_effect = [[_hit("same-attr", "table-1")], []]
    column = {
        "id": "source-column",
        "name": "id",
        "table_id": "table-1",
        "table_name": "tags",
    }

    assert semantic_fk._resolve_via_vdb(column, retriever, "database") is None
    mock_llm_pick.assert_not_called()


@patch(
    "auto_ontology.semantic.semantic_fk._match_hit_by_sample_values",
    return_value="target-attr",
)
@patch("auto_ontology.semantic.semantic_fk._llm_pick_hit", return_value=None)
def test_vdb_resolution_uses_sql_fallback_when_llm_abstains(
    _mock_llm_pick: MagicMock,
    mock_sql_fallback: MagicMock,
) -> None:
    retriever = MagicMock()
    hit = _hit("target-attr", "table-2")
    retriever.query.side_effect = [[hit], []]
    column = {
        "id": "source-column",
        "name": "id",
        "table_id": "table-1",
        "table_name": "tags",
        "sample_values": '["72", "35"]',
    }
    connector = MagicMock()

    selected = semantic_fk._resolve_via_vdb(
        column,
        retriever,
        "database",
        connector,
    )

    assert selected == "target-attr"
    mock_sql_fallback.assert_called_once_with(column, [hit], connector)


def test_column_attribute_embedding_includes_physical_table() -> None:
    rows = _build_rows(
        "database",
        {},
        [
            {
                "id": "attr-1",
                "name": "Tag id",
                "term_name": "Tag",
                "source_column": "id",
                "table_id": "table-1",
                "table_name": "tags",
                "is_unique": True,
            }
        ],
    )

    assert rows[0]["metadata"]["table_id"] == "table-1"
    assert rows[0]["metadata"]["table_name"] == "tags"
    assert rows[0]["metadata"]["is_unique"] is True
    assert rows[0]["metadata"]["content_metadata"]["table_id"] == "table-1"
    assert rows[0]["metadata"]["content_metadata"]["table_name"] == "tags"
    assert rows[0]["metadata"]["content_metadata"]["is_unique"] is True


def _probe_result(*values: object, ok: bool = True) -> dict:
    return {
        "ok": ok,
        "rows": [{"matched_value": value} for value in values],
    }


def _run_sample_fallback(
    column: dict,
    hits: list[dict],
    *probe_results: dict,
) -> tuple[str | None, MagicMock]:
    executor = MagicMock()
    executor.__enter__.return_value = executor
    executor.run.side_effect = list(probe_results)
    with patch(
        "auto_ontology.semantic.semantic_fk.ProbeExecutor",
        return_value=executor,
    ):
        selected = semantic_fk._match_hit_by_sample_values(
            column,
            hits,
            MagicMock(dialect="postgresql"),
        )
    return selected, executor


def test_sql_fallback_selects_candidate_containing_all_samples() -> None:
    column = {"name": "id", "table_name": "tags", "sample_values": '["72", "35"]'}

    selected, executor = _run_sample_fallback(
        column,
        [_hit("torrent-id", "torrent-table", source_column="ID")],
        _probe_result(72, 35),
    )

    assert selected == "torrent-id"
    assert executor.run.call_count == 1


def test_sql_fallback_rejects_partial_or_missing_matches() -> None:
    column = {"name": "id", "table_name": "tags", "sample_values": '["72", "35"]'}

    selected, _executor = _run_sample_fallback(
        column,
        [
            _hit("partial", "table-2"),
            _hit("missing", "table-3"),
        ],
        _probe_result(72),
        _probe_result(),
    )

    assert selected is None


def test_sql_fallback_uses_best_vdb_score_when_multiple_candidates_match() -> None:
    column = {"name": "id", "table_name": "tags", "sample_values": '["72", "35"]'}

    selected, _executor = _run_sample_fallback(
        column,
        [
            _hit("weaker", "table-2", score=0.4),
            _hit("stronger", "table-3", score=0.1),
        ],
        _probe_result(72, 35),
        _probe_result(72, 35),
    )

    assert selected == "stronger"


def test_sql_fallback_continues_after_probe_failure() -> None:
    column = {"name": "id", "table_name": "tags", "sample_values": '["72", "35"]'}

    selected, _executor = _run_sample_fallback(
        column,
        [
            _hit("failed", "table-2"),
            _hit("matched", "table-3"),
        ],
        _probe_result(ok=False),
        _probe_result(72, 35),
    )

    assert selected == "matched"


@patch("auto_ontology.semantic.semantic_fk.ProbeExecutor")
def test_sql_fallback_skips_without_samples_or_connector(
    mock_executor: MagicMock,
) -> None:
    hit = _hit("target", "table-2")

    assert (
        semantic_fk._match_hit_by_sample_values(
            {"sample_values": None}, [hit], MagicMock()
        )
        is None
    )
    assert (
        semantic_fk._match_hit_by_sample_values(
            {"sample_values": '["72"]'}, [hit], None
        )
        is None
    )
    mock_executor.assert_not_called()


@patch("auto_ontology.semantic.semantic_fk.ProbeExecutor")
def test_sql_fallback_skips_candidates_with_different_column_names(
    mock_executor: MagicMock,
) -> None:
    column = {
        "name": "id",
        "table_name": "tags",
        "sample_values": '["1"]',
    }
    hit = _hit(
        "timestamp-attr",
        "table-2",
        source_column="lasteditedwhen",
    )

    assert semantic_fk._match_hit_by_sample_values(column, [hit], MagicMock()) is None
    mock_executor.assert_not_called()


def test_sample_match_sql_quotes_location_and_compares_as_text() -> None:
    sql = semantic_fk._sample_match_sql(
        {
            "schema_name": "music",
            "table_name": "torrents",
            "source_column": "id",
        },
        ["72", "35"],
        "postgresql",
    )

    assert sql is not None
    assert 'FROM "music"."torrents"' in sql
    assert "CAST(\"id\" AS TEXT) IN ('72', '35')" in sql
    assert sql.endswith("LIMIT 2")


def test_resolution_logs_candidate_progress(caplog: pytest.LogCaptureFixture) -> None:
    candidates = [
        {"id": "col-1", "name": "customer_id", "table_name": "orders"},
        {"id": "col-2", "name": "product_id", "table_name": "order_items"},
    ]
    caplog.set_level(logging.INFO, logger=semantic_fk.__name__)

    with (
        patch(
            "auto_ontology.semantic.semantic_fk.find_unlinked_fk_columns",
            return_value=candidates,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk._build_retriever",
            return_value=MagicMock(),
        ),
        patch(
            "auto_ontology.semantic.semantic_fk._resolve_connector", return_value=None
        ),
        patch("auto_ontology.semantic.semantic_fk._resolve_via_vdb", return_value=None),
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=[],
        ),
    ):
        assert semantic_fk.resolve_semantic_fks("database") == 0

    assert "Resolving semantic FK edges… (1/2): orders.customer_id" in caplog.messages
    assert (
        "Resolving semantic FK edges… (2/2): order_items.product_id" in caplog.messages
    )


def _join_group_row(
    column_id: str,
    column_name: str,
    *,
    target_attribute_id: str = "customer-attribute",
    target_table_id: str = "customers-table",
    target_table_name: str = "customers",
    target_column_name: str = "id",
) -> dict:
    return {
        "source_table_id": "orders-table",
        "source_table_name": "orders",
        "source_table_description": "Customer orders.",
        "source_column_id": column_id,
        "source_column_name": column_name,
        "source_column_description": f"The {column_name} role.",
        "target_attribute_id": target_attribute_id,
        "target_attribute_name": "Customer ID",
        "target_table_id": target_table_id,
        "target_table_name": target_table_name,
        "target_table_description": "Customers.",
        "target_column_id": "customer-id",
        "target_column_name": target_column_name,
        "target_column_description": "Customer identifier.",
        "use_in_join_paths": True,
    }


def test_single_semantic_fk_is_default_without_llm() -> None:
    group = [_join_group_row("customer-id-column", "customer_id")]
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=group,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.set_semantic_fk_join_default"
        ) as set_default,
        patch("auto_ontology.semantic.semantic_fk.get_llm_client") as get_llm_client,
    ):
        assert semantic_fk.select_semantic_fk_join_defaults("database") == 1

    get_llm_client.assert_not_called()
    set_default.assert_called_once_with(
        "orders-table",
        "customers-table",
        "customer-id-column",
    )


@pytest.mark.parametrize(
    ("target_table_name", "general_name", "qualified_name"),
    [
        ("Player", "player_api_id", "player_fifa_api_id"),
        ("Team", "team_api_id", "team_fifa_api_id"),
        ("customers", "customer_id", "billing_customer_id"),
    ],
)
def test_generic_identifier_relationship_outranks_qualified_variant(
    target_table_name: str,
    general_name: str,
    qualified_name: str,
) -> None:
    group = [
        _join_group_row(
            "qualified-column",
            qualified_name,
            target_attribute_id="qualified-attribute",
            target_table_id="entity-table",
            target_table_name=target_table_name,
            target_column_name=qualified_name,
        ),
        _join_group_row(
            "general-column",
            general_name,
            target_attribute_id="general-attribute",
            target_table_id="entity-table",
            target_table_name=target_table_name,
            target_column_name=general_name,
        ),
    ]

    assert semantic_fk._select_generic_join_default(group) == "general-column"


def test_peer_roles_have_no_deterministic_generic_relationship() -> None:
    group = [
        _join_group_row("home-column", "home_customer_id"),
        _join_group_row("away-column", "away_customer_id"),
    ]

    assert semantic_fk._select_generic_join_default(group) is None


def test_bare_identifier_does_not_outrank_entity_identifier() -> None:
    group = [
        _join_group_row(
            "bare-column",
            "api_id",
            target_table_name="Player",
            target_column_name="api_id",
        ),
        _join_group_row(
            "player-column",
            "player_api_id",
            target_table_name="Player",
            target_column_name="player_api_id",
        ),
    ]

    assert semantic_fk._select_generic_join_default(group) is None


def test_generic_relationship_is_selected_without_initializing_llm() -> None:
    group = [
        _join_group_row(
            "fifa-column",
            "player_fifa_api_id",
            target_attribute_id="fifa-attribute",
            target_table_id="players-table",
            target_table_name="Player",
            target_column_name="player_fifa_api_id",
        ),
        _join_group_row(
            "api-column",
            "player_api_id",
            target_attribute_id="api-attribute",
            target_table_id="players-table",
            target_table_name="Player",
            target_column_name="player_api_id",
        ),
    ]
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=group,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.set_semantic_fk_join_default"
        ) as set_default,
        patch("auto_ontology.semantic.semantic_fk.get_llm_client") as get_llm_client,
        patch(
            "auto_ontology.semantic.semantic_fk.invoke_with_structured_output"
        ) as invoke_llm,
    ):
        semantic_fk.select_semantic_fk_join_defaults("database")

    get_llm_client.assert_not_called()
    invoke_llm.assert_not_called()
    set_default.assert_called_once_with(
        "orders-table",
        "players-table",
        "api-column",
    )


def test_llm_selects_default_from_parallel_roles() -> None:
    group = [
        _join_group_row("home-column", "home_customer_id"),
        _join_group_row("away-column", "away_customer_id"),
    ]
    result = SemanticFkJoinDefaultSelection(
        selected_column_id="home-column",
        rationale="The home relationship is the likely default.",
    )
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=group,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.get_llm_client",
            return_value=object(),
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.invoke_with_structured_output",
            return_value=result,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.set_semantic_fk_join_default"
        ) as set_default,
    ):
        semantic_fk.select_semantic_fk_join_defaults("database")

    set_default.assert_called_once_with(
        "orders-table",
        "customers-table",
        "home-column",
    )


def test_invalid_llm_selection_uses_deterministic_default() -> None:
    group = [
        _join_group_row("z-column", "z_customer_id"),
        _join_group_row("a-column", "a_customer_id"),
    ]
    result = SemanticFkJoinDefaultSelection(
        selected_column_id="invented-column",
        rationale="Invalid.",
    )
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=group,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.get_llm_client",
            return_value=object(),
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.invoke_with_structured_output",
            return_value=result,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.set_semantic_fk_join_default"
        ) as set_default,
    ):
        semantic_fk.select_semantic_fk_join_defaults("database")

    set_default.assert_called_once_with(
        "orders-table",
        "customers-table",
        "a-column",
    )


def test_missing_llm_uses_deterministic_default() -> None:
    group = [
        _join_group_row("z-column", "z_customer_id"),
        _join_group_row("a-column", "a_customer_id"),
    ]
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=group,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.get_llm_client",
            side_effect=EnvironmentError("not configured"),
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.set_semantic_fk_join_default"
        ) as set_default,
    ):
        semantic_fk.select_semantic_fk_join_defaults("database")

    set_default.assert_called_once_with(
        "orders-table",
        "customers-table",
        "a-column",
    )


def test_resolution_always_normalizes_join_defaults() -> None:
    with (
        patch(
            "auto_ontology.semantic.semantic_fk._resolve_semantic_fks",
            return_value=3,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.select_semantic_fk_join_defaults"
        ) as select_defaults,
    ):
        assert semantic_fk.resolve_semantic_fks("database") == 3

    select_defaults.assert_called_once_with("database")
