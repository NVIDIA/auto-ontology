"""Tests for deterministic fallbacks in FK suggestion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gsf.semantic.fk_suggester import (
    _format_sample_values,
    suggest_potential_foreign_keys,
)
from gsf.semantic.models import FkAndPkResult


@patch(
    "gsf.semantic.fk_suggester.get_non_reasoning_llm_client", return_value=MagicMock()
)
@patch(
    "gsf.semantic.fk_suggester.invoke_with_structured_output",
    return_value=FkAndPkResult(fk_suggestions=[], pk_column_names=[]),
)
def test_non_unique_uuid_columns_suggested(
    _mock_invoke: MagicMock,
    _mock_client: MagicMock,
) -> None:
    table = {"id": "t1", "name": "events", "schema_name": "public"}
    ctx = {
        "columns": [
            {"name": "owner_uuid", "data_type": "uuid"},
            {"name": "external_uuid", "data_type": "uuid"},
            {"name": "vendor_code", "data_type": "varchar"},
        ],
        "fks": [],
    }
    profiling = {
        "owner_uuid": {"sample_values": ["u1", "u1", "u2"], "is_unique": False},
        "external_uuid": {"sample_values": ["u1", "u2"], "is_unique": True},
        "vendor_code": {"sample_values": ["A", "A", "B"], "is_unique": False},
    }

    result = suggest_potential_foreign_keys(table, ctx, profiling)
    suggested = {s.column_name for s in result.suggestions}

    # Non-unique uuid column is suggested by the fallback.
    assert "owner_uuid" in suggested
    # Unique uuid column is not (likely a key, not a many-to-one FK).
    assert "external_uuid" not in suggested
    # Non-uuid types are left to the LLM, not the deterministic fallback.
    assert "vendor_code" not in suggested

    messages = _mock_invoke.call_args.args[1]
    prompt = messages[1].content
    assert "owner_uuid (uuid) [is_unique: false]" in prompt
    assert "external_uuid (uuid) [is_unique: true]" in prompt
    assert "vendor_code (varchar) [is_unique: false]" in prompt


@patch(
    "gsf.semantic.fk_suggester.get_non_reasoning_llm_client", return_value=MagicMock()
)
@patch(
    "gsf.semantic.fk_suggester.invoke_with_structured_output",
    return_value=FkAndPkResult(fk_suggestions=[], pk_column_names=["owner_uuid"]),
)
def test_non_unique_uuid_cannot_be_inferred_as_primary_key(
    _mock_invoke: MagicMock,
    _mock_client: MagicMock,
) -> None:
    table = {"id": "t1", "name": "events", "schema_name": "public"}
    ctx = {
        "columns": [{"name": "owner_uuid", "data_type": "uuid"}],
        "fks": [],
    }
    profiling = {
        "owner_uuid": {"sample_values": ["u1", "u1"], "is_unique": False},
    }

    result = suggest_potential_foreign_keys(table, ctx, profiling)

    assert [suggestion.column_name for suggestion in result.suggestions] == [
        "owner_uuid"
    ]


@patch(
    "gsf.semantic.fk_suggester.get_non_reasoning_llm_client", return_value=MagicMock()
)
@patch(
    "gsf.semantic.fk_suggester.invoke_with_structured_output",
    return_value=FkAndPkResult(fk_suggestions=[], pk_column_names=[]),
)
def test_uuid_without_profiling_not_suggested(
    _mock_invoke: MagicMock,
    _mock_client: MagicMock,
) -> None:
    table = {"id": "t1", "name": "events", "schema_name": "public"}
    ctx = {
        "columns": [{"name": "owner_uuid", "data_type": "uuid"}],
        "fks": [],
    }

    # No profiling -> is_unique unknown -> not suggested by the fallback.
    result = suggest_potential_foreign_keys(table, ctx, None)
    assert result.suggestions == []

    messages = _mock_invoke.call_args.args[1]
    assert "is_unique:" not in messages[1].content


def test_format_sample_values_handles_legacy_json_string_and_native_list() -> None:
    # Legacy Column nodes still store sample_values as a JSON string.
    assert _format_sample_values('["a", "b", "b"]') == "samples: a, b, b"
    # Current writers store a native list.
    assert _format_sample_values(["a", "b", "b"]) == "samples: a, b, b"
    assert _format_sample_values(None) == ""
    assert _format_sample_values([]) == ""
    assert _format_sample_values(["a", None, "b"]) == "samples: a, b"
    assert _format_sample_values('["a", null, "b"]') == "samples: a, b"


@patch(
    "gsf.semantic.fk_suggester.get_non_reasoning_llm_client", return_value=MagicMock()
)
@patch(
    "gsf.semantic.fk_suggester.invoke_with_structured_output",
    return_value=FkAndPkResult(fk_suggestions=[], pk_column_names=[]),
)
def test_candidate_column_samples_included_regardless_of_stored_shape(
    _mock_invoke: MagicMock,
    _mock_client: MagicMock,
) -> None:
    table = {"id": "t1", "name": "events", "schema_name": "public"}
    ctx = {
        "columns": [
            # Legacy JSON-string shape.
            {
                "name": "legacy_col",
                "data_type": "varchar",
                "sample_values": '["x", "y"]',
            },
            # Current native-list shape.
            {
                "name": "current_col",
                "data_type": "varchar",
                "sample_values": ["x", "y"],
            },
        ],
        "fks": [],
    }

    suggest_potential_foreign_keys(table, ctx)

    prompt = _mock_invoke.call_args.args[1][1].content
    assert "legacy_col (varchar) [samples: x, y]" in prompt
    assert "current_col (varchar) [samples: x, y]" in prompt


@patch(
    "gsf.semantic.fk_suggester.get_non_reasoning_llm_client", return_value=MagicMock()
)
@patch(
    "gsf.semantic.fk_suggester.invoke_with_structured_output",
    return_value=FkAndPkResult(fk_suggestions=[], pk_column_names=[]),
)
def test_declared_fk_target_is_not_suggested(
    mock_invoke: MagicMock,
    _mock_client: MagicMock,
) -> None:
    table = {"id": "players", "name": "players", "schema_name": "public"}
    ctx = {
        "columns": [
            {
                "name": "playerID",
                "data_type": "varchar",
                "is_foreign_key_target": True,
            },
            {"name": "team_id", "data_type": "integer"},
        ],
        "fks": [],
    }

    result = suggest_potential_foreign_keys(table, ctx)

    assert result.suggestions == []
    prompt = mock_invoke.call_args.args[1][1].content
    assert "playerID" not in prompt
    assert "team_id" in prompt
