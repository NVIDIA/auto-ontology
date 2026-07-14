"""Tests for deterministic fallbacks in FK suggestion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gsf.semantic.fk_suggester import suggest_potential_foreign_keys
from gsf.semantic.models import FkAndPkResult


@patch("gsf.semantic.fk_suggester.get_llm_client", return_value=MagicMock())
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


@patch("gsf.semantic.fk_suggester.get_llm_client", return_value=MagicMock())
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
