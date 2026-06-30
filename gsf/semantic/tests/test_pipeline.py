"""Tests for the flat table-by-table compile_semantic_layer loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gsf.semantic.pipeline import compile_semantic_layer


@patch("gsf.semantic.pipeline.process_table")
@patch("gsf.semantic.pipeline.fetch_table_context")
@patch("gsf.semantic.pipeline.load_domain_summary", return_value=None)
@patch("gsf.semantic.pipeline.fetch_all_tables_without_term")
def test_compile_processes_all_tables(
    mock_fetch_tables: MagicMock,
    _mock_summary: MagicMock,
    mock_fetch_ctx: MagicMock,
    mock_process: MagicMock,
) -> None:
    """Two tables → two process_table calls → count == 2."""
    mock_fetch_tables.return_value = [
        {"id": "t1", "name": "orders", "description": ""},
        {"id": "t2", "name": "customers", "description": ""},
    ]
    mock_fetch_ctx.side_effect = [
        {"columns": [{"name": "amount"}], "fks": []},
        {"columns": [{"name": "name"}], "fks": []},
    ]

    count = compile_semantic_layer("dbx")

    assert count == 2
    assert mock_process.call_count == 2


@patch("gsf.semantic.pipeline.process_table")
@patch("gsf.semantic.pipeline.fetch_table_context")
@patch("gsf.semantic.pipeline.load_domain_summary", return_value=None)
@patch("gsf.semantic.pipeline.fetch_all_tables_without_term")
def test_compile_skips_columnless_table(
    mock_fetch_tables: MagicMock,
    _mock_summary: MagicMock,
    mock_fetch_ctx: MagicMock,
    mock_process: MagicMock,
) -> None:
    """Table with no columns is skipped without calling process_table."""
    mock_fetch_tables.return_value = [
        {"id": "t1", "name": "empty_table", "description": ""}
    ]
    mock_fetch_ctx.return_value = {"columns": [], "fks": []}

    count = compile_semantic_layer("dbx")

    assert count == 0
    mock_process.assert_not_called()


@patch("gsf.semantic.pipeline.process_table", side_effect=RuntimeError("boom"))
@patch("gsf.semantic.pipeline.fetch_table_context")
@patch("gsf.semantic.pipeline.load_domain_summary", return_value=None)
@patch("gsf.semantic.pipeline.fetch_all_tables_without_term")
def test_compile_continues_after_error(
    mock_fetch_tables: MagicMock,
    _mock_summary: MagicMock,
    mock_fetch_ctx: MagicMock,
    _mock_process: MagicMock,
) -> None:
    """process_table raising an exception should not abort the loop."""
    mock_fetch_tables.return_value = [{"id": "t1", "name": "orders", "description": ""}]
    mock_fetch_ctx.return_value = {"columns": [{"name": "x"}], "fks": []}

    count = compile_semantic_layer("dbx")

    assert count == 0
