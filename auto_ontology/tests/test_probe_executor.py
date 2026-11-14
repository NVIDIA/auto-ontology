from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor


def test_tsql_probe_rewrites_explicit_limit_to_top() -> None:
    connector = MagicMock(dialect="tsql", database_name="db")
    connector.execute.return_value = pd.DataFrame()

    result = ProbeExecutor(connector, max_rows=7).run(
        "SELECT DISTINCT [id] FROM [dbo].[orders] LIMIT 3"
    )

    assert result["ok"] is True
    sql = connector.execute.call_args.args[0]
    assert sql.startswith("SELECT DISTINCT TOP 3 ")
    assert "LIMIT" not in sql


def test_tsql_probe_adds_top_when_limit_is_missing() -> None:
    connector = MagicMock(dialect="tsql", database_name="db")
    connector.execute.return_value = pd.DataFrame()

    ProbeExecutor(connector, max_rows=7).run("SELECT [id] FROM [dbo].[orders]")

    sql = connector.execute.call_args.args[0]
    assert sql.startswith("SELECT TOP 7 ")
    assert "LIMIT" not in sql


def test_tsql_probe_clamps_existing_top_to_max_rows() -> None:
    connector = MagicMock(dialect="tsql", database_name="db")
    connector.execute.return_value = pd.DataFrame()

    result = ProbeExecutor(connector, max_rows=7).run(
        "SELECT TOP 1000000 [id] FROM [dbo].[orders]"
    )

    assert result["ok"] is True
    assert connector.execute.call_args.args[0].startswith("SELECT TOP 7 ")


def test_tsql_probe_clamps_fetch_next_to_max_rows() -> None:
    connector = MagicMock(dialect="tsql", database_name="db")
    connector.execute.return_value = pd.DataFrame()

    result = ProbeExecutor(connector, max_rows=7).run(
        "SELECT [id] FROM [dbo].[orders] ORDER BY [id] "
        "OFFSET 0 ROWS FETCH NEXT 100000 ROWS ONLY"
    )

    assert result["ok"] is True
    assert "FETCH NEXT 7 ROWS ONLY" in connector.execute.call_args.args[0]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TOP 20 PERCENT [id] FROM [dbo].[orders]",
        "SELECT TOP 20 WITH TIES [id] FROM [dbo].[orders] ORDER BY [id]",
    ],
    ids=["percent", "with-ties"],
)
def test_tsql_probe_rejects_non_hard_row_caps(sql: str) -> None:
    connector = MagicMock(dialect="tsql", database_name="db")

    result = ProbeExecutor(connector, max_rows=7).run(sql)

    assert result["ok"] is False
    assert "hard row cap" in result["error"]
    connector.execute.assert_not_called()


def test_tsql_probe_rejects_sqlglot_parse_failure_instead_of_running_uncapped() -> None:
    connector = MagicMock(dialect="tsql", database_name="db")

    result = ProbeExecutor(connector, max_rows=7).run(
        "SELECT [id] FROM [dbo].[orders] OPTION (USE HINT ('DISABLE_PARAMETER_SNIFFING'))"
    )

    assert result["ok"] is False
    assert "could not apply SQL Server row cap" in result["error"]
    connector.execute.assert_not_called()
