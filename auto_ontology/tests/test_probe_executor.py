from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

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
