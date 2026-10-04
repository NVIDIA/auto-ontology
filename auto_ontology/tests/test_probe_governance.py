# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-movement policy tests for live database probes."""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd

from auto_ontology.retrieval.text_to_sql.db_probe.executor import (
    ProbeExecutor,
    allowed_table_scope,
)


def _connector(frame: pd.DataFrame) -> MagicMock:
    connector = MagicMock()
    connector.database_name = "warehouse"
    connector.dialect = "postgres"
    connector.execute.return_value = frame
    return connector


def test_probe_clamps_existing_limit_and_redacts_audit(
    caplog,
) -> None:
    connector = _connector(
        pd.DataFrame(
            {
                "a": range(10),
                "b": range(10),
                "c": range(10),
            }
        )
    )
    executor = ProbeExecutor(
        connector,
        max_rows=2,
        max_result_columns=2,
    )

    with caplog.at_level(logging.INFO):
        result = executor.run(
            "SELECT a, b, c FROM events WHERE a = 'private-canary' LIMIT 999",
            purpose="test",
        )

    executed = connector.execute.call_args.args[0]
    assert executed.endswith("LIMIT 2")
    assert "_bounded_probe" in executed
    assert result["rows"] == [{"a": 0, "b": 0}, {"a": 1, "b": 1}]
    assert result["truncated"] is True
    assert executor.log[0]["rows"] is None
    assert executor.log[0]["sql"] is None
    assert executor.log[0]["purpose"] == "test"
    assert "private-canary" not in caplog.text


@patch(
    "auto_ontology.retrieval.text_to_sql.db_probe.executor."
    "is_column_safe_for_data_movement",
    return_value=False,
)
def test_probe_blocks_pii_or_unprocessed_column(mock_safe: MagicMock) -> None:
    connector = _connector(pd.DataFrame({"email": ["person@example.com"]}))
    executor = ProbeExecutor(
        connector,
        enforce_data_policy=True,
        database_name="warehouse",
        allowed_tables=frozenset({"public.customers"}),
    )

    result = executor.run(
        "SELECT email FROM public.customers",
        purpose="literal_check",
    )

    assert result["ok"] is False
    assert result["error"] == "rejected: column is PII, unprocessed, or unresolved"
    assert executor.log[0]["error"] == "failed"
    assert executor.log[0]["sql"] is None
    connector.execute.assert_not_called()
    mock_safe.assert_called_once_with(
        database_name="warehouse",
        schema_name="public",
        table_name="customers",
        column_name="email",
    )


def test_probe_blocks_table_outside_authorized_scope() -> None:
    connector = _connector(pd.DataFrame({"secret": ["value"]}))
    executor = ProbeExecutor(
        connector,
        enforce_data_policy=True,
        database_name="warehouse",
        allowed_tables=frozenset({"orders"}),
    )

    result = executor.run("SELECT secret FROM payroll", purpose="literal_check")

    assert result["error"] == "rejected: table is outside the authorized query scope"
    connector.execute.assert_not_called()


@patch(
    "auto_ontology.retrieval.text_to_sql.db_probe.executor."
    "is_column_safe_for_data_movement",
    return_value=True,
)
def test_probe_blocks_bare_wildcard_projection(mock_safe: MagicMock) -> None:
    connector = _connector(pd.DataFrame({"email": ["private-canary"]}))
    executor = ProbeExecutor(
        connector,
        enforce_data_policy=True,
        database_name="warehouse",
        allowed_tables=frozenset({"public.customers"}),
    )

    for sql in (
        "SELECT * FROM public.customers",
        "SELECT c.* FROM public.customers AS c",
    ):
        result = executor.run(sql, purpose="literal_check")
        assert result["error"] == "rejected: wildcard probes are not permitted"
    connector.execute.assert_not_called()
    mock_safe.assert_not_called()


@patch(
    "auto_ontology.retrieval.text_to_sql.db_probe.executor."
    "is_column_safe_for_data_movement",
    return_value=True,
)
def test_probe_allows_count_star(_mock_safe: MagicMock) -> None:
    connector = _connector(pd.DataFrame({"n": [3]}))
    executor = ProbeExecutor(
        connector,
        enforce_data_policy=True,
        database_name="warehouse",
        allowed_tables=frozenset({"public.customers"}),
    )

    result = executor.run(
        "SELECT COUNT(*) AS n FROM public.customers WHERE status = 'open'",
        purpose="literal_check",
    )

    assert result["ok"] is True
    connector.execute.assert_called_once()


def test_probe_blocks_table_in_another_catalog() -> None:
    connector = _connector(pd.DataFrame({"status": ["open"]}))
    executor = ProbeExecutor(
        connector,
        enforce_data_policy=True,
        database_name="warehouse",
        allowed_tables=frozenset({"public.customers"}),
    )

    result = executor.run(
        "SELECT status FROM other.public.customers",
        purpose="literal_check",
    )

    assert result["error"] == "rejected: table is outside the authorized query scope"
    connector.execute.assert_not_called()


def test_table_scope_keeps_duplicate_names_schema_qualified() -> None:
    scope = allowed_table_scope(
        [
            {"name": "events", "schema_name": "public"},
            {"name": "events", "schema_name": "restricted"},
            {"name": "orders", "schema_name": "public"},
        ]
    )

    assert scope == frozenset(
        {"public.events", "restricted.events", "public.orders", "orders"}
    )


def test_probe_blocks_same_table_name_in_unauthorized_schema() -> None:
    connector = _connector(pd.DataFrame({"email": ["private-canary"]}))
    executor = ProbeExecutor(
        connector,
        enforce_data_policy=True,
        database_name="warehouse",
        allowed_tables=frozenset({"public.customers"}),
    )

    result = executor.run(
        "SELECT email FROM restricted.customers",
        purpose="literal_check",
    )

    assert result["error"] == "rejected: table is outside the authorized query scope"
    connector.execute.assert_not_called()
