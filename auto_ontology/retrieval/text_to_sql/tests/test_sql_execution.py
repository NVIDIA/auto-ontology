# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from auto_ontology.connectors.sqlite import SQLiteDatabase
from auto_ontology.retrieval.text_to_sql.agents import sql_execution


def test_duplicate_result_columns_are_deduplicated_before_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame([[1, 2, 3]], columns=["ID", "ID", "ID__2"])
    monkeypatch.setattr(
        sql_execution, "execute_chat_sql", lambda _connector, _sql: frame
    )

    response = sql_execution._run_sql(
        "SELECT first_id, second_id, third_id",
        SimpleNamespace(dialect="postgres"),
        record_numeric_kinds=True,
    )

    assert response.error is None
    assert json.loads(response.result[0]) == [{"ID": 1, "ID__3": 2, "ID__2": 3}]
    assert response.numeric_kinds == [None, None, None]
    assert response.values_digest is not None
    assert list(frame.columns) == ["ID", "ID", "ID__2"]


def test_unique_result_columns_remain_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame([[1, "Ada"]], columns=["ID", "Name"])
    monkeypatch.setattr(
        sql_execution, "execute_chat_sql", lambda _connector, _sql: frame
    )

    response = sql_execution._run_sql(
        "SELECT id, name",
        SimpleNamespace(dialect="postgres"),
    )

    assert response.error is None
    assert json.loads(response.result[0]) == [{"ID": 1, "Name": "Ada"}]


def test_metadata_errors_are_returned_as_query_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame([[1]], columns=["ID"])
    monkeypatch.setattr(
        sql_execution, "execute_chat_sql", lambda _connector, _sql: frame
    )
    monkeypatch.setattr(
        sql_execution,
        "numeric_column_kinds",
        lambda _frame: (_ for _ in ()).throw(ValueError("metadata failed")),
    )

    response = sql_execution._run_sql(
        "SELECT id",
        SimpleNamespace(dialect="postgres"),
        record_numeric_kinds=True,
    )

    assert response.error == "metadata failed"
    assert response.result is None


def test_sqlite_connector_preserves_duplicate_column_values_through_run_sql(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "shop.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (id INT, name TEXT);
        CREATE TABLE orders (id INT, customer_id INT, name TEXT);
        INSERT INTO customers VALUES (1, 'Ada');
        INSERT INTO orders VALUES (100, 1, 'first order');
        """
    )
    conn.close()

    connector = SQLiteDatabase(f"sqlite:///{db_path}")
    try:
        sql = (
            "SELECT c.id, o.id, c.name, o.name "
            "FROM customers c JOIN orders o ON o.customer_id = c.id"
        )
        frame = connector.execute(sql)
        assert list(frame.columns) == ["id", "id", "name", "name"]
        assert frame.values.tolist() == [[1, 100, "Ada", "first order"]]

        response = sql_execution._run_sql(sql, connector)
        assert response.error is None
        assert json.loads(response.result[0]) == [
            {"id": 1, "id__2": 100, "name": "Ada", "name__2": "first order"}
        ]
    finally:
        connector.close()
