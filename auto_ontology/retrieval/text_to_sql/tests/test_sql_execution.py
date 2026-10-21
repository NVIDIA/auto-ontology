# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pandas as pd

from auto_ontology.retrieval.text_to_sql.agents import sql_execution


def test_duplicate_result_columns_are_deduplicated_before_serialization(
    monkeypatch,
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


def test_unique_result_columns_remain_unchanged(monkeypatch) -> None:
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


def test_metadata_errors_are_returned_as_query_errors(monkeypatch) -> None:
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
