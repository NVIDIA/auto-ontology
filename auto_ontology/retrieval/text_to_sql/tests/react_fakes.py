# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fakes shared by the ReAct agent tests: a catalog, a SQLite database, a model."""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage


class FakeCatalog:
    """Table name → columns, standing in for the catalog's Schema objects."""

    def __init__(self, tables: dict[str, list[str]]):
        self._tables = {name.lower(): cols for name, cols in tables.items()}

    def columns(self, table: str, schema: str | None) -> list[str] | None:
        return self._tables.get(table.lower())


class SQLiteConnector:
    """An in-memory SQLite database behind the connector interface the tools use."""

    dialect = "sqlite"
    database_name = "shop"

    def __init__(self, script: str):
        self._conn = sqlite3.connect(":memory:")
        self._conn.executescript(script)
        self.executed: list[str] = []

    def execute(self, sql: str) -> pd.DataFrame:
        self.executed.append(sql)
        return pd.read_sql_query(sql, self._conn)


SHOP_SCRIPT = """
CREATE TABLE customers (customer_id INTEGER, status TEXT);
CREATE TABLE orders (order_id INTEGER, cust_ref INTEGER, amount REAL, discount REAL);
INSERT INTO customers VALUES (1, 'ACTIVE'), (2, 'CHURNED'), (3, 'ACTIVE');
INSERT INTO orders VALUES (10, 1, 100, 10), (11, 1, 50, 0), (12, 2, 80, 0), (13, 3, 20, 0);
"""

SHOP_CATALOG = {
    "customers": ["customer_id", "status"],
    "orders": ["order_id", "cust_ref", "amount", "discount"],
}


def call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class ScriptedModel:
    """A chat model that replies with a fixed sequence of messages.

    An exception in the sequence is raised instead of returned.

    Records which tools were bound for each turn and what it was sent, so a
    test can assert what the model saw.
    """

    def __init__(self, replies: list[AIMessage | Exception]):
        self._replies = list(replies)
        self.bound_tool_names: list[list[str]] = []
        self.seen: list[list[Any]] = []
        self._current: list[str] = []

    def bind_tools(self, tools: list[dict]) -> "ScriptedModel":
        self._current = [t["function"]["name"] for t in tools]
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.bound_tool_names.append(self._current)
        self.seen.append(list(messages))
        if not self._replies:
            return AIMessage(content="I am done.")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
