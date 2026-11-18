# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic PII canaries through semantic compilation of one table.

A real SQLite warehouse holds canary values in PII-tagged columns, and the
catalog context carries a canary persisted before the tag existed. The whole of
``process_table`` runs against it -- row sample, DISTINCT probes, key probes --
with every place a value can leave recorded: warehouse results, LLM prompts,
and catalog writes. No canary may appear in any of them.

Each test also asserts that a non-PII value *does* move, so a run that samples
nothing at all cannot pass for one that withholds the right columns.
"""

from __future__ import annotations

import sqlite3
from contextlib import ExitStack
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from auto_ontology.semantic.models import PotentialFkResult, TableTermsResult
from auto_ontology.semantic.visit_enter import process_table, reset_sampling_breaker
from auto_ontology.utils.sql_identifiers import qualified_name

CANARY = "pii-canary-7f3a9c"
STALE_CANARY = "pii-canary-stale-41d2"

_MODULE = "auto_ontology.semantic.visit_enter"
_SINKS = (
    "store_column_sample_values",
    "store_column_distinct_values",
    "store_column_uniqueness",
    "store_column_distinct_ratios",
    "store_column_date_formats",
    "clear_column_values",
    "mark_table_as_junction",
    "upsert_table_term",
    "merge_column_attribute",
    "fill_missing_column_descriptions",
)


class _SqliteWarehouse:
    """A connector over in-memory SQLite that records every result it returns."""

    dialect = "sqlite"
    database_name = "canary_db"

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE users (id INTEGER, email TEXT, phone TEXT, status TEXT)"
        )
        self.conn.executemany(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            [
                (i, f"{CANARY}-{i}@example.invalid", f"{CANARY}-555", status)
                for i, status in enumerate(["open", "open", "closed"] * 4)
            ],
        )
        self.statements: list[str] = []
        self.results: list[pd.DataFrame] = []

    def qualify(self, schema: str, table: str) -> str:
        return qualified_name(schema, table, dialect=self.dialect)

    def execute(self, sql: str, parameters: Any = None) -> pd.DataFrame:
        self.statements.append(sql)
        df = pd.read_sql_query(sql, self.conn)
        self.results.append(df)
        return df


TABLE = {
    "id": "t-users",
    "name": "users",
    "description": "",
    "schema_name": "main",
    "pk": ["email"],
}


def _ctx() -> dict[str, Any]:
    return {
        "columns": [
            {"id": "c-id", "name": "id", "data_type": "integer"},
            {
                "id": "c-email",
                "name": "email",
                "data_type": "text",
                "sample_values": f'["{STALE_CANARY}@example.invalid"]',
            },
            {"id": "c-phone", "name": "phone", "data_type": "text"},
            {"id": "c-status", "name": "status", "data_type": "text"},
        ],
        "fks": [],
    }


@pytest.fixture(autouse=True)
def _reset_breaker():
    reset_sampling_breaker(_SqliteWarehouse.database_name)
    yield
    reset_sampling_breaker(_SqliteWarehouse.database_name)


def _compile(
    warehouse: _SqliteWarehouse, pii_lookup: MagicMock
) -> dict[str, MagicMock]:
    """Run ``process_table`` over *warehouse*; return every recorded sink."""
    prompts: dict[str, MagicMock] = {
        # The LLM is baited with the PII columns as keys, so only the PII
        # guard -- not the model -- keeps their values from being probed.
        "suggest_potential_foreign_keys": MagicMock(
            return_value=PotentialFkResult(pk_column_names=["email", "phone", "id"])
        ),
        "extract_term": MagicMock(return_value=TableTermsResult(terms=[])),
    }
    sinks = {name: MagicMock() for name in _SINKS}
    with ExitStack() as stack:
        stack.enter_context(
            patch(f"{_MODULE}._resolve_connector", return_value=warehouse)
        )
        stack.enter_context(patch(f"{_MODULE}.pii_tagged_column_ids", pii_lookup))
        for name, mock in {**prompts, **sinks}.items():
            stack.enter_context(patch(f"{_MODULE}.{name}", mock))
        process_table(TABLE, _ctx(), domain_summary=None, database_name="canary_db")
    return {**prompts, **sinks}


def _leaks(
    warehouse: _SqliteWarehouse, recorded: dict[str, MagicMock], canary: str
) -> list[str]:
    places = [
        f"warehouse result {i}"
        for i, df in enumerate(warehouse.results)
        if canary in df.to_string()
    ]
    places += [
        name for name, mock in recorded.items() if canary in repr(mock.call_args_list)
    ]
    return places


def test_pii_canaries_never_leave_the_warehouse() -> None:
    warehouse = _SqliteWarehouse()

    recorded = _compile(warehouse, MagicMock(return_value={"c-email", "c-phone"}))

    assert _leaks(warehouse, recorded, CANARY) == []
    assert _leaks(warehouse, recorded, STALE_CANARY) == []
    assert not any("email" in sql or "phone" in sql for sql in warehouse.statements), (
        warehouse.statements
    )
    recorded["clear_column_values"].assert_called_once_with(
        "t-users", ["email", "phone"]
    )
    # Control: the untagged columns were sampled and probed as usual.
    assert "open" in repr(recorded["store_column_sample_values"].call_args_list)
    assert "id" in recorded["store_column_distinct_values"].call_args[0][1]


def test_pii_canaries_stay_put_when_the_tags_cannot_be_read() -> None:
    warehouse = _SqliteWarehouse()

    recorded = _compile(warehouse, MagicMock(side_effect=RuntimeError("store down")))

    assert warehouse.statements == []
    assert _leaks(warehouse, recorded, CANARY) == []
    assert _leaks(warehouse, recorded, STALE_CANARY) == []
    # Profiles persisted by earlier runs may hold PII; with no tags, all go.
    recorded["clear_column_values"].assert_called_once_with(
        "t-users", ["email", "id", "phone", "status"]
    )


def test_canaries_do_move_without_a_pii_tag() -> None:
    """The harness can see a leak: untagged, the same canaries are everywhere."""
    warehouse = _SqliteWarehouse()

    recorded = _compile(warehouse, MagicMock(return_value=set()))

    leaks = _leaks(warehouse, recorded, CANARY)
    assert "store_column_sample_values" in leaks
    assert "store_column_distinct_values" in leaks
    assert "suggest_potential_foreign_keys" in leaks
    assert "suggest_potential_foreign_keys" in _leaks(warehouse, recorded, STALE_CANARY)
