# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dialect-safe value lookup SQL used by typed entity grounding."""

from types import SimpleNamespace
from typing import Any

from auto_ontology.retrieval.text_to_sql.db_probe.value_lookup import lookup_text_value


class _Executor:
    def __init__(self, stored_value: str) -> None:
        self.stored_value = stored_value
        self.calls: list[tuple[str, str]] = []

    def run(self, sql: str, purpose: str = "") -> dict[str, Any]:
        self.calls.append((sql, purpose))
        return {"ok": True, "rows": [{"value": self.stored_value}]}


class _Connector(SimpleNamespace):
    def qualify(self, schema_name: str | None, table_name: str) -> str:
        return f'"{schema_name}"."{table_name}"'


def test_equal_lookup_is_case_insensitive_and_returns_stored_spelling() -> None:
    executor = _Executor("Angel of Mercy")
    connector = _Connector(dialect="sqlite")

    result = lookup_text_value(
        executor,
        connector,
        schema_name="main",
        table_name="cards",
        column_name="name",
        value="angel OF mercy",
        match_kind="equal",
    )

    assert result == "Angel of Mercy"
    sql, purpose = executor.calls[0]
    assert 'LOWER(TRIM(CAST("name" AS TEXT)))' in sql
    assert "LOWER(TRIM('angel OF mercy'))" in sql
    assert purpose == "typed_entity_value_equal"


def test_contains_lookup_escapes_like_wildcards() -> None:
    executor = _Executor("100%_complete")
    connector = _Connector(dialect="sqlite")

    lookup_text_value(
        executor,
        connector,
        schema_name="main",
        table_name="labels",
        column_name="value",
        value="100%_complete",
        match_kind="contains",
    )

    sql, purpose = executor.calls[0]
    assert "LIKE LOWER('%100!%!_complete%') ESCAPE '!'" in sql
    assert purpose == "typed_entity_value_contains"
