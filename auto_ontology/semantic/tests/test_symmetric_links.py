# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric link tables are told apart from directed ones by the data alone."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from auto_ontology.semantic.symmetric_links import (
    PROBE_SAMPLE_ROWS,
    candidate_column_pairs,
    detect_symmetric_column_pairs,
    mirror_probe_sql,
)
from auto_ontology.utils.sql_identifiers import quoted_identifier


class _SqliteConnector:
    dialect = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.queries: list[str] = []

    def qualify(self, schema: str | None, table: str) -> str:
        return quoted_identifier(table, self.dialect)

    def execute(self, sql: str) -> pd.DataFrame:
        self.queries.append(sql)
        with sqlite3.connect(self.path) as conn:
            return pd.read_sql_query(sql, conn)


def _execute(connector: _SqliteConnector, sql: str) -> pd.DataFrame:
    return connector.execute(sql)


def _link_table(
    tmp_path: Path, name: str, rows: list[tuple], with_key: bool = False
) -> _SqliteConnector:
    path = tmp_path / "links.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE node (node_id TEXT PRIMARY KEY)")
        key = ", link_id TEXT" if with_key else ""
        conn.execute(f'CREATE TABLE "{name}" (a TEXT, b TEXT{key})')
        marks = ", ".join("?" for _ in rows[0]) if rows else "?, ?"
        conn.executemany(f'INSERT INTO "{name}" VALUES ({marks})', rows)
    return _SqliteConnector(path)


def _columns(with_key: bool = False) -> list[dict[str, Any]]:
    columns = [
        {"name": "a", "data_type": "TEXT"},
        {"name": "b", "data_type": "TEXT"},
    ]
    if with_key:
        columns.append({"name": "link_id", "data_type": "TEXT"})
    return columns


def _fks(with_key: bool = False) -> list[dict[str, Any]]:
    fks = [
        {"source_column": c, "target_table": "node", "target_column": "node_id"}
        for c in ("a", "b")
    ]
    if with_key:
        fks.append(
            {"source_column": "link_id", "target_table": "link", "target_column": "id"}
        )
    return fks


def _detect(connector, name: str, with_key: bool = False, **table: Any):
    return detect_symmetric_column_pairs(
        {"name": name, **table},
        _columns(with_key),
        _fks(with_key),
        set(),
        connector,
        _execute,
    )


def _mirrored(count: int, with_key: bool = False) -> list[tuple]:
    rows = []
    for i in range(count):
        key = (f"l{i}",) if with_key else ()
        rows.append((f"n{i}", f"m{i}", *key))
        rows.append((f"m{i}", f"n{i}", *key))
    return rows


def test_a_fully_mirrored_table_is_detected(tmp_path: Path) -> None:
    connector = _link_table(tmp_path, "friends", _mirrored(15))
    assert _detect(connector, "friends") == [["a", "b"]]


def test_a_directed_table_is_not_detected(tmp_path: Path) -> None:
    rows = [(f"u{i}", f"u{i + 1}") for i in range(30)]
    connector = _link_table(tmp_path, "follows", rows)
    assert _detect(connector, "follows") == []


def test_a_mostly_mirrored_table_is_not_detected(tmp_path: Path) -> None:
    rows = _mirrored(19) + [(f"x{i}", f"y{i}") for i in range(2)]
    assert len(rows) == 40
    connector = _link_table(tmp_path, "similar", rows)
    assert _detect(connector, "similar") == []


def test_a_table_under_twenty_rows_is_skipped(tmp_path: Path) -> None:
    connector = _link_table(tmp_path, "pairs", _mirrored(9))
    assert _detect(connector, "pairs") == []


def test_self_loops_count_as_mirrored(tmp_path: Path) -> None:
    rows = _mirrored(10) + [("s", "s")]
    connector = _link_table(tmp_path, "linked", rows)
    assert _detect(connector, "linked") == [["a", "b"]]


def test_the_other_key_column_must_match_too(tmp_path: Path) -> None:
    mirrored = _link_table(tmp_path, "connected", _mirrored(15, True), with_key=True)
    assert _detect(mirrored, "connected", with_key=True) == [["a", "b"]]
    assert '"link_id"' in mirrored.queries[0]


def test_a_mirror_under_another_key_is_not_a_mirror(tmp_path: Path) -> None:
    rows = []
    for i in range(15):
        rows.append((f"n{i}", f"m{i}", f"l{i}"))
        rows.append((f"m{i}", f"n{i}", f"other{i}"))
    connector = _link_table(tmp_path, "connected", rows, with_key=True)
    assert _detect(connector, "connected", with_key=True) == []


def test_a_row_id_primary_key_is_not_matched(tmp_path: Path) -> None:
    connector = _link_table(tmp_path, "friends", _mirrored(15))
    assert _detect(connector, "friends", pk=["id"]) == [["a", "b"]]
    assert '"id"' not in connector.queries[0]


def test_a_failing_probe_skips_the_table(tmp_path: Path) -> None:
    connector = _link_table(tmp_path, "friends", _mirrored(15))

    def _timeout(connector, sql):
        raise TimeoutError("probe exceeded 120s")

    assert (
        detect_symmetric_column_pairs(
            {"name": "friends"}, _columns(), _fks(), set(), connector, _timeout
        )
        == []
    )


def test_only_columns_sharing_a_target_are_paired() -> None:
    columns = _columns(with_key=True)
    assert candidate_column_pairs(
        {"name": "connected"}, columns, _fks(with_key=True)
    ) == [("a", "b")]


def test_three_columns_on_one_target_are_not_paired() -> None:
    columns = [{"name": f"player_{i}", "data_type": "INTEGER"} for i in range(3)]
    fks = [
        {"source_column": c["name"], "target_table": "player", "target_column": "id"}
        for c in columns
    ]
    assert candidate_column_pairs({"name": "match"}, columns, fks) == []


def test_two_columns_on_the_tables_own_key_are_not_paired() -> None:
    assert candidate_column_pairs({"name": "node"}, _columns(), _fks()) == []


def test_suggested_foreign_keys_are_not_paired(tmp_path: Path) -> None:
    connector = _link_table(tmp_path, "friends", _mirrored(15))

    assert (
        detect_symmetric_column_pairs(
            {"name": "friends"}, _columns(), [], {"a", "b"}, connector, _execute
        )
        == []
    )
    assert connector.queries == []


def test_only_a_sample_of_rows_is_checked(tmp_path: Path) -> None:
    connector = _link_table(tmp_path, "friends", _mirrored(PROBE_SAMPLE_ROWS))

    assert _detect(connector, "friends") == [["a", "b"]]
    assert f"LIMIT {PROBE_SAMPLE_ROWS}) t" in connector.queries[0]


def test_a_table_without_two_links_to_one_target_is_never_probed(
    tmp_path: Path,
) -> None:
    connector = _link_table(tmp_path, "orders", _mirrored(15))
    fks = [{"source_column": "a", "target_table": "node", "target_column": "node_id"}]

    assert (
        detect_symmetric_column_pairs(
            {"name": "orders"}, _columns(), fks, set(), connector, _execute
        )
        == []
    )
    assert connector.queries == []


@pytest.mark.parametrize("dialect", ["sqlite", "postgres", "mysql", "spark"])
def test_the_probe_quotes_for_the_dialect_and_casts_nothing(dialect: str) -> None:
    sql = mirror_probe_sql("t_ref", ("a", "b"), ["k"], dialect)
    quoted = quoted_identifier("a", dialect)
    assert f"r.{quoted} = t.{quoted_identifier('b', dialect)}" in sql
    assert "CASE WHEN EXISTS" in sql
    assert "::" not in sql and "CAST" not in sql.upper()
