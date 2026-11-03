# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy table detection.

The end-to-end tests run the real probes against a SQLite file; only the
catalog read, the flag write and the LLM are stubbed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from auto_ontology.connectors.sqlite import SQLiteDatabase
from auto_ontology.semantic import legacy_tables as lt
from auto_ontology.semantic.models import LegacyTableJudgement


def _table(
    table_id: str,
    name: str,
    columns: list[str],
    *,
    query_count: int = 0,
    schema: str = "main",
) -> dict[str, Any]:
    return {
        "id": table_id,
        "name": name,
        "schema_name": schema,
        "pk": [],
        "query_count": query_count,
        "columns": [{"name": c, "data_type": "TEXT"} for c in columns],
    }


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_prefix_pair_matches_columns_without_the_tables_own_prefix() -> None:
    first = _table("a", "XA_ITEM", ["XA_ITEM_KEY", "ITEM_CODE", "LOAD_DATE"])
    second = _table("b", "XB_ITEM", ["XB_ITEM_KEY", "ITEM_CODE", "LOAD_DATE"])

    [pair] = lt._find_pairs([first, second])

    assert pair.kind == "prefix"
    assert pair.prefixes == ("XA", "XB")
    assert pair.similarity == 1.0
    assert pair.shared["ITEM_KEY"] == ("XA_ITEM_KEY", "XB_ITEM_KEY")


def test_tables_with_different_columns_are_not_paired() -> None:
    first = _table("a", "XA_ITEM", ["ITEM_CODE", "COLOR", "SIZE"])
    second = _table("b", "XB_ITEM", ["ITEM_CODE", "PRICE", "QTY"])

    assert lt._find_pairs([first, second]) == []


def test_tables_in_different_schemas_are_not_paired() -> None:
    first = _table("a", "XA_ITEM", ["ITEM_CODE"], schema="one")
    second = _table("b", "XB_ITEM", ["ITEM_CODE"], schema="two")

    assert lt._find_pairs([first, second]) == []


def test_marker_pair_pairs_a_table_with_its_marked_copy() -> None:
    current = _table("a", "ORDERS", ["ORDER_ID", "AMOUNT"])
    old = _table("b", "ORDERS_OLD", ["ORDER_ID", "AMOUNT"])

    [pair] = lt._find_pairs([current, old])

    assert pair.kind == "marker"
    lt._score_catalog(pair)
    first_is_old = pair.first["id"] == "b"
    assert (pair.score > 0) == first_is_old


def test_version_tokens_point_to_the_older_version() -> None:
    v1 = _table("a", "ORDERS_V1", ["ORDER_ID", "AMOUNT"])
    v2 = _table("b", "ORDERS_V2", ["ORDER_ID", "AMOUNT"])

    [pair] = lt._find_pairs([v1, v2])
    lt._score_catalog(pair)

    assert pair.first["id"] == "a"
    assert pair.score > 0


# ---------------------------------------------------------------------------
# Catalog and family evidence
# ---------------------------------------------------------------------------


def test_usage_and_extra_columns_favour_the_current_table() -> None:
    first = _table("a", "XA_ITEM", ["ITEM_CODE", "A", "B", "C", "D", "E"])
    second = _table(
        "b",
        "XB_ITEM",
        ["ITEM_CODE", "A", "B", "C", "D", "E", "F"],
        query_count=5,
    )
    [pair] = lt._find_pairs([first, second])

    lt._score_catalog(pair)

    assert pair.score == lt._W_COLUMNS + lt._W_USAGE


def test_family_evidence_counts_history_tables_and_outside_references() -> None:
    tables = [
        _table("a1", "XA_ITEM", ["ITEM_CODE"]),
        _table("b1", "XB_ITEM", ["ITEM_CODE"]),
        _table("a2", "XA_STORE", ["STORE_CODE"]),
        _table("b2", "XB_STORE", ["STORE_CODE"]),
        _table("h1", "XB_ITEM_HIST", ["ITEM_CODE", "PERIOD"]),
        _table("r1", "SALES", ["SALE_ID", "XB_ITEM_KEY"]),
    ]
    pairs = lt._find_pairs(tables)

    lt._score_families(pairs, tables)

    assert len(pairs) == 2
    assert all(p.score == 2 * lt._W_FAMILY for p in pairs)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def _scored_pair(score: float, *, key_overlap: float | None = None) -> lt._Pair:
    [pair] = lt._find_pairs(
        [_table("a", "XA_ITEM", ["ITEM_CODE"]), _table("b", "XB_ITEM", ["ITEM_CODE"])]
    )
    pair.score = score
    pair.key_overlap = key_overlap
    return pair


def test_a_decisive_score_needs_no_llm() -> None:
    with patch.object(lt, "_judge_with_llm") as judge:
        decided = lt._decide([_scored_pair(3.0)])

    judge.assert_not_called()
    assert decided["a"][0] == "b"


def test_an_inconclusive_score_goes_to_the_llm_and_null_leaves_it_unmarked() -> None:
    with (
        patch.object(lt, "get_llm_client"),
        patch.object(
            lt,
            "invoke_with_structured_output",
            return_value=LegacyTableJudgement(legacy_table_id=None, reason="unsure"),
        ) as invoke,
    ):
        decided = lt._decide([_scored_pair(1.0)])

    invoke.assert_called_once()
    assert decided == {}


def test_the_llm_cannot_name_a_table_outside_the_pair() -> None:
    with (
        patch.object(lt, "get_llm_client"),
        patch.object(
            lt,
            "invoke_with_structured_output",
            return_value=LegacyTableJudgement(legacy_table_id="zzz", reason=""),
        ),
    ):
        assert lt._decide([_scored_pair(0.0)]) == {}


def test_low_key_overlap_blocks_even_a_decisive_score() -> None:
    with patch.object(lt, "_judge_with_llm") as judge:
        decided = lt._decide([_scored_pair(9.0, key_overlap=0.1)])

    judge.assert_not_called()
    assert decided == {}


def test_a_confirmed_prefix_marks_unpaired_tables_with_it() -> None:
    tables = [
        _table("a1", "XA_ITEM", ["ITEM_CODE"]),
        _table("b1", "XB_ITEM", ["ITEM_CODE"]),
        _table("a2", "XA_STORE", ["STORE_CODE"]),
        _table("b2", "XB_STORE", ["STORE_CODE"]),
        # Drifted too far to pair, and one with no counterpart at all.
        _table("a3", "XA_ZONE", ["ZONE_CODE", "P", "Q", "R"]),
        _table("b3", "XB_ZONE", ["ZONE_CODE", "S", "T", "U"]),
        _table("a4", "XA_LONELY", ["LONELY_CODE"]),
        _table("o1", "OTHER", ["X"]),
    ]
    decided = {"a1": ("b1", "evidence"), "a2": ("b2", "evidence")}

    extra = lt._generalise(decided, lt._find_pairs(tables), tables)

    assert extra["a3"][0] == "b3"
    assert extra["a4"][0] is None
    assert set(extra) == {"a3", "a4"}


def test_one_decided_prefix_pair_is_not_a_pattern() -> None:
    tables = [
        _table("a1", "XA_ITEM", ["ITEM_CODE"]),
        _table("b1", "XB_ITEM", ["ITEM_CODE"]),
        _table("a4", "XA_LONELY", ["LONELY_CODE"]),
    ]

    extra = lt._generalise({"a1": ("b1", "")}, lt._find_pairs(tables), tables)

    assert extra == {}


def test_a_confirmed_marker_marks_other_tables_carrying_it() -> None:
    tables = [
        _table("c", "ORDERS", ["ORDER_ID"]),
        _table("o", "ORDERS_OLD", ["ORDER_ID"]),
        _table("p", "PAYMENTS_OLD", ["PAYMENT_ID"]),
    ]

    extra = lt._generalise({"o": ("c", "")}, [], tables)

    assert set(extra) == {"p"}


def test_chains_point_to_the_final_replacement_and_cycles_are_dropped() -> None:
    chained = lt._resolve_chains({"a": ("b", ""), "b": ("c", "")})
    assert {d["table_id"]: d["replacement_table_id"] for d in chained} == {
        "a": "c",
        "b": "c",
    }
    assert lt._resolve_chains({"a": ("b", ""), "b": ("a", "")}) == []


# ---------------------------------------------------------------------------
# End to end against SQLite
# ---------------------------------------------------------------------------


def _build_sqlite(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE XA_ITEM (XA_ITEM_KEY INT, ITEM_CODE TEXT, ITEM_NAME TEXT,
                              PRICE INT, LOAD_DATE TEXT);
        CREATE TABLE XB_ITEM (XB_ITEM_KEY INT, ITEM_CODE TEXT, ITEM_NAME TEXT,
                              PRICE INT, COLOR TEXT, LOAD_DATE TEXT);
        CREATE TABLE XB_ITEM_HIST (XB_ITEM_KEY INT, ITEM_CODE TEXT, PERIOD TEXT);
        CREATE TABLE XA_STORE (XA_STORE_KEY INT, STORE_CODE TEXT, LOAD_DATE TEXT);
        CREATE TABLE XB_STORE (XB_STORE_KEY INT, STORE_CODE TEXT, LOAD_DATE TEXT);
        CREATE TABLE XB_STORE_HIST (XB_STORE_KEY INT, STORE_CODE TEXT, PERIOD TEXT);
        CREATE TABLE XA_LONELY (LONELY_CODE TEXT);
        CREATE TABLE SALES (SALE_ID INT, XB_ITEM_KEY INT);
        CREATE TABLE YA_PART (PART_CODE TEXT, WEIGHT INT);
        CREATE TABLE YB_PART (PART_CODE TEXT, WEIGHT INT);
        """
    )
    for i in range(10):
        conn.execute(
            "INSERT INTO XB_ITEM VALUES (?, ?, ?, 5, 'red', '2026-06-01')",
            (i, f"I{i}", f"item {i}"),
        )
        conn.execute("INSERT INTO XB_STORE VALUES (?, ?, '2026-06-01')", (i, f"S{i}"))
        conn.execute("INSERT INTO YA_PART VALUES (?, 1)", (f"Q{i}",))
        conn.execute("INSERT INTO YB_PART VALUES (?, 2)", (f"P{i}",))
    for i in range(6):
        conn.execute(
            "INSERT INTO XA_ITEM VALUES (?, ?, ?, 5, '2024-01-01')",
            (100 + i, f"I{i}", f"item {i}"),
        )
        conn.execute(
            "INSERT INTO XA_STORE VALUES (?, ?, '2024-01-01')", (100 + i, f"S{i}")
        )
    conn.commit()
    conn.close()


def _catalog() -> list[dict[str, Any]]:
    return [
        _table(
            "xa_item",
            "XA_ITEM",
            ["XA_ITEM_KEY", "ITEM_CODE", "ITEM_NAME", "PRICE", "LOAD_DATE"],
        ),
        _table(
            "xb_item",
            "XB_ITEM",
            ["XB_ITEM_KEY", "ITEM_CODE", "ITEM_NAME", "PRICE", "COLOR", "LOAD_DATE"],
        ),
        _table("xb_item_hist", "XB_ITEM_HIST", ["XB_ITEM_KEY", "ITEM_CODE", "PERIOD"]),
        _table("xa_store", "XA_STORE", ["XA_STORE_KEY", "STORE_CODE", "LOAD_DATE"]),
        _table("xb_store", "XB_STORE", ["XB_STORE_KEY", "STORE_CODE", "LOAD_DATE"]),
        _table(
            "xb_store_hist", "XB_STORE_HIST", ["XB_STORE_KEY", "STORE_CODE", "PERIOD"]
        ),
        _table("xa_lonely", "XA_LONELY", ["LONELY_CODE"]),
        _table("sales", "SALES", ["SALE_ID", "XB_ITEM_KEY"]),
        _table("ya_part", "YA_PART", ["PART_CODE", "WEIGHT"]),
        _table("yb_part", "YB_PART", ["PART_CODE", "WEIGHT"]),
    ]


@pytest.fixture
def sqlite_db(tmp_path: Path) -> SQLiteDatabase:
    path = tmp_path / "shop.sqlite"
    _build_sqlite(path)
    return SQLiteDatabase(str(path))


def test_detects_the_legacy_family_end_to_end(sqlite_db: SQLiteDatabase) -> None:
    writes: list[list[dict[str, Any]]] = []
    with (
        patch.object(lt, "fetch_tables_for_legacy_detection", return_value=_catalog()),
        patch.object(lt, "set_legacy_tables", lambda _db, d: writes.append(d)),
        patch.object(lt, "_resolve_connector", return_value=sqlite_db),
        patch.object(lt, "_judge_with_llm", return_value=None) as judge,
    ):
        first = lt.detect_legacy_tables("shop")
        second = lt.detect_legacy_tables("shop")

    marked = {d["table_id"]: d["replacement_table_id"] for d in writes[0]}
    assert marked == {
        "xa_item": "xb_item",
        "xa_store": "xb_store",
        "xa_lonely": None,
    }
    # YA_PART / YB_PART share no key values, so they never reach the LLM.
    judge.assert_not_called()
    assert first == second == 3
    assert writes[0] == writes[1]


def test_without_a_connector_the_catalog_evidence_still_decides() -> None:
    writes: list[list[dict[str, Any]]] = []
    with (
        patch.object(lt, "fetch_tables_for_legacy_detection", return_value=_catalog()),
        patch.object(lt, "set_legacy_tables", lambda _db, d: writes.append(d)),
        patch.object(lt, "_resolve_connector", return_value=None),
        patch.object(lt, "_judge_with_llm", return_value=None),
    ):
        lt.detect_legacy_tables("shop")

    marked = {d["table_id"] for d in writes[0]}
    assert {"xa_item", "xa_store"} <= marked
    assert not marked & {"xb_item", "xb_store", "ya_part", "yb_part"}


def test_no_pairs_clears_previous_flags() -> None:
    writes: list[list[dict[str, Any]]] = []
    with (
        patch.object(
            lt,
            "fetch_tables_for_legacy_detection",
            return_value=[_table("t", "ONLY", ["X"])],
        ),
        patch.object(lt, "set_legacy_tables", lambda _db, d: writes.append(d)),
    ):
        assert lt.detect_legacy_tables("shop") == 0

    assert writes == [[]]
