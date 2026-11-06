# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Table roles, grain and sibling tables.

The catalog read, the writes and the LLM are stubbed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from auto_ontology.semantic import table_roles as tr
from auto_ontology.semantic.models import SiblingTableNotes, TableRoleJudgement


def _table(
    table_id: str,
    name: str,
    columns: list[str],
    *,
    schema: str = "main",
    is_junction_table: bool = False,
) -> dict[str, Any]:
    return {
        "id": table_id,
        "name": name,
        "schema_name": schema,
        "description": "",
        "pk": [],
        "is_junction_table": is_junction_table,
        "columns": [{"name": c, "data_type": "TEXT"} for c in columns],
        "references": [],
        "referenced_by": [],
    }


_ORDER_COLUMNS = ["ORDER_KEY", "CUSTOMER_CODE", "ORDER_DATE", "AMOUNT", "STATUS"]


# ---------------------------------------------------------------------------
# Sibling pairs
# ---------------------------------------------------------------------------


def test_tables_with_near_identical_columns_pair_up() -> None:
    detail = _table("d", "ORDERS", _ORDER_COLUMNS + ["LINE_NUMBER"])
    summary = _table("s", "ORDERS_SUMMARY", _ORDER_COLUMNS + ["LINE_COUNT"])
    other = _table("c", "CUSTOMERS", ["CUSTOMER_CODE", "CUSTOMER_NAME", "REGION"])

    [pair] = tr.find_sibling_pairs([detail, summary, other])

    assert {pair.first["id"], pair.second["id"]} == {"d", "s"}
    assert pair.shared == tuple(sorted(_ORDER_COLUMNS))
    assert pair.first_only == ("LINE_NUMBER",)
    assert pair.second_only == ("LINE_COUNT",)


def test_a_shared_name_token_lowers_the_similarity_needed() -> None:
    shared = ["ORDER_KEY", "CUSTOMER_CODE", "ORDER_DATE"]
    first = _table("a", "ORDERS", shared + [f"A{i}" for i in range(9)])
    second = _table("b", "ORDERS_ALL", shared + [f"B{i}" for i in range(4)])
    unnamed = _table("c", "SHIPMENTS", shared + [f"C{i}" for i in range(4)])

    pairs = tr.find_sibling_pairs([first, second, unnamed])

    assert [(p.first["id"], p.second["id"]) for p in pairs] == [("a", "b")]


def test_too_few_shared_columns_never_pair() -> None:
    first = _table("a", "ORDERS", ["ORDER_KEY", "AMOUNT"])
    second = _table("b", "ORDERS_ALL", ["ORDER_KEY", "AMOUNT"])

    assert tr.find_sibling_pairs([first, second]) == []


def test_tables_in_different_schemas_are_not_siblings() -> None:
    first = _table("a", "ORDERS", _ORDER_COLUMNS, schema="sales")
    second = _table("b", "ORDERS", _ORDER_COLUMNS, schema="archive")

    assert tr.find_sibling_pairs([first, second]) == []


def test_columns_on_most_tables_do_not_make_tables_siblings() -> None:
    housekeeping = ["LOAD_DATE", "LOADED_BY", "SOURCE_SYSTEM"]
    tables = [
        _table(str(i), f"T{i}", housekeeping + [f"T{i}_COL{j}" for j in range(3)])
        for i in range(tr._MIN_UBIQUITOUS_TABLES + 2)
    ]

    assert tr.find_sibling_pairs(tables) == []


def test_each_table_keeps_only_its_closest_siblings() -> None:
    base = [f"COL{i}" for i in range(10)]
    hub = _table("hub", "ORDERS", base)
    copies = [
        _table(f"c{i}", f"ORDERS_{i}", base[: 10 - i] + [f"EXTRA{i}"])
        for i in range(tr._MAX_SIBLINGS_PER_TABLE + 2)
    ]

    pairs = tr.find_sibling_pairs([hub, *copies])
    with_hub = [p for p in pairs if "hub" in (p.first["id"], p.second["id"])]

    kept = {
        p.second["id"] if p.first["id"] == "hub" else p.first["id"] for p in with_hub
    }
    assert kept == {f"c{i}" for i in range(tr._MAX_SIBLINGS_PER_TABLE)}


# ---------------------------------------------------------------------------
# Role and grain
# ---------------------------------------------------------------------------


def _judged(result: Any):
    return (
        patch.object(tr, "get_llm_client"),
        patch.object(tr, "invoke_with_structured_output", return_value=result),
    )


def test_role_and_grain_come_from_the_model() -> None:
    llm, invoke = _judged(
        TableRoleJudgement(role="dimension", grain=" one row per customer ")
    )
    with llm, invoke:
        decision = tr._classify_one(_table("c", "CUSTOMERS", ["CUSTOMER_CODE"]))

    assert decision == (True, {"role": "dimension", "grain": "one row per customer"})


def test_a_compiled_junction_stays_a_junction() -> None:
    llm, invoke = _judged(TableRoleJudgement(role="fact", grain="one row per link"))
    with llm, invoke:
        decision = tr._classify_one(
            _table("j", "ORDER_TAGS", ["ORDER_KEY", "TAG_KEY"], is_junction_table=True)
        )

    assert decision == (True, {"role": "junction", "grain": "one row per link"})


def test_a_failed_call_is_reported_as_unanswered() -> None:
    llm, invoke = _judged(None)
    with llm, invoke:
        assert tr._classify_one(_table("o", "ORDERS", _ORDER_COLUMNS)) == (False, None)


def test_an_empty_answer_leaves_the_table_unclassified() -> None:
    llm, invoke = _judged(TableRoleJudgement(role=None, grain=""))
    with llm, invoke:
        assert tr._classify_one(_table("o", "ORDERS", _ORDER_COLUMNS)) == (True, None)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_classification_writes_roles_and_both_sibling_notes() -> None:
    detail = _table("d", "ORDERS", _ORDER_COLUMNS + ["LINE_NUMBER"])
    summary = _table("s", "ORDERS_SUMMARY", _ORDER_COLUMNS + ["LINE_COUNT"])
    customers = _table("c", "CUSTOMERS", ["CUSTOMER_CODE", "CUSTOMER_NAME"])

    def answer(_llm, _messages, model):
        if model is TableRoleJudgement:
            return TableRoleJudgement(role="fact", grain="one row per order")
        return SiblingTableNotes(
            first_note="Use for order lines.", second_note="Use for order totals."
        )

    written: dict[str, Any] = {}
    with (
        patch.object(
            tr,
            "fetch_tables_for_role_classification",
            return_value=[detail, summary, customers],
        ),
        patch.object(
            tr,
            "set_table_roles",
            side_effect=lambda db, r, keep: written.update(roles=r, kept_roles=keep),
        ),
        patch.object(
            tr,
            "set_table_siblings",
            side_effect=lambda db, s, keep: written.update(siblings=s, kept=keep),
        ),
        patch.object(tr, "get_llm_client"),
        patch.object(tr, "invoke_with_structured_output", side_effect=answer),
    ):
        assert tr.classify_table_roles("shop") == (3, 1)

    assert set(written["roles"]) == {"d", "s", "c"}
    assert sorted(
        (s["table_id"], s["sibling_table_id"], s["note"]) for s in written["siblings"]
    ) == [
        ("d", "s", "Use for order lines."),
        ("s", "d", "Use for order totals."),
    ]


def test_a_pair_the_model_cannot_tell_apart_is_dropped() -> None:
    detail = _table("d", "ORDERS", _ORDER_COLUMNS)
    summary = _table("s", "ORDERS_SUMMARY", _ORDER_COLUMNS)

    def answer(_llm, _messages, model):
        if model is TableRoleJudgement:
            return TableRoleJudgement(role="fact", grain="")
        return SiblingTableNotes(first_note=None, second_note=None)

    written: dict[str, Any] = {}
    with (
        patch.object(
            tr, "fetch_tables_for_role_classification", return_value=[detail, summary]
        ),
        patch.object(tr, "set_table_roles"),
        patch.object(
            tr,
            "set_table_siblings",
            side_effect=lambda db, s, keep: written.update(siblings=s, kept=keep),
        ),
        patch.object(tr, "get_llm_client"),
        patch.object(tr, "invoke_with_structured_output", side_effect=answer),
    ):
        assert tr.classify_table_roles("shop") == (2, 0)

    assert written["siblings"] == []


def test_failed_calls_keep_what_an_earlier_compile_stored() -> None:
    detail = _table("d", "ORDERS", _ORDER_COLUMNS)
    summary = _table("s", "ORDERS_SUMMARY", _ORDER_COLUMNS)

    def answer(_llm, messages, model):
        if model is TableRoleJudgement:
            if "ORDERS_SUMMARY" in messages[-1].content:
                return None
            return TableRoleJudgement(role="fact", grain="one row per order")
        return None

    written: dict[str, Any] = {}
    with (
        patch.object(
            tr, "fetch_tables_for_role_classification", return_value=[detail, summary]
        ),
        patch.object(
            tr,
            "set_table_roles",
            side_effect=lambda db, r, keep: written.update(roles=r, kept_roles=keep),
        ),
        patch.object(
            tr,
            "set_table_siblings",
            side_effect=lambda db, s, keep: written.update(siblings=s, kept=keep),
        ),
        patch.object(tr, "get_llm_client"),
        patch.object(tr, "invoke_with_structured_output", side_effect=answer),
    ):
        assert tr.classify_table_roles("shop") == (1, 0)

    assert set(written["roles"]) == {"d"}
    assert written["kept_roles"] == ["s"]
    assert written["siblings"] == []
    assert written["kept"] == [("d", "s")]
