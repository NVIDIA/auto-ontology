# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.provenance import (
    custom_analyses_used,
)

# The saved definition names a derived column that does not exist physically.
SAVED = (
    "SELECT SUM(order_line__quantity) FROM order_lines WHERE order_line__quantity >= 20"
)
# The obvious physical-table answer the agent writes instead.
PHYSICAL = "SELECT SUM(quantity) FROM order_lines WHERE quantity >= 20"


def _ctx(sql: str = SAVED) -> ToolContext:
    ctx = ToolContext(connectors=[])
    ctx.shown_analyses["a1"] = {"id": "a1", "name": "bulk_units", "sql": sql}
    return ctx


def _stage(entries: list[dict]) -> str:
    assert len(entries) == 1
    return entries[0]["stage"]


def test_nothing_retrieved_is_an_empty_list() -> None:
    assert custom_analyses_used(ToolContext(connectors=[]), PHYSICAL) == []


def test_an_analysis_the_model_only_saw_is_retrieved() -> None:
    assert _stage(custom_analyses_used(_ctx(), None)) == "retrieved"
    assert _stage(custom_analyses_used(_ctx(), "SELECT 1 FROM customers")) == (
        "retrieved"
    )


def test_a_generated_repair_is_a_reference_not_an_execution() -> None:
    entries = custom_analyses_used(_ctx(), PHYSICAL)

    assert _stage(entries) == "reference"
    assert entries[0]["id"] == "a1"
    assert entries[0]["name"] == "bulk_units"


def test_the_saved_sql_run_as_it_is_counts_as_executed() -> None:
    reformatted = SAVED.lower().replace(" FROM", "\nFROM") + ";"

    assert _stage(custom_analyses_used(_ctx(), SAVED)) == "executed"
    assert _stage(custom_analyses_used(_ctx(), reformatted)) == "executed"


def test_the_fingerprint_follows_the_saved_definition_not_the_submission() -> None:
    first = custom_analyses_used(_ctx(), PHYSICAL)[0]["fingerprint"]
    assert custom_analyses_used(_ctx(), SAVED)[0]["fingerprint"] == first
    assert custom_analyses_used(_ctx(PHYSICAL), SAVED)[0]["fingerprint"] != first


def test_one_of_two_analyses_can_be_executed_while_the_other_is_context() -> None:
    ctx = _ctx()
    ctx.shown_analyses["a2"] = {"id": "a2", "name": "other", "sql": PHYSICAL}

    stages = {e["id"]: e["stage"] for e in custom_analyses_used(ctx, PHYSICAL)}

    assert stages == {"a1": "reference", "a2": "executed"}
