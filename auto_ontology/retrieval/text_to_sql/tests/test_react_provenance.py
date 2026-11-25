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


def test_sharing_a_table_alone_is_not_a_reference() -> None:
    unrelated = "SELECT COUNT(*) FROM order_lines WHERE quantity < 5"

    assert _stage(custom_analyses_used(_ctx(), unrelated)) == "retrieved"


def test_two_different_ctes_with_the_same_name_are_not_a_reference() -> None:
    saved = "WITH x AS (SELECT id FROM orders) SELECT COUNT(*) FROM x"
    other = "WITH x AS (SELECT id FROM customers) SELECT id FROM x"

    assert _stage(custom_analyses_used(_ctx(saved), other)) == "retrieved"


def test_the_saved_sql_embedded_as_a_cte_counts_as_executed() -> None:
    wrapped = f"WITH b AS ({SAVED}) SELECT * FROM b"

    assert _stage(custom_analyses_used(_ctx(), wrapped)) == "executed"


def test_a_shared_aggregate_with_a_different_filter_is_only_retrieved() -> None:
    saved = "SELECT SUM(quantity) FROM order_lines WHERE quantity >= 20"
    for other in (
        "SELECT SUM(quantity) FROM order_lines WHERE quantity < 20",
        "SELECT SUM(quantity) FROM order_lines WHERE quantity >= 5",
        "SELECT SUM(quantity) FROM order_lines",
    ):
        assert _stage(custom_analyses_used(_ctx(saved), other)) == "retrieved"


def test_an_opposite_equality_filter_is_only_retrieved() -> None:
    saved = "SELECT COUNT(*) FROM orders WHERE status = 'open'"
    other = "SELECT COUNT(*) FROM orders WHERE status = 'closed'"

    assert _stage(custom_analyses_used(_ctx(saved), other)) == "retrieved"


def test_the_kept_filter_with_extra_conditions_is_a_reference() -> None:
    other = "SELECT SUM(quantity) FROM order_lines WHERE quantity >= 20 AND id > 0"

    assert _stage(custom_analyses_used(_ctx(), other)) == "reference"


def test_an_unreferenced_cte_is_not_an_execution() -> None:
    unused = f"WITH b AS ({SAVED}) SELECT 1"

    assert _stage(custom_analyses_used(_ctx(), unused)) != "executed"


def test_the_saved_sql_as_a_subquery_counts_as_executed() -> None:
    wrapped = f"SELECT * FROM ({SAVED}) AS b"

    assert _stage(custom_analyses_used(_ctx(), wrapped)) == "executed"


def _stage_of(saved: str, submitted: str) -> str:
    return _stage(custom_analyses_used(_ctx(saved), submitted))


def test_a_filter_on_another_table_does_not_satisfy_the_saved_filter() -> None:
    saved = (
        "SELECT COUNT(*) FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.status = 'open'"
    )
    on_customers = saved.replace("WHERE o.status", "WHERE c.status")

    assert _stage_of(saved, on_customers) == "retrieved"


def test_the_same_filter_under_other_aliases_still_counts() -> None:
    saved = (
        "SELECT COUNT(*) FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.status = 'open'"
    )
    renamed = (
        "SELECT COUNT(*) FROM orders AS ord JOIN customers AS cu "
        "ON cu.id = ord.customer_id WHERE ord.status = 'open' AND cu.tier = 1"
    )
    bare = "SELECT COUNT(*) FROM orders WHERE status = 'open'"
    semantic = "SELECT COUNT(*) FROM orders WHERE orders__status = 'open'"

    assert _stage_of(saved, renamed) == "reference"
    assert _stage_of(saved, bare) == "reference"
    assert _stage_of(semantic, bare) == "reference"


def test_a_filter_inside_an_unused_cte_is_not_a_filter_of_the_answer() -> None:
    saved = "SELECT SUM(quantity) FROM order_lines WHERE quantity >= 20"
    submitted = (
        "WITH unused AS (SELECT id FROM order_lines WHERE quantity >= 20) "
        "SELECT SUM(quantity) FROM order_lines WHERE quantity < 20"
    )

    assert _stage_of(saved, submitted) == "retrieved"


def test_a_filter_inside_a_cte_the_answer_reads_still_counts() -> None:
    saved = "SELECT SUM(quantity) FROM order_lines WHERE quantity >= 20"
    submitted = (
        "WITH big AS (SELECT quantity FROM order_lines WHERE quantity >= 20) "
        "SELECT SUM(quantity) FROM big"
    )

    assert _stage_of(saved, submitted) == "reference"


def test_a_cte_only_an_unused_cte_reads_did_not_run_the_saved_sql() -> None:
    saved = "SELECT SUM(quantity) AS total FROM order_lines"
    chained = f"WITH b AS ({saved}), unused AS (SELECT * FROM b) SELECT 1 AS one"
    direct = f"WITH b AS ({saved}) SELECT * FROM b"
    through = f"WITH b AS ({saved}), c AS (SELECT * FROM b) SELECT * FROM c"

    assert _stage_of(saved, chained) == "retrieved"
    assert _stage_of(saved, direct) == "executed"
    assert _stage_of(saved, through) == "executed"
