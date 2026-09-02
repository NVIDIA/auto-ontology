# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seams between the pieces, which each piece's own tests do not reach.

Every part of this can be correct on its own and still be wired up wrongly: a
cache key that never receives the connector it distinguishes, a projection that
is built and then not used, a per-request field replaced on the wrong object.
These go through the public entry point.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from gsf.retrieval.kumo import predictor
from gsf.retrieval.kumo.graph_cache import _term

CATALOG = [
    {
        "name": "customers",
        "schema_name": "main",
        "database_name": "sales",
        "pk": ["customer_id"],
        "columns": [
            {"name": "customer_id", "data_type": "int"},
            {"name": "tier", "data_type": "text"},
        ],
    },
    {
        "name": "orders",
        "schema_name": "main",
        "database_name": "sales",
        "pk": ["order_id"],
        "columns": [
            {"name": "order_id", "data_type": "int"},
            {"name": "customer_id", "data_type": "int"},
            {"name": "amount", "data_type": "double"},
            {"name": "placed_at", "data_type": "timestamp"},
        ],
    },
]
JOINS = [
    {
        "path": [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "customer_id",
            }
        ]
    }
]


class _Warehouse:
    database_name = "sales"
    dialect = "sqlite"

    def __init__(self, connection_string: str = "sqlite:///sales.db") -> None:
        self._connection_string = connection_string
        self.queries: list[str] = []

    def execute(self, sql: str) -> pd.DataFrame:
        self.queries.append(sql)
        if "customers" in sql:
            return pd.DataFrame(
                {"customer_id": range(30), "tier": ["Enterprise", "SMB"] * 15}
            )
        return pd.DataFrame(
            {
                "order_id": range(90),
                "customer_id": [i % 30 for i in range(90)],
                "amount": [10.0 + i for i in range(90)],
                "placed_at": pd.to_datetime(["2025-06-01"] * 90),
            }
        )


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No live NIM. What is under test is the wiring, not the model."""

    class Client:
        def relational(self, graph: Any) -> object:
            return object()

    monkeypatch.setattr(predictor, "_ensure_init", Client)


_DDL = (
    '"customers"("customer_id" ID, "tier" categorical)  -- PRIMARY KEY ("customer_id")\n'
    '"orders"("order_id" ID, "customer_id" ID, "amount" numerical, '
    '"placed_at" timestamp)  -- PRIMARY KEY ("order_id"); TIME COLUMN ("placed_at")'
)


class _LLM:
    model_name = "stub"


def _llm(pql: str) -> Any:
    """An LLM that answers with one fixed PQL statement."""

    class Fixed:
        model_name = "stub"

        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            return type("Reply", (), {"content": pql})()

    return Fixed()


class _KumoModel:
    """Accepts any PQL, so the refusal is what decides the outcome."""

    def validate_pql(self, query: str) -> None:
        return None

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("a refused question must never reach the model")


def _build(warehouse: Any, examples: list[dict[str, str]]) -> Any:
    return predictor.build_prediction_context([warehouse], CATALOG, JOINS, examples)


def _build_no_joins(warehouse: Any, examples: list[dict[str, str]]) -> Any:
    """No catalog joins, so the engine has to guess the relationships."""
    return predictor.build_prediction_context([warehouse], CATALOG, None, examples)


def test_only_the_columns_the_catalog_names_are_read() -> None:
    """A projection that is computed and then not used reads the whole table."""
    predictor._GRAPH_CACHE.clear()
    warehouse = _Warehouse()

    _build(warehouse, [])

    reads = [q for q in warehouse.queries if q.upper().startswith("SELECT")]
    assert reads
    for query in reads:
        selected = query.split("FROM")[0]
        assert "*" not in selected, query
        assert "customer_id" in selected


def test_a_projection_falls_back_when_the_catalog_names_no_columns() -> None:
    """Reading too much beats building a graph missing a column it needed."""
    assert predictor._projection({"name": "orders"}) == "*"
    assert predictor._projection({"name": "orders", "columns": []}) == "*"
    assert "*" not in predictor._projection(
        {"name": "orders", "columns": [{"name": "amount"}]}
    )


def test_the_key_actually_carries_the_connector_it_distinguishes() -> None:
    """_connector_identity can be right while nothing passes its result on."""
    one = predictor._cache_key([_Warehouse("sqlite:///prod.db")], CATALOG, JOINS)
    other = predictor._cache_key([_Warehouse("sqlite:///staging.db")], CATALOG, JOINS)

    assert one is not None and other is not None
    assert one.connector != other.connector
    assert one != other


def test_two_deployments_do_not_answer_each_others_questions() -> None:
    """The whole point of the connector in the key, through the entry point."""
    predictor._GRAPH_CACHE.clear()
    prod = _Warehouse("sqlite:///prod.db")
    staging = _Warehouse("sqlite:///staging.db")

    _build(prod, [])
    _build(staging, [])

    assert staging.queries, "staging was served prod's graph"


def test_a_follow_up_gets_its_own_examples_through_the_entry_point() -> None:
    """The cached context is shared; the examples on it must not be."""
    predictor._GRAPH_CACHE.clear()

    first = _build(_Warehouse(), [{"question": "first", "pql": "PREDICT a"}])
    second = _build(_Warehouse(), [{"question": "second", "pql": "PREDICT b"}])

    assert second.identity.cache == "hit"
    assert [e["question"] for e in first.examples] == ["first"]
    assert [e["question"] for e in second.examples] == ["second"]


def test_a_value_cannot_pose_as_two_fields() -> None:
    """Joining on a separator lets a value containing it forge another schema."""
    assert _term("a:b") != _term("a") + _term("b")
    assert _term("ab") + _term("c") != _term("a") + _term("bc")
    assert _term("") != _term(None) or _term(None) == "0:"


def test_a_renamed_table_is_not_the_same_schema() -> None:
    """The collision the length prefix exists to stop, through the real key."""
    shifted = [
        {**CATALOG[0], "name": "customer", "schema_name": "smain"},
        CATALOG[1],
    ]

    one = predictor._cache_key([_Warehouse()], CATALOG, JOINS)
    other = predictor._cache_key([_Warehouse()], shifted, JOINS)

    assert one is not None and other is not None
    assert one.schema_fingerprint != other.schema_fingerprint


class _TooMuchData(_Warehouse):
    """Returns a frame far larger than any sane byte budget."""

    def execute(self, sql: str) -> pd.DataFrame:
        self.queries.append(sql)
        return pd.DataFrame({"blob": ["x" * 500] * 40_000})


def test_a_refusal_over_budget_reaches_the_caller_as_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Spend tests prove it raises. This proves the raise is not swallowed."""
    monkeypatch.setenv("KUMO_MAX_BYTES", str(4 * 1024 * 1024))
    predictor._GRAPH_CACHE.clear()

    answer = _build(_TooMuchData(), [])

    assert isinstance(answer, dict), "an over-budget request was answered anyway"
    assert "more than it can hold in memory" in answer["response"]


def test_a_refusal_over_the_table_budget_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUMO_MAX_TABLES", "1")
    predictor._GRAPH_CACHE.clear()

    answer = _build(_Warehouse(), [])

    assert isinstance(answer, dict)
    assert "more than 1 table" in answer["response"]


def test_giving_up_on_a_hung_build_reaches_the_caller_as_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BuildTimedOut must become a graceful answer, not escape as an exception."""
    from gsf.retrieval.kumo.graph_cache import BuildTimedOut

    def timed_out(self: Any, key: Any, build: Any, **kwargs: Any) -> Any:
        raise BuildTimedOut("Preparing this prediction is taking longer than 300s")

    monkeypatch.setattr(predictor.GraphCache, "get_or_build", timed_out)
    predictor._GRAPH_CACHE.clear()

    answer = _build(_Warehouse(), [])

    assert isinstance(answer, dict)
    assert "taking longer" in answer["response"]


def test_a_population_too_large_is_refused_not_annotated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal downgraded to a note would answer over a slice and look fine."""
    from gsf.retrieval.kumo import pql_gen

    monkeypatch.setattr(
        pql_gen, "_resolve_indices", lambda *a, **k: ([1, 2, 3], 48_391)
    )

    result = pql_gen.generate_pql(
        "what will total revenue be?",
        llm=_llm("PREDICT SUM(orders.amount, 0, 30) FOR EACH customers.customer_id"),
        kumo_model=_KumoModel(),
        connector=_Warehouse(),
        graph_ddl=_DDL,
        graph_edges=[("orders", "customer_id", "customers")],
        graph_col_stypes={"customers": {"customer_id": "ID"}, "orders": {}},
        time_columns={"orders": "placed_at"},
        group_by="tier",
        max_tries=1,
    )

    assert result.success is False
    assert "48,391" in (result.error or "")
    assert not result.rows


def test_every_prediction_leaves_a_record(caplog: Any) -> None:
    """A record emitted nowhere explains nothing when an answer is questioned."""
    import json
    import logging

    from gsf.retrieval.kumo import pql_gen
    from gsf.retrieval.kumo.pql_gen import PqlGenerationResult

    predictor._GRAPH_CACHE.clear()
    context = _build(_Warehouse(), [])
    original = pql_gen.generate_pql
    pql_gen.generate_pql = lambda *a, **k: PqlGenerationResult(
        question="q",
        success=True,
        attempts=1,
        num_entities=2,
        population=2,
        rows=[{"a": 1}],
        columns=["a"],
        pql="PREDICT x FOR customers.customer_id IN (1)",
    )
    try:
        with caplog.at_level(logging.INFO, logger="gsf.retrieval.kumo.telemetry"):
            predictor.run_prediction("q", _LLM(), context)
    finally:
        pql_gen.generate_pql = original

    written = [
        json.loads(r.getMessage().removeprefix("kumo.run "))
        for r in caplog.records
        if r.getMessage().startswith("kumo.run ")
    ]
    assert len(written) == 1
    assert written[0]["outcome"] == "answered"


class _OneTableFails(_Warehouse):
    """The customers table reads; the orders table does not."""

    def execute(self, sql: str) -> pd.DataFrame:
        if "orders" in sql:
            raise RuntimeError("connection reset by peer")
        return super().execute(sql)


def test_a_table_that_cannot_be_read_stops_the_prediction() -> None:
    """Building from the tables that did load answers a narrower question silently."""
    predictor._GRAPH_CACHE.clear()

    answer = _build(_OneTableFails(), [])

    assert isinstance(answer, dict), "a partial graph was built and returned"
    assert "orders" in answer["response"]
    assert "could not be read" in answer["response"]


def test_a_table_that_reads_back_empty_is_not_a_failure() -> None:
    """Empty is an answer; unreadable is not. They must not be conflated."""

    class Empty(_Warehouse):
        def execute(self, sql: str) -> pd.DataFrame:
            if "orders" in sql:
                return pd.DataFrame()
            return super().execute(sql)

    predictor._GRAPH_CACHE.clear()

    answer = _build(Empty(), [])

    assert not isinstance(answer, dict) or "could not be read" not in answer.get(
        "response", ""
    )


def test_a_graph_whose_edges_were_guessed_is_not_held() -> None:
    """Inference can pick different edges for the same tables, so freezing one
    serves whichever build won the race to everyone for the whole TTL."""
    predictor._GRAPH_CACHE.clear()

    first = _build_no_joins(_Warehouse(), [])
    second = _build_no_joins(_Warehouse(), [])

    assert first.identity.edges_from == "inferred"
    assert second.identity.cache != "hit", "a guessed graph was served from cache"


def test_a_graph_built_from_catalog_joins_is_held() -> None:
    """Those edges were stated, so a second request gets the same shape."""
    predictor._GRAPH_CACHE.clear()

    first = _build(_Warehouse(), [])
    second = _build(_Warehouse(), [])

    assert first.identity.edges_from == "catalog"
    assert second.identity.cache == "hit"


def test_a_prediction_that_raised_still_leaves_a_record(caplog: Any) -> None:
    """The module promises a record whether it answered, refused, or failed."""
    import json
    import logging

    from gsf.retrieval.kumo import pql_gen

    predictor._GRAPH_CACHE.clear()
    context = _build(_Warehouse(), [])
    original = pql_gen.generate_pql

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the gateway timed out after 60s")

    pql_gen.generate_pql = boom
    try:
        with caplog.at_level(logging.INFO, logger="gsf.retrieval.kumo.telemetry"):
            with pytest.raises(RuntimeError):
                predictor.run_prediction("who churns?", _LLM(), context)
    finally:
        pql_gen.generate_pql = original

    written = [
        json.loads(r.getMessage().removeprefix("kumo.run "))
        for r in caplog.records
        if r.getMessage().startswith("kumo.run ")
    ]
    assert len(written) == 1
    assert written[0]["outcome"] == "failed"
    assert "gateway timed out" in written[0]["error"]
    assert written[0]["graph"]["fingerprint"] == context.identity.fingerprint


def _records(caplog: Any) -> list[dict[str, Any]]:
    import json

    return [
        json.loads(r.getMessage().removeprefix("kumo.run "))
        for r in caplog.records
        if r.getMessage().startswith("kumo.run ")
    ]


def test_a_request_refused_before_predicting_still_leaves_a_record(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """A refusal exits before run_prediction, so nothing recorded it at all."""
    import logging

    monkeypatch.setenv("KUMO_MAX_TABLES", "1")
    predictor._GRAPH_CACHE.clear()

    with caplog.at_level(logging.INFO, logger="gsf.retrieval.kumo.telemetry"):
        answer = _build(_Warehouse(), [])

    assert isinstance(answer, dict)
    written = _records(caplog)
    assert len(written) == 1
    assert written[0]["outcome"] == "refused"
    assert "more than 1 table" in written[0]["error"]


def test_an_unreadable_table_leaves_a_record(caplog: Any) -> None:
    import logging

    predictor._GRAPH_CACHE.clear()

    with caplog.at_level(logging.INFO, logger="gsf.retrieval.kumo.telemetry"):
        _build(_OneTableFails(), [])

    written = _records(caplog)
    assert written and written[0]["outcome"] == "refused"
    assert "could not be read" in written[0]["error"]


def test_a_refusal_record_carries_no_unredacted_values(caplog: Any) -> None:
    """The refusal message is prose, but it goes through the same redaction."""
    import inspect

    assert "redact_error(" in inspect.getsource(predictor._refused)


def test_the_deadline_is_checked_after_the_reads_finish() -> None:
    """A build that overran during graph construction used to run on unchecked."""
    import inspect

    source = inspect.getsource(predictor._build_context_within_budget)

    for phase in ("declaring keys", "building the graph", "creating the model"):
        assert f'check_deadline("{phase}")' in source, phase


def test_a_build_that_overruns_after_reading_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUMO_MAX_BUILD_SECONDS", "1")
    predictor._GRAPH_CACHE.clear()

    class Slow(_Warehouse):
        def execute(self, sql: str) -> pd.DataFrame:
            import time

            time.sleep(0.7)
            return super().execute(sql)

    answer = _build(Slow(), [])

    assert isinstance(answer, dict)
    assert "passed" in answer["response"] or "narrower" in answer["response"]
